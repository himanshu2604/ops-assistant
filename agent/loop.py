"""Hand-built agent loop: prompt in, tool call out, tool result back in, repeat.

`client` is anything shaped like the OpenAI SDK client (client.chat.completions.create).
Keeping it injected means tests can script the model, and a Galileo-wrapped client can be
dropped in later without touching this file.
"""
import json
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .tools import run_tool, tool_specs

DEFAULT_MAX_STEPS = 6


@dataclass
class ToolCallRecord:
    id: str
    name: str
    arguments: Any  # a dict, or the raw string when the model sent invalid JSON
    result: dict[str, Any]

    @property
    def error_type(self) -> str | None:
        return self.result.get("error_type")


@dataclass
class AgentResult:
    question: str
    answer: str
    stopped_reason: str  # "answer" or "max_steps"
    steps: int
    model: str
    prompt_version: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


def _openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]},
        }
        for s in tool_specs()
    ]


def _parse_arguments(raw: str | None) -> tuple[Any, str | None]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return raw, f"arguments were not valid JSON: {exc}"
    if not isinstance(args, dict):
        return args, "arguments must be a JSON object"
    return args, None


def _assistant_message(message: Any, calls: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.content}
    if calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in calls
        ]
    return out


def run_agent(
    question: str,
    client: Any,
    model: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float | None = None,
) -> AgentResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user", "content": question},
    ]
    tools = _openai_tools()
    records: list[ToolCallRecord] = []

    def result(answer: str, reason: str, steps: int) -> AgentResult:
        return AgentResult(
            question=question,
            answer=answer,
            stopped_reason=reason,
            steps=steps,
            model=model,
            prompt_version=prompts.PROMPT_VERSION,
            tool_calls=records,
        )

    for step in range(1, max_steps + 1):
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
        if temperature is not None:
            kwargs["temperature"] = temperature
        message = client.chat.completions.create(**kwargs).choices[0].message
        calls = message.tool_calls or []
        messages.append(_assistant_message(message, calls))

        if not calls:
            return result(message.content or "", "answer", step)

        for call in calls:
            args, problem = _parse_arguments(call.function.arguments)
            if problem:
                outcome = {"error": problem, "error_type": "bad_arguments"}
            else:
                outcome = run_tool(call.function.name, args)
            records.append(ToolCallRecord(call.id, call.function.name, args, outcome))
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(outcome, default=str)}
            )

    return result(f"Stopped after {max_steps} steps without a final answer.", "max_steps", max_steps)
