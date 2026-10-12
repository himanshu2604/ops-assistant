"""Hand-built agent loop: prompt in, tool call out, tool result back in, repeat.

`client` is anything shaped like the OpenAI SDK client (client.chat.completions.create).
Keeping it injected means tests can script the model. Tracing goes through a small tracer
interface (see tracing.py), so this file never imports Galileo.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .tools import run_tool, tool_specs
from .tracing import NoopTracer, run_metadata

DEFAULT_MAX_STEPS = 6


@dataclass
class ToolCallRecord:
    id: str
    name: str
    arguments: Any  # a dict, or the raw string when the model sent invalid JSON
    result: dict[str, Any]
    duration_ms: float = 0.0

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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0  # wall time for the whole question


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


def _usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        key: getattr(usage, key, None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def run_agent(
    question: str,
    client: Any,
    model: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float | None = None,
    tracer: Any = None,
) -> AgentResult:
    tracer = tracer or NoopTracer()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.system_prompt()},
        {"role": "user", "content": question},
    ]
    tools = _openai_tools()
    records: list[ToolCallRecord] = []
    prompt_tokens = completion_tokens = 0
    started = time.perf_counter_ns()

    tracer.start(question, run_metadata(model, prompts.PROMPT_VERSION, temperature))

    def result(answer: str, reason: str, steps: int) -> AgentResult:
        total_ns = time.perf_counter_ns() - started
        tracer.finish(answer=answer, stopped_reason=reason, duration_ns=total_ns)
        return AgentResult(
            question=question,
            answer=answer,
            stopped_reason=reason,
            steps=steps,
            model=model,
            prompt_version=prompts.PROMPT_VERSION,
            tool_calls=records,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=total_ns / 1e6,
        )

    for step in range(1, max_steps + 1):
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
        if temperature is not None:
            kwargs["temperature"] = temperature
        sent = list(messages)  # what the model sees on this step, for the trace

        call_started = time.perf_counter_ns()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            tracer.finish(
                answer=error, stopped_reason="model_error", duration_ns=time.perf_counter_ns() - started
            )
            raise
        llm_ns = time.perf_counter_ns() - call_started

        message = response.choices[0].message
        usage = _usage(response)
        prompt_tokens += usage["prompt_tokens"] or 0
        completion_tokens += usage["completion_tokens"] or 0
        calls = message.tool_calls or []
        assistant = _assistant_message(message, calls)
        messages.append(assistant)
        tracer.llm_call(
            step=step,
            model=model,
            messages=sent,
            output=assistant,
            tools=tools,
            duration_ns=llm_ns,
            usage=usage,
            temperature=temperature,
        )

        if not calls:
            return result(message.content or "", "answer", step)

        for call in calls:
            args, problem = _parse_arguments(call.function.arguments)
            tool_started = time.perf_counter_ns()
            if problem:
                outcome = {"error": problem, "error_type": "bad_arguments"}
            else:
                outcome = run_tool(call.function.name, args)
            tool_ns = time.perf_counter_ns() - tool_started
            records.append(ToolCallRecord(call.id, call.function.name, args, outcome, tool_ns / 1e6))
            tracer.tool_call(
                call_id=call.id,
                name=call.function.name,
                arguments=args,
                result=outcome,
                duration_ns=tool_ns,
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(outcome, default=str)}
            )

    return result(f"Stopped after {max_steps} steps without a final answer.", "max_steps", max_steps)
