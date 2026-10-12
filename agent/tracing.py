"""Best-effort tracing. Tracing must never break or slow down an answer.

The loop talks to a small tracer interface (start, llm_call, tool_call, finish, flush).
NoopTracer does nothing. GalileoTracer logs one trace per question: an LLM span per model
call and a tool span per tool call, linked by tool_call_id.
"""
import json
import logging
import os
import subprocess
from typing import Any

log = logging.getLogger("ops_assistant.tracing")


def git_commit() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_metadata(model: str, prompt_version: str, temperature: float | None = None) -> dict[str, str]:
    meta = {
        "model": model,
        "prompt_version": prompt_version,
        "git_commit": git_commit(),
        "source": "ci" if os.getenv("GITHUB_ACTIONS") else os.getenv("OPS_RUN_SOURCE", "manual"),
    }
    if temperature is not None:
        meta["temperature"] = str(temperature)
    return meta


class NoopTracer:
    def start(self, question: str, metadata: dict[str, str]) -> None: ...

    def llm_call(self, **kwargs: Any) -> None: ...

    def tool_call(self, **kwargs: Any) -> None: ...

    def finish(self, **kwargs: Any) -> None: ...

    def flush(self) -> None: ...


class GalileoTracer:
    """Every method is best-effort. The first failure disables tracing until the next
    question starts, and nothing here ever raises into the agent."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.disabled = False

    def _safe(self, what: str, fn: Any) -> None:
        if self.disabled:
            return
        try:
            fn()
        except Exception as exc:
            self.disabled = True
            log.warning("Galileo tracing disabled after an error in %s: %s", what, exc)

    def start(self, question: str, metadata: dict[str, str]) -> None:
        self.disabled = False
        self._safe(
            "start",
            lambda: self.logger.start_trace(
                input=question, name="ops_question", metadata=metadata, tags=["ops-assistant"]
            ),
        )

    def llm_call(
        self,
        *,
        step: int,
        model: str,
        messages: list[dict[str, Any]],
        output: dict[str, Any],
        tools: list[dict[str, Any]],
        duration_ns: int,
        usage: dict[str, int | None],
        temperature: float | None,
    ) -> None:
        self._safe(
            "llm_call",
            lambda: self.logger.add_llm_span(
                input=messages,
                output=output,
                model=model,
                tools=tools,
                name=f"llm_step_{step}",
                duration_ns=duration_ns,
                num_input_tokens=usage.get("prompt_tokens"),
                num_output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                temperature=temperature,
                status_code=200,
            ),
        )

    def tool_call(
        self, *, call_id: str, name: str, arguments: Any, result: dict[str, Any], duration_ns: int
    ) -> None:
        failed = "error" in result
        self._safe(
            "tool_call",
            lambda: self.logger.add_tool_span(
                input=json.dumps(arguments, default=str),
                output=json.dumps(result, default=str),
                name=name,
                duration_ns=duration_ns,
                tool_call_id=call_id,
                status_code=500 if failed else 200,
                metadata={"error_type": str(result.get("error_type"))} if failed else None,
            ),
        )

    def finish(self, *, answer: str, stopped_reason: str, duration_ns: int) -> None:
        self._safe(
            "finish",
            lambda: self.logger.conclude(
                output=answer,
                duration_ns=duration_ns,
                status_code=200 if stopped_reason == "answer" else 500,
            ),
        )

    def flush(self) -> None:
        try:
            self.logger.flush()
        except Exception as exc:
            log.warning("Galileo flush failed: %s", exc)


def make_tracer() -> Any:
    """Galileo tracing when GALILEO_API_KEY is set, otherwise a no-op."""
    if not os.getenv("GALILEO_API_KEY"):
        return NoopTracer()
    try:
        from galileo import GalileoLogger

        logger = GalileoLogger(
            project=os.getenv("GALILEO_PROJECT"), log_stream=os.getenv("GALILEO_LOG_STREAM")
        )
        return GalileoTracer(logger)
    except Exception as exc:
        log.warning("Galileo tracing unavailable, continuing without it: %s", exc)
        return NoopTracer()
