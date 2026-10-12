"""Usage: python -m agent "which pods are unhealthy in mlops-fraud"
Tool calls go to stderr, the answer to stdout. Tracing to Galileo is on when GALILEO_API_KEY is set."""
import json
import os
import sys
from typing import Any

from . import config
from .loop import run_agent
from .tracing import GalileoTracer, make_tracer


def main(argv: list[str], client: Any = None, tracer: Any = None) -> int:
    if len(argv) < 2:
        print('usage: python -m agent "<question>"', file=sys.stderr)
        return 2
    model = config.openai_model()
    if not model:
        print(
            "OPS_MODEL is not set. Set it to a tool-calling model your provider can serve.",
            file=sys.stderr,
        )
        return 2

    from openai import OpenAI, OpenAIError  # imported here so tests and tools don't need the SDK

    tracer = tracer or make_tracer()
    try:
        result = run_agent(
            " ".join(argv[1:]),
            client or OpenAI(),
            model,
            temperature=config.temperature(),
            tracer=tracer,
        )
    except OpenAIError as exc:  # bad key, no credits, rate limit, provider rejected the request
        print(f"API error: {str(exc)[:500]}", file=sys.stderr)
        return 1
    finally:
        tracer.flush()

    for call in result.tool_calls:
        print(f"[tool] {call.name} {json.dumps(call.arguments)} -> {call.error_type or 'ok'}", file=sys.stderr)
    if isinstance(tracer, GalileoTracer):
        print(
            f"[trace] sent to Galileo project={os.getenv('GALILEO_PROJECT')} "
            f"log_stream={os.getenv('GALILEO_LOG_STREAM')}",
            file=sys.stderr,
        )
    print(result.answer)
    print(
        f"[usage] {result.prompt_tokens} in, {result.completion_tokens} out, "
        f"{result.steps} steps, {result.latency_ms:.0f} ms",
        file=sys.stderr,
    )
    if result.stopped_reason != "answer":
        print(f"(stopped: {result.stopped_reason} after {result.steps} steps)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from dotenv import load_dotenv  # only for the CLI, so tests never read a local .env

    load_dotenv()
    sys.exit(main(sys.argv))
