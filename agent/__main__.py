"""Usage: python -m agent "which pods are unhealthy in mlops-fraud"
Tool calls go to stderr, the answer to stdout."""
import json
import sys
from typing import Any

from . import config
from .loop import run_agent


def main(argv: list[str], client: Any = None) -> int:
    if len(argv) < 2:
        print('usage: python -m agent "<question>"', file=sys.stderr)
        return 2
    model = config.openai_model()
    if not model:
        print(
            "OPS_MODEL is not set. Set it to a tool-calling model your OpenAI account can use.",
            file=sys.stderr,
        )
        return 2

    from openai import OpenAI, OpenAIError  # imported here so tests and tools don't need the SDK

    try:
        result = run_agent(
            " ".join(argv[1:]), client or OpenAI(), model, temperature=config.temperature()
        )
    except OpenAIError as exc:  # bad key, no credits, rate limit, provider rejected the request
        text = str(exc)
        print(f"API error: {text[:500]}", file=sys.stderr)
        return 1

    for call in result.tool_calls:
        print(f"[tool] {call.name} {json.dumps(call.arguments)} -> {call.error_type or 'ok'}", file=sys.stderr)
    print(result.answer)
    if result.stopped_reason != "answer":
        print(f"(stopped: {result.stopped_reason} after {result.steps} steps)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
