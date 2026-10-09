"""Tool registry. Everything the agent can do is listed here, and all of it is read-only."""
from typing import Any

from . import ci_status, k8s_status, prom_query
from .base import Tool

TOOLS: dict[str, Tool] = {t.name: t for t in (k8s_status.TOOL, ci_status.TOOL, prom_query.TOOL)}


def tool_specs() -> list[dict[str, Any]]:
    """Provider-neutral tool definitions; adapt to the LLM API in the agent loop."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in TOOLS.values()
    ]


def _check_args(tool: Tool, args: dict[str, Any]) -> str | None:
    props = tool.parameters.get("properties", {})
    unexpected = sorted(set(args) - set(props))
    if unexpected:
        return f"unexpected arguments: {', '.join(unexpected)}"
    missing = [k for k in tool.parameters.get("required", []) if k not in args]
    if missing:
        return f"missing required arguments: {', '.join(missing)}"
    return None


def run_tool(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Run a tool and always return a dict. Failures come back as {'error', 'error_type'}
    so the loop can hand them to the model and the trace can record them."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool: {name}", "error_type": "unknown_tool"}
    args = args or {}
    problem = _check_args(tool, args)
    if problem:
        return {"error": problem, "error_type": "bad_arguments"}
    try:
        return tool.fn(**args)
    except ValueError as exc:
        return {"error": str(exc), "error_type": "invalid_input"}
    except Exception as exc:  # tool failures are data for the agent, not crashes
        return {"error": f"{type(exc).__name__}: {exc}", "error_type": "tool_failure"}
