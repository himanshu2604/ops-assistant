"""Pod status through kubectl. Read-only: the command is fixed, only the namespace varies."""
import json
import re
import subprocess
from typing import Any, Callable

from .base import Tool

NAMESPACE_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

Runner = Callable[[list[str]], str]


def _kubectl(args: list[str]) -> str:
    proc = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, timeout=15, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"kubectl exited with {proc.returncode}")
    return proc.stdout


def _summarize_pod(item: dict[str, Any]) -> dict[str, Any]:
    status = item.get("status", {})
    phase = status.get("phase", "Unknown")
    containers = status.get("containerStatuses") or []
    ready = sum(1 for c in containers if c.get("ready"))
    total = len(containers)
    restarts = sum(c.get("restartCount", 0) for c in containers)

    reason = None
    for c in containers:
        state = c.get("state", {})
        waiting = state.get("waiting")
        if waiting and waiting.get("reason"):
            reason = waiting["reason"]
            break
        terminated = state.get("terminated")
        if terminated and terminated.get("exitCode", 0) != 0:
            reason = terminated.get("reason") or "Error"
            break
    if reason is None:
        for cond in status.get("conditions") or []:
            if cond.get("status") == "False" and cond.get("reason"):
                reason = cond["reason"]
                break
    if reason is None and phase in ("Pending", "Failed", "Unknown"):
        reason = status.get("reason") or phase

    healthy = phase == "Succeeded" or (
        phase == "Running" and total > 0 and ready == total and not _has_bad_state(containers)
    )
    return {
        "name": item.get("metadata", {}).get("name", "?"),
        "phase": phase,
        "ready": f"{ready}/{total}",
        "restarts": restarts,
        "healthy": healthy,
        "reason": None if healthy else reason,
    }


def _has_bad_state(containers: list[dict[str, Any]]) -> bool:
    return any(c.get("state", {}).get("waiting") for c in containers)


def get_pod_status(namespace: str, runner: Runner = _kubectl) -> dict[str, Any]:
    if not isinstance(namespace, str) or len(namespace) > 63 or not NAMESPACE_RE.match(namespace):
        raise ValueError(f"invalid namespace: {namespace!r}")
    raw = runner(["get", "pods", "-n", namespace, "-o", "json"])
    items = json.loads(raw).get("items", [])
    pods = [_summarize_pod(i) for i in items]
    return {
        "namespace": namespace,
        "total": len(pods),
        "unhealthy_count": sum(1 for p in pods if not p["healthy"]),
        "pods": pods,
    }


TOOL = Tool(
    name="get_pod_status",
    description=(
        "List the pods in a Kubernetes namespace with phase, ready containers, restart count "
        "and whether each pod is healthy. Use it for questions about unhealthy or crashing pods."
    ),
    parameters={
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "description": "Namespace name, e.g. mlops-fraud"}
        },
        "required": ["namespace"],
        "additionalProperties": False,
    },
    fn=get_pod_status,
)
