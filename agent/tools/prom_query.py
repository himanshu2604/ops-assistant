"""Instant PromQL query against Prometheus. The instant-query endpoint is read-only."""
import math
from typing import Any

from .. import config
from ._http import HttpGet, http_get_json
from .base import Tool

MAX_QUERY_CHARS = 500
MAX_SERIES = 50


def _number(raw: str) -> float | None:
    value = float(raw)
    return None if math.isnan(value) or math.isinf(value) else value


def query_prometheus(query: str, http_get: HttpGet = http_get_json) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query too long (max {MAX_QUERY_CHARS} characters)")

    body = http_get(f"{config.prometheus_url()}/api/v1/query", {}, {"query": query})
    if body.get("status") != "success":
        raise RuntimeError(body.get("error") or "prometheus query failed")

    data = body.get("data", {})
    result_type = data.get("resultType")
    result = data.get("result", [])

    if result_type == "scalar":
        series = [{"metric": {}, "value": _number(result[1])}]
    elif result_type == "vector":
        series = [
            {"metric": r.get("metric", {}), "value": _number(r["value"][1])}
            for r in result
            if r.get("value")
        ]
    else:
        raise ValueError(f"unsupported result type: {result_type!r}; use an instant vector query")

    return {
        "query": query,
        "result_type": result_type,
        "empty": len(series) == 0,
        "truncated": len(series) > MAX_SERIES,
        "series": series[:MAX_SERIES],
    }


TOOL = Tool(
    name="query_prometheus",
    description=(
        "Run an instant PromQL query and return the current value of each matching series. "
        "Use it for current model metrics such as ROC AUC. Empty result means no such series."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "PromQL expression, e.g. a metric name"}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    fn=query_prometheus,
)
