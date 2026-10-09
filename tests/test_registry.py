from agent.tools import TOOLS, Tool, run_tool, tool_specs


def test_exactly_three_tools():
    names = [s["name"] for s in tool_specs()]
    assert sorted(names) == ["get_latest_ci_run", "get_pod_status", "query_prometheus"]


def test_every_spec_is_complete():
    for spec in tool_specs():
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"
        assert spec["parameters"]["additionalProperties"] is False


def test_unknown_tool():
    assert run_tool("delete_pod", {"name": "x"})["error_type"] == "unknown_tool"


def test_missing_required_argument():
    out = run_tool("get_pod_status", {})
    assert out["error_type"] == "bad_arguments" and "namespace" in out["error"]


def test_unexpected_argument_is_rejected():
    out = run_tool("get_pod_status", {"namespace": "mlops-fraud", "verb": "delete"})
    assert out["error_type"] == "bad_arguments" and "verb" in out["error"]


def test_none_args_treated_as_empty(monkeypatch):
    spec = TOOLS["get_latest_ci_run"]
    stub = Tool(spec.name, spec.description, spec.parameters, lambda: {"ok": True})
    monkeypatch.setitem(TOOLS, "get_latest_ci_run", stub)
    assert run_tool("get_latest_ci_run", None) == {"ok": True}


def _swap(monkeypatch, fn):
    monkeypatch.setitem(TOOLS, "get_pod_status", Tool("get_pod_status", "d", {"type": "object", "properties": {}}, fn))


def test_value_error_becomes_invalid_input(monkeypatch):
    def bad():
        raise ValueError("nope")

    _swap(monkeypatch, bad)
    assert run_tool("get_pod_status", {}) == {"error": "nope", "error_type": "invalid_input"}


def test_other_exceptions_become_tool_failure(monkeypatch):
    def boom():
        raise RuntimeError("cluster unreachable")

    _swap(monkeypatch, boom)
    out = run_tool("get_pod_status", {})
    assert out["error_type"] == "tool_failure" and "cluster unreachable" in out["error"]


def test_optional_ci_arguments_accept_null_in_the_schema():
    props = TOOLS["get_latest_ci_run"].parameters["properties"]
    for name in ("repo", "workflow", "branch"):
        assert "null" in props[name]["type"]


def test_long_errors_are_clipped(monkeypatch):
    def boom():
        raise RuntimeError("<html>" + "x" * 5000)

    _swap(monkeypatch, boom)
    out = run_tool("get_pod_status", {})
    assert len(out["error"]) < 600 and out["error"].endswith("[truncated]")
