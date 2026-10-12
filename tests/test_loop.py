import json

import pytest

from agent import prompts
from agent.__main__ import main
from agent.loop import run_agent
from agent.tools import TOOLS
from helpers import POD_RESULT, FakeClient, reply, stub_tool, tool_call, usage


class RecordingTracer:
    def __init__(self):
        self.events = []

    def start(self, question, metadata):
        self.events.append(("start", question, metadata))

    def llm_call(self, **kw):
        self.events.append(("llm_call", kw))

    def tool_call(self, **kw):
        self.events.append(("tool_call", kw))

    def finish(self, **kw):
        self.events.append(("finish", kw))

    def flush(self):
        self.events.append(("flush",))


def test_answers_without_tools():
    client = FakeClient(reply("Hello."))
    out = run_agent("hi", client, "m")
    assert out.answer == "Hello." and out.steps == 1 and out.tool_calls == []
    assert out.stopped_reason == "answer" and out.model == "m"
    assert out.prompt_version == prompts.PROMPT_VERSION


def test_tool_call_then_answer(monkeypatch):
    stub_tool(monkeypatch, "get_pod_status", lambda namespace: POD_RESULT)
    client = FakeClient(
        reply(calls=[tool_call("get_pod_status", {"namespace": "mlops-fraud"})]),
        reply("1 of 2 pods is unhealthy."),
    )
    out = run_agent("which pods are unhealthy?", client, "m")

    assert out.answer == "1 of 2 pods is unhealthy." and out.steps == 2
    assert out.tool_calls[0].name == "get_pod_status"
    assert out.tool_calls[0].arguments == {"namespace": "mlops-fraud"}
    assert out.tool_calls[0].result == POD_RESULT

    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["function"]["name"] == "get_pod_status"
    assert second[3] == {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(POD_RESULT)}


def test_parallel_tool_calls_all_get_answered(monkeypatch):
    stub_tool(monkeypatch, "get_pod_status", lambda namespace: POD_RESULT)
    stub_tool(monkeypatch, "get_latest_ci_run", lambda: {"found": True, "passed": True})
    client = FakeClient(
        reply(
            calls=[
                tool_call("get_pod_status", {"namespace": "mlops-fraud"}, "call_a"),
                tool_call("get_latest_ci_run", {}, "call_b"),
            ]
        ),
        reply("done"),
    )
    out = run_agent("status?", client, "m")
    tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]
    assert [c.name for c in out.tool_calls] == ["get_pod_status", "get_latest_ci_run"]


def test_invalid_json_arguments_become_a_tool_error():
    client = FakeClient(reply(calls=[tool_call("get_pod_status", "{not json")]), reply("sorry"))
    out = run_agent("x", client, "m")
    assert out.tool_calls[0].error_type == "bad_arguments"
    assert out.tool_calls[0].arguments == "{not json"
    assert out.answer == "sorry"


def test_unknown_tool_is_reported_back_to_the_model():
    client = FakeClient(reply(calls=[tool_call("delete_pod", {"name": "x"})]), reply("can't do that"))
    out = run_agent("delete the pod", client, "m")
    assert out.tool_calls[0].error_type == "unknown_tool"
    tool_msg = client.requests[1]["messages"][-1]
    assert json.loads(tool_msg["content"])["error_type"] == "unknown_tool"


def test_step_cap_stops_a_runaway_loop(monkeypatch):
    stub_tool(monkeypatch, "get_pod_status", lambda namespace: POD_RESULT)
    loop_forever = [reply(calls=[tool_call("get_pod_status", {"namespace": "a"}, f"c{i}")]) for i in range(10)]
    client = FakeClient(*loop_forever)
    out = run_agent("x", client, "m", max_steps=3)
    assert out.stopped_reason == "max_steps" and out.steps == 3
    assert len(client.requests) == 3 and len(out.tool_calls) == 3


def test_request_shape():
    client = FakeClient(reply("ok"))
    run_agent("q", client, "my-model")
    req = client.requests[0]
    assert req["model"] == "my-model"
    assert req["messages"][0]["role"] == "system"
    assert "read-only" in req["messages"][0]["content"]
    assert req["messages"][1] == {"role": "user", "content": "q"}
    assert all(t["type"] == "function" for t in req["tools"])
    assert {t["function"]["name"] for t in req["tools"]} == set(TOOLS)


def test_temperature_only_sent_when_given():
    client = FakeClient(reply("a"), reply("b"))
    run_agent("q", client, "m")
    run_agent("q", client, "m", temperature=0)
    assert "temperature" not in client.requests[0]
    assert client.requests[1]["temperature"] == 0


def test_known_metrics_reach_the_system_prompt(monkeypatch):
    monkeypatch.setattr(prompts, "KNOWN_METRICS", {"fraud_roc_auc": "ROC AUC of the live model"})
    text = prompts.system_prompt()
    assert "fraud_roc_auc" in text and "could not find the metric" not in text


def test_cli_needs_a_question_and_a_model(monkeypatch):
    monkeypatch.delenv("OPS_MODEL", raising=False)
    assert main(["agent"]) == 2
    assert main(["agent", "which pods are unhealthy?"]) == 2


def test_tokens_and_latency_are_captured(monkeypatch):
    stub_tool(monkeypatch, "get_pod_status", lambda namespace: POD_RESULT)
    client = FakeClient(
        reply(calls=[tool_call("get_pod_status", {"namespace": "a"})], usage=usage(100, 20)),
        reply("done", usage=usage(150, 12)),
    )
    out = run_agent("q", client, "m")
    assert (out.prompt_tokens, out.completion_tokens) == (250, 32)
    assert out.latency_ms > 0 and out.tool_calls[0].duration_ms >= 0


def test_missing_usage_counts_as_zero():
    out = run_agent("q", FakeClient(reply("hi")), "m")
    assert (out.prompt_tokens, out.completion_tokens) == (0, 0)


def test_tracer_sees_every_step_in_order(monkeypatch):
    stub_tool(monkeypatch, "get_pod_status", lambda namespace: POD_RESULT)
    tracer = RecordingTracer()
    client = FakeClient(
        reply(calls=[tool_call("get_pod_status", {"namespace": "a"}, "c1")], usage=usage(10, 5)),
        reply("done", usage=usage(20, 6)),
    )
    run_agent("which pods?", client, "m", tracer=tracer)
    assert [e[0] for e in tracer.events] == ["start", "llm_call", "tool_call", "llm_call", "finish"]

    _, question, meta = tracer.events[0]
    assert question == "which pods?"
    assert meta["model"] == "m" and meta["prompt_version"] == prompts.PROMPT_VERSION

    first_llm = tracer.events[1][1]
    assert first_llm["step"] == 1 and first_llm["usage"]["prompt_tokens"] == 10
    assert first_llm["output"]["tool_calls"][0]["id"] == "c1"
    assert [m["role"] for m in first_llm["messages"]] == ["system", "user"]  # what the model saw

    tool_event = tracer.events[2][1]
    assert tool_event["call_id"] == "c1" and tool_event["result"] == POD_RESULT

    finish = tracer.events[-1][1]
    assert finish["answer"] == "done" and finish["stopped_reason"] == "answer"


def test_model_error_is_traced_then_raised():
    tracer = RecordingTracer()
    client = FakeClient(error=RuntimeError("provider down"))
    with pytest.raises(RuntimeError):
        run_agent("q", client, "m", tracer=tracer)
    finish = tracer.events[-1][1]
    assert finish["stopped_reason"] == "model_error" and "provider down" in finish["answer"]
