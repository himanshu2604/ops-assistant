"""Runs the tracer through the real Galileo SDK, offline. The SDK's ingestion hook captures what
would have been sent, so this checks the logged structure without a network or an API key."""
import pytest

pytest.importorskip("galileo")

from galileo import GalileoLogger  # noqa: E402

from agent.loop import run_agent  # noqa: E402
from agent.tracing import GalileoTracer  # noqa: E402
from helpers import POD_RESULT, FakeClient, reply, stub_tool, tool_call, usage  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Dummy key and an unreachable URL, so a mistake fails fast instead of reaching real Galileo.
    monkeypatch.setenv("GALILEO_API_KEY", "dummy")
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "http://127.0.0.1:9")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_one_question_becomes_one_trace_with_linked_spans(monkeypatch):
    stub_tool(monkeypatch, "get_pod_status", lambda namespace: POD_RESULT)
    sent = []
    logger = GalileoLogger(project="p", log_stream="l", mode="batch", ingestion_hook=sent.append)
    tracer = GalileoTracer(logger)
    client = FakeClient(
        reply(calls=[tool_call("get_pod_status", {"namespace": "mlops-fraud"}, "call_9")], usage=usage(100, 20)),
        reply("1 of 2 pods is unhealthy.", usage=usage(150, 12)),
    )

    run_agent("which pods are unhealthy?", client, "test-model", tracer=tracer)
    tracer.flush()

    assert not tracer.disabled
    assert len(sent) == 1
    trace = sent[0].model_dump()["traces"][0]
    assert trace["name"] == "ops_question"
    assert trace["input"] == "which pods are unhealthy?"
    assert trace["output"] == "1 of 2 pods is unhealthy."
    assert trace["user_metadata"]["model"] == "test-model"
    assert trace["user_metadata"]["prompt_version"]

    kinds = [str(getattr(s["type"], "value", s["type"])) for s in trace["spans"]]
    assert kinds == ["llm", "tool", "llm"]
    llm1, tool, llm2 = trace["spans"]
    assert tool["name"] == "get_pod_status" and tool["tool_call_id"] == "call_9"
    assert llm1["metrics"]["num_input_tokens"] == 100 and llm1["metrics"]["num_output_tokens"] == 20
    assert llm2["metrics"]["num_input_tokens"] == 150
