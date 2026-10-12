import sys
from types import SimpleNamespace

from agent.tracing import GalileoTracer, NoopTracer, make_tracer, run_metadata


class FakeLogger:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def _record(self, method, /, **kwargs):
        self.calls.append((method, kwargs))
        if method == self.fail_on:
            raise RuntimeError("galileo unreachable")

    def start_trace(self, **kw):
        self._record("start_trace", **kw)

    def add_llm_span(self, **kw):
        self._record("add_llm_span", **kw)

    def add_tool_span(self, **kw):
        self._record("add_tool_span", **kw)

    def conclude(self, **kw):
        self._record("conclude", **kw)

    def flush(self):
        self._record("flush")


def llm_kwargs():
    return dict(
        step=1, model="m", messages=[], output={"role": "assistant", "content": "x"}, tools=[],
        duration_ns=5, usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}, temperature=None,
    )


def test_noop_tracer_accepts_every_call():
    t = NoopTracer()
    t.start("q", {})
    t.llm_call(**llm_kwargs())
    t.tool_call(call_id="c", name="n", arguments={}, result={}, duration_ns=1)
    t.finish(answer="a", stopped_reason="answer", duration_ns=1)
    t.flush()


def test_galileo_tracer_maps_calls_to_logger_methods():
    logger = FakeLogger()
    t = GalileoTracer(logger)
    t.start("which pods?", {"model": "m"})
    t.llm_call(**llm_kwargs())
    t.tool_call(call_id="c1", name="get_pod_status", arguments={"namespace": "a"}, result={"ok": 1}, duration_ns=9)
    t.finish(answer="done", stopped_reason="answer", duration_ns=20)
    names = [c[0] for c in logger.calls]
    assert names == ["start_trace", "add_llm_span", "add_tool_span", "conclude"]
    start, llm, tool, end = (c[1] for c in logger.calls)
    assert start["input"] == "which pods?" and start["metadata"] == {"model": "m"}
    assert llm["num_input_tokens"] == 7 and llm["total_tokens"] == 10 and llm["name"] == "llm_step_1"
    assert tool["tool_call_id"] == "c1" and tool["status_code"] == 200 and tool["metadata"] is None
    assert end["status_code"] == 200


def test_tool_errors_are_marked_on_the_span():
    logger = FakeLogger()
    t = GalileoTracer(logger)
    t.tool_call(call_id="c", name="n", arguments={}, result={"error": "x", "error_type": "tool_failure"}, duration_ns=1)
    span = logger.calls[0][1]
    assert span["status_code"] == 500 and span["metadata"] == {"error_type": "tool_failure"}


def test_unfinished_runs_are_marked_failed():
    logger = FakeLogger()
    GalileoTracer(logger).finish(answer="Stopped", stopped_reason="max_steps", duration_ns=1)
    assert logger.calls[0][1]["status_code"] == 500


def test_a_failing_logger_never_raises_and_disables_further_calls():
    logger = FakeLogger(fail_on="start_trace")
    t = GalileoTracer(logger)
    t.start("q", {})
    t.llm_call(**llm_kwargs())
    t.finish(answer="a", stopped_reason="answer", duration_ns=1)
    assert [c[0] for c in logger.calls] == ["start_trace"]  # nothing after the first failure
    assert t.disabled


def test_next_question_retries_after_a_failure():
    logger = FakeLogger(fail_on="add_llm_span")
    t = GalileoTracer(logger)
    t.start("q1", {})
    t.llm_call(**llm_kwargs())
    assert t.disabled
    t.start("q2", {})
    assert not t.disabled and logger.calls[-1][0] == "start_trace"


def test_flush_failure_is_swallowed():
    GalileoTracer(FakeLogger(fail_on="flush")).flush()


def test_run_metadata(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("OPS_RUN_SOURCE", raising=False)
    assert run_metadata("m", "v2", 0.0) == {
        "model": "m", "prompt_version": "v2", "git_commit": "abcdef1", "source": "manual", "temperature": "0.0",
    }
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert run_metadata("m", "v2")["source"] == "ci"


def test_no_key_means_no_tracing(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)
    assert isinstance(make_tracer(), NoopTracer)


def test_setup_failure_falls_back_to_no_tracing(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("bad credentials")

    monkeypatch.setenv("GALILEO_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "galileo", SimpleNamespace(GalileoLogger=boom))
    assert isinstance(make_tracer(), NoopTracer)
