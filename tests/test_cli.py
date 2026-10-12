from types import SimpleNamespace

import openai
import pytest

from agent.__main__ import main


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    monkeypatch.delenv("GALILEO_API_KEY", raising=False)


class CountingTracer:
    def __init__(self):
        self.flushes = 0

    def start(self, question, metadata): ...

    def llm_call(self, **kw): ...

    def tool_call(self, **kw): ...

    def finish(self, **kw): ...

    def flush(self):
        self.flushes += 1


class FakeClient:
    def __init__(self, response=None, error=None):
        self._response, self._error = response, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


def answer(text):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_prints_the_answer(monkeypatch, capsys):
    monkeypatch.setenv("OPS_MODEL", "m")
    assert main(["agent", "hi"], client=FakeClient(answer("hello"))) == 0
    assert capsys.readouterr().out.strip() == "hello"


def test_api_errors_are_one_line_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("OPS_MODEL", "m")
    error = openai.OpenAIError("no credits remaining")
    assert main(["agent", "hi"], client=FakeClient(error=error)) == 1
    assert capsys.readouterr().err.startswith("API error: no credits remaining")


def test_tracer_is_flushed_on_success_and_on_api_error(monkeypatch):
    monkeypatch.setenv("OPS_MODEL", "m")
    ok = CountingTracer()
    main(["agent", "hi"], client=FakeClient(answer("hello")), tracer=ok)
    assert ok.flushes == 1
    failed = CountingTracer()
    main(["agent", "hi"], client=FakeClient(error=openai.OpenAIError("boom")), tracer=failed)
    assert failed.flushes == 1
