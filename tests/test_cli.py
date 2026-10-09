from types import SimpleNamespace

import openai

from agent.__main__ import main


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
