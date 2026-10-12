import copy
import json
from types import SimpleNamespace

from agent.tools import TOOLS, Tool


def tool_call(name, args, id_="call_1"):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=id_, type="function", function=SimpleNamespace(name=name, arguments=raw))


def reply(content=None, calls=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


class FakeClient:
    """Scripted stand-in for the OpenAI client. Snapshots each request so later mutation can't hide bugs."""

    def __init__(self, *responses, error=None):
        self.requests = []
        self._responses = list(responses)
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if self._error:
            raise self._error
        return self._responses.pop(0)


def stub_tool(monkeypatch, name, fn):
    spec = TOOLS[name]
    monkeypatch.setitem(TOOLS, name, Tool(name, spec.description, spec.parameters, fn))


POD_RESULT = {"namespace": "mlops-fraud", "total": 2, "unhealthy_count": 1, "pods": []}
