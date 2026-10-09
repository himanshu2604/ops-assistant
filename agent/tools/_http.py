from typing import Any, Callable

import requests

# (url, headers, params) -> parsed JSON body. Tools take this as an argument so tests can fake it.
HttpGet = Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]]


def http_get_json(url: str, headers: dict[str, str], params: dict[str, str]) -> dict[str, Any]:
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code == 404:
        raise LookupError("not found (check the name, or token access for private repos)")
    resp.raise_for_status()
    return resp.json()
