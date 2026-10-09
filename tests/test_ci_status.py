import pytest

from agent.tools.ci_status import get_latest_ci_run


def run_payload(**overrides):
    run = {
        "name": "CI",
        "status": "completed",
        "conclusion": "success",
        "head_branch": "main",
        "head_sha": "a1b2c3d4e5f6a7b8",
        "event": "push",
        "created_at": "2026-10-09T04:00:00Z",
        "html_url": "https://github.com/himanshu2604/mlops-fraud-pipeline/actions/runs/1",
    }
    run.update(overrides)
    return {"workflow_runs": [run]}


def fake_http(body):
    calls = []

    def get(url, headers, params):
        calls.append((url, headers, params))
        return body

    get.calls = calls
    return get


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPS_GITHUB_REPO", raising=False)


def test_passed_run():
    http = fake_http(run_payload())
    out = get_latest_ci_run(http_get=http)
    assert out["found"] and out["passed"] is True
    assert out["commit"] == "a1b2c3d"
    url, _, params = http.calls[0]
    assert url == "https://api.github.com/repos/himanshu2604/mlops-fraud-pipeline/actions/runs"
    assert params == {"per_page": "1"}


def test_failed_run():
    out = get_latest_ci_run(http_get=fake_http(run_payload(conclusion="failure")))
    assert out["passed"] is False and out["conclusion"] == "failure"


def test_in_progress_run_has_no_verdict():
    body = run_payload(status="in_progress", conclusion=None)
    out = get_latest_ci_run(http_get=fake_http(body))
    assert out["passed"] is None and out["status"] == "in_progress"


def test_no_runs():
    out = get_latest_ci_run(http_get=fake_http({"workflow_runs": []}))
    assert out == {"repo": "himanshu2604/mlops-fraud-pipeline", "found": False}


def test_workflow_and_branch_narrow_the_query():
    http = fake_http(run_payload())
    get_latest_ci_run(repo="a/b", workflow="ci.yml", branch="release/1", http_get=http)
    url, _, params = http.calls[0]
    assert url == "https://api.github.com/repos/a/b/actions/workflows/ci.yml/runs"
    assert params == {"per_page": "1", "branch": "release/1"}


def test_token_sent_only_when_set(monkeypatch):
    http = fake_http(run_payload())
    get_latest_ci_run(http_get=http)
    assert "Authorization" not in http.calls[0][1]
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    get_latest_ci_run(http_get=http)
    assert http.calls[1][1]["Authorization"] == "Bearer tok"


@pytest.mark.parametrize(
    "kwargs",
    [{"repo": "no-slash"}, {"repo": "a/b/c"}, {"workflow": "../x"}, {"branch": "a b"}, {"repo": "a/b;ls"}],
)
def test_rejects_bad_input_without_calling_api(kwargs):
    http = fake_http(run_payload())
    with pytest.raises(ValueError):
        get_latest_ci_run(http_get=http, **kwargs)
    assert http.calls == []


def test_null_optional_arguments_behave_like_omitted():
    http = fake_http(run_payload())
    out = get_latest_ci_run(repo=None, workflow=None, branch=None, http_get=http)
    assert out["found"]
    url, _, params = http.calls[0]
    assert url.endswith("/actions/runs") and params == {"per_page": "1"}
