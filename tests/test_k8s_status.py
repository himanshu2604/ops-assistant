from pathlib import Path

import pytest

from agent.tools.k8s_status import get_pod_status

FIXTURE = (Path(__file__).parent / "fixtures" / "pods_mlops_fraud.json").read_text()


def fake_runner(output: str):
    calls = []

    def run(args):
        calls.append(args)
        return output

    run.calls = calls
    return run


def test_summarizes_pods_and_counts_unhealthy():
    out = get_pod_status("mlops-fraud", runner=fake_runner(FIXTURE))
    assert out["total"] == 4
    assert out["unhealthy_count"] == 2
    by_name = {p["name"]: p for p in out["pods"]}

    ok = by_name["fraud-api-7d9f8b6c5-abcde"]
    assert ok["healthy"] and ok["ready"] == "1/1" and ok["reason"] is None

    crash = by_name["fraud-api-7d9f8b6c5-fghij"]
    assert not crash["healthy"]
    assert crash["reason"] == "CrashLoopBackOff"
    assert crash["restarts"] == 7

    pending = by_name["fraud-retrain-28800000-xk2lp"]
    assert not pending["healthy"]
    assert pending["reason"] == "Unschedulable"

    done = by_name["fraud-drift-check-28799000-q9wzt"]
    assert done["healthy"] and done["reason"] is None


def test_command_is_a_fixed_read_only_get():
    runner = fake_runner(FIXTURE)
    get_pod_status("mlops-fraud", runner=runner)
    assert runner.calls == [["get", "pods", "-n", "mlops-fraud", "-o", "json"]]


def test_empty_namespace():
    out = get_pod_status("empty", runner=fake_runner('{"items": []}'))
    assert out == {"namespace": "empty", "total": 0, "unhealthy_count": 0, "pods": []}


@pytest.mark.parametrize("bad", ["", "UPPER", "a b", "x; rm -rf /", "-lead", "a" * 64, "--all-namespaces"])
def test_rejects_bad_namespace_without_calling_kubectl(bad):
    runner = fake_runner(FIXTURE)
    with pytest.raises(ValueError):
        get_pod_status(bad, runner=runner)
    assert runner.calls == []


def test_kubectl_failure_propagates():
    def boom(args):
        raise RuntimeError("connection refused")

    with pytest.raises(RuntimeError, match="connection refused"):
        get_pod_status("mlops-fraud", runner=boom)
