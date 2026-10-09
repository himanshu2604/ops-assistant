import pytest

from agent.tools.prom_query import MAX_QUERY_CHARS, MAX_SERIES, query_prometheus


def vector(*values):
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"__name__": "m", "model": f"v{i}"}, "value": [1760000000, str(v)]}
                for i, v in enumerate(values)
            ],
        },
    }


def fake_http(body):
    calls = []

    def get(url, headers, params):
        calls.append((url, headers, params))
        return body

    get.calls = calls
    return get


@pytest.fixture(autouse=True)
def default_url(monkeypatch):
    monkeypatch.delenv("OPS_PROMETHEUS_URL", raising=False)


def test_vector_result():
    http = fake_http(vector(0.9731))
    out = query_prometheus("model_roc_auc", http_get=http)
    assert out["series"][0]["value"] == pytest.approx(0.9731)
    assert out["empty"] is False and out["truncated"] is False
    url, _, params = http.calls[0]
    assert url == "http://localhost:9090/api/v1/query"
    assert params == {"query": "model_roc_auc"}


def test_uses_configured_url(monkeypatch):
    monkeypatch.setenv("OPS_PROMETHEUS_URL", "http://prom.internal:9090/")
    http = fake_http(vector(1))
    query_prometheus("up", http_get=http)
    assert http.calls[0][0] == "http://prom.internal:9090/api/v1/query"


def test_empty_result_is_reported_not_hidden():
    out = query_prometheus("nope", http_get=fake_http(vector()))
    assert out["empty"] is True and out["series"] == []


def test_nan_becomes_null():
    out = query_prometheus("x", http_get=fake_http(vector("NaN")))
    assert out["series"][0]["value"] is None


def test_scalar_result():
    body = {"status": "success", "data": {"resultType": "scalar", "result": [1760000000, "3"]}}
    assert query_prometheus("1+2", http_get=fake_http(body))["series"][0]["value"] == 3.0


def test_series_are_capped():
    out = query_prometheus("x", http_get=fake_http(vector(*range(MAX_SERIES + 5))))
    assert out["truncated"] is True and len(out["series"]) == MAX_SERIES


def test_prometheus_error_raises():
    body = {"status": "error", "error": "parse error at char 3"}
    with pytest.raises(RuntimeError, match="parse error"):
        query_prometheus("sum(", http_get=fake_http(body))


def test_range_vector_is_rejected():
    body = {"status": "success", "data": {"resultType": "matrix", "result": []}}
    with pytest.raises(ValueError, match="unsupported result type"):
        query_prometheus("up[5m]", http_get=fake_http(body))


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_QUERY_CHARS + 1)])
def test_rejects_bad_query_without_calling_api(bad):
    http = fake_http(vector(1))
    with pytest.raises(ValueError):
        query_prometheus(bad, http_get=http)
    assert http.calls == []
