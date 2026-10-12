# ops-assistant

A small agent that answers infrastructure questions about the mlops-fraud-pipeline project by calling read-only tools. Calls are traced and evaluated with Galileo AI.

Status: tools, agent loop and Galileo tracing built and unit-tested. Eval dataset, experiment harness and CI gate are not built yet.

## Tools

| Tool | What it does |
|---|---|
| `get_pod_status` | `kubectl get pods -n <namespace> -o json`, summarized per pod |
| `get_latest_ci_run` | Latest GitHub Actions run, optionally by workflow or branch |
| `query_prometheus` | Instant PromQL query against Prometheus |

All three are read-only. `kubectl` runs with a fixed command and only the namespace varies. Tool errors come back as data (`error`, `error_type`) instead of raising, so the loop can pass them to the model and the trace can record them.

## Agent loop

`agent/loop.py` is hand-built, no framework: send the question and tool definitions to the model, run any tool calls it asks for, append the results, repeat until it answers or hits the step cap (6). Unknown tools and malformed arguments are returned to the model as tool errors. `run_agent` returns the answer, every tool call with its result, the step count, model and prompt version.

The OpenAI client is passed in, so tests script the model with a fake and a Galileo-wrapped client can be dropped in later.

## Tracing

Set `GALILEO_API_KEY`, `GALILEO_PROJECT` and `GALILEO_LOG_STREAM` and every question is logged to Galileo as one trace: an LLM span per model call (input messages, tool calls, token counts, latency) and a tool span per tool call (arguments, result, latency, linked to the model's call by `tool_call_id`). Trace metadata carries the model, prompt version, git commit and whether the run came from CI or a person. Without a key, tracing is off.

Tracing is best-effort. If Galileo is unreachable or a call fails, tracing switches itself off for that question and the agent still answers. `loop.py` never imports Galileo; it talks to a small tracer interface in `agent/tracing.py`. Spans are logged manually, so don't also import `galileo.openai`, which would log every model call a second time.

`tests/test_tracing_galileo.py` runs the tracer through the real Galileo SDK offline (its ingestion hook captures what would be sent), so the logged structure is tested without a network or a key.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # then fill it in
pytest
```

## Run the agent

```
export OPENAI_API_KEY=...
export OPS_MODEL=<a tool-calling model your account can use>
python -m agent "which pods are unhealthy in mlops-fraud"
```

Tool calls print to stderr, the answer to stdout. Before ROC AUC questions can work, fill in `KNOWN_METRICS` in `agent/prompts.py` with the metric names from your serving app's `/metrics` endpoint.

## Try a tool without the model

```
python scripts/try_tool.py get_pod_status '{"namespace": "mlops-fraud"}'
python scripts/try_tool.py get_latest_ci_run
python scripts/try_tool.py query_prometheus '{"query": "up"}'
```

Prometheus needs to be reachable. For a cluster install, port-forward first, for example:

```
kubectl -n monitoring port-forward svc/<prometheus-service> 9090:9090
```

## Settings (environment variables)

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | required for the agent |
| `OPS_MODEL` | required for the agent, no default |
| `OPENAI_BASE_URL` | unset (set it to use a non-OpenAI provider; set it together with that provider's key) |
| `GALILEO_API_KEY` | unset (tracing off) |
| `GALILEO_PROJECT`, `GALILEO_LOG_STREAM` | required when tracing is on |
| `OPS_TEMPERATURE` | unset (some newer models reject a custom temperature) |
| `OPS_GITHUB_REPO` | `himanshu2604/mlops-fraud-pipeline` |
| `OPS_PROMETHEUS_URL` | `http://localhost:9090` |
| `GITHUB_TOKEN` | unset (unauthenticated, rate-limited; needed for private repos) |
