# ops-assistant

A small agent that answers infrastructure questions about the mlops-fraud-pipeline project by calling read-only tools. Calls are traced and evaluated with Galileo AI.

Status: tools and agent loop built and unit-tested. Galileo tracing, eval suite and CI gate are not built yet.

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

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
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
| `OPS_TEMPERATURE` | unset (some newer models reject a custom temperature) |
| `OPS_GITHUB_REPO` | `himanshu2604/mlops-fraud-pipeline` |
| `OPS_PROMETHEUS_URL` | `http://localhost:9090` |
| `GITHUB_TOKEN` | unset (unauthenticated, rate-limited; needed for private repos) |
