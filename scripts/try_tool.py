"""Run one tool against real infra. Example:
    python scripts/try_tool.py get_pod_status '{"namespace": "mlops-fraud"}'
"""
import json
import sys

sys.path.insert(0, ".")
from agent.tools import run_tool  # noqa: E402

name = sys.argv[1]
args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
print(json.dumps(run_tool(name, args), indent=2))
