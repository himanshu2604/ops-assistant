"""Latest GitHub Actions run for a repo, optionally narrowed to a workflow or branch."""
import re
from typing import Any

from .. import config
from ._http import HttpGet, http_get_json
from .base import Tool

GITHUB_API = "https://api.github.com"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
WORKFLOW_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def get_latest_ci_run(
    repo: str | None = None,
    workflow: str | None = None,
    branch: str | None = None,
    http_get: HttpGet = http_get_json,
) -> dict[str, Any]:
    repo = repo or config.github_repo()
    if not REPO_RE.match(repo):
        raise ValueError(f"invalid repo, expected owner/name: {repo!r}")
    if workflow is not None and not WORKFLOW_RE.match(workflow):
        raise ValueError(f"invalid workflow: {workflow!r}")
    if branch is not None and not BRANCH_RE.match(branch):
        raise ValueError(f"invalid branch: {branch!r}")

    if workflow:
        url = f"{GITHUB_API}/repos/{repo}/actions/workflows/{workflow}/runs"
    else:
        url = f"{GITHUB_API}/repos/{repo}/actions/runs"
    params = {"per_page": "1"}
    if branch:
        params["branch"] = branch

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = config.github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    runs = http_get(url, headers, params).get("workflow_runs", [])
    if not runs:
        return {"repo": repo, "found": False}

    run = runs[0]
    completed = run.get("status") == "completed"
    return {
        "repo": repo,
        "found": True,
        "workflow": run.get("name"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        # None while the run is still going, so the agent can say "not finished yet".
        "passed": (run.get("conclusion") == "success") if completed else None,
        "branch": run.get("head_branch"),
        "commit": (run.get("head_sha") or "")[:7],
        "event": run.get("event"),
        "created_at": run.get("created_at"),
        "url": run.get("html_url"),
    }


TOOL = Tool(
    name="get_latest_ci_run",
    description=(
        "Get the most recent GitHub Actions run: status, conclusion, branch, commit. "
        "Defaults to the project's own repo. 'passed' is null while the run is still in progress."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "owner/name. Omit to use the default repo."},
            "workflow": {"type": "string", "description": "Workflow file name, e.g. ci.yml"},
            "branch": {"type": "string", "description": "Only consider runs on this branch"},
        },
        "required": [],
        "additionalProperties": False,
    },
    fn=get_latest_ci_run,
)
