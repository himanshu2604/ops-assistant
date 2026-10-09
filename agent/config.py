"""Runtime settings, read from the environment at call time so tests can override them."""
import os


def github_repo() -> str:
    return os.getenv("OPS_GITHUB_REPO", "himanshu2604/mlops-fraud-pipeline")


def prometheus_url() -> str:
    return os.getenv("OPS_PROMETHEUS_URL", "http://localhost:9090").rstrip("/")


def github_token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or None


def openai_model() -> str | None:
    """Required for the CLI. No default on purpose: pick a tool-calling model your account can use."""
    return os.getenv("OPS_MODEL") or None


def temperature() -> float | None:
    """Unset by default. Some newer models reject a custom temperature."""
    raw = os.getenv("OPS_TEMPERATURE")
    return float(raw) if raw else None
