"""FastAPI dependency providers for the prediction-market API.

Each provider returns a small, swappable object so endpoints can be tested
without touching the filesystem or instantiating heavy dependencies.

Trade-store discovery
---------------------
The trade store directory (where ``trade_execution_<id>.json`` logs live) is
configurable via the ``AUTOPILOT_TRADE_STORE`` environment variable. The
default matches the orchestrator's current behavior (the repo root). Tests
should monkey-patch this env var to point at a temp dir before constructing
the app.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from outcome_review_agent.logger import PerformanceTracker
from risk_management_agent.risk_engine import RiskCalculator
from storage import SQLITE_PATH_ENV_VAR, SQLiteStore, get_default_store


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRADE_STORE_ENV_VAR = "AUTOPILOT_TRADE_STORE"


def get_trade_store_dir() -> Path:
    """Return the directory where trade execution logs are read / written.

    Resolves ``AUTOPILOT_TRADE_STORE`` lazily on every call so tests can
    monkey-patch the env var between requests. Falls back to the repo root,
    matching ``orchestrator.run_final_risk_gate``'s current write location.
    """

    raw = os.environ.get(TRADE_STORE_ENV_VAR, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT


def get_audit_file_path() -> Path:
    """Return the path to ``performance_audit.json`` inside the trade store dir."""

    return get_trade_store_dir() / "performance_audit.json"


def get_risk_calculator() -> RiskCalculator:
    """Construct a default RiskCalculator. Override per-request via env vars."""

    return RiskCalculator()


def _noop_review_agent_factory() -> Any:
    """A do-nothing review agent stub used when no real agent is wired in.

    PerformanceTracker requires *some* agent. For the read-only ``/postmortems``
    endpoint we never call ``process_settled_trades`` so the agent is unused;
    this stub keeps construction cheap and import-side-effect free (no Gemini
    client built at app startup).
    """

    class _NoopReviewAgent:
        def review_trade(self, trade_payload: dict) -> dict:  # noqa: ARG002
            return {
                "matrix_classification": "Deserved Success",
                "thesis_held": True,
                "unknown_at_entry": False,
                "calibration_reasonable": True,
                "resulting_detected": False,
                "key_takeaways": [],
                "reasoning": "Stub review (no LLM call).",
            }

    return _NoopReviewAgent()


def get_sqlite_store() -> Optional[SQLiteStore]:
    """Return the process-wide SQLite store when ``AUTOPILOT_SQLITE_PATH`` is set.

    Returns ``None`` when the env var is unset so callers can fall back to the
    canonical JSON read path. Resolved lazily on every call so tests can flip
    the env var between requests.
    """

    return get_default_store()


def get_performance_tracker(
    *,
    review_agent: Any | None = None,
    trade_store_dir: Path | None = None,
) -> PerformanceTracker:
    """Construct a ``PerformanceTracker`` pointed at the trade store dir.

    The agent defaults to a stub so we never block on a real Gemini call when
    the caller only wants to read the audit file.
    """

    store_dir = trade_store_dir if trade_store_dir is not None else get_trade_store_dir()
    agent = review_agent if review_agent is not None else _noop_review_agent_factory()
    return PerformanceTracker(
        trade_store_dir=store_dir,
        outcome_review_agent=agent,
        audit_file=store_dir / "performance_audit.json",
    )
