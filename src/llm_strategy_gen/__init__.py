"""LLM Strategy-Generation Loop (CEO Plan Phase 5 / E3) — SKELETON.

This package implements the weekly Claude/Gemini-driven loop that:

1. Reads recent loss postmortem reports + the legacy
   ``performance_audit.json`` (:class:`OutcomeAnalyst`).
2. Asks the LLM to propose 3 new feature/strategy ideas as compilable
   Python (:class:`FeatureProposalGenerator`).
3. Backtests each proposal and applies a Sharpe-lift gate
   (:class:`BacktestRunner`).
4. Opens a draft PR (via ``gh``) for any proposal that clears the gate
   (:class:`PROpener`).
5. Orchestrates the full pipeline (:class:`WeeklyJob`).

This is the *skeleton* — interfaces and smoke-test-level functionality.
Real LLM integration, secure sandboxing of proposed code, and real
worktree+PR plumbing are tracked as follow-up PRs. See
``INTEGRATION.md`` in this package for required env, cron schedule,
operator runbook, and the explicit security-deferred items.

Submodules are lazily imported so importing this package is cheap and
side-effect free; pulling in :mod:`requests` / :mod:`pyarrow` etc. only
happens when callers reach for the actual classes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "OutcomeAnalyst",
    "OutcomePattern",
    "FeatureProposal",
    "FeatureProposalGenerator",
    "BacktestResult",
    "BacktestRunner",
    "PROpener",
    "WeeklyJob",
]


if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from llm_strategy_gen.outcome_analyst import OutcomeAnalyst, OutcomePattern
    from llm_strategy_gen.feature_proposal import (
        FeatureProposal,
        FeatureProposalGenerator,
    )
    from llm_strategy_gen.backtest_runner import BacktestResult, BacktestRunner
    from llm_strategy_gen.pr_opener import PROpener
    from llm_strategy_gen.weekly_job import WeeklyJob


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy re-export
    """Lazy submodule re-export so importing this package is cheap.

    Resolves names like ``OutcomeAnalyst`` to the right submodule on
    first access.
    """
    if name in {"OutcomeAnalyst", "OutcomePattern"}:
        from llm_strategy_gen.outcome_analyst import OutcomeAnalyst, OutcomePattern
        return {"OutcomeAnalyst": OutcomeAnalyst, "OutcomePattern": OutcomePattern}[name]
    if name in {"FeatureProposal", "FeatureProposalGenerator"}:
        from llm_strategy_gen.feature_proposal import (
            FeatureProposal,
            FeatureProposalGenerator,
        )
        return {
            "FeatureProposal": FeatureProposal,
            "FeatureProposalGenerator": FeatureProposalGenerator,
        }[name]
    if name in {"BacktestResult", "BacktestRunner"}:
        from llm_strategy_gen.backtest_runner import BacktestResult, BacktestRunner
        return {
            "BacktestResult": BacktestResult,
            "BacktestRunner": BacktestRunner,
        }[name]
    if name == "PROpener":
        from llm_strategy_gen.pr_opener import PROpener
        return PROpener
    if name == "WeeklyJob":
        from llm_strategy_gen.weekly_job import WeeklyJob
        return WeeklyJob
    raise AttributeError(f"module 'llm_strategy_gen' has no attribute {name!r}")
