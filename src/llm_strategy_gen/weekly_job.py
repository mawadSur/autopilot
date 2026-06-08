"""WeeklyJob — top-level orchestrator for the E3 LLM Strategy-Gen loop.

Runs the four-stage pipeline end-to-end:

1. :class:`OutcomeAnalyst.analyze()` — read postmortems + audit log,
   return ranked patterns.
2. :class:`FeatureProposalGenerator.propose()` — turn patterns into
   validated Python feature proposals.
3. :class:`BacktestRunner.run()` — Sharpe-gate each proposal.
4. :class:`PROpener.open_pr()` — for each proposal that cleared the
   gate, materialize files and (in live mode) open a draft PR.

Returns a small dict of counts so the cron job has a one-line audit
trail per run.

CLI
---
``python -m llm_strategy_gen.weekly_job --dry-run`` runs the pipeline
end-to-end with mocked-out wiring. The full live invocation is
documented in ``INTEGRATION.md`` next to this module.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_strategy_gen.backtest_runner import BacktestRunner
from llm_strategy_gen.feature_proposal import FeatureProposalGenerator
from llm_strategy_gen.outcome_analyst import OutcomeAnalyst
from llm_strategy_gen.pr_opener import PROpener

LOGGER = logging.getLogger(__name__)


# Cron-tunable env var for the Sharpe-lift gate. Read here so an
# operator can bump it without code changes.
_GATE_ENV_VAR = "AUTOPILOT_LLM_STRATEGY_GATE"


class WeeklyJob:
    """Top-level orchestrator. Glue, not logic.

    Parameters
    ----------
    analyst, generator, runner, opener:
        Pre-constructed components. The CLI builds these with default
        wiring; tests pass mocks.
    window_days:
        How many days of postmortems / audits the analyst should
        consume.
    dry_run:
        Whether to skip the actual ``gh pr create`` invocation. Defaults
        to ``True`` because the skeleton's gh-runner integration is a
        stub.
    """

    def __init__(
        self,
        analyst: OutcomeAnalyst,
        generator: FeatureProposalGenerator,
        runner: BacktestRunner,
        opener: PROpener,
        *,
        window_days: int = 7,
        dry_run: bool = True,
    ) -> None:
        self.analyst = analyst
        self.generator = generator
        self.runner = runner
        self.opener = opener
        self.window_days = int(window_days)
        self.dry_run = bool(dry_run)

    def run(self) -> Dict[str, int]:
        """Execute the pipeline; return per-stage counts."""
        counts: Dict[str, int] = {
            "patterns": 0,
            "proposals": 0,
            "backtests_run": 0,
            "passed_gate": 0,
            "prs_opened": 0,
        }

        try:
            patterns = self.analyst.analyze(window_days=self.window_days)
        except Exception as exc:  # noqa: BLE001 - never raise from a cron job
            LOGGER.error("WeeklyJob.analyze crashed: %r", exc)
            return counts
        counts["patterns"] = len(patterns)

        if not patterns:
            LOGGER.info("WeeklyJob: no patterns found in %dd window", self.window_days)
            return counts

        try:
            proposals = self.generator.propose(patterns)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("WeeklyJob.propose crashed: %r", exc)
            return counts
        counts["proposals"] = len(proposals)

        if not proposals:
            LOGGER.info("WeeklyJob: no valid proposals from %d patterns", len(patterns))
            return counts

        for proposal in proposals:
            try:
                result = self.runner.run(proposal)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "WeeklyJob.runner crashed for %r (%r); skipping",
                    proposal.name,
                    exc,
                )
                continue
            counts["backtests_run"] += 1
            if not result.passed_gate:
                continue
            counts["passed_gate"] += 1

            try:
                pr = self.opener.open_pr(
                    proposal, result, dry_run=self.dry_run
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "WeeklyJob.opener crashed for %r (%r); skipping",
                    proposal.name,
                    exc,
                )
                continue
            if pr is not None:
                counts["prs_opened"] += 1

        return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_default_job(
    *,
    repo_root: Path,
    dry_run: bool,
    gate_threshold: float,
) -> WeeklyJob:
    """Construct a WeeklyJob with default wiring.

    The skeleton wires every component up with NO LLM caller — so a
    real ``--no-dry-run`` run with no env config is still a no-op
    (analyst returns empty list when the postmortems dir is empty).
    """
    postmortems_dir = repo_root / "runs" / "postmortems"
    audit_path = repo_root / "runs" / "performance_audit.json"
    dataset_path = repo_root / "data" / "crypto" / "datasets"

    analyst = OutcomeAnalyst(
        postmortems_dir=postmortems_dir,
        performance_audit_path=audit_path if audit_path.exists() else None,
    )
    generator = FeatureProposalGenerator()
    runner = BacktestRunner(
        dataset_path=dataset_path,
        gate_threshold=gate_threshold,
    )
    opener = PROpener(repo_root=repo_root)
    return WeeklyJob(
        analyst, generator, runner, opener,
        dry_run=dry_run,
    )


def _resolve_gate_threshold(cli_value: Optional[float]) -> float:
    if cli_value is not None:
        return float(cli_value)
    env_value = os.getenv(_GATE_ENV_VAR)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            LOGGER.warning(
                "WeeklyJob: %s=%r is not a number; using default 0.2",
                _GATE_ENV_VAR,
                env_value,
            )
    return 0.2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm_strategy_gen.weekly_job",
        description="E3 LLM Strategy-Gen weekly job (skeleton).",
    )
    parser.add_argument(
        "--repo-root", default=".",
        help="Repository root path (defaults to cwd)",
    )
    parser.add_argument(
        "--window-days", type=int, default=7,
        help="How many days of postmortems to scan",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=None,
        help=f"Sharpe lift gate; falls back to env {_GATE_ENV_VAR} (default 0.2)",
    )
    dry_group = parser.add_mutually_exclusive_group()
    dry_group.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True,
        help="Skip gh pr create; only write artifacts (default)",
    )
    dry_group.add_argument(
        "--no-dry-run", dest="dry_run", action="store_false",
        help="Live mode — would call gh pr create (requires GH_TOKEN)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase logging verbosity",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(args.repo_root).resolve()
    gate_threshold = _resolve_gate_threshold(args.gate_threshold)
    job = _build_default_job(
        repo_root=repo_root,
        dry_run=args.dry_run,
        gate_threshold=gate_threshold,
    )
    job.window_days = int(args.window_days)
    counts = job.run()

    print(
        "WeeklyJob run complete: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )
    return 0


__all__ = ["WeeklyJob", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    sys.exit(main())
