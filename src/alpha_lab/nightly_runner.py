"""Orchestrator entrypoint for the alpha-lab nightly run.

Phase 4 / E2 commit 3. Wires :class:`alpha_lab.correlation_miner.CorrelationMiner`
to :class:`alpha_lab.auto_promotion_gate.AutoPromotionGate`, then writes a
daily JSON summary to ``<output_dir>/<UTC-date>.json``. Designed to be the
single entrypoint a cron job calls, e.g.::

    0 5 * * *  cd /srv/autopilot && \
        ./.venv/bin/python -m alpha_lab.nightly_runner \
            --output-dir runs/alpha_lab/

The CLI is a thin wrapper around :class:`NightlyRunner` so the same code
path is reachable from a cron line, a one-off operator invocation, or an
in-process job runner. The wire-up of real data sources (Coinbase REST,
Polymarket macro fetches) is intentionally a future PR — see
``INTEGRATION.md`` for the plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from alpha_lab.auto_promotion_gate import AutoPromotionGate, PromotionCandidate
from alpha_lab.correlation_miner import CorrelationMiner, CorrelationResult

LOGGER = logging.getLogger(__name__)


__all__ = [
    "NightlyRunner",
    "build_arg_parser",
    "main",
]


# Number of top-by-|rank_ic| results to embed in the daily summary JSON. The
# full result set is potentially large (cartesian product over many features)
# so we cap the on-disk summary at 5 to keep it human-skim-able. The full
# set is implicitly retained in the gate's rolling window (Redis or memory).
_TOP_N_IN_SUMMARY: int = 5


@dataclass
class NightlyRunner:
    """Coordinates one nightly mine -> record -> promote -> summarize cycle.

    Attributes:
        miner: a constructed :class:`CorrelationMiner` (with FeatureSources
            already wired). The runner doesn't own the miner's lifecycle —
            the caller is responsible for passing in the right sources.
        gate: a constructed :class:`AutoPromotionGate`. Same lifecycle note.
        output_dir: directory the per-day summary JSON is written into. The
            runner creates it if missing. One file per UTC date; re-running
            on the same day overwrites that day's summary (the audit trail
            for promotions lives in the gate's JSONL queue, not the daily
            summary, so overwriting is safe).
        window_days: rolling window the miner pulls per source. Defaults
            to 30 — matches the gate's min_samples default.
    """

    miner: CorrelationMiner
    gate: AutoPromotionGate
    output_dir: Path
    window_days: int = 30

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_once(self, *, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute one full mine -> record -> promote -> summarize cycle.

        Returns a counts dict with the shape::

            {
                "results_mined": int,
                "promotions_emitted": int,
                "summary_path": str,         # absolute path of the JSON file
                "run_date_utc": "YYYY-MM-DD",
            }

        The same dict is also persisted to disk as the bulk of the daily
        summary. Tests assert against the dict; production cron logs it.
        """
        anchor = now_utc or datetime.now(timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)

        LOGGER.info(
            "alpha_lab nightly: starting (anchor=%s, window_days=%d)",
            anchor.isoformat(),
            self.window_days,
        )

        # 1. Mine.
        results: List[CorrelationResult] = self.miner.mine(
            window_days=self.window_days, end_utc=anchor
        )
        LOGGER.info("alpha_lab nightly: %d results from miner", len(results))

        # 2. Record into the gate.
        for r in results:
            self.gate.record(r)

        # 3. Check for promotions (writes to JSONL queue internally).
        candidates: List[PromotionCandidate] = self.gate.check_promotions()
        LOGGER.info(
            "alpha_lab nightly: %d promotion candidates emitted", len(candidates)
        )

        # 4. Daily summary JSON (top-N by |rank_ic|).
        summary_path = self._write_daily_summary(
            anchor=anchor, results=results, candidates=candidates
        )

        return {
            "results_mined": len(results),
            "promotions_emitted": len(candidates),
            "summary_path": str(summary_path),
            "run_date_utc": anchor.strftime("%Y-%m-%d"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _write_daily_summary(
        self,
        *,
        anchor: datetime,
        results: Sequence[CorrelationResult],
        candidates: Sequence[PromotionCandidate],
    ) -> Path:
        """Write ``<output_dir>/<UTC-date>.json``.

        Format: counts + the top-5 most-correlated pairs (by |rank_ic|) +
        the candidates emitted on this run. Designed so an operator can
        ``cat`` the file and immediately see whether tonight's run produced
        anything worth a follow-up.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_str = anchor.strftime("%Y-%m-%d")
        path = self.output_dir / f"{date_str}.json"

        top_n = list(results[:_TOP_N_IN_SUMMARY])
        payload: Dict[str, Any] = {
            "run_date_utc": date_str,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "window_days": self.window_days,
            "counts": {
                "results_mined": len(results),
                "promotions_emitted": len(candidates),
            },
            "top_results": [
                {
                    "pair": {
                        "feature_a": r.pair.feature_a,
                        "feature_b": r.pair.feature_b,
                        "horizon_bars": r.pair.horizon_bars,
                        "asset_class_a": r.pair.asset_class_a,
                        "asset_class_b": r.pair.asset_class_b,
                        "stable_id": r.pair.stable_id(),
                    },
                    "rank_ic": r.rank_ic,
                    "n_samples": r.n_samples,
                    "pvalue": r.pvalue,
                }
                for r in top_n
            ],
            "promotion_candidates": [c.to_dict() for c in candidates],
        }

        # Atomic write: tmp -> rename so a kill mid-write can't leave a
        # half-formatted JSON file. Mirrors the outcome-weight adjuster's
        # discipline.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser used by :func:`main`.

    Exposed as a module-level helper so tests can introspect the parser
    without paying the import cost of the real miner / gate.
    """
    parser = argparse.ArgumentParser(
        prog="alpha_lab.nightly_runner",
        description=(
            "Run one alpha-lab nightly cycle: mine cross-asset correlations, "
            "feed into the auto-promotion gate, write a daily summary JSON."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/alpha_lab/"),
        help="Directory to write the per-day summary JSON into. Created if "
        "missing. Defaults to runs/alpha_lab/.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling window the miner pulls per source. Defaults to 30.",
    )
    parser.add_argument(
        "--threshold-rank-ic",
        type=float,
        default=0.05,
        help="|rank_ic| floor for the promotion gate. Defaults to 0.05.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum rolling samples before a pair is eligible for promotion. "
        "Defaults to 30.",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Optional Redis connection string. When set, gate history persists "
        "across runs in Redis.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,15,60",
        help="Comma-separated list of forward horizons (in bars). Defaults to 5,15,60.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def _parse_horizons(raw: str) -> List[int]:
    """Parse a comma-separated horizon list. Empty / malformed -> default."""
    out: List[int] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            raise SystemExit(f"--horizons: invalid integer {chunk!r}")
        if value <= 0:
            raise SystemExit(f"--horizons: must be positive, got {value}")
        out.append(value)
    if not out:
        raise SystemExit("--horizons: must contain at least one positive integer")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code.

    Real wiring of feature sources is intentionally TODO — once the
    :class:`alpha_lab.feature_sources.CryptoFeatureSource` and
    :class:`PolymarketFeatureSource` adapters are production-ready (currently
    skeleton in commit 4), this main() builds them from env vars / CLI flags
    and constructs a :class:`NightlyRunner`. For now, main() validates the
    args and prints a "wire your sources" hint so operators don't think the
    cron job is running headlessly.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    horizons = _parse_horizons(args.horizons)

    # Build the gate eagerly — its construction is cheap and validates the
    # threshold / min_samples / redis_url combo before we report a wiring
    # error from the source side.
    gate = AutoPromotionGate(
        threshold_rank_ic=args.threshold_rank_ic,
        min_samples=args.min_samples,
        redis_url=args.redis_url,
        promotion_queue_path=Path(args.output_dir) / "promotion_queue.jsonl",
    )

    # Source wiring is the future-PR seam. We import here so unit tests can
    # mock the entire factory without the import cost of the data adapters.
    try:
        from alpha_lab.feature_sources import build_default_feature_sources  # noqa: F401
        sources = build_default_feature_sources()
    except (ImportError, NotImplementedError):
        sources = []

    if not sources:
        LOGGER.warning(
            "alpha_lab nightly: no feature sources wired yet — see "
            "src/alpha_lab/INTEGRATION.md for how to wire CryptoFeatureSource "
            "and PolymarketFeatureSource. Exiting without running the miner."
        )
        return 0

    miner = CorrelationMiner(sources, horizon_bars_options=horizons)
    runner = NightlyRunner(
        miner=miner, gate=gate, output_dir=args.output_dir, window_days=args.window_days
    )
    counts = runner.run_once()
    LOGGER.info("alpha_lab nightly: done (%s)", counts)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
