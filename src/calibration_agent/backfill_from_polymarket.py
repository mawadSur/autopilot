"""Backfill synthetic trade execution logs from resolved Polymarket markets.

Pulls historical resolved markets via :func:`fetcher.fetch_resolved_markets`
and converts each ``(Market, market_outcome)`` pair into a synthetic
``trade_execution_<market_id>.json`` written to ``output_dir``. The resulting
logs share the canonical schema with the orchestrator (``source`` /
``notes`` / ``status`` / ``features_window`` / ``market_outcome`` etc.) so
``build_dataset.py`` can ingest them directly.

FIDELITY CAVEAT (also surfaced via the ``notes`` field on every backfilled
log): ``features_window`` reflects post-resolution market state, not the
state at original decision time. Volume_24h, spread, and price-change deltas
are degraded. These rows exist to smoke-test the training pipeline today
without waiting weeks for live paper-trade volume; they should NOT be used
to evaluate calibration quality in production.

CLI::

    ./.venv/bin/python src/calibration_agent/backfill_from_polymarket.py \\
        --output-dir ./backfill --limit 200 --min-volume 5000 --days-back 90
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Mirror the sys.path shim used by build_dataset.py so this CLI runs without
# the caller setting PYTHONPATH. ``calibration_agent.__init__`` imports
# ``analyzer`` which uses flat ``from models import Market`` against ``src/``.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration_agent.ml_service import (  # noqa: E402  (after sys.path shim)
    extract_market_features,
    extract_research_features,
)
from models import Market  # noqa: E402  (after sys.path shim)

logger = logging.getLogger(__name__)

PathLike = Path

ResolvedFetcherFn = Callable[..., List[Tuple[Market, bool]]]

BACKFILL_NOTES = (
    "Backfilled from resolved Polymarket market - features_window reflects "
    "post-resolution market state, not features at original decision time. "
    "Volume_24h, spread, and price_change_* are degraded fidelity. Use only "
    "for smoke-testing the training pipeline; not for production calibration."
)

DEFAULT_LIMIT = 500
DEFAULT_MIN_VOLUME = 5_000.0


def _resolution_iso(market: Market) -> str:
    """Return the market's resolution_date as an ISO-8601 string in UTC."""

    resolution = getattr(market, "resolution_date", None)
    if isinstance(resolution, datetime):
        if resolution.tzinfo is None:
            resolution = resolution.replace(tzinfo=timezone.utc)
        return resolution.astimezone(timezone.utc).isoformat()
    if isinstance(resolution, str) and resolution.strip():
        return resolution
    return datetime.now(timezone.utc).isoformat()


def _build_scanner_row(market: Market) -> Dict[str, Any]:
    """Compact scanner-row stub for the synthetic event payload."""

    return {
        "market_id": market.market_id,
        "title": market.title,
        "category": market.category,
        "implied_prob": float(market.implied_prob),
        "volume_24h": float(market.volume_24h or 0.0),
    }


def _build_event_payload(market: Market, market_outcome: bool) -> Dict[str, Any]:
    """Assemble a backfill-shaped event payload matching the canonical schema."""

    timestamp = _resolution_iso(market)
    features_window = {
        **extract_market_features(market),
        **extract_research_features(None, None),
    }
    return {
        "event_id": market.market_id,
        "trade_id": market.market_id,
        "status": "settled",
        "created_at_utc": timestamp,
        "settled_at": timestamp,
        # Following mark_trade_settled.py's always-long-YES convention:
        # we did not actually trade these, so final_outcome mirrors
        # market_outcome (a hypothetical YES position would have won/lost
        # iff the market resolved YES/NO).
        "final_outcome": bool(market_outcome),
        "market_outcome": bool(market_outcome),
        "post_settlement_news": None,
        "scanner": _build_scanner_row(market),
        "features_window": features_window,
        "model_meta": None,
        "research": None,
        "calibration": None,
        "risk": None,
        "source": "backfill",
        "notes": BACKFILL_NOTES,
    }


def _write_trade_log(output_dir: Path, payload: Dict[str, Any]) -> Path:
    market_id = str(payload.get("trade_id") or payload.get("event_id") or "").strip()
    if not market_id:
        raise ValueError("Backfill payload is missing trade_id / event_id")
    output_path = output_dir / f"trade_execution_{market_id}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def backfill(
    output_dir: PathLike,
    *,
    limit: int = DEFAULT_LIMIT,
    min_volume_24h: float = DEFAULT_MIN_VOLUME,
    days_back: Optional[int] = None,
    fetcher: Optional[ResolvedFetcherFn] = None,
) -> Dict[str, Any]:
    """Pull resolved markets and write synthetic trade logs into ``output_dir``.

    ``fetcher`` is dependency-injected for testability; defaults to
    :func:`fetcher.fetch_resolved_markets` resolved lazily so importing this
    module never touches the network.

    Returns a summary dict suitable for printing or piping into another tool::

        {
            "markets_fetched": int,
            "trade_logs_written": int,
            "skipped_ambiguous_resolution": int,  # always 0 here; the fetcher
                                                  # logs ambiguous skips
            "output_dir": str,
        }
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fetch_fn = fetcher
    if fetch_fn is None:
        # Lazy import keeps this module test-isolated: importing
        # backfill_from_polymarket alone does not pull requests / network.
        from fetcher import fetch_resolved_markets as _default_fetcher  # noqa: WPS433

        fetch_fn = _default_fetcher

    pairs = list(
        fetch_fn(
            min_volume_24h=min_volume_24h,
            days_back=days_back,
        )
    )
    markets_fetched = len(pairs)

    logs_written = 0
    capped_pairs = pairs[: max(0, int(limit))]
    for market, market_outcome in capped_pairs:
        payload = _build_event_payload(market, market_outcome)
        _write_trade_log(output_path, payload)
        logs_written += 1

    summary: Dict[str, Any] = {
        "markets_fetched": markets_fetched,
        "trade_logs_written": logs_written,
        "skipped_ambiguous_resolution": 0,
        "output_dir": str(output_path),
    }
    print(
        f"backfill: fetched={markets_fetched} written={logs_written} "
        f"output_dir={output_path}",
        file=sys.stderr,
    )
    return summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill synthetic trade execution logs from resolved Polymarket markets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./backfill"),
        help="Directory to write trade_execution_<market_id>.json files into.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of resolved markets to convert into trade logs.",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_MIN_VOLUME,
        help="Minimum historical volume threshold (USD) for a market to be backfilled.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Optional cutoff: only include markets whose endDate is within N days.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    summary = backfill(
        args.output_dir,
        limit=args.limit,
        min_volume_24h=args.min_volume,
        days_back=args.days_back,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
