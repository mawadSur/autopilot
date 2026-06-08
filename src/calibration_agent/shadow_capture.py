"""Shadow-capture CLI for the calibration baseline.

Walks the active Polymarket markets at configurable intervals and writes
pseudo-trade-execution logs that look exactly like orchestrator output, except
marked ``source="shadow"``. The goal is to accumulate clean
``(features_at_decision_time, market_outcome)`` data faster than relying on the
orchestrator's per-trade decision flow.

Each capture writes one JSON file per market into ``output_dir``, named
``shadow_<UTC-timestamp>_<market_id>.json``. The schema mirrors the
orchestrator's ``_write_trade_execution_log`` payload (see
:mod:`orchestrator.run_final_risk_gate`) but with research / calibration / risk
slots set to ``None`` because shadow capture does not run those agents. The
``features_window`` therefore contains only the 8 market-microstructure columns
plus ``captured_at_utc``; downstream
:func:`calibration_agent.ml_service._full_feature_vector` defaults the 6
research columns to 0.0 when those keys are absent.

The companion ``mark_trade_settled`` CLI (used by the orchestrator's settled
trades workflow) can later mark these shadow logs as ``settled`` once the
underlying market resolves, producing the ``(features, label)`` pair the
calibration baseline needs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Mirror the sys.path shim used by build_dataset.py so this CLI runs without
# the caller setting PYTHONPATH. ``calibration_agent.__init__`` imports
# ``analyzer`` which uses flat ``from models import Market`` against ``src/``.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration_agent.ml_service import extract_market_features
from models import Market

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
FetcherFn = Callable[..., Sequence[Market]]
NowFn = Callable[[], datetime]
SleepFn = Callable[[float], None]

# Filename timestamp uses ``YYYYMMDDTHHMMSSZ`` for filesystem-friendliness and
# lexicographic sort order. The ISO timestamp baked into the JSON payload uses
# the canonical ``isoformat()`` form for downstream parsers.
_FILENAME_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _utc_now() -> datetime:
    """Default ``now`` factory; isolated for test injection."""

    return datetime.now(timezone.utc)


def _format_filename_timestamp(now: datetime) -> str:
    """Return a filesystem-friendly UTC timestamp for the filename prefix."""

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return now.strftime(_FILENAME_TIMESTAMP_FORMAT)


def _sanitize_market_id(market_id: str) -> str:
    """Make ``market_id`` safe to embed in a filename.

    Polymarket market ids are typically numeric strings, but we defensively
    replace path separators / whitespace just in case a future API returns
    something exotic.
    """

    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(market_id)
    )
    return cleaned or "unknown"


def _build_shadow_payload(
    market: Market,
    *,
    now: datetime,
) -> Dict[str, Any]:
    """Construct the shadow trade-log payload for a single market.

    Schema mirrors the orchestrator's ``event_payload`` (see
    :func:`orchestrator.run_final_risk_gate`). ``research``, ``calibration``,
    ``risk`` and ``model_meta`` are ``None`` because shadow capture does not
    invoke those agents. ``source="shadow"`` and ``notes=None`` align with the
    canonical schema being added in the parallel Pass 1a work.
    """

    market.refresh_derived_fields()
    return {
        "event_id": market.market_id,
        "trade_id": market.market_id,
        "status": "open",
        "created_at_utc": now.isoformat(),
        "settled_at": None,
        "final_outcome": None,
        "market_outcome": None,
        "post_settlement_news": None,
        "scanner": {
            "market_id": market.market_id,
            "title": market.title,
            "category": market.category,
            "implied_prob": market.implied_prob,
            "volume_24h": market.volume_24h,
        },
        "features_window": extract_market_features(market),
        "model_meta": None,
        "research": None,
        "calibration": None,
        "risk": None,
        "source": "shadow",
        "notes": None,
    }


def _write_shadow_log(
    payload: Dict[str, Any],
    *,
    output_dir: Path,
    now: datetime,
) -> Path:
    """Serialize one shadow payload; returns the written path."""

    market_id = _sanitize_market_id(payload.get("trade_id") or "unknown")
    timestamp = _format_filename_timestamp(now)
    file_path = output_dir / f"shadow_{timestamp}_{market_id}.json"
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return file_path


def capture_once(
    output_dir: PathLike,
    *,
    min_volume_24h: float = 5_000.0,
    page_size: int = 100,
    max_pages: Optional[int] = None,
    fetcher: Optional[FetcherFn] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Fetch active markets once and write a shadow log per market.

    Parameters
    ----------
    output_dir:
        Directory to write ``shadow_<timestamp>_<market_id>.json`` files into.
        Created (with parents) if missing.
    min_volume_24h, page_size, max_pages:
        Forwarded to :func:`fetcher.fetch_active_markets` (overridable for tests).
    fetcher:
        Optional fetch function for testability. Defaults to the real
        :func:`fetcher.fetch_active_markets`. Imported lazily so the module can
        be imported without making any network call.
    now:
        Optional ``datetime`` to use as the capture timestamp (UTC). Defaults
        to :func:`datetime.now` ``(timezone.utc)``. Useful for deterministic
        tests.

    Returns
    -------
    dict
        Summary ``{"markets_captured": N, "output_dir": str(...),
        "captured_at_utc": iso}``. Also printed to stderr.
    """

    captured_at = (now or _utc_now()).astimezone(timezone.utc)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if fetcher is None:
        # Lazy import: keep network-touching modules out of import-time graph.
        from fetcher import fetch_active_markets as _real_fetch  # noqa: WPS433

        fetcher = _real_fetch

    markets = list(
        fetcher(
            min_volume_24h=min_volume_24h,
            page_size=page_size,
            max_pages=max_pages,
        )
    )

    written: List[Path] = []
    for market in markets:
        payload = _build_shadow_payload(market, now=captured_at)
        written.append(
            _write_shadow_log(payload, output_dir=output_path, now=captured_at)
        )

    summary: Dict[str, Any] = {
        "markets_captured": len(written),
        "output_dir": str(output_path),
        "captured_at_utc": captured_at.isoformat(),
    }
    print(
        f"[shadow_capture] Captured {summary['markets_captured']} markets "
        f"into {summary['output_dir']} at {summary['captured_at_utc']}",
        file=sys.stderr,
    )
    return summary


def capture_loop(
    output_dir: PathLike,
    *,
    interval_seconds: float,
    max_iterations: Optional[int] = None,
    sleep_fn: SleepFn = time.sleep,
    capture_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    **capture_kwargs: Any,
) -> Dict[str, Any]:
    """Run :func:`capture_once` in a loop until interrupted.

    ``KeyboardInterrupt`` is caught cleanly so Ctrl+C produces a final summary
    line instead of a traceback. Per-iteration exceptions are logged at ERROR
    level and the loop continues — a transient API outage should not kill a
    long-running shadow daemon.

    Parameters
    ----------
    output_dir, capture_kwargs:
        Forwarded to :func:`capture_once`.
    interval_seconds:
        Seconds to sleep between successful iterations. Sleep also runs after
        a failed iteration so we don't tight-loop against a broken upstream.
    max_iterations:
        Optional cap on the number of iterations. ``None`` (the default) runs
        forever until ``KeyboardInterrupt``. Tests pass a small integer to keep
        the loop bounded.
    sleep_fn:
        Sleep function (defaults to :func:`time.sleep`). Tests pass a no-op /
        counting stub so they don't actually sleep.
    capture_fn:
        Optional override for the per-iteration capture callable, primarily
        for tests. Defaults to :func:`capture_once`.

    Returns
    -------
    dict
        Summary ``{"iterations": N, "errors": M, "interrupted": bool,
        "last_summary": <last capture_once summary or None>}``.
    """

    capture = capture_fn or capture_once
    iterations = 0
    errors = 0
    last_summary: Optional[Dict[str, Any]] = None
    interrupted = False

    try:
        while True:
            if max_iterations is not None and iterations >= max_iterations:
                break
            try:
                last_summary = capture(output_dir, **capture_kwargs)
            except KeyboardInterrupt:
                # Re-raise so the outer handler emits the final summary; do
                # not count this as an iteration error.
                raise
            except Exception as exc:  # noqa: BLE001 - log + keep the loop alive
                errors += 1
                logger.error(
                    "shadow_capture iteration %d failed: %s", iterations + 1, exc,
                    exc_info=True,
                )
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            try:
                sleep_fn(interval_seconds)
            except KeyboardInterrupt:
                raise
    except KeyboardInterrupt:
        interrupted = True
        print(
            "[shadow_capture] Interrupted by user after "
            f"{iterations} iteration(s); {errors} error(s).",
            file=sys.stderr,
        )

    summary = {
        "iterations": iterations,
        "errors": errors,
        "interrupted": interrupted,
        "last_summary": last_summary,
    }
    print(
        "[shadow_capture] Loop finished: "
        f"iterations={summary['iterations']} errors={summary['errors']} "
        f"interrupted={summary['interrupted']}",
        file=sys.stderr,
    )
    return summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk active Polymarket markets at intervals and write shadow "
            "trade-execution logs for the calibration baseline dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write shadow_<timestamp>_<market_id>.json into.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run a single capture and exit (default if neither mode is given).",
    )
    mode.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Run capture_loop, sleeping this many seconds between iterations.",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=5_000.0,
        dest="min_volume_24h",
        help="Minimum 24h volume filter (USD). Default: 5000.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Polymarket Gamma API page size. Default: 100.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Cap on Gamma API pages per capture. Default: no cap.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap on loop iterations (only meaningful with --interval-seconds).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level. Default: INFO.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    common_kwargs: Dict[str, Any] = {
        "min_volume_24h": args.min_volume_24h,
        "page_size": args.page_size,
        "max_pages": args.max_pages,
    }

    if args.interval_seconds is not None:
        capture_loop(
            args.output_dir,
            interval_seconds=args.interval_seconds,
            max_iterations=args.max_iterations,
            **common_kwargs,
        )
        return 0

    # Default: --once.
    capture_once(args.output_dir, **common_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
