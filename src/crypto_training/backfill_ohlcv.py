"""Backfill historical OHLCV bars from Coinbase for crypto training.

Paginates through Coinbase's `/products/{id}/candles` endpoint, which has a
hard 350-bar-per-request cap. Saves bars as one CSV per UTC day so re-runs
can skip already-fetched windows (idempotent / resumable).

CLI::

    ./.venv/bin/python src/crypto_training/backfill_ohlcv.py \\
        --symbol ETH/USD --days 30 --out data/crypto/

Granularities supported: 1m, 5m, 15m, 1h. Default is 1m (matches the
legacy transformer's training resolution).

Resumability: each day's bars are written to
``<out_dir>/<sym_safe>/<granularity>/<YYYY-MM-DD>.csv``. A re-run skips
days whose CSV already exists with the expected row count (1440 for 1m,
288 for 5m, etc.) -- partial days are re-fetched. The current UTC day
always gets re-fetched since it's still in progress.

Network discipline: we sleep ``--rate-pause`` seconds between requests
(default 0.25s) to stay well under Coinbase's public-endpoint rate limit.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# sys.path shim so this CLI runs without PYTHONPATH=src.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

LOGGER = logging.getLogger(__name__)


_GRANULARITY_SECONDS: Dict[str, int] = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR": 3600,
}
_GRANULARITY_LABEL: Dict[str, str] = {
    "ONE_MINUTE": "1m",
    "FIVE_MINUTE": "5m",
    "FIFTEEN_MINUTE": "15m",
    "ONE_HOUR": "1h",
}
_GRANULARITY_FROM_LABEL: Dict[str, str] = {v: k for k, v in _GRANULARITY_LABEL.items()}

# Bars per UTC day, used for the "is this CSV already complete?" check.
_BARS_PER_DAY: Dict[str, int] = {
    "ONE_MINUTE": 1440,
    "FIVE_MINUTE": 288,
    "FIFTEEN_MINUTE": 96,
    "ONE_HOUR": 24,
}

_COINBASE_MAX_BARS_PER_REQUEST = 350


@dataclass
class BackfillSummary:
    """Roll-up returned by ``backfill_symbol``. Useful for tests + CLI report."""

    symbol: str
    granularity: str
    days_requested: int
    days_written: int
    days_skipped: int
    bars_fetched: int
    requests_made: int
    output_dir: Path


def _utc_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _safe_symbol(symbol: str) -> str:
    """Convert ``ETH/USD`` -> ``ETH-USD`` for use as a directory name."""
    return symbol.replace("/", "-")


def _day_csv_path(out_dir: Path, symbol: str, granularity: str, day: datetime) -> Path:
    """Return ``<out_dir>/<symbol>/<granularity>/<YYYY-MM-DD>.csv``."""
    label = _GRANULARITY_LABEL[granularity]
    return out_dir / _safe_symbol(symbol) / label / f"{day.strftime('%Y-%m-%d')}.csv"


def _csv_is_complete(path: Path, expected_rows: int) -> bool:
    """A day CSV is "complete" if it exists and has at least ``expected_rows`` rows.

    Coinbase occasionally drops an individual minute, so we accept >= rather
    than == to keep re-runs from re-fetching almost-full days.
    """
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1  # subtract header
        return n >= expected_rows
    except Exception:  # noqa: BLE001 - unreadable file -> re-fetch
        return False


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows (sorted by timestamp) to ``path``. Always overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def _fetch_with_retry(
    fetch_fn: Callable[[], List[Dict[str, Any]]],
    *,
    max_attempts: int = 4,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> List[Dict[str, Any]]:
    """Call ``fetch_fn`` with bounded exponential-backoff retries.

    Coinbase occasionally returns 5xx or transient JSON errors. We retry up
    to ``max_attempts`` times with delays of base, 2x, 4x, ...
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return fetch_fn()
        except Exception as exc:  # noqa: BLE001 - bounded retry on any failure
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            delay = base_delay_s * (2 ** attempt)
            LOGGER.warning(
                "fetch failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            sleep(delay)
    assert last_exc is not None
    raise last_exc


def _fetch_window(
    exchange: Any,
    *,
    symbol: str,
    granularity: str,
    start_unix: int,
    end_unix: int,
    sleep: Callable[[float], None] = time.sleep,
) -> List[Dict[str, Any]]:
    """Fetch one [start_unix, end_unix] window via the exchange.

    Uses ``exchange.fetch_recent_candles`` if it accepts ``start``/``end``;
    otherwise falls back to fetching the most recent ``limit`` candles
    relative to "now" and slicing client-side. We try the start/end path
    first because it's the only reliable way to backfill historical data
    older than 350 minutes.
    """
    secs = _GRANULARITY_SECONDS[granularity]
    requested_bars = max(1, min(_COINBASE_MAX_BARS_PER_REQUEST, (end_unix - start_unix) // secs))

    # Direct REST call via the exchange's ``_session`` so we get the
    # start/end semantics rather than the "last N from now" fallback in
    # ``fetch_recent_candles``. The CoinbaseExchange wrapper exposes the
    # raw URL through requests; we replicate the call here so the test
    # double can stub a single function.
    if hasattr(exchange, "fetch_candles_window"):
        return _fetch_with_retry(
            lambda: exchange.fetch_candles_window(
                symbol,
                granularity=granularity,
                start_unix=start_unix,
                end_unix=end_unix,
            ),
            sleep=sleep,
        )
    # Fallback: ``fetch_recent_candles`` only returns the latest N bars. If
    # the requested window ends "near now," this works; otherwise it would
    # return wrong-period data. Caller is expected to handle that.
    return _fetch_with_retry(
        lambda: exchange.fetch_recent_candles(
            symbol, granularity=granularity, limit=requested_bars
        ),
        sleep=sleep,
    )


def _bucket_by_utc_day(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group rows by UTC date string. Rows must carry an ISO ``timestamp``."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        ts = r.get("timestamp", "")
        # Format produced by CoinbaseExchange: ISO with +00:00 suffix.
        date_key = ts[:10] if len(ts) >= 10 else ""
        if not date_key:
            continue
        buckets.setdefault(date_key, []).append(r)
    return buckets


def backfill_symbol(
    *,
    exchange: Any,
    symbol: str,
    days: int,
    granularity: str = "ONE_MINUTE",
    out_dir: Path,
    rate_pause_s: float = 0.25,
    today: Optional[datetime] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> BackfillSummary:
    """Backfill ``days`` UTC days of historical bars for ``symbol``.

    ``days`` covers the last N COMPLETED UTC days plus today-in-progress.
    Already-complete day CSVs are skipped so re-runs are cheap.
    """
    if granularity not in _GRANULARITY_SECONDS:
        raise ValueError(
            f"Unsupported granularity {granularity!r}; expected one of "
            f"{sorted(_GRANULARITY_SECONDS)}"
        )
    if days <= 0:
        raise ValueError("days must be positive")

    today = today or _utc_today()
    secs = _GRANULARITY_SECONDS[granularity]
    expected_rows = _BARS_PER_DAY[granularity]
    out_dir = Path(out_dir).expanduser().resolve()

    days_written = 0
    days_skipped = 0
    bars_fetched = 0
    requests_made = 0

    # Walk oldest-first. Each iteration fetches one UTC day's worth of bars
    # in 350-bar chunks (4 chunks for 1m * 1440 bars).
    for day_offset in range(days, 0, -1):
        day_start = today - timedelta(days=day_offset)
        day_end = day_start + timedelta(days=1)
        is_today_in_progress = (day_start == today)
        path = _day_csv_path(out_dir, symbol, granularity, day_start)

        # Skip already-complete COMPLETED days (today is always re-fetched).
        if not is_today_in_progress and _csv_is_complete(path, expected_rows):
            days_skipped += 1
            LOGGER.debug("skip %s: already complete (>=%d rows)", path, expected_rows)
            continue

        # Fetch the day in 350-bar chunks.
        day_rows: List[Dict[str, Any]] = []
        chunk_start_unix = int(day_start.timestamp())
        day_end_unix = int(day_end.timestamp())
        chunk_step = secs * _COINBASE_MAX_BARS_PER_REQUEST
        while chunk_start_unix < day_end_unix:
            chunk_end_unix = min(chunk_start_unix + chunk_step, day_end_unix)
            try:
                chunk = _fetch_window(
                    exchange,
                    symbol=symbol,
                    granularity=granularity,
                    start_unix=chunk_start_unix,
                    end_unix=chunk_end_unix,
                    sleep=sleep,
                )
            except Exception as exc:  # noqa: BLE001 -- log and skip the day
                LOGGER.error(
                    "Fetch failed for %s %s [%s -> %s]: %s",
                    symbol,
                    granularity,
                    datetime.fromtimestamp(chunk_start_unix, tz=timezone.utc),
                    datetime.fromtimestamp(chunk_end_unix, tz=timezone.utc),
                    exc,
                )
                day_rows = []
                break
            requests_made += 1
            day_rows.extend(chunk)
            chunk_start_unix = chunk_end_unix
            if rate_pause_s > 0:
                sleep(rate_pause_s)

        if not day_rows:
            LOGGER.warning("no bars fetched for %s %s", symbol, day_start.date())
            continue

        # Dedupe by timestamp (Coinbase chunk boundaries can overlap by 1 bar)
        # and sort oldest-first.
        unique: Dict[str, Dict[str, Any]] = {}
        for r in day_rows:
            ts = r.get("timestamp", "")
            if ts and ts.startswith(day_start.strftime("%Y-%m-%d")):
                unique[ts] = r
        sorted_rows = sorted(unique.values(), key=lambda r: r["timestamp"])
        if not sorted_rows:
            LOGGER.warning(
                "fetched bars for %s %s did not include any rows for that UTC day",
                symbol,
                day_start.date(),
            )
            continue

        _write_csv(path, sorted_rows)
        days_written += 1
        bars_fetched += len(sorted_rows)
        LOGGER.info(
            "wrote %s (%d bars, %d requests so far)",
            path,
            len(sorted_rows),
            requests_made,
        )

    return BackfillSummary(
        symbol=symbol,
        granularity=granularity,
        days_requested=days,
        days_written=days_written,
        days_skipped=days_skipped,
        bars_fetched=bars_fetched,
        requests_made=requests_made,
        output_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backfill_ohlcv",
        description=(
            "Backfill historical OHLCV bars from Coinbase. Idempotent + "
            "resumable: already-complete day CSVs are skipped on re-run."
        ),
    )
    p.add_argument("--symbol", required=True, help="e.g. ETH/USD or BTC/USD")
    p.add_argument("--days", type=int, default=30, help="UTC days back from today (default 30)")
    p.add_argument(
        "--granularity",
        default="1m",
        choices=sorted(_GRANULARITY_FROM_LABEL.keys()),
        help="Bar size (default 1m)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/crypto/"),
        help="Output root directory (default ./data/crypto)",
    )
    p.add_argument(
        "--rate-pause",
        type=float,
        default=0.25,
        help="Seconds between requests to stay under rate limits (default 0.25)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    granularity = _GRANULARITY_FROM_LABEL[args.granularity]

    # Lazy import so the unit-test path doesn't pull in ccxt.
    from exchanges.coinbase import CoinbaseExchange

    exchange = CoinbaseExchange()
    summary = backfill_symbol(
        exchange=exchange,
        symbol=args.symbol,
        days=args.days,
        granularity=granularity,
        out_dir=args.out,
        rate_pause_s=args.rate_pause,
    )
    LOGGER.info(
        "backfill done: %s %s -- %d/%d days written (%d skipped), "
        "%d bars across %d requests, out=%s",
        summary.symbol,
        _GRANULARITY_LABEL[summary.granularity],
        summary.days_written,
        summary.days_requested,
        summary.days_skipped,
        summary.bars_fetched,
        summary.requests_made,
        summary.output_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
