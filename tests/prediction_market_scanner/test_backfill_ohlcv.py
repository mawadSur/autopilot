"""Tests for src/crypto_training/backfill_ohlcv.py.

Hermetic: a fake exchange implements ``fetch_candles_window`` and produces
canned candle dicts. No network, no Coinbase REST.
"""

from __future__ import annotations

import csv
import io
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from crypto_training.backfill_ohlcv import (
    BackfillSummary,
    _bucket_by_utc_day,
    _csv_is_complete,
    _day_csv_path,
    _fetch_with_retry,
    _safe_symbol,
    backfill_symbol,
)


# ---------------------------------------------------------------------------
# Fake exchange
# ---------------------------------------------------------------------------


class _FakeCandlesExchange:
    """Returns synthetic candles for any [start_unix, end_unix] window."""

    def __init__(
        self,
        *,
        granularity_seconds: int = 60,
        gap_unix: int = 0,
        fail_first_n_calls: int = 0,
    ) -> None:
        self.granularity_seconds = granularity_seconds
        self.gap_unix = gap_unix  # synthesise a "missing bar" range
        self.fail_first_n_calls = fail_first_n_calls
        self.calls: List[Dict[str, Any]] = []

    def fetch_candles_window(
        self, symbol: str, *, granularity: str, start_unix: int, end_unix: int
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "symbol": symbol,
                "granularity": granularity,
                "start": start_unix,
                "end": end_unix,
            }
        )
        if len(self.calls) <= self.fail_first_n_calls:
            raise RuntimeError(f"synthetic transient error #{len(self.calls)}")

        rows: List[Dict[str, Any]] = []
        ts = start_unix
        i = 0
        while ts < end_unix:
            if not (self.gap_unix and ts >= self.gap_unix and ts < self.gap_unix + 600):
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(
                            ts, tz=timezone.utc
                        ).isoformat(),
                        "_unix": ts,
                        "open": 100.0 + i * 0.01,
                        "high": 100.5 + i * 0.01,
                        "low": 99.5 + i * 0.01,
                        "close": 100.2 + i * 0.01,
                        "volume": 1.0,
                    }
                )
            ts += self.granularity_seconds
            i += 1
        # Strip _unix to mirror CoinbaseExchange's return shape.
        for r in rows:
            r.pop("_unix", None)
        return rows


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class HelperTests(unittest.TestCase):
    def test_safe_symbol_replaces_slash(self) -> None:
        self.assertEqual(_safe_symbol("ETH/USD"), "ETH-USD")
        self.assertEqual(_safe_symbol("BTC-USD"), "BTC-USD")

    def test_day_csv_path_layout(self) -> None:
        out = Path("/tmp/data")
        day = datetime(2026, 4, 27, tzinfo=timezone.utc)
        path = _day_csv_path(out, "ETH/USD", "ONE_MINUTE", day)
        self.assertEqual(path, Path("/tmp/data/ETH-USD/1m/2026-04-27.csv"))

    def test_csv_is_complete_returns_false_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(_csv_is_complete(Path(td) / "x.csv", 1440))

    def test_csv_is_complete_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.csv"
            path.write_text("h\n" + "row\n" * 1440, encoding="utf-8")
            self.assertTrue(_csv_is_complete(path, 1440))
            self.assertFalse(_csv_is_complete(path, 1441))

    def test_bucket_by_utc_day(self) -> None:
        rows = [
            {"timestamp": "2026-04-27T23:59:00+00:00", "x": 1},
            {"timestamp": "2026-04-28T00:00:00+00:00", "x": 2},
            {"timestamp": "2026-04-28T00:01:00+00:00", "x": 3},
            {"timestamp": "", "x": 4},
        ]
        buckets = _bucket_by_utc_day(rows)
        self.assertEqual(set(buckets.keys()), {"2026-04-27", "2026-04-28"})
        self.assertEqual(len(buckets["2026-04-27"]), 1)
        self.assertEqual(len(buckets["2026-04-28"]), 2)

    def test_fetch_with_retry_succeeds_after_transient_failures(self) -> None:
        sleeps: List[float] = []
        attempts = {"n": 0}

        def fn() -> List[Dict[str, Any]]:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return [{"ok": True}]

        rows = _fetch_with_retry(
            fn, max_attempts=4, base_delay_s=0.01, sleep=lambda s: sleeps.append(s)
        )
        self.assertEqual(rows, [{"ok": True}])
        self.assertEqual(attempts["n"], 3)
        # Two backoffs (between attempt 1->2 and 2->3); exponential.
        self.assertEqual(len(sleeps), 2)
        self.assertLess(sleeps[0], sleeps[1])

    def test_fetch_with_retry_raises_after_exhaustion(self) -> None:
        def always_fail() -> List[Dict[str, Any]]:
            raise RuntimeError("persistent")

        with self.assertRaises(RuntimeError):
            _fetch_with_retry(
                always_fail, max_attempts=2, base_delay_s=0.01, sleep=lambda s: None
            )


# ---------------------------------------------------------------------------
# backfill_symbol integration (synthetic)
# ---------------------------------------------------------------------------


class BackfillTests(unittest.TestCase):
    def test_backfill_writes_one_csv_per_day_with_correct_header(self) -> None:
        exch = _FakeCandlesExchange(granularity_seconds=60)
        today = datetime(2026, 4, 27, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            summary = backfill_symbol(
                exchange=exch,
                symbol="ETH/USD",
                days=2,
                granularity="ONE_MINUTE",
                out_dir=Path(td),
                rate_pause_s=0.0,
                today=today,
                sleep=lambda s: None,
            )
            day_paths = sorted((Path(td) / "ETH-USD" / "1m").glob("*.csv"))
            # 2 days requested -> 2 CSVs (one per UTC day BEFORE today).
            self.assertEqual(len(day_paths), 2)
            # Headers match the legacy compute_features contract.
            with day_paths[0].open() as fh:
                header = next(csv.reader(fh))
            self.assertEqual(
                header, ["timestamp", "open", "high", "low", "close", "volume"]
            )
            self.assertEqual(summary.symbol, "ETH/USD")
            self.assertEqual(summary.days_requested, 2)
            self.assertEqual(summary.days_written, 2)
            self.assertEqual(summary.bars_fetched, 2 * 1440)

    def test_backfill_skips_already_complete_days(self) -> None:
        exch = _FakeCandlesExchange(granularity_seconds=60)
        today = datetime(2026, 4, 27, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            # First run: writes both days.
            backfill_symbol(
                exchange=exch,
                symbol="ETH/USD",
                days=2,
                granularity="ONE_MINUTE",
                out_dir=Path(td),
                rate_pause_s=0.0,
                today=today,
                sleep=lambda s: None,
            )
            calls_after_first = len(exch.calls)
            # Second run: should skip both (today is NOT in the requested
            # 2-days window since days are days BEFORE today).
            summary2 = backfill_symbol(
                exchange=exch,
                symbol="ETH/USD",
                days=2,
                granularity="ONE_MINUTE",
                out_dir=Path(td),
                rate_pause_s=0.0,
                today=today,
                sleep=lambda s: None,
            )
            self.assertEqual(summary2.days_skipped, 2)
            self.assertEqual(summary2.days_written, 0)
            self.assertEqual(summary2.bars_fetched, 0)
            # No additional fetch calls.
            self.assertEqual(len(exch.calls), calls_after_first)

    def test_backfill_chunks_one_day_into_multiple_350_bar_requests(self) -> None:
        exch = _FakeCandlesExchange(granularity_seconds=60)
        today = datetime(2026, 4, 27, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            summary = backfill_symbol(
                exchange=exch,
                symbol="ETH/USD",
                days=1,
                granularity="ONE_MINUTE",
                out_dir=Path(td),
                rate_pause_s=0.0,
                today=today,
                sleep=lambda s: None,
            )
        # 1 UTC day = 1440 minutes. 1440 / 350 = 4.11 -> 5 requests.
        self.assertEqual(summary.requests_made, 5)
        self.assertEqual(summary.bars_fetched, 1440)

    def test_backfill_dedupes_overlapping_chunk_boundaries(self) -> None:
        # Stub a chunk fetcher that returns overlapping bars at chunk seams.
        rows_per_chunk = 350

        class _OverlapExchange:
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def fetch_candles_window(
                self, symbol: str, *, granularity: str, start_unix: int, end_unix: int
            ) -> List[Dict[str, Any]]:
                self.calls.append({"start": start_unix, "end": end_unix})
                rows: List[Dict[str, Any]] = []
                # Always include 1 extra overlap bar at the front.
                ts = max(0, start_unix - 60)
                for _ in range(rows_per_chunk + 1):
                    if ts >= end_unix:
                        break
                    rows.append(
                        {
                            "timestamp": datetime.fromtimestamp(
                                ts, tz=timezone.utc
                            ).isoformat(),
                            "open": 1.0,
                            "high": 1.0,
                            "low": 1.0,
                            "close": 1.0,
                            "volume": 1.0,
                        }
                    )
                    ts += 60
                return rows

        ex = _OverlapExchange()
        today = datetime(2026, 4, 27, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            summary = backfill_symbol(
                exchange=ex,
                symbol="BTC/USD",
                days=1,
                granularity="ONE_MINUTE",
                out_dir=Path(td),
                rate_pause_s=0.0,
                today=today,
                sleep=lambda s: None,
            )
            # All bars must be unique -- exactly 1440 in the day, no duplicates.
            self.assertLessEqual(summary.bars_fetched, 1440)
            # Day CSV should contain unique rows only.
            day_csv = (
                Path(td) / "BTC-USD" / "1m"
                / f"{(today - timedelta(days=1)).strftime('%Y-%m-%d')}.csv"
            )
            with day_csv.open() as fh:
                timestamps = [row[0] for row in csv.reader(fh)][1:]  # skip header
            self.assertEqual(len(timestamps), len(set(timestamps)))

    def test_backfill_propagates_persistent_failure_as_skipped_day(self) -> None:
        # Exchange that fails every call -> day is logged + skipped, no crash.
        class _AlwaysFails:
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def fetch_candles_window(
                self, symbol: str, *, granularity: str, start_unix: int, end_unix: int
            ) -> List[Dict[str, Any]]:
                self.calls.append({})
                raise RuntimeError("persistent network error")

        ex = _AlwaysFails()
        today = datetime(2026, 4, 27, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            summary = backfill_symbol(
                exchange=ex,
                symbol="ETH/USD",
                days=1,
                granularity="ONE_MINUTE",
                out_dir=Path(td),
                rate_pause_s=0.0,
                today=today,
                sleep=lambda s: None,
            )
        self.assertEqual(summary.days_written, 0)
        self.assertEqual(summary.bars_fetched, 0)

    def test_backfill_rejects_unknown_granularity(self) -> None:
        exch = _FakeCandlesExchange()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                backfill_symbol(
                    exchange=exch,
                    symbol="ETH/USD",
                    days=1,
                    granularity="WHAT_MINUTE",
                    out_dir=Path(td),
                    rate_pause_s=0.0,
                    today=datetime(2026, 4, 27, tzinfo=timezone.utc),
                    sleep=lambda s: None,
                )


if __name__ == "__main__":
    unittest.main()
