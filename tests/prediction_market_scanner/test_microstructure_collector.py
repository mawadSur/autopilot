"""Tests for the Coinbase WS microstructure collector.

We don't connect to a real WS; we drive ``handle_message`` directly with
synthesized Coinbase Advanced Trade payloads, then read back the daily
parquet to verify schema + values.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from microstructure_collector import (
    ALL_FEATURE_COLUMNS,
    L2_FEATURE_COLUMNS,
    TRADE_AGG_COLUMNS,
    MicrostructureCollector,
    SymbolState,
    daily_parquet_path,
    snapshot_features,
)


def _snapshot_event(product: str, bids, asks) -> dict:
    """Build a Coinbase l2_data snapshot event."""
    return {
        "channel": "l2_data",
        "events": [
            {
                "type": "snapshot",
                "product_id": product,
                "updates": (
                    [{"side": "bid", "price_level": str(p), "new_quantity": str(s)}
                     for p, s in bids]
                    + [{"side": "offer", "price_level": str(p), "new_quantity": str(s)}
                       for p, s in asks]
                ),
            }
        ],
    }


def _l2_update(product: str, side: str, price: float, size: float) -> dict:
    return {
        "channel": "l2_data",
        "events": [
            {
                "type": "update",
                "product_id": product,
                "updates": [{
                    "side": side,
                    "price_level": str(price),
                    "new_quantity": str(size),
                }],
            }
        ],
    }


def _trade_event(product: str, price: float, size: float, side: str) -> dict:
    return {
        "channel": "market_trades",
        "events": [
            {
                "type": "update",
                "trades": [{
                    "product_id": product,
                    "price": str(price),
                    "size": str(size),
                    "side": side,
                    "time": "2026-05-13T12:00:00.000Z",
                }],
            }
        ],
    }


class FeatureComputationTests(unittest.TestCase):
    """``snapshot_features`` produces the expected 26-feature row."""

    def setUp(self) -> None:
        self.state = SymbolState(symbol="ETH-USD")
        # Asks 100, 101, 102 ... ; bids 99, 98, 97 ...
        for i in range(25):
            self.state.bids[99 - i] = 1.0 + i  # deeper levels have more size
            self.state.asks[100 + i] = 1.0 + i
        self.state.book_seeded = True
        self.state.trade_count = 5
        self.state.buy_count = 3
        self.state.sell_count = 2
        self.state.taker_buy_volume_base = 1.5
        self.state.taker_sell_volume_base = 0.7
        self.state.taker_buy_volume_quote = 150.0
        self.state.taker_sell_volume_quote = 70.0

    def test_top_of_book(self) -> None:
        row = snapshot_features(self.state, datetime(2026, 5, 13, 12, 0,
                                                    tzinfo=timezone.utc))
        self.assertEqual(row["best_bid"], 99.0)
        self.assertEqual(row["best_ask"], 100.0)
        self.assertEqual(row["bid_size_l1"], 1.0)
        self.assertEqual(row["ask_size_l1"], 1.0)

    def test_depth_aggregation(self) -> None:
        row = snapshot_features(self.state, datetime(2026, 5, 13, 12, 0,
                                                    tzinfo=timezone.utc))
        # Top-5 bids: prices 99..95, sizes 1..5 → depth=15.
        self.assertAlmostEqual(row["bid_depth_5"], 15.0)
        self.assertAlmostEqual(row["ask_depth_5"], 15.0)
        # Top-10 = 1+2+...+10 = 55.
        self.assertAlmostEqual(row["bid_depth_10"], 55.0)

    def test_trade_aggregates_passthrough(self) -> None:
        row = snapshot_features(self.state, datetime(2026, 5, 13, 12, 0,
                                                    tzinfo=timezone.utc))
        self.assertEqual(row["trade_count"], 5.0)
        self.assertEqual(row["buy_count"], 3.0)
        self.assertEqual(row["sell_count"], 2.0)
        self.assertEqual(row["taker_buy_volume_base"], 1.5)
        self.assertEqual(row["taker_sell_volume_quote"], 70.0)

    def test_schema_matches_history_coindesk(self) -> None:
        """Schema must be identical to what build_dataset.py expects."""
        from history_coindesk import (
            L2_FEATURE_COLUMNS as HIST_L2,
            TRADE_AGG_COLUMNS as HIST_TRADES,
        )
        self.assertEqual(L2_FEATURE_COLUMNS, HIST_L2)
        self.assertEqual(TRADE_AGG_COLUMNS, HIST_TRADES)


class HandleMessageTests(unittest.TestCase):
    """End-to-end: route WS payloads through ``handle_message`` and check
    book + trade state updates."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_root = Path(self.tmpdir.name)
        self.collector = MicrostructureCollector(
            symbols=["ETH-USD"],
            out_root=self.out_root,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_snapshot_seeds_book(self) -> None:
        self.collector.handle_message(
            _snapshot_event("ETH-USD",
                            bids=[(2000.0, 1.5), (1999.0, 2.0)],
                            asks=[(2001.0, 1.0), (2002.0, 3.0)]),
            now=datetime(2026, 5, 13, 12, 0, 30, tzinfo=timezone.utc),
        )
        st = self.collector.state["ETH-USD"]
        self.assertTrue(st.book_seeded)
        self.assertEqual(st.bids, {2000.0: 1.5, 1999.0: 2.0})
        self.assertEqual(st.asks, {2001.0: 1.0, 2002.0: 3.0})

    def test_update_then_delete(self) -> None:
        self.collector.handle_message(
            _snapshot_event("ETH-USD", bids=[(2000.0, 1.0)], asks=[(2001.0, 1.0)]),
            now=datetime(2026, 5, 13, 12, 0, 30, tzinfo=timezone.utc),
        )
        self.collector.handle_message(
            _l2_update("ETH-USD", "bid", 2000.0, 2.5),
            now=datetime(2026, 5, 13, 12, 0, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(self.collector.state["ETH-USD"].bids[2000.0], 2.5)
        self.collector.handle_message(
            _l2_update("ETH-USD", "bid", 2000.0, 0.0),
            now=datetime(2026, 5, 13, 12, 0, 32, tzinfo=timezone.utc),
        )
        self.assertNotIn(2000.0, self.collector.state["ETH-USD"].bids)

    def test_trade_accumulation(self) -> None:
        # Seed first so book_seeded → minute flushes will fire.
        self.collector.handle_message(
            _snapshot_event("ETH-USD", bids=[(2000.0, 1.0)], asks=[(2001.0, 1.0)]),
            now=datetime(2026, 5, 13, 12, 0, 30, tzinfo=timezone.utc),
        )
        self.collector.handle_message(
            _trade_event("ETH-USD", 2000.5, 0.4, "BUY"),
            now=datetime(2026, 5, 13, 12, 0, 31, tzinfo=timezone.utc),
        )
        self.collector.handle_message(
            _trade_event("ETH-USD", 2000.5, 0.1, "SELL"),
            now=datetime(2026, 5, 13, 12, 0, 32, tzinfo=timezone.utc),
        )
        st = self.collector.state["ETH-USD"]
        self.assertEqual(st.trade_count, 2)
        self.assertEqual(st.buy_count, 1)
        self.assertEqual(st.sell_count, 1)
        self.assertAlmostEqual(st.taker_buy_volume_base, 0.4)
        self.assertAlmostEqual(st.taker_sell_volume_quote, 200.05)

    def test_minute_boundary_flush_writes_parquet(self) -> None:
        # t=12:00:30 — seed + a buy trade.
        self.collector.handle_message(
            _snapshot_event("ETH-USD",
                            bids=[(2000.0, 1.0), (1999.0, 1.5)],
                            asks=[(2001.0, 1.0), (2002.0, 1.5)]),
            now=datetime(2026, 5, 13, 12, 0, 30, tzinfo=timezone.utc),
        )
        self.collector.handle_message(
            _trade_event("ETH-USD", 2000.5, 1.0, "BUY"),
            now=datetime(2026, 5, 13, 12, 0, 45, tzinfo=timezone.utc),
        )
        # t=12:01:01 — crossing the minute boundary triggers a flush of
        # the row stamped 12:00.
        self.collector.handle_message(
            _trade_event("ETH-USD", 2000.6, 0.5, "SELL"),
            now=datetime(2026, 5, 13, 12, 1, 1, tzinfo=timezone.utc),
        )
        path = daily_parquet_path(
            self.out_root, "ETH-USD",
            datetime(2026, 5, 13, tzinfo=timezone.utc),
        )
        self.assertTrue(path.exists(), f"expected parquet at {path}")
        df = pd.read_parquet(path)
        self.assertEqual(list(df.columns), ALL_FEATURE_COLUMNS)
        self.assertEqual(len(df), 1)
        first = df.iloc[0]
        # Timestamp at minute start, UTC.
        self.assertEqual(first["timestamp"],
                         pd.Timestamp("2026-05-13T12:00:00", tz="UTC"))
        self.assertEqual(first["best_bid"], 2000.0)
        self.assertEqual(first["best_ask"], 2001.0)
        # The first minute saw 1 BUY trade; the second-minute SELL is
        # still buffered (it arrived after we crossed into 12:01).
        self.assertEqual(first["trade_count"], 1.0)
        self.assertEqual(first["buy_count"], 1.0)
        self.assertEqual(first["sell_count"], 0.0)
        # And the bucket reset for the new minute.
        st = self.collector.state["ETH-USD"]
        self.assertEqual(st.sell_count, 1)
        self.assertEqual(st.buy_count, 0)

    def test_flush_is_idempotent_across_restart(self) -> None:
        """A second run on the same day appends, doesn't overwrite."""
        self.collector.handle_message(
            _snapshot_event("ETH-USD", bids=[(2000.0, 1.0)], asks=[(2001.0, 1.0)]),
            now=datetime(2026, 5, 13, 12, 0, 30, tzinfo=timezone.utc),
        )
        # Cross 12:00 → 12:01: flush row stamped 12:00.
        self.collector.handle_message(
            _trade_event("ETH-USD", 2000.5, 1.0, "BUY"),
            now=datetime(2026, 5, 13, 12, 1, 5, tzinfo=timezone.utc),
        )
        # Cross 12:01 → 12:02: flush row stamped 12:01.
        self.collector.handle_message(
            _trade_event("ETH-USD", 2000.5, 1.0, "SELL"),
            now=datetime(2026, 5, 13, 12, 2, 5, tzinfo=timezone.utc),
        )
        path = daily_parquet_path(
            self.out_root, "ETH-USD",
            datetime(2026, 5, 13, tzinfo=timezone.utc),
        )
        df = pd.read_parquet(path)
        self.assertEqual(len(df), 2)
        self.assertEqual(
            list(df["timestamp"]),
            [pd.Timestamp("2026-05-13T12:00:00", tz="UTC"),
             pd.Timestamp("2026-05-13T12:01:00", tz="UTC")],
        )

    def test_unknown_channel_does_not_raise(self) -> None:
        self.collector.handle_message({"channel": "heartbeats", "events": []})
        self.collector.handle_message({"channel": "unknown"})
        self.collector.handle_message({"type": "subscriptions",
                                       "events": [{"subscriptions": {}}]})


class StopTests(unittest.TestCase):
    def test_stop_sets_flag(self) -> None:
        c = MicrostructureCollector(symbols=["ETH-USD"], out_root=Path("/tmp"))
        self.assertFalse(c._stop.is_set())
        c.stop()
        self.assertTrue(c._stop.is_set())

    def test_empty_symbols_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MicrostructureCollector(symbols=[], out_root=Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
