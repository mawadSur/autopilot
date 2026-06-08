"""Unit tests for the shadow PnL ledger (``state.pnl_ledger``).

Each test uses a fresh :class:`tempfile.TemporaryDirectory` so the append-only
JSONL file is isolated and nothing leaks between cases. The ledger is the
single auditable per-trade truth record, so these tests pin down the
event-log fold semantics, the no-look-ahead settle guard, and the summary math.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from state.pnl_ledger import PnlLedger, TradeRecord


def _record(
    *,
    trade_id: str = "t-1",
    ts_utc: str = "2026-05-31T12:00:00+00:00",
    venue: str = "polymarket",
    market_id: str = "mkt-abc",
    side: str = "YES",
    entry_price: float = 0.42,
    size: float = 100.0,
    fees_usd: float = 1.50,
    slippage_bps: float = 5.0,
    strategy: str = "ev_brain",
    status: str = "open",
    notes: str = "",
) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        ts_utc=ts_utc,
        venue=venue,
        market_id=market_id,
        side=side,
        entry_price=entry_price,
        size=size,
        fees_usd=fees_usd,
        slippage_bps=slippage_bps,
        strategy=strategy,
        status=status,
        notes=notes,
    )


class PnlLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Point at a nested path to also exercise parent-dir creation.
        self.path = os.path.join(self._tmp.name, "nested", "pnl_ledger.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_append_then_reload_roundtrip(self) -> None:
        ledger = PnlLedger(self.path)
        rec = _record(trade_id="t-roundtrip", entry_price=0.37, notes="hi")
        ledger.append(rec)

        # Parent dir was created and the file exists on disk.
        self.assertTrue(os.path.exists(self.path))

        # Re-open from the same path: state must fold back identically.
        reopened = PnlLedger(self.path)
        records = reopened.all_records()
        self.assertEqual(len(records), 1)
        loaded = records[0]
        self.assertEqual(loaded.trade_id, "t-roundtrip")
        self.assertEqual(loaded.entry_price, 0.37)
        self.assertEqual(loaded.side, "YES")
        self.assertEqual(loaded.notes, "hi")
        self.assertEqual(loaded.status, "open")
        self.assertIsNone(loaded.exit_price)

    def test_settle_updates_realized_pnl_and_win_rate(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_record(trade_id="win", strategy="s"))
        ledger.append(_record(trade_id="loss", strategy="s"))

        ledger.settle(
            "win",
            exit_price=0.80,
            exit_ts_utc="2026-05-31T18:00:00+00:00",
            market_outcome="YES",
            realized_pnl_usd=38.0,
        )
        ledger.settle(
            "loss",
            exit_price=0.10,
            exit_ts_utc="2026-05-31T18:00:00+00:00",
            market_outcome="NO",
            realized_pnl_usd=-42.0,
        )

        summary = ledger.summary()
        self.assertEqual(summary["n_trades"], 2)
        self.assertEqual(summary["n_settled"], 2)
        self.assertEqual(summary["n_open"], 0)
        self.assertAlmostEqual(summary["total_realized_pnl_usd"], -4.0)
        # 1 win out of 2 settled.
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        # 2 records * 1.50 fee each.
        self.assertAlmostEqual(summary["total_fees_usd"], 3.0)

        # The returned record from settle reflects the new state immediately.
        again = PnlLedger(self.path).settled()
        self.assertEqual(len(again), 2)
        win = next(r for r in again if r.trade_id == "win")
        self.assertEqual(win.status, "settled")
        self.assertEqual(win.exit_price, 0.80)
        self.assertEqual(win.market_outcome, "YES")
        self.assertEqual(win.realized_pnl_usd, 38.0)

    def test_settle_before_entry_raises_lookahead(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_record(trade_id="la", ts_utc="2026-05-31T12:00:00+00:00"))

        with self.assertRaises(ValueError):
            ledger.settle(
                "la",
                exit_price=0.50,
                # One hour BEFORE entry — must be rejected.
                exit_ts_utc="2026-05-31T11:00:00+00:00",
                realized_pnl_usd=5.0,
            )

        # The rejected settle must not have mutated state: still open.
        self.assertEqual(len(ledger.open_positions()), 1)
        self.assertEqual(len(ledger.settled()), 0)

    def test_settle_at_exactly_entry_is_allowed(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_record(trade_id="eq", ts_utc="2026-05-31T12:00:00+00:00"))
        # exit == entry is not look-ahead; should succeed.
        rec = ledger.settle(
            "eq",
            exit_price=0.50,
            exit_ts_utc="2026-05-31T12:00:00+00:00",
            realized_pnl_usd=1.0,
        )
        self.assertEqual(rec.status, "settled")

    def test_empty_and_missing_file_returns_empty_summary(self) -> None:
        # Missing file (never written).
        ledger = PnlLedger(self.path)
        self.assertFalse(os.path.exists(self.path))
        summary = ledger.summary()
        self.assertEqual(summary["n_trades"], 0)
        self.assertEqual(summary["n_settled"], 0)
        self.assertEqual(summary["n_open"], 0)
        self.assertEqual(summary["total_realized_pnl_usd"], 0.0)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["total_fees_usd"], 0.0)
        self.assertEqual(ledger.open_positions(), [])
        self.assertEqual(ledger.settled(), [])

        # Explicitly-empty file (touched but no events).
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8"):
            pass
        self.assertEqual(PnlLedger(self.path).summary()["n_trades"], 0)

    def test_by_strategy_filter(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_record(trade_id="a", strategy="alpha"))
        ledger.append(_record(trade_id="b", strategy="beta"))
        ledger.append(_record(trade_id="c", strategy="alpha"))

        alpha = ledger.by_strategy("alpha")
        self.assertEqual({r.trade_id for r in alpha}, {"a", "c"})
        beta = ledger.by_strategy("beta")
        self.assertEqual({r.trade_id for r in beta}, {"b"})
        self.assertEqual(ledger.by_strategy("missing"), [])

    def test_open_vs_settled_partition(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_record(trade_id="still-open"))
        ledger.append(_record(trade_id="closed"))
        ledger.settle(
            "closed",
            exit_price=0.6,
            exit_ts_utc="2026-05-31T20:00:00+00:00",
            realized_pnl_usd=10.0,
        )

        open_ids = {r.trade_id for r in ledger.open_positions()}
        settled_ids = {r.trade_id for r in ledger.settled()}
        self.assertEqual(open_ids, {"still-open"})
        self.assertEqual(settled_ids, {"closed"})
        # Partition is exhaustive and disjoint over the two records.
        self.assertEqual(open_ids & settled_ids, set())
        self.assertEqual(len(ledger.all_records()), 2)

    def test_settle_appends_event_keeps_entry_line_immutable(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_record(trade_id="audit", entry_price=0.33))
        ledger.settle(
            "audit",
            exit_price=0.9,
            exit_ts_utc="2026-05-31T20:00:00+00:00",
            realized_pnl_usd=57.0,
        )

        with open(self.path, "r", encoding="utf-8") as handle:
            lines = [ln for ln in handle.read().splitlines() if ln.strip()]
        # Two events: the original open and the settle, both preserved.
        self.assertEqual(len(lines), 2)
        self.assertIn('"_event":"open"', lines[0])
        self.assertIn('"entry_price":0.33', lines[0])
        self.assertIn('"_event":"settle"', lines[1])

    def test_settle_unknown_trade_raises(self) -> None:
        ledger = PnlLedger(self.path)
        with self.assertRaises(KeyError):
            ledger.settle(
                "nope",
                exit_price=0.5,
                exit_ts_utc="2026-05-31T20:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
