"""Tests for open-position deduplication (SHADOW-ONLY).

Two ``whale_follow_runner`` loops writing the SAME ledger can each append an
``open`` event for the same convergence (each snapshots the open set before the
other's append lands), leaving DUPLICATE open records for one (market, outcome).
Left alone those duplicates show twice on the dashboard AND settle twice
(double-counted realized P/L). This module pins down the fix at every layer:

  * ``dedupe_open_positions`` collapses by ``(market_id, side)``, keeping the
    EARLIEST entry (the honest decision-time fill);
  * ``PnlLedger.cancel`` retires a record append-only (dropped from both
    ``open_positions`` and ``settled``) and survives a reload;
  * ``settle_resolved_positions`` settles a duplicated position ONCE, never twice;
  * ``build_state`` shows one card per position and ``n_open`` matches;
  * ``_heal_duplicate_opens`` cancels the extras and is idempotent;
  * ``_acquire_ledger_lock`` refuses a second LIVE writer and reclaims a stale lock.

No network, no orders — pure ledger bookkeeping and a fake resolver.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import whale_follow_runner as runner
from dashboard.state import build_state
from exit_rules import apply_exit_rules
from portfolio_reporter import build_report
from shadow_settlement import settle_resolved_positions
from state.pnl_ledger import (
    PnlLedger,
    TradeRecord,
    dedupe_open_positions,
    duplicate_open_records,
)

ENTRY_TS_1 = "2026-06-01T12:00:00+00:00"  # earliest entry (the one to keep)
ENTRY_TS_2 = "2026-06-01T12:00:00.020000+00:00"  # 20ms later — the racing dup
SETTLE_TS = "2026-06-02T12:00:00+00:00"


def _open(
    *,
    trade_id: str,
    market_id: str = "0xMKT",
    side: str = "Yes",
    ts_utc: str = ENTRY_TS_1,
    entry_price: float = 0.40,
    outcome_index: int = 0,
) -> TradeRecord:
    """An OPEN whale-convergence record with the notes markers the stack reads."""
    return TradeRecord(
        trade_id=trade_id,
        ts_utc=ts_utc,
        venue="polymarket",
        market_id=market_id,
        side=side,
        entry_price=entry_price,
        size=100.0,
        fees_usd=0.0,
        slippage_bps=0.0,
        strategy="whale_convergence",
        status="open",
        notes=(
            "SHADOW MODE - NO ORDERS; whale_convergence n=4 "
            f"confidence=0.90 (high); outcomeIndex={outcome_index}; "
            "entry_price=0.4000 (latest /trades mark); wallets=0xA,0xB,0xC,0xD"
        ),
    )


class DedupePureFunctionTests(unittest.TestCase):
    def test_collapses_by_market_and_side_keeping_earliest(self) -> None:
        keep = _open(trade_id="t-keep", ts_utc=ENTRY_TS_1, entry_price=0.40)
        dup = _open(trade_id="t-dup", ts_utc=ENTRY_TS_2, entry_price=0.55)
        deduped = dedupe_open_positions([dup, keep])  # input order shouldn't matter
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].trade_id, "t-keep")  # earliest ts wins
        self.assertEqual(deduped[0].entry_price, 0.40)

    def test_distinct_positions_are_not_collapsed(self) -> None:
        a = _open(trade_id="a", market_id="0xMKT", side="Yes")
        b = _open(trade_id="b", market_id="0xMKT", side="No")  # same market, diff outcome
        c = _open(trade_id="c", market_id="0xOTHER", side="Yes")
        deduped = dedupe_open_positions([a, b, c])
        self.assertEqual({r.trade_id for r in deduped}, {"a", "b", "c"})

    def test_duplicate_open_records_returns_only_the_extras(self) -> None:
        keep = _open(trade_id="t-keep", ts_utc=ENTRY_TS_1)
        dup = _open(trade_id="t-dup", ts_utc=ENTRY_TS_2)
        extras = duplicate_open_records([keep, dup])
        self.assertEqual([r.trade_id for r in extras], ["t-dup"])

    def test_empty_and_singleton(self) -> None:
        self.assertEqual(dedupe_open_positions([]), [])
        one = _open(trade_id="only")
        self.assertEqual([r.trade_id for r in dedupe_open_positions([one])], ["only"])


class LedgerCancelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "ledger.jsonl")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_cancel_drops_from_open_and_settled_and_persists(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1))
        ledger.append(_open(trade_id="t-dup", ts_utc=ENTRY_TS_2))
        self.assertEqual(len(ledger.open_positions()), 2)

        rec = ledger.cancel("t-dup", reason="duplicate_open")
        self.assertEqual(rec.status, "cancelled")

        open_ids = {r.trade_id for r in ledger.open_positions()}
        self.assertEqual(open_ids, {"t-keep"})
        self.assertNotIn("t-dup", {r.trade_id for r in ledger.settled()})

        # Survives a reload (the cancel is a real append-only event).
        reloaded = PnlLedger(self.path)
        self.assertEqual({r.trade_id for r in reloaded.open_positions()}, {"t-keep"})

    def test_cancel_unknown_trade_raises(self) -> None:
        ledger = PnlLedger(self.path)
        with self.assertRaises(KeyError):
            ledger.cancel("nope")

    def test_unique_open_positions_collapses(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1))
        ledger.append(_open(trade_id="t-dup", ts_utc=ENTRY_TS_2))
        uniq = ledger.unique_open_positions()
        self.assertEqual(len(uniq), 1)
        self.assertEqual(uniq[0].trade_id, "t-keep")


class _Resolver:
    """A fake resolver: every market is closed with token[0] the winner."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, market_id: str) -> Optional[Dict[str, Any]]:
        self.calls.append(market_id)
        return {
            "closed": True,
            "tokens": [
                {"outcome": "Yes", "winner": True, "price": 1.0},
                {"outcome": "No", "winner": False, "price": 0.0},
            ],
        }


class SettlementDoesNotDoubleCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "ledger.jsonl")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_duplicate_open_settles_exactly_once(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1, entry_price=0.40))
        ledger.append(_open(trade_id="t-dup", ts_utc=ENTRY_TS_2, entry_price=0.40))

        resolver = _Resolver()
        counts = settle_resolved_positions(ledger, resolver, now_iso=SETTLE_TS)

        # Only ONE of the two duplicate records is settled — the P/L is booked once.
        self.assertEqual(counts["settled"], 1)
        self.assertEqual(counts["won"], 1)
        self.assertEqual(len(ledger.settled()), 1)
        # The realized total equals a single winning position, not two.
        single = ledger.settled()[0].realized_pnl_usd
        self.assertAlmostEqual(counts["total_realized_pnl_usd"], single, places=9)


class ExitRulesDoNotDoubleExitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "ledger.jsonl")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_duplicate_open_take_profits_exactly_once(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1, entry_price=0.40))
        ledger.append(_open(trade_id="t-dup", ts_utc=ENTRY_TS_2, entry_price=0.40))

        # A current mark of 0.96 >= take_profit_price (0.95) → take-profit fires.
        counts = apply_exit_rules(
            ledger,
            lambda record: 0.96,
            stop_loss_pct=0.20,
            take_profit_pct=0.30,
            take_profit_price=0.95,
            now_iso=SETTLE_TS,
        )
        self.assertEqual(counts["take_profit"], 1)  # not 2
        self.assertEqual(len(ledger.settled()), 1)


class BuildStateDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "ledger.jsonl")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_open_cards_and_count_are_deduped(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_open(trade_id="t-keep", side="Frances Tiafoe", ts_utc=ENTRY_TS_1))
        ledger.append(_open(trade_id="t-dup", side="Frances Tiafoe", ts_utc=ENTRY_TS_2))
        ledger.append(_open(trade_id="t-other", side="Matteo Arnaldi", ts_utc=ENTRY_TS_1))

        state = build_state(ledger, bankroll_usd=1000.0)
        sides = sorted(p["side"] for p in state["open_positions"])
        self.assertEqual(sides, ["Frances Tiafoe", "Matteo Arnaldi"])  # dup collapsed
        self.assertEqual(state["summary"]["n_open"], 2)  # count matches the cards


class PortfolioReportDedupTests(unittest.TestCase):
    """build_report marks ONE position per (market, outcome) — a duplicate open
    must not inflate unrealized P/L or equity (the arb runner reports without the
    runner's self-heal, so build_report has to be correct on its own)."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _ledger(self, name: str) -> PnlLedger:
        return PnlLedger(os.path.join(self._dir.name, name))

    def test_duplicate_open_does_not_inflate_marked_equity(self) -> None:
        # A ledger with a duplicate open (same market+outcome).
        dup = self._ledger("dup.jsonl")
        dup.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1, entry_price=0.40))
        dup.append(_open(trade_id="t-dup", ts_utc=ENTRY_TS_2, entry_price=0.40))
        # A ledger with the single true position.
        single = self._ledger("single.jsonl")
        single.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1, entry_price=0.40))

        price_fn = lambda record: 0.70  # a real mark > entry
        dup_report = build_report(dup, price_fn=price_fn, bankroll_usd=1000.0)
        single_report = build_report(single, price_fn=price_fn, bankroll_usd=1000.0)

        self.assertEqual(dup_report["n_open"], 1)
        self.assertAlmostEqual(
            dup_report["unrealized_pnl_usd"],
            single_report["unrealized_pnl_usd"],
            places=9,
        )
        self.assertAlmostEqual(
            dup_report["equity_usd"], single_report["equity_usd"], places=9
        )


class HealDuplicateOpensTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "ledger.jsonl")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_heal_cancels_extras_and_is_idempotent(self) -> None:
        ledger = PnlLedger(self.path)
        ledger.append(_open(trade_id="t-keep", ts_utc=ENTRY_TS_1))
        ledger.append(_open(trade_id="t-dup", ts_utc=ENTRY_TS_2))

        n = runner._heal_duplicate_opens(ledger)
        self.assertEqual(n, 1)
        self.assertEqual({r.trade_id for r in ledger.open_positions()}, {"t-keep"})

        # Running again on a now-clean ledger heals nothing.
        self.assertEqual(runner._heal_duplicate_opens(ledger), 0)


class LedgerLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "ledger.jsonl")
        self._orig_pid_alive = runner._pid_alive

    def tearDown(self) -> None:
        runner._pid_alive = self._orig_pid_alive
        self._dir.cleanup()

    def test_refuses_when_a_live_writer_holds_the_lock(self) -> None:
        lock_path = f"{self.path}.lock"
        with open(lock_path, "w", encoding="utf-8") as handle:
            handle.write("999999")  # a different PID
        runner._pid_alive = lambda pid: True  # pretend it's alive
        self.assertIsNone(runner._acquire_ledger_lock(self.path))

    def test_reclaims_a_stale_lock(self) -> None:
        lock_path = f"{self.path}.lock"
        with open(lock_path, "w", encoding="utf-8") as handle:
            handle.write("999999")
        runner._pid_alive = lambda pid: False  # the holder is gone
        acquired = runner._acquire_ledger_lock(self.path)
        self.assertEqual(acquired, lock_path)
        with open(lock_path, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), str(os.getpid()))
        runner._release_ledger_lock(acquired)
        self.assertFalse(os.path.exists(lock_path))


if __name__ == "__main__":
    unittest.main()
