"""Tests for src/shadow_settlement.py (SHADOW-ONLY).

No network. ``settle_resolved_positions`` is driven with an injected fake
``resolver`` callable and a real tempfile :class:`PnlLedger` carrying open
records (entry_price > 0, notes with ``outcomeIndex=N``). We assert:

  * compute_settlement_pnl: a win nets less than the staked notional (the ~2%
    settlement fee bites the gross payout), a loss returns ``-size``, and an
    unsizable entry (entry_price 0) returns 0.0;
  * settle_resolved_positions closes a resolved+won position with a positive
    realized PnL and a resolved+lost position at ``-size``, leaves an OPEN
    (closed=false) market open, leaves a resolver-returns-None market open,
    skips an unpriced (entry_price<=0) position, and a resolver that RAISES for
    one market does not abort settling the others;
  * the ledger reflects the settlements (``settled()`` / ``summary()``), and the
    no-look-ahead guard is respected (``now_iso`` is AFTER each entry ts).

SHADOW-ONLY: settlement is pure ledger bookkeeping — there is no order/execution
surface anywhere in the module under test.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import shadow_settlement
from shadow_settlement import compute_settlement_pnl, settle_resolved_positions
from state.pnl_ledger import PnlLedger, TradeRecord


# A settlement timestamp comfortably AFTER every record's entry ts below, so the
# ledger's no-look-ahead guard is satisfied (exit_ts >= entry ts).
ENTRY_TS = "2026-06-01T12:00:00+00:00"
SETTLE_TS = "2026-06-02T12:00:00+00:00"


def _open_record(
    *,
    trade_id: str,
    market_id: str,
    outcome_index: int,
    side: str = "Yes",
    entry_price: float = 0.50,
    size: float = 100.0,
    ts_utc: str = ENTRY_TS,
) -> TradeRecord:
    """An OPEN whale_convergence record with an outcomeIndex marker in notes."""
    return TradeRecord(
        trade_id=trade_id,
        ts_utc=ts_utc,
        venue="polymarket",
        market_id=market_id,
        side=side,
        entry_price=entry_price,
        size=size,
        fees_usd=0.0,
        slippage_bps=0.0,
        strategy="whale_convergence",
        status="open",
        notes=(
            "SHADOW MODE - NO ORDERS; whale_convergence n=3 "
            f"outcomeIndex={outcome_index}; entry_price={entry_price:.4f}; "
            "wallets=0xT1,0xT2,0xT3"
        ),
    )


def _resolution(closed: bool, winners: List[bool]) -> Dict[str, Any]:
    """A normalized resolution dict (the shape get_market_resolution returns)."""
    tokens = [
        {"outcome": ("Yes" if i == 0 else "No"),
         "price": (1.0 if w else 0.0),
         "winner": w}
        for i, w in enumerate(winners)
    ]
    return {"closed": closed, "tokens": tokens}


# ---------------------------------------------------------------------------
# compute_settlement_pnl
# ---------------------------------------------------------------------------


class ComputeSettlementPnlTest(unittest.TestCase):
    def test_won_is_positive_but_below_notional_due_to_fee(self) -> None:
        # entry 0.5, size 100 -> units 200; gross $200; net after 2% fee = $196;
        # PnL = 196 - 100 = 96. Positive and < the +100 a fee-free win would give.
        pnl = compute_settlement_pnl(0.5, 100.0, won=True)
        self.assertGreater(pnl, 0.0)
        self.assertLess(pnl, 100.0)
        self.assertAlmostEqual(pnl, 96.0, places=4)

    def test_lost_returns_negative_size(self) -> None:
        self.assertAlmostEqual(
            compute_settlement_pnl(0.5, 100.0, won=False), -100.0, places=4
        )

    def test_zero_entry_price_returns_zero(self) -> None:
        # Unsizable: caller skips rather than fabricate a realized number.
        self.assertEqual(compute_settlement_pnl(0.0, 100.0, won=True), 0.0)
        self.assertEqual(compute_settlement_pnl(0.0, 100.0, won=False), 0.0)

    def test_zero_size_returns_zero(self) -> None:
        self.assertEqual(compute_settlement_pnl(0.5, 0.0, won=True), 0.0)

    def test_fee_bps_zero_recovers_full_payout(self) -> None:
        # No fee: entry 0.25, size 100 -> units 400, gross $400, PnL = +300.
        self.assertAlmostEqual(
            compute_settlement_pnl(0.25, 100.0, won=True, fee_bps=0.0),
            300.0,
            places=4,
        )

    def test_cheap_entry_win_is_large(self) -> None:
        # entry 0.10, size 100 -> units 1000, gross $1000, net $980, PnL +880.
        self.assertAlmostEqual(
            compute_settlement_pnl(0.10, 100.0, won=True), 880.0, places=4
        )


# ---------------------------------------------------------------------------
# settle_resolved_positions: sweep open ledger, settle resolved markets
# ---------------------------------------------------------------------------


class _FakeResolver:
    """callable(market_id) -> resolution | None, recording calls.

    A market_id listed in ``raises_for`` makes the call raise, to prove one bad
    market does not abort the others.
    """

    def __init__(
        self,
        resolutions: Dict[str, Optional[Dict[str, Any]]],
        raises_for: Any = None,
    ) -> None:
        self._resolutions = resolutions
        self._raises_for = set(raises_for or ())
        self.calls: List[str] = []

    def __call__(self, market_id: str) -> Optional[Dict[str, Any]]:
        self.calls.append(market_id)
        if market_id in self._raises_for:
            raise RuntimeError(f"boom for {market_id}")
        return self._resolutions.get(market_id)


class SettleResolvedPositionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "pnl_ledger.jsonl")
        self.ledger = PnlLedger(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_resolved_win_settled_with_positive_pnl(self) -> None:
        self.ledger.append(
            _open_record(trade_id="win", market_id="0xWON", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        resolver = _FakeResolver({"0xWON": _resolution(True, [True, False])})

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["settled"], 1)
        self.assertEqual(res["won"], 1)
        self.assertEqual(res["lost"], 0)
        self.assertEqual(res["still_open"], 0)
        self.assertGreater(res["total_realized_pnl_usd"], 0.0)
        self.assertAlmostEqual(res["total_realized_pnl_usd"], 96.0, places=4)

        # The ledger now shows it settled with the same realized PnL.
        settled = self.ledger.settled()
        self.assertEqual(len(settled), 1)
        rec = settled[0]
        self.assertEqual(rec.status, "settled")
        self.assertEqual(rec.exit_price, 1.0)
        self.assertEqual(rec.market_outcome, "won:Yes")
        self.assertAlmostEqual(rec.realized_pnl_usd, 96.0, places=4)
        self.assertEqual(self.ledger.open_positions(), [])

    def test_resolved_loss_settled_at_negative_size(self) -> None:
        self.ledger.append(
            _open_record(trade_id="loss", market_id="0xLOST", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        # Held outcome (index 0) lost; index 1 won.
        resolver = _FakeResolver({"0xLOST": _resolution(True, [False, True])})

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["settled"], 1)
        self.assertEqual(res["won"], 0)
        self.assertEqual(res["lost"], 1)
        self.assertAlmostEqual(res["total_realized_pnl_usd"], -100.0, places=4)

        rec = self.ledger.settled()[0]
        self.assertEqual(rec.exit_price, 0.0)
        self.assertEqual(rec.market_outcome, "lost:Yes")
        self.assertAlmostEqual(rec.realized_pnl_usd, -100.0, places=4)

    def test_open_market_left_open(self) -> None:
        self.ledger.append(
            _open_record(trade_id="still", market_id="0xOPEN", outcome_index=0)
        )
        # closed=false -> do NOT settle (no look-ahead on an unresolved book).
        resolver = _FakeResolver({"0xOPEN": _resolution(False, [False, False])})

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["settled"], 0)
        self.assertEqual(res["still_open"], 1)
        self.assertEqual(len(self.ledger.open_positions()), 1)
        self.assertEqual(self.ledger.settled(), [])

    def test_resolver_returns_none_leaves_open(self) -> None:
        self.ledger.append(
            _open_record(trade_id="unknown", market_id="0xNONE", outcome_index=0)
        )
        resolver = _FakeResolver({"0xNONE": None})

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["settled"], 0)
        self.assertEqual(res["still_open"], 1)
        self.assertEqual(len(self.ledger.open_positions()), 1)

    def test_unpriced_position_skipped(self) -> None:
        # entry_price 0.0 (the /holders unmarkable case): cannot compute realized
        # PnL honestly, so it is counted unpriced and left open.
        self.ledger.append(
            _open_record(trade_id="np", market_id="0xWON", outcome_index=0,
                         entry_price=0.0)
        )
        resolver = _FakeResolver({"0xWON": _resolution(True, [True, False])})

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["unpriced"], 1)
        self.assertEqual(res["settled"], 0)
        self.assertEqual(len(self.ledger.open_positions()), 1)
        self.assertEqual(self.ledger.settled(), [])

    def test_out_of_range_outcome_index_counted_error(self) -> None:
        # Notes say outcomeIndex=5 but the resolved market has only 2 tokens.
        self.ledger.append(
            _open_record(trade_id="oor", market_id="0xWON", outcome_index=5)
        )
        resolver = _FakeResolver({"0xWON": _resolution(True, [True, False])})

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["errors"], 1)
        self.assertEqual(res["settled"], 0)
        self.assertEqual(len(self.ledger.open_positions()), 1)

    def test_one_raising_resolver_does_not_abort_the_others(self) -> None:
        # Three open positions: 0xBAD raises, 0xWON wins, 0xLOST loses. The bad
        # market is counted as an error but the other two still settle.
        self.ledger.append(
            _open_record(trade_id="bad", market_id="0xBAD", outcome_index=0)
        )
        self.ledger.append(
            _open_record(trade_id="win", market_id="0xWON", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        self.ledger.append(
            _open_record(trade_id="loss", market_id="0xLOST", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        resolver = _FakeResolver(
            {
                "0xWON": _resolution(True, [True, False]),
                "0xLOST": _resolution(True, [False, True]),
            },
            raises_for=["0xBAD"],
        )

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["errors"], 1)
        self.assertEqual(res["settled"], 2)
        self.assertEqual(res["won"], 1)
        self.assertEqual(res["lost"], 1)
        # Net realized = +96 (win) - 100 (loss) = -4.
        self.assertAlmostEqual(res["total_realized_pnl_usd"], -4.0, places=4)

        # The bad market was attempted (proving the skip, not a silent omission).
        self.assertIn("0xBAD", resolver.calls)
        # The bad position stays open; the two good ones are settled.
        open_ids = {r.trade_id for r in self.ledger.open_positions()}
        settled_ids = {r.trade_id for r in self.ledger.settled()}
        self.assertEqual(open_ids, {"bad"})
        self.assertEqual(settled_ids, {"win", "loss"})

    def test_ledger_summary_reflects_settlements(self) -> None:
        self.ledger.append(
            _open_record(trade_id="win", market_id="0xWON", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        self.ledger.append(
            _open_record(trade_id="loss", market_id="0xLOST", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        resolver = _FakeResolver(
            {
                "0xWON": _resolution(True, [True, False]),
                "0xLOST": _resolution(True, [False, True]),
            }
        )

        settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        summary = self.ledger.summary()
        self.assertEqual(summary["n_trades"], 2)
        self.assertEqual(summary["n_settled"], 2)
        self.assertEqual(summary["n_open"], 0)
        self.assertAlmostEqual(summary["win_rate"], 0.5)  # 1 win of 2 settled
        self.assertAlmostEqual(summary["total_realized_pnl_usd"], -4.0, places=4)

    def test_winner_inferred_from_price_when_flag_absent(self) -> None:
        # Some payloads may carry only a resolved price (no winner flag): a
        # price >= 0.5 marks the held outcome a winner.
        self.ledger.append(
            _open_record(trade_id="pw", market_id="0xPRICE", outcome_index=0,
                         entry_price=0.5, size=100.0)
        )
        resolver = _FakeResolver(
            {
                "0xPRICE": {
                    "closed": True,
                    "tokens": [
                        {"outcome": "Yes", "price": 1.0, "winner": False},
                        {"outcome": "No", "price": 0.0, "winner": False},
                    ],
                }
            }
        )

        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)

        self.assertEqual(res["won"], 1)
        rec = self.ledger.settled()[0]
        self.assertEqual(rec.market_outcome, "won:Yes")

    def test_no_now_iso_uses_current_utc(self) -> None:
        # Entry ts in the past so the auto-generated now-ish exit ts passes the
        # ledger's no-look-ahead guard.
        self.ledger.append(
            _open_record(trade_id="win", market_id="0xWON", outcome_index=0,
                         entry_price=0.5, size=100.0,
                         ts_utc="2020-01-01T00:00:00+00:00")
        )
        resolver = _FakeResolver({"0xWON": _resolution(True, [True, False])})

        res = settle_resolved_positions(self.ledger, resolver)  # now_iso omitted
        self.assertEqual(res["settled"], 1)
        self.assertTrue(self.ledger.settled())

    def test_empty_ledger_is_noop(self) -> None:
        resolver = _FakeResolver({})
        res = settle_resolved_positions(self.ledger, resolver, now_iso=SETTLE_TS)
        self.assertEqual(
            res,
            {
                "settled": 0,
                "won": 0,
                "lost": 0,
                "still_open": 0,
                "unpriced": 0,
                "errors": 0,
                "total_realized_pnl_usd": 0.0,
            },
        )
        self.assertEqual(resolver.calls, [])


# ---------------------------------------------------------------------------
# Module-level safety: no order/execution symbols
# ---------------------------------------------------------------------------


class ModuleSafetyTest(unittest.TestCase):
    def test_module_has_no_order_helpers(self) -> None:
        for forbidden in (
            "place_order",
            "create_order",
            "submit_order",
            "sign",
            "sign_order",
            "redeem",
        ):
            self.assertFalse(hasattr(shadow_settlement, forbidden))


if __name__ == "__main__":
    unittest.main()
