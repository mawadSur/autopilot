"""Tests for src/exit_rules.py (SHADOW-ONLY).

No network. ``apply_exit_rules`` is driven with an injected fake ``price_fn`` and
a real tempfile :class:`PnlLedger` carrying open records (entry_price > 0). We
assert:

  * evaluate_exit: a position past the stop-loss reads 'stop_loss', past a
    take-profit pct or price reads 'take_profit', a small move holds (None), an
    unpriced/None/out-of-range mark holds, and stop-loss is checked BEFORE
    take-profit;
  * compute_exit_pnl: a losing exit is negative but capped well above -size (the
    whole point — we recover most of the stake instead of riding to $0), a
    winning exit is positive, and an unsizable position returns 0.0;
  * apply_exit_rules settles a stop-loss position at the current mark with the
    capped (smaller) loss, settles a take-profit position with a gain, leaves an
    in-between position open, leaves a None-mark position open, and a price_fn
    that RAISES for one position does not abort the sweep of the others.

SHADOW-ONLY: exit rules are pure ledger bookkeeping over the current observable
mark — there is no order/execution surface anywhere in the module under test.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import exit_rules
from exit_rules import apply_exit_rules, compute_exit_pnl, evaluate_exit
from state.pnl_ledger import PnlLedger, TradeRecord


# Exit timestamps comfortably AFTER every record's entry ts below, so the
# ledger's no-look-ahead guard is satisfied (exit_ts >= entry ts).
ENTRY_TS = "2026-06-01T12:00:00+00:00"
EXIT_TS = "2026-06-02T12:00:00+00:00"


def _open_record(
    *,
    trade_id: str,
    market_id: str = "0xCOND",
    side: str = "Yes",
    entry_price: float = 0.50,
    size: float = 100.0,
    ts_utc: str = ENTRY_TS,
) -> TradeRecord:
    """An OPEN whale_convergence record (entry priced for exit math)."""
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
            "outcomeIndex=0; entry_price=0.5000; wallets=0xT1,0xT2,0xT3"
        ),
    )


# ---------------------------------------------------------------------------
# evaluate_exit
# ---------------------------------------------------------------------------


class EvaluateExitTest(unittest.TestCase):
    def test_down_45pct_triggers_stop_loss(self) -> None:
        # entry 0.50 -> 0.275 is -45% on cost; stop at 40% -> exit.
        reason = evaluate_exit(
            0.50, 0.275,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
        )
        self.assertEqual(reason, "stop_loss")

    def test_up_60pct_triggers_take_profit_pct(self) -> None:
        # entry 0.50 -> 0.80 is +60% on cost; tp pct 50% -> exit.
        reason = evaluate_exit(
            0.50, 0.80,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=None,
        )
        self.assertEqual(reason, "take_profit")

    def test_price_above_take_profit_price_triggers(self) -> None:
        # current 0.92 >= tp price 0.90 -> exit, even though +/- pct don't fire.
        reason = evaluate_exit(
            0.90, 0.92,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
        )
        self.assertEqual(reason, "take_profit")

    def test_small_move_holds(self) -> None:
        # entry 0.50 -> 0.52 is +4%: below tp, above stop -> hold.
        reason = evaluate_exit(
            0.50, 0.52,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
        )
        self.assertIsNone(reason)

    def test_zero_entry_price_holds(self) -> None:
        self.assertIsNone(
            evaluate_exit(
                0.0, 0.30,
                stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            )
        )

    def test_none_or_out_of_range_mark_holds(self) -> None:
        for bad in (None, -0.1, 1.5, "0.3", True):
            self.assertIsNone(
                evaluate_exit(
                    0.50, bad,
                    stop_loss_pct=0.40, take_profit_pct=0.50,
                    take_profit_price=0.90,
                ),
                msg=f"mark {bad!r} should hold",
            )

    def test_stop_loss_checked_before_take_profit(self) -> None:
        # A degenerate case where BOTH could nominally fire: price is past the
        # absolute take-profit price AND past the stop-loss on return-on-cost.
        # With entry 0.95 and current 0.91: r = 0.91/0.95 - 1 = -0.042 (NOT a
        # stop). To force the both-true case, use a stop_loss_pct of 0.01 and a
        # take_profit_price of 0.90: r = -0.042 <= -0.01 -> stop fires first.
        reason = evaluate_exit(
            0.95, 0.91,
            stop_loss_pct=0.01, take_profit_pct=None, take_profit_price=0.90,
        )
        self.assertEqual(reason, "stop_loss")

    def test_disabled_legs_hold(self) -> None:
        # All thresholds None -> always hold regardless of move.
        self.assertIsNone(
            evaluate_exit(
                0.50, 0.05,
                stop_loss_pct=None, take_profit_pct=None, take_profit_price=None,
            )
        )


# ---------------------------------------------------------------------------
# compute_exit_pnl
# ---------------------------------------------------------------------------


class ComputeExitPnlTest(unittest.TestCase):
    def test_losing_exit_negative_but_capped_above_full_loss(self) -> None:
        # entry 0.5, size 100 -> 200 units; sell at 0.3 -> gross $60, net ~$58.8;
        # PnL ~ -41.2. Negative, but FAR above the -100 a ride-to-$0 would cost.
        pnl = compute_exit_pnl(0.5, 100.0, 0.3)
        self.assertLess(pnl, 0.0)
        self.assertGreater(pnl, -100.0)
        # ~ -40 net of fee (60 gross * 0.98 = 58.8; 58.8 - 100 = -41.2).
        self.assertAlmostEqual(pnl, -41.2, places=4)

    def test_winning_exit_positive(self) -> None:
        # entry 0.5, size 100 -> 200 units; sell at 0.8 -> gross $160, net $156.8;
        # PnL = +56.8.
        pnl = compute_exit_pnl(0.5, 100.0, 0.8)
        self.assertGreater(pnl, 0.0)
        self.assertAlmostEqual(pnl, 56.8, places=4)

    def test_zero_entry_or_size_returns_zero(self) -> None:
        self.assertEqual(compute_exit_pnl(0.0, 100.0, 0.3), 0.0)
        self.assertEqual(compute_exit_pnl(0.5, 0.0, 0.3), 0.0)

    def test_fee_bps_zero_recovers_full_proceeds(self) -> None:
        # No haircut: entry 0.5, size 100, exit 0.3 -> gross 60, PnL = -40 exactly.
        self.assertAlmostEqual(
            compute_exit_pnl(0.5, 100.0, 0.3, fee_bps=0.0), -40.0, places=4
        )


# ---------------------------------------------------------------------------
# apply_exit_rules: sweep open ledger, settle stop / take-profit at the mark
# ---------------------------------------------------------------------------


class _FakePriceFn:
    """callable(record) -> mark | None, by trade_id, recording calls.

    A trade_id listed in ``raises_for`` makes the call raise, to prove one bad
    position does not abort the sweep of the others.
    """

    def __init__(
        self,
        prices: Dict[str, Optional[float]],
        raises_for: Any = None,
    ) -> None:
        self._prices = prices
        self._raises_for = set(raises_for or ())
        self.calls: List[str] = []

    def __call__(self, record: Any) -> Optional[float]:
        self.calls.append(record.trade_id)
        if record.trade_id in self._raises_for:
            raise RuntimeError(f"boom for {record.trade_id}")
        return self._prices.get(record.trade_id)


class ApplyExitRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "pnl_ledger.jsonl")
        self.ledger = PnlLedger(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_stop_loss_settled_at_mark_with_capped_loss(self) -> None:
        # entry 0.50, current 0.25 -> -50% on cost -> stop fires; settle at 0.25.
        self.ledger.append(
            _open_record(trade_id="sl", entry_price=0.50, size=100.0)
        )
        price_fn = _FakePriceFn({"sl": 0.25})

        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            now_iso=EXIT_TS,
        )

        self.assertEqual(res["exited"], 1)
        self.assertEqual(res["stop_loss"], 1)
        self.assertEqual(res["take_profit"], 0)
        self.assertEqual(res["still_open"], 0)
        # Capped loss: ~ -51 (50 gross * 0.98 - 100), NOT the -100 of a $0 ride.
        self.assertLess(res["realized_pnl_usd"], 0.0)
        self.assertGreater(res["realized_pnl_usd"], -100.0)

        settled = self.ledger.settled()
        self.assertEqual(len(settled), 1)
        rec = settled[0]
        self.assertEqual(rec.status, "settled")
        self.assertEqual(rec.exit_price, 0.25)
        self.assertEqual(rec.market_outcome, "exit:stop_loss")
        self.assertAlmostEqual(rec.realized_pnl_usd, -51.0, places=4)
        self.assertEqual(self.ledger.open_positions(), [])

    def test_take_profit_settled_with_gain(self) -> None:
        # entry 0.50, current 0.80 -> +60% -> take-profit; settle at 0.80.
        self.ledger.append(
            _open_record(trade_id="tp", entry_price=0.50, size=100.0)
        )
        price_fn = _FakePriceFn({"tp": 0.80})

        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            now_iso=EXIT_TS,
        )

        self.assertEqual(res["exited"], 1)
        self.assertEqual(res["take_profit"], 1)
        self.assertEqual(res["stop_loss"], 0)
        self.assertGreater(res["realized_pnl_usd"], 0.0)

        rec = self.ledger.settled()[0]
        self.assertEqual(rec.exit_price, 0.80)
        self.assertEqual(rec.market_outcome, "exit:take_profit")
        self.assertAlmostEqual(rec.realized_pnl_usd, 56.8, places=4)

    def test_in_between_position_stays_open(self) -> None:
        # entry 0.50, current 0.55 -> +10%: below tp, above stop -> hold.
        self.ledger.append(
            _open_record(trade_id="hold", entry_price=0.50, size=100.0)
        )
        price_fn = _FakePriceFn({"hold": 0.55})

        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            now_iso=EXIT_TS,
        )

        self.assertEqual(res["exited"], 0)
        self.assertEqual(res["still_open"], 1)
        self.assertEqual(res["realized_pnl_usd"], 0.0)
        self.assertEqual(len(self.ledger.open_positions()), 1)
        self.assertEqual(self.ledger.settled(), [])

    def test_none_mark_leaves_open(self) -> None:
        # price_fn returns None (unmarkable) -> never settle on a guessed price.
        self.ledger.append(
            _open_record(trade_id="np", entry_price=0.50, size=100.0)
        )
        price_fn = _FakePriceFn({"np": None})

        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            now_iso=EXIT_TS,
        )

        self.assertEqual(res["exited"], 0)
        self.assertEqual(res["still_open"], 1)
        self.assertEqual(len(self.ledger.open_positions()), 1)
        self.assertEqual(self.ledger.settled(), [])

    def test_one_raising_price_fn_does_not_abort_the_sweep(self) -> None:
        # Three open positions: "bad" raises in price_fn, "sl" stops out, "tp"
        # takes profit. The bad one is left open; the other two settle.
        self.ledger.append(
            _open_record(trade_id="bad", market_id="0xBAD", entry_price=0.50)
        )
        self.ledger.append(
            _open_record(trade_id="sl", market_id="0xSL", entry_price=0.50)
        )
        self.ledger.append(
            _open_record(trade_id="tp", market_id="0xTP", entry_price=0.50)
        )
        price_fn = _FakePriceFn(
            {"sl": 0.25, "tp": 0.80}, raises_for=["bad"]
        )

        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            now_iso=EXIT_TS,
        )

        self.assertEqual(res["exited"], 2)
        self.assertEqual(res["stop_loss"], 1)
        self.assertEqual(res["take_profit"], 1)
        # bad -> still_open (it raised but did not abort the others).
        self.assertEqual(res["still_open"], 1)
        # The bad position WAS attempted (proving the skip, not a silent omission).
        self.assertIn("bad", price_fn.calls)

        open_ids = {r.trade_id for r in self.ledger.open_positions()}
        settled_ids = {r.trade_id for r in self.ledger.settled()}
        self.assertEqual(open_ids, {"bad"})
        self.assertEqual(settled_ids, {"sl", "tp"})

    def test_no_now_iso_uses_current_utc(self) -> None:
        # Entry ts in the past so the auto-generated now-ish exit ts passes the
        # ledger's no-look-ahead guard.
        self.ledger.append(
            _open_record(trade_id="sl", entry_price=0.50, size=100.0,
                         ts_utc="2020-01-01T00:00:00+00:00")
        )
        price_fn = _FakePriceFn({"sl": 0.25})

        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
        )  # now_iso omitted
        self.assertEqual(res["exited"], 1)
        self.assertTrue(self.ledger.settled())

    def test_empty_ledger_is_noop(self) -> None:
        price_fn = _FakePriceFn({})
        res = apply_exit_rules(
            self.ledger, price_fn,
            stop_loss_pct=0.40, take_profit_pct=0.50, take_profit_price=0.90,
            now_iso=EXIT_TS,
        )
        self.assertEqual(
            res,
            {
                "exited": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "still_open": 0,
                "realized_pnl_usd": 0.0,
            },
        )
        self.assertEqual(price_fn.calls, [])


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
            self.assertFalse(hasattr(exit_rules, forbidden))


if __name__ == "__main__":
    unittest.main()
