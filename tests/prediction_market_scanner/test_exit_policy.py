"""Unit tests for ``src/exit_policy.py``.

These tests exercise each exit branch in isolation, verify the documented
priority order (SL > TP > trail > time > reversal), confirm the trailing-
stop high-water-mark bookkeeping, confirm the all-disabled no-op, and
confirm that a malformed position raises rather than silently no-opping.

Synthetic positions / ticks are built from ``types.SimpleNamespace`` so
the tests do not depend on the live pydantic Position model. The exit
policy is duck-typed; this is the intended contract.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from exit_policy import ExitDecision, ExitPolicy


def _pos(**overrides):
    """Build a benign long-side test position with all relevant fields."""
    base = dict(
        side="long",
        entry_price=100.0,
        bars_held=0,
        high_water_mark=100.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tick(price: float, signal_prob: float = 0.7):
    return SimpleNamespace(price=price, signal_prob=signal_prob)


class StopLossOnlyTests(unittest.TestCase):
    """Stop loss fires; TP / trail / time / reversal are all disabled."""

    def setUp(self):
        self.policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=-0.004,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )

    def test_long_below_stop_closes_with_sl(self):
        # 100 -> 99.59 is -0.41%, past the -0.40% stop.
        decision = self.policy.evaluate(_pos(), _tick(price=99.59))
        self.assertEqual(decision, ExitDecision(close=True, reason="sl", exit_price=99.59))

    def test_long_at_stop_threshold_closes(self):
        # Exactly -0.4% — the SL is inclusive (``<=``).
        decision = self.policy.evaluate(_pos(), _tick(price=99.6))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "sl")

    def test_long_above_stop_does_not_close(self):
        decision = self.policy.evaluate(_pos(), _tick(price=99.7))
        self.assertEqual(decision, ExitDecision(close=False, reason="", exit_price=None))

    def test_short_inverse(self):
        # Short entered at 100; price rises to 100.5 — that is a -0.5% PnL
        # for the short, deeper than the -0.4% SL.
        decision = self.policy.evaluate(
            _pos(side="short"), _tick(price=100.5)
        )
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "sl")


class TakeProfitOnlyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=0.008,
            trailing_stop_pct=None,
            signal_reversal=False,
        )

    def test_long_above_tp_closes(self):
        decision = self.policy.evaluate(_pos(), _tick(price=100.81))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "tp")
        self.assertEqual(decision.exit_price, 100.81)

    def test_long_at_tp_threshold_closes(self):
        decision = self.policy.evaluate(_pos(), _tick(price=100.8))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "tp")

    def test_long_below_tp_does_not_close(self):
        decision = self.policy.evaluate(_pos(), _tick(price=100.5))
        self.assertFalse(decision.close)


class TrailingStopOnlyTests(unittest.TestCase):
    def setUp(self):
        # 0.5% trail, all other branches off.
        self.policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.005,
            signal_reversal=False,
        )

    def test_high_water_mark_advances_on_rising_price(self):
        position = _pos()
        # Rising sequence — watermark should track the max.
        for px in (101.0, 102.0, 101.5, 103.0):
            self.policy.update_high_water_mark(position, _tick(price=px))
        self.assertEqual(position.high_water_mark, 103.0)

    def test_no_close_while_price_climbs(self):
        position = _pos()
        for px in (101.0, 102.0, 103.0):
            self.policy.update_high_water_mark(position, _tick(price=px))
            decision = self.policy.evaluate(position, _tick(price=px))
            self.assertFalse(decision.close, f"unexpected close at price {px}")

    def test_close_when_retracement_breaches_trail(self):
        position = _pos()
        # Walk price up to 103, then back down to 102.4 — that's
        # (103 - 102.4) / 103 = ~0.583% retracement, past the 0.5% trail.
        for px in (101.0, 102.0, 103.0):
            self.policy.update_high_water_mark(position, _tick(price=px))
        self.assertEqual(position.high_water_mark, 103.0)
        decision = self.policy.evaluate(position, _tick(price=102.4))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "trail")
        self.assertEqual(decision.exit_price, 102.4)

    def test_no_close_when_retracement_below_trail(self):
        position = _pos()
        for px in (101.0, 102.0, 103.0):
            self.policy.update_high_water_mark(position, _tick(price=px))
        # Drop to 102.7 — only ~0.29% retracement; should not fire.
        decision = self.policy.evaluate(position, _tick(price=102.7))
        self.assertFalse(decision.close)

    def test_short_uses_low_water_mark(self):
        # Short trail: track the running MIN, close when price rebounds.
        position = _pos(side="short", high_water_mark=100.0)
        for px in (99.0, 98.0, 97.0):
            self.policy.update_high_water_mark(position, _tick(price=px))
        self.assertEqual(position.high_water_mark, 97.0)
        # Bounce to 97.6 — (97.6 - 97) / 97 = ~0.619%, past 0.5%.
        decision = self.policy.evaluate(position, _tick(price=97.6))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "trail")

    def test_seeds_high_water_mark_when_none(self):
        # Brand-new position with no watermark yet — the helper should
        # seed it from entry_price.
        position = _pos(high_water_mark=None)
        self.policy.update_high_water_mark(position, _tick(price=101.0))
        # Seeded at entry (100), then advanced to 101.
        self.assertEqual(position.high_water_mark, 101.0)


class TimeStopOnlyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ExitPolicy(
            time_stop_bars=20,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )

    def test_close_when_bars_reach_limit(self):
        decision = self.policy.evaluate(_pos(bars_held=20), _tick(price=100.0))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "time")

    def test_close_when_bars_exceed_limit(self):
        decision = self.policy.evaluate(_pos(bars_held=25), _tick(price=100.0))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "time")

    def test_no_close_when_bars_below_limit(self):
        decision = self.policy.evaluate(_pos(bars_held=19), _tick(price=100.0))
        self.assertFalse(decision.close)


class SignalReversalOnlyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=True,
        )

    def test_long_closes_when_signal_flips_bearish(self):
        decision = self.policy.evaluate(_pos(), _tick(price=100.0, signal_prob=0.45))
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "reversal")

    def test_long_does_not_close_when_signal_still_bullish(self):
        decision = self.policy.evaluate(_pos(), _tick(price=100.0, signal_prob=0.55))
        self.assertFalse(decision.close)

    def test_short_closes_when_signal_flips_bullish(self):
        decision = self.policy.evaluate(
            _pos(side="short"), _tick(price=100.0, signal_prob=0.55)
        )
        self.assertTrue(decision.close)
        self.assertEqual(decision.reason, "reversal")


class PriorityOrderTests(unittest.TestCase):
    """All four branches enabled; verify SL > TP > trail > time > reversal."""

    def setUp(self):
        self.policy = ExitPolicy(
            time_stop_bars=20,
            stop_loss_pct=-0.004,
            take_profit_pct=0.008,
            trailing_stop_pct=0.005,
            signal_reversal=True,
        )

    def test_sl_beats_everything(self):
        # Loss past SL AND time-stop hit AND signal flipped — SL must win.
        position = _pos(bars_held=99, high_water_mark=100.0)
        decision = self.policy.evaluate(
            position, _tick(price=99.0, signal_prob=0.1)
        )
        self.assertEqual(decision.reason, "sl")

    def test_tp_beats_trail_time_reversal(self):
        # Strong winner: TP fires. Watermark equals the current price, so
        # retracement is 0 — but TP must take precedence regardless.
        position = _pos(bars_held=99, high_water_mark=101.0)
        decision = self.policy.evaluate(
            position, _tick(price=101.0, signal_prob=0.1)
        )
        self.assertEqual(decision.reason, "tp")

    def test_trail_beats_time_and_reversal(self):
        # Price modestly up from entry (in profit, below TP), bars_held >=
        # limit, signal flipped, AND retracement past trail. Trail wins.
        position = _pos(bars_held=99, high_water_mark=100.7)
        decision = self.policy.evaluate(
            position, _tick(price=100.1, signal_prob=0.1)
        )
        self.assertEqual(decision.reason, "trail")

    def test_time_beats_reversal(self):
        # In-profit but not at TP, no trail breach (watermark == price), at
        # time-stop, AND signal flipped. Time wins.
        position = _pos(bars_held=20, high_water_mark=100.3)
        decision = self.policy.evaluate(
            position, _tick(price=100.3, signal_prob=0.1)
        )
        self.assertEqual(decision.reason, "time")

    def test_reversal_fires_when_no_other_branch_does(self):
        # Tiny profit, well inside all PnL bands, bars under the limit,
        # signal flips bearish. Reversal is the last line of defence.
        position = _pos(bars_held=5, high_water_mark=100.2)
        decision = self.policy.evaluate(
            position, _tick(price=100.2, signal_prob=0.1)
        )
        self.assertEqual(decision.reason, "reversal")


class AllDisabledTests(unittest.TestCase):
    """Every branch off — evaluate should be a no-op regardless of state."""

    def setUp(self):
        self.policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )

    def test_no_op_on_huge_loss(self):
        decision = self.policy.evaluate(_pos(), _tick(price=50.0))
        self.assertEqual(decision, ExitDecision(close=False, reason="", exit_price=None))

    def test_no_op_on_huge_gain(self):
        decision = self.policy.evaluate(_pos(), _tick(price=200.0))
        self.assertEqual(decision, ExitDecision(close=False, reason="", exit_price=None))

    def test_no_op_with_old_position(self):
        decision = self.policy.evaluate(_pos(bars_held=1000), _tick(price=100.0))
        self.assertFalse(decision.close)

    def test_zero_time_stop_disables_branch(self):
        # ``time_stop_bars=0`` is the documented "disabled" sentinel for
        # the bar threshold; even very old positions must not close.
        policy = ExitPolicy(
            time_stop_bars=0,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        decision = policy.evaluate(_pos(bars_held=10_000), _tick(price=100.0))
        self.assertFalse(decision.close)


class MalformedPositionTests(unittest.TestCase):
    """Missing required fields must raise — never silently close."""

    def test_missing_entry_price_raises(self):
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=-0.004,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        broken = SimpleNamespace(side="long")  # no entry_price
        with self.assertRaises(AttributeError):
            policy.evaluate(broken, _tick(price=99.0))

    def test_missing_side_raises(self):
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=-0.004,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        broken = SimpleNamespace(entry_price=100.0)  # no side
        with self.assertRaises(AttributeError):
            policy.evaluate(broken, _tick(price=99.0))

    def test_missing_bars_held_raises_when_time_stop_on(self):
        policy = ExitPolicy(
            time_stop_bars=20,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        broken = SimpleNamespace(side="long", entry_price=100.0)  # no bars_held
        with self.assertRaises(AttributeError):
            policy.evaluate(broken, _tick(price=100.0))

    def test_missing_high_water_mark_raises_when_trail_on(self):
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.005,
            signal_reversal=False,
        )
        broken = SimpleNamespace(side="long", entry_price=100.0)  # no hwm
        with self.assertRaises(AttributeError):
            policy.evaluate(broken, _tick(price=99.0))

    def test_missing_signal_prob_raises_when_reversal_on(self):
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=True,
        )
        broken_tick = SimpleNamespace(price=100.0)  # no signal_prob
        with self.assertRaises(AttributeError):
            policy.evaluate(_pos(), broken_tick)


class ConstructorValidationTests(unittest.TestCase):
    """The constructor catches obvious sign / magnitude mistakes early."""

    def test_positive_stop_loss_rejected(self):
        with self.assertRaises(ValueError):
            ExitPolicy(stop_loss_pct=0.004)

    def test_negative_take_profit_rejected(self):
        with self.assertRaises(ValueError):
            ExitPolicy(take_profit_pct=-0.008)

    def test_non_positive_trail_rejected(self):
        with self.assertRaises(ValueError):
            ExitPolicy(trailing_stop_pct=0.0)
        with self.assertRaises(ValueError):
            ExitPolicy(trailing_stop_pct=-0.005)

    def test_negative_time_stop_rejected(self):
        with self.assertRaises(ValueError):
            ExitPolicy(time_stop_bars=-1)


if __name__ == "__main__":
    unittest.main()
