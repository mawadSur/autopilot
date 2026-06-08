"""Tests for src/exit_backtest.py (PURE — no network, no orders).

The module is a pure exit-overlay backtester: every function is a computation over
numbers the test supplies, so there is no client/session/ledger to fake. We assert:

  * simulate_exit:
      - a path that DIPS past the stop-loss exits at the stop (capped loss, strictly
        better than riding to $0);
      - a path that RISES to a take_profit_price exits with a gain;
      - a flat path with no trigger HOLDS to final_price (won -> positive at
        entry < 1; lost -> ~ -1, the whole stake);
      - when a single price would trip BOTH stop-loss and take-profit, stop-loss
        wins (capital-preserving ordering);
      - entry_price <= 0 -> ('skip');
      - the exact return_on_cost math (entry 0.5, exit 0.3, 200 bps -> -0.412;
        exit 1.0 won -> +0.96).
  * grid_search:
      - combos are ranked by mean_return descending;
      - the baseline (all-None hold-to-resolution) combo is always present;
      - the risk_adjusted pick excludes any combo whose worst_return < -0.6;
      - each combo's exit_mix counts sum to its n.

PURE: there is no order/sign/redeem/web3/network surface anywhere in the module
under test — these tests need no I/O of any kind.
"""

from __future__ import annotations

import math
import unittest

import exit_backtest
from exit_backtest import grid_search, simulate_exit


class SimulateExitTest(unittest.TestCase):
    def test_stop_loss_caps_the_loss_below_riding_to_zero(self):
        # Entry 0.50; price dips to 0.30 (r = -0.40) then would recover. A 30% stop
        # fires at 0.30 BEFORE the recovery, and the capped loss beats riding to $0.
        ret, reason = simulate_exit(
            0.50,
            [0.40, 0.30, 0.45],
            0.0,
            stop_loss_pct=0.30,
            take_profit_pct=None,
            take_profit_price=None,
        )
        self.assertEqual(reason, "stop_loss")
        # Exit at 0.30: (1/0.5)*0.30*0.98 - 1 = -0.412.
        self.assertAlmostEqual(ret, -0.412, places=6)
        # Riding to resolution at $0 would be a full -1.0 loss; the stop is better.
        ride, ride_reason = simulate_exit(
            0.50,
            [0.40, 0.30, 0.45],
            0.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            take_profit_price=None,
        )
        self.assertEqual(ride_reason, "resolution")
        self.assertAlmostEqual(ride, -1.0, places=6)
        self.assertGreater(ret, ride)

    def test_take_profit_price_exits_with_a_gain(self):
        # Entry 0.50; rises to 0.90 which hits the absolute take-profit target.
        ret, reason = simulate_exit(
            0.50,
            [0.60, 0.90, 0.95],
            1.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            take_profit_price=0.90,
        )
        self.assertEqual(reason, "take_profit")
        # Exit at 0.90: (1/0.5)*0.90*0.98 - 1 = 0.764 > 0.
        self.assertAlmostEqual(ret, 0.764, places=6)
        self.assertGreater(ret, 0.0)

    def test_take_profit_pct_exits_when_relative_target_hit(self):
        # Entry 0.40; rises to 0.65 -> r = +0.625 clears a 0.5 take_profit_pct.
        # (0.60 would give r = 0.4999999999999998 in binary float — just shy of the
        # threshold — so use a price that genuinely exceeds +50%.)
        ret, reason = simulate_exit(
            0.40,
            [0.45, 0.65, 0.50],
            0.0,
            stop_loss_pct=None,
            take_profit_pct=0.50,
            take_profit_price=None,
        )
        self.assertEqual(reason, "take_profit")
        self.assertAlmostEqual(ret, (1.0 / 0.40) * 0.65 * 0.98 - 1.0, places=6)

    def test_flat_path_holds_to_resolution_won_is_positive(self):
        # No trigger across a flat path; entry 0.50 won -> +0.96.
        ret, reason = simulate_exit(
            0.50,
            [0.50, 0.51, 0.49],
            1.0,
            stop_loss_pct=0.90,
            take_profit_pct=None,
            take_profit_price=0.99,
        )
        self.assertEqual(reason, "resolution")
        self.assertAlmostEqual(ret, 0.96, places=6)
        self.assertGreater(ret, 0.0)

    def test_flat_path_holds_to_resolution_lost_is_near_minus_one(self):
        # No trigger; entry 0.50 lost -> exit at 0.0 -> -1.0 (the whole stake).
        ret, reason = simulate_exit(
            0.50,
            [0.50, 0.48, 0.52],
            0.0,
            stop_loss_pct=None,
            take_profit_pct=2.0,
            take_profit_price=None,
        )
        self.assertEqual(reason, "resolution")
        self.assertAlmostEqual(ret, -1.0, places=6)

    def test_stop_loss_checked_before_take_profit_at_same_point(self):
        # Construct a single price that would trip BOTH a stop and a take-profit if
        # both were checked: stop_loss_pct=0.0 (any r<=0 trips) AND
        # take_profit_pct=0.0 (any r>=0 trips). At the FIRST price r==0 satisfies
        # both; stop-loss must win.
        ret, reason = simulate_exit(
            0.50,
            [0.50],
            1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            take_profit_price=0.10,
        )
        self.assertEqual(reason, "stop_loss")
        # Exit at the entry price 0.50: (1/0.5)*0.5*0.98 - 1 = -0.02 (just the fee).
        self.assertAlmostEqual(ret, -0.02, places=6)

    def test_entry_price_zero_returns_skip(self):
        ret, reason = simulate_exit(
            0.0,
            [0.5, 0.6],
            1.0,
            stop_loss_pct=0.3,
            take_profit_pct=0.5,
            take_profit_price=0.9,
        )
        self.assertEqual(reason, "skip")
        self.assertEqual(ret, 0.0)

    def test_return_on_cost_math_loss_and_win(self):
        # Spec-pinned arithmetic. Loss: entry 0.5, exit 0.3, 200 bps.
        self.assertAlmostEqual(
            exit_backtest._return_on_cost(0.5, 0.3, 200.0), -0.412, places=6
        )
        # Win: entry 0.5, exit 1.0, 200 bps.
        self.assertAlmostEqual(
            exit_backtest._return_on_cost(0.5, 1.0, 200.0), 0.96, places=6
        )

    def test_out_of_range_prices_are_skipped_not_acted_on(self):
        # A bogus 1.5 price (out of [0,1]) must NOT trip the take-profit; the valid
        # 0.90 that follows does.
        ret, reason = simulate_exit(
            0.50,
            [1.5, 0.90],
            1.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            take_profit_price=0.85,
        )
        self.assertEqual(reason, "take_profit")
        self.assertAlmostEqual(ret, (1.0 / 0.5) * 0.90 * 0.98 - 1.0, places=6)


class GridSearchTest(unittest.TestCase):
    def _samples(self):
        # Two winners (rose then resolved at 1.0) and two losers (dipped through a
        # 35c level — r = -0.30, which a 0.30 stop catches with a capped loss of
        # -0.314, above the -0.6 floor — then on to 0.0). Entry 0.50 throughout.
        return [
            {"entry_price": 0.50, "price_path": [0.60, 0.95], "final_price": 1.0},
            {"entry_price": 0.50, "price_path": [0.70, 0.92], "final_price": 1.0},
            {"entry_price": 0.50, "price_path": [0.35, 0.20], "final_price": 0.0},
            {"entry_price": 0.50, "price_path": [0.35, 0.10], "final_price": 0.0},
        ]

    def test_combos_ranked_by_mean_return_desc(self):
        res = grid_search(
            self._samples(),
            sl_values=[None, 0.3, 0.5],
            tp_pct_values=[None, 0.5],
            tp_price_values=[None, 0.90],
        )
        means = [c["mean_return"] for c in res["combos"]]
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertIs(res["best_by_meanret"], res["combos"][0])

    def test_baseline_all_none_present(self):
        res = grid_search(
            self._samples(),
            # Deliberately OMIT None on every axis; baseline must still be injected.
            sl_values=[0.3, 0.5],
            tp_pct_values=[0.5],
            tp_price_values=[0.90],
        )
        baselines = [
            c
            for c in res["combos"]
            if c["stop_loss_pct"] is None
            and c["take_profit_pct"] is None
            and c["take_profit_price"] is None
        ]
        self.assertEqual(len(baselines), 1)
        self.assertIsNotNone(res["baseline"])
        # Baseline holds to resolution: 2 wins (+0.96 each) + 2 losses (-1.0 each).
        self.assertEqual(res["baseline"]["exit_mix"], {"resolution": 4})
        self.assertAlmostEqual(
            res["baseline"]["mean_return"], (0.96 + 0.96 - 1.0 - 1.0) / 4, places=6
        )

    def test_risk_adjusted_excludes_combos_below_worst_floor(self):
        res = grid_search(
            self._samples(),
            sl_values=[None, 0.3, 0.5],
            tp_pct_values=[None, 0.5],
            tp_price_values=[None, 0.90],
        )
        ra = res["risk_adjusted"]
        self.assertIsNotNone(ra)
        # The pick must respect the -0.6 worst-single-trade floor...
        self.assertGreaterEqual(ra["worst_return"], exit_backtest.RISK_ADJUSTED_WORST_FLOOR)
        # ...and be the highest mean_return among all eligible combos.
        eligible = [
            c
            for c in res["combos"]
            if c["worst_return"] >= exit_backtest.RISK_ADJUSTED_WORST_FLOOR
        ]
        self.assertTrue(eligible)
        self.assertAlmostEqual(
            ra["mean_return"], max(c["mean_return"] for c in eligible), places=12
        )
        # The baseline rides losers to -1.0, which is below the floor -> ineligible.
        self.assertLess(res["baseline"]["worst_return"], exit_backtest.RISK_ADJUSTED_WORST_FLOOR)

    def test_risk_adjusted_none_when_no_combo_clears_floor(self):
        # All-loser dataset with no stop available: every combo rides to -1.0.
        samples = [
            {"entry_price": 0.50, "price_path": [0.40, 0.20], "final_price": 0.0},
            {"entry_price": 0.50, "price_path": [0.30, 0.10], "final_price": 0.0},
        ]
        res = grid_search(
            samples,
            sl_values=[None],
            tp_pct_values=[None],
            tp_price_values=[None],
        )
        self.assertIsNone(res["risk_adjusted"])

    def test_exit_mix_counts_sum_to_n(self):
        res = grid_search(
            self._samples(),
            sl_values=[None, 0.3, 0.5],
            tp_pct_values=[None, 0.5],
            tp_price_values=[None, 0.90],
        )
        for combo in res["combos"]:
            self.assertEqual(sum(combo["exit_mix"].values()), combo["n"])

    def test_skipped_unsizable_samples_excluded_from_n(self):
        # One sample has entry 0 -> simulate_exit returns 'skip' -> dropped from n.
        samples = [
            {"entry_price": 0.50, "price_path": [0.60, 0.95], "final_price": 1.0},
            {"entry_price": 0.0, "price_path": [0.60, 0.95], "final_price": 1.0},
        ]
        res = grid_search(
            samples,
            sl_values=[None],
            tp_pct_values=[None],
            tp_price_values=[None],
        )
        baseline = res["baseline"]
        self.assertEqual(baseline["n"], 1)
        self.assertEqual(sum(baseline["exit_mix"].values()), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
