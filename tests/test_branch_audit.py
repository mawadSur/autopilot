"""Unit tests for the post-fee branch audit + regime router.

Dependency-light on purpose: exercises the pure decision logic (regime
classification, keep/cut verdicts, honest post-fee reconstruction, consensus)
without loading a model, torch, TA-Lib, or the 1.3G datasets.

Run:
    env PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_branch_audit
"""

import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from regime_router import RegimeRouter, Regime, classify_regime  # noqa: E402
from crypto_training.branch_audit import (  # noqa: E402
    ConsensusFilter, _verdict, _post_fee_metrics, _regime_breakdown,
    CLS_LONG, CLS_HOLD,
)


class TestRegimeClassifier(unittest.TestCase):
    def test_trend_up_down_chop(self):
        self.assertEqual(classify_regime({"adx": 40, "close_over_ema_50": 0.002}), Regime.TREND_UP)
        self.assertEqual(classify_regime({"adx": 40, "close_over_ema_50": -0.002}), Regime.TREND_DOWN)
        # weak trend strength -> chop regardless of direction
        self.assertEqual(classify_regime({"adx": 10, "close_over_ema_50": 0.05}), Regime.CHOP)

    def test_nan_is_chop_failsafe(self):
        self.assertEqual(classify_regime({"adx": float("nan"), "close_over_ema_50": 0.01}), Regime.CHOP)
        self.assertEqual(classify_regime({}), Regime.CHOP)

    def test_router_enable_and_threshold(self):
        r = RegimeRouter()
        self.assertEqual(r.route({"adx": 40, "close_over_ema_50": 0.01}), (True, 0.55, Regime.TREND_UP))
        enabled, thr, reg = r.route({"adx": 40, "close_over_ema_50": -0.01})
        self.assertFalse(enabled)  # do not long into a downtrend
        self.assertEqual(reg, Regime.TREND_DOWN)

    def test_from_config_override(self):
        r = RegimeRouter.from_config({"adx_trend_min": 30, "params": {
            "trend_up": {"enabled": True, "thr_long": 0.42}}})
        self.assertEqual(r.adx_trend_min, 30)
        self.assertAlmostEqual(r.route({"adx": 40, "close_over_ema_50": 0.01})[1], 0.42)
        # adx below the raised min is now chop, not trend_up
        self.assertEqual(r.classify({"adx": 28, "close_over_ema_50": 0.01}), Regime.CHOP)


class TestVerdict(unittest.TestCase):
    def test_cost_always_on(self):
        self.assertEqual(_verdict("cost", -0.01, 0.05, 100), "COST (always on)")

    def test_remove_branch_keep_vs_cut(self):
        # Removing branch HURTS EV (ablate < base) -> the branch pays -> KEEP.
        self.assertEqual(_verdict("remove", base_exp=-0.010, ablate_exp=-0.020, ablate_trades=50), "KEEP")
        # Removing branch IMPROVES EV -> branch is dead weight -> CUT.
        self.assertEqual(_verdict("remove", base_exp=-0.020, ablate_exp=-0.010, ablate_trades=50), "CUT")
        # No real difference -> drop the complexity.
        self.assertTrue(_verdict("remove", -0.010, -0.010, 50).startswith("CUT (neutral)"))

    def test_add_branch_keep_requires_gain_and_trades(self):
        self.assertTrue(_verdict("add", base_exp=-0.020, ablate_exp=-0.005, ablate_trades=50).startswith("KEEP"))
        # gain but too few trades -> not trustworthy
        self.assertTrue(_verdict("add", -0.020, -0.005, 3).startswith("SKIP (too few"))
        # no gain -> skip
        self.assertEqual(_verdict("add", -0.010, -0.011, 50), "SKIP")


class _StubSim:
    """Minimal PortfolioSimulator stand-in for _post_fee_metrics/_regime_breakdown."""

    def __init__(self, trade_log, last_equity):
        self.trade_log = trade_log
        self.last_equity = last_equity

    def report(self):
        return {"portfolio": {"max_drawdown": 0.0, "maker_fills": 0, "taker_fills": 0,
                              "missed_entries": 0, "exposure": 0.0}}


class TestPostFeeReconstruction(unittest.TestCase):
    def test_net_returns_from_equity_sequence(self):
        # Two round-trips: 10000 -> 10100 (+1%) -> 9999 (~-1%). The per-trade net
        # return must come from equity ratios, NOT the gross trade_log 'ret'.
        tl = [
            {"action": "enter", "timestamp": "t0"},
            {"action": "exit", "timestamp": "t0", "equity": 10100.0, "ret": 0.05},  # gross ret is a lie here
            {"action": "enter", "timestamp": "t1"},
            {"action": "exit", "timestamp": "t1", "equity": 9999.0, "ret": 0.05},
        ]
        m = _post_fee_metrics(_StubSim(tl, 9999.0), start_capital=10000.0)
        self.assertEqual(m["n_trades"], 2)
        self.assertAlmostEqual(m["post_fee_expectancy"],
                               ((10100/10000 - 1) + (9999/10100 - 1)) / 2, places=8)
        self.assertAlmostEqual(m["post_fee_win_rate"], 0.5, places=8)

    def test_regime_breakdown_tags_entry_regime(self):
        tl = [
            {"action": "enter", "timestamp": "a"},
            {"action": "exit", "timestamp": "a", "equity": 10100.0},
            {"action": "enter", "timestamp": "b"},
            {"action": "exit", "timestamp": "b", "equity": 10000.0},
        ]
        ts2reg = {"a": Regime.TREND_UP, "b": Regime.CHOP}
        out = _regime_breakdown(_StubSim(tl, 10000.0), 10000.0, ts2reg)
        self.assertEqual(out[Regime.TREND_UP]["n_trades"], 1)
        self.assertGreater(out[Regime.TREND_UP]["post_fee_expectancy"], 0.0)
        self.assertEqual(out[Regime.CHOP]["n_trades"], 1)
        self.assertLess(out[Regime.CHOP]["post_fee_expectancy"], 0.0)


class TestConsensusFilter(unittest.TestCase):
    def test_requires_consecutive(self):
        cf = ConsensusFilter(2)
        # isolated long signals never confirm -> the sparse-model silencing bug
        self.assertEqual(cf.step(CLS_LONG), CLS_HOLD)
        self.assertEqual(cf.step(CLS_HOLD), CLS_HOLD)
        self.assertEqual(cf.step(CLS_LONG), CLS_HOLD)
        # two in a row confirms
        self.assertEqual(cf.step(CLS_LONG), CLS_LONG)

    def test_consensus_one_is_passthrough(self):
        cf = ConsensusFilter(1)
        self.assertEqual(cf.step(CLS_LONG), CLS_LONG)


if __name__ == "__main__":
    unittest.main()
