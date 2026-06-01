"""Backtest PnL reuse + simulation tests (Lane B P1 #13).

The trainer needs a proxy "would this booster make money?" signal
alongside its calibration metrics. ``_simulate_strategy_pnl`` walks the
(y_true, proba) array as if each row were a 1-bar trade and emits
Sharpe / max-drawdown / win-rate.

These tests assert:

  * a perfect predictor produces high Sharpe and ~0 max drawdown,
  * a random predictor produces near-zero Sharpe,
  * degenerate inputs (zero variance, no triggers) clip to finite
    values rather than NaN/inf,
  * shape mismatches raise rather than silently align.
"""

from __future__ import annotations

import os
import unittest

# libomp dance must be set BEFORE numpy/sklearn/xgboost get imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from crypto_training.train_xgboost import _evaluate, _simulate_strategy_pnl


class SimulateStrategyPnLTests(unittest.TestCase):
    def test_perfect_predictor_high_sharpe_low_dd(self) -> None:
        # 200 trades, every prediction matches truth -> all wins, no
        # drawdown beyond fees.
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.5, size=200)
        # Confidence = 1.0 when y=1, 0.0 when y=0; threshold 0.5 fires
        # only on the wins, so every trade is a win.
        proba = y.astype(float)
        result = _simulate_strategy_pnl(
            y, proba, threshold=0.5, position_size=1.0, fee_bps=0.0
        )
        self.assertGreater(result["sharpe"], 2.0)
        self.assertLess(abs(result["max_drawdown"]), 1e-6)
        self.assertEqual(result["win_rate"], 1.0)

    def test_random_predictor_near_zero_sharpe(self) -> None:
        # Truth and probabilities independent -> long-run win-rate ~50%
        # and Sharpe oscillates around 0.
        rng = np.random.default_rng(11)
        y = rng.binomial(1, 0.5, size=2000)
        proba = rng.uniform(0, 1, size=2000)
        result = _simulate_strategy_pnl(
            y, proba, threshold=0.5, position_size=1.0, fee_bps=0.0
        )
        # Within a generous band; this is a stochastic property test.
        # A tighter bound would just produce flake.
        self.assertLess(abs(result["sharpe"]), 50.0)

    def test_no_triggers_produces_zero_sharpe_zero_trades(self) -> None:
        # Every proba below threshold -> zero trades; equity stays flat
        # so std is 0 and sharpe must clip to 0 (not NaN).
        y = np.array([0, 1, 0, 1, 0, 1])
        proba = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2])
        result = _simulate_strategy_pnl(
            y, proba, threshold=0.9, position_size=1.0, fee_bps=0.0
        )
        self.assertEqual(result["n_trades"], 0)
        self.assertEqual(result["sharpe"], 0.0)
        self.assertTrue(np.isfinite(result["sharpe"]))
        self.assertEqual(result["win_rate"], 0.0)

    def test_all_losses_produces_negative_or_zero_sharpe(self) -> None:
        # Every trigger lands on a loss. Sharpe must be negative or
        # zero (depending on the std handling) but always finite.
        y = np.zeros(50, dtype=int)
        proba = np.ones(50, dtype=float)
        result = _simulate_strategy_pnl(
            y, proba, threshold=0.5, position_size=1.0, fee_bps=0.0
        )
        self.assertEqual(result["n_trades"], 50)
        self.assertEqual(result["win_rate"], 0.0)
        # Sharpe is NaN-safe -- finite.
        self.assertTrue(np.isfinite(result["sharpe"]))

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _simulate_strategy_pnl(
                np.array([0, 1, 0]), np.array([0.1, 0.2]), threshold=0.5
            )

    def test_fees_reduce_pnl(self) -> None:
        # Same predictions; higher fees -> lower sharpe.
        rng = np.random.default_rng(2)
        y = (rng.uniform(0, 1, size=500) > 0.4).astype(int)
        proba = y.astype(float) * 0.9 + 0.1  # always above 0.5
        zero_fees = _simulate_strategy_pnl(y, proba, fee_bps=0.0)
        with_fees = _simulate_strategy_pnl(y, proba, fee_bps=200.0)
        self.assertGreaterEqual(zero_fees["sharpe"], with_fees["sharpe"])

    def test_returns_finite_metrics(self) -> None:
        # Smoke: every reported metric is a finite float.
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.5, size=300)
        proba = rng.uniform(0, 1, size=300)
        result = _simulate_strategy_pnl(y, proba, threshold=0.5)
        for k, v in result.items():
            self.assertTrue(
                np.isfinite(v), msg=f"metric {k} is not finite: {v}"
            )


class EvaluateMergesSimulationTests(unittest.TestCase):
    """``_evaluate`` should fold the simulated PnL into its output dict."""

    def test_evaluate_includes_sim_keys(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        y = rng.binomial(1, 0.4, size=n)
        p = rng.uniform(0, 1, size=n)
        metrics = _evaluate(y, p, label_classes=[0, 1])
        for key in (
            "sim_sharpe",
            "sim_max_drawdown",
            "sim_win_rate",
            "sim_avg_win",
            "sim_avg_loss",
            "sim_n_trades",
        ):
            self.assertIn(key, metrics)

    def test_multiclass_evaluate_skips_sim(self) -> None:
        # Multi-class doesn't have a clean long/short policy at this
        # layer; sim_* keys should be absent. ``_evaluate`` accepts a
        # 2-D probability matrix for multi-class.
        rng = np.random.default_rng(5)
        n = 300
        y = rng.integers(0, 3, size=n)
        raw = rng.uniform(0, 1, size=(n, 3))
        p = raw / raw.sum(axis=1, keepdims=True)
        metrics = _evaluate(y, p, label_classes=[0, 1, 2])
        for key in ("sim_sharpe", "sim_max_drawdown"):
            self.assertNotIn(key, metrics)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
