"""Sharpe-weighted threshold optimization tests (Lane B P1 #12).

D2 lock: pick the threshold maximising simulated Sharpe on the
validation set, NOT F1. (Higher F1 favours threshold balance, but a
60% win-rate at +5% beats a 70% win-rate at +1% and only Sharpe
captures that.)

Tests:

  * synthetic bimodal proba / matching y -> optimal threshold lands
    in the basin where wins outnumber losses; simulated Sharpe > 0,
  * degenerate inputs (all-zero or all-one labels) -> graceful
    default to 0.5 with a warning,
  * meta.json carries optimal_threshold + threshold_metrics,
  * XGBoostPredictor reads optimal_threshold via the precedence chain
    (explicit kwarg > meta.json > 0.5 default).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path

# libomp dance must be set BEFORE numpy/sklearn/xgboost get imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from crypto_training.train_xgboost import (
    _sweep_thresholds_for_sharpe,
    train,
)


class SweepThresholdsTests(unittest.TestCase):
    def test_bimodal_proba_picks_separating_threshold(self) -> None:
        # 100 wins clustered at proba=0.85, 100 losses at proba=0.15.
        # Any threshold in [0.3, 0.8] perfectly separates them so the
        # simulated trades are all wins -> high Sharpe at most thresholds
        # with a non-empty trade set. The picked threshold must yield
        # Sharpe > 0 and trigger trades.
        rng = np.random.default_rng(0)
        n_each = 100
        y_wins = np.ones(n_each, dtype=int)
        y_losses = np.zeros(n_each, dtype=int)
        p_wins = rng.normal(0.85, 0.02, size=n_each)
        p_losses = rng.normal(0.15, 0.02, size=n_each)
        y = np.concatenate([y_wins, y_losses])
        proba = np.clip(np.concatenate([p_wins, p_losses]), 0.0, 1.0)

        best_thr, per_threshold = _sweep_thresholds_for_sharpe(y, proba)
        # The optimal threshold should sit comfortably below 0.85.
        self.assertGreaterEqual(best_thr, 0.3)
        self.assertLess(best_thr, 0.85)
        self.assertGreater(per_threshold[f"{best_thr:.4f}"]["sharpe"], 0)

    def test_all_zero_labels_defaults_to_05(self) -> None:
        # No wins anywhere -> every triggered trade loses; some
        # candidate may produce negative Sharpe but no positive Sharpe
        # is reachable. Sweep returns 0.5 as the safe default with a
        # logged warning.
        y = np.zeros(100, dtype=int)
        proba = np.linspace(0.1, 0.9, 100)
        with self.assertLogs(
            "crypto_training.train_xgboost", level=logging.WARNING
        ) as cm:
            best_thr, _ = _sweep_thresholds_for_sharpe(y, proba)
        self.assertAlmostEqual(best_thr, 0.5, places=6)
        self.assertTrue(any("0.5" in m for m in cm.output))

    def test_all_one_labels_returns_finite_threshold(self) -> None:
        # All wins -> simulated Sharpe is positive at every threshold
        # that triggers >0 trades. Best threshold should be a real
        # candidate in [0.3, 0.8].
        y = np.ones(100, dtype=int)
        proba = np.linspace(0.2, 0.95, 100)
        best_thr, per_threshold = _sweep_thresholds_for_sharpe(y, proba)
        self.assertTrue(np.isfinite(best_thr))
        self.assertGreaterEqual(best_thr, 0.3)
        self.assertLessEqual(best_thr, 0.8)

    def test_no_triggers_anywhere_defaults_to_05(self) -> None:
        # Probas all 0; every threshold triggers 0 trades; warn + 0.5.
        y = np.array([0, 1, 0, 1, 0, 1])
        proba = np.zeros(6)
        best_thr, _ = _sweep_thresholds_for_sharpe(y, proba)
        self.assertAlmostEqual(best_thr, 0.5, places=6)

    def test_per_threshold_dict_keyed_on_candidate(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.5, size=200)
        p = rng.uniform(0, 1, size=200)
        candidates = np.array([0.4, 0.5, 0.6])
        _, per_threshold = _sweep_thresholds_for_sharpe(
            y, p, candidates=candidates
        )
        self.assertEqual(set(per_threshold.keys()), {"0.4000", "0.5000", "0.6000"})


def _signal_dataset(n: int = 1000, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    score = 0.9 * f1 + 0.4 * f2
    label = (score > np.median(score)).astype(int)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    return pd.DataFrame(
        {"timestamp": timestamps, "f1": f1, "f2": f2, "label": label}
    )


class TrainPersistsOptimalThresholdTests(unittest.TestCase):
    def test_meta_carries_optimal_threshold_and_metrics(self) -> None:
        df = _signal_dataset(n=900)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            out_dir = Path(td) / "model"
            train(
                dataset_path=ds_path,
                output_dir=out_dir,
                val_frac=0.2,
                test_frac=0.2,
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
            )
            meta = json.loads((out_dir / "meta.json").read_text())
            self.assertIn("optimal_threshold", meta)
            self.assertIsNotNone(meta["optimal_threshold"])
            self.assertGreaterEqual(meta["optimal_threshold"], 0.3)
            self.assertLessEqual(meta["optimal_threshold"], 0.8)
            self.assertIn("threshold_metrics", meta)
            # threshold_metrics is keyed by 4-decimal string of candidate.
            self.assertGreater(len(meta["threshold_metrics"]), 0)
            for thr_str, m in meta["threshold_metrics"].items():
                # Each entry has the sim_ payload keys.
                for k in (
                    "sharpe",
                    "max_drawdown",
                    "win_rate",
                    "n_trades",
                ):
                    self.assertIn(k, m)


class XGBoostPredictorReadsMetaThresholdTests(unittest.TestCase):
    """End-to-end: trained meta with optimal_threshold flows through to
    XGBoostPredictor.thr_long when no explicit override is given."""

    def test_predictor_picks_up_meta_threshold(self) -> None:
        from predictor import XGBoostPredictor

        df = _signal_dataset(n=900)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            out_dir = Path(td) / "model"
            train(
                dataset_path=ds_path,
                output_dir=out_dir,
                val_frac=0.2,
                test_frac=0.2,
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
            )
            meta = json.loads((out_dir / "meta.json").read_text())
            optimal = float(meta["optimal_threshold"])

            # No explicit thr_long -> predictor reads from meta.
            p = XGBoostPredictor(model_dir=str(out_dir), exchange=None)
            self.assertAlmostEqual(p.thr_long, optimal, places=6)
            self.assertEqual(p._thr_source, "meta.optimal_threshold")

            # Explicit thr_long wins.
            p2 = XGBoostPredictor(
                model_dir=str(out_dir), exchange=None, thr_long=0.91
            )
            self.assertAlmostEqual(p2.thr_long, 0.91, places=6)
            self.assertEqual(p2._thr_source, "explicit")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
