"""Tests for src/crypto_training/train_xgboost.py.

Hermetic: builds a tiny synthetic feature/label DataFrame in-memory, runs
a real XGBoost train + isotonic calibration, then asserts metrics fall
in sane ranges and the persisted model bundle is loadable.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

# libomp dance must be set BEFORE numpy/sklearn/xgboost get imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from crypto_training.train_xgboost import (
    _evaluate,
    _load_dataset,
    _reliability_slope,
    _time_based_split,
    train,
)


# ---------------------------------------------------------------------------
# Synthetic dataset: f1 + f2 are predictive (positive correlation with label)
# ---------------------------------------------------------------------------


def _synthetic_dataset(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    noise = rng.normal(0, 1, size=n)
    # Linear-ish signal with noise -> binary label.
    score = 0.7 * f1 + 0.5 * f2 + 0.3 * noise
    label = (score > np.median(score)).astype(int)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "f1": f1,
            "f2": f2,
            "f3_noise": noise,
            "label": label,
        }
    )


class TimeSplitTests(unittest.TestCase):
    def test_split_proportions(self) -> None:
        df = _synthetic_dataset(n=100)
        train_df, val_df, test_df = _time_based_split(
            df, val_frac=0.2, test_frac=0.2
        )
        self.assertEqual(len(train_df), 60)
        self.assertEqual(len(val_df), 20)
        self.assertEqual(len(test_df), 20)
        # Time order preserved -- last train timestamp is older than first val.
        self.assertLess(train_df["timestamp"].iloc[-1], val_df["timestamp"].iloc[0])

    def test_split_raises_when_train_empty(self) -> None:
        df = _synthetic_dataset(n=10)
        with self.assertRaises(ValueError):
            _time_based_split(df, val_frac=0.6, test_frac=0.6)


class ReliabilitySlopeTests(unittest.TestCase):
    def test_perfect_calibration_slope_near_one(self) -> None:
        # Build a smoothly distributed prob -> outcome pair: each row's
        # outcome is drawn from Bernoulli(p) where p is the predicted prob.
        # Aggregated reliability slope should land near 1.0.
        rng = np.random.default_rng(0)
        n = 20000
        probs = rng.uniform(0.05, 0.95, size=n)
        labels = (rng.uniform(0, 1, size=n) < probs).astype(int)
        slope = _reliability_slope(labels, probs, bins=10)
        self.assertGreater(slope, 0.85)
        self.assertLess(slope, 1.15)

    def test_constant_probs_returns_nan(self) -> None:
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.isnan(_reliability_slope(labels, probs)))


class EvaluateTests(unittest.TestCase):
    def test_binary_metrics_keys(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        y = rng.binomial(1, 0.4, size=n)
        p = rng.uniform(0, 1, size=n)
        metrics = _evaluate(y, p, label_classes=[0, 1])
        for key in ("brier", "log_loss", "accuracy", "auc", "reliability_slope"):
            self.assertIn(key, metrics)


class LoadDatasetTests(unittest.TestCase):
    def test_load_csv(self) -> None:
        df = _synthetic_dataset(n=20)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ds.csv"
            df.to_csv(path, index=False)
            loaded = _load_dataset(path)
            self.assertEqual(len(loaded), 20)
            # Always returned sorted by timestamp.
            self.assertTrue(loaded["timestamp"].is_monotonic_increasing)

    def test_load_raises_when_label_missing(self) -> None:
        df = _synthetic_dataset(n=10).drop(columns=["label"])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                _load_dataset(path)


class EndToEndTrainTests(unittest.TestCase):
    def test_train_persists_model_and_meta(self) -> None:
        df = _synthetic_dataset(n=600)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            out_dir = Path(td) / "model_out"
            summary = train(
                dataset_path=ds_path,
                output_dir=out_dir,
                val_frac=0.2,
                test_frac=0.2,
                # Smaller booster to keep the test fast.
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
            )
            self.assertEqual(summary.feature_count, 3)  # f1, f2, f3_noise
            self.assertGreater(summary.rows_train, 0)
            # Model + meta files persisted.
            model_path = out_dir / "model.joblib"
            meta_path = out_dir / "meta.json"
            self.assertTrue(model_path.exists())
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text())
            self.assertIn("metrics_test", meta)
            self.assertEqual(meta["calibration_method"], "isotonic")
            self.assertEqual(meta["feature_cols"], ["f1", "f2", "f3_noise"])
            # Loadable + can predict.
            model = joblib.load(model_path)
            X_demo = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
            probs = model.predict_proba(X_demo)
            self.assertEqual(probs.shape, (1, 2))

    def test_train_metrics_show_some_signal_above_random(self) -> None:
        # f1 + f2 carry signal; AUC on test should beat ~0.5.
        df = _synthetic_dataset(n=1200)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            summary = train(
                dataset_path=ds_path,
                output_dir=Path(td) / "out",
                val_frac=0.2,
                test_frac=0.2,
                xgb_kwargs={"n_estimators": 60, "max_depth": 4},
            )
            self.assertGreater(summary.metrics_test["auc"], 0.6)
            self.assertLess(summary.metrics_test["log_loss"], 0.7)


if __name__ == "__main__":
    unittest.main()
