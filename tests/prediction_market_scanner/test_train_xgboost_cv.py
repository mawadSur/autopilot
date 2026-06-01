"""Walk-forward CV tests for ``crypto_training.train_xgboost`` (Lane B P1 #11).

Property-based assertions (hand-rolled because hypothesis is not a
required dep here):

  * For every fold, ``max(train.timestamp) < min(val.timestamp)`` -- no
    leakage between the slice the booster fits on and the slice it's
    scored on.
  * Validation slices tile the dataset without overlap (an anchored
    rolling design means the val *windows* don't overlap; train always
    grows).
  * ``walk_forward_cv`` returns ``[]`` when the dataset is smaller than
    ``window_size`` (caller falls back to single-split ``train()``).
"""

from __future__ import annotations

import os
import unittest

# libomp dance must be set BEFORE numpy/sklearn/xgboost get imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from crypto_training.train_xgboost import (
    WalkForwardSummary,
    train_with_walk_forward_cv,
    walk_forward_cv,
)


def _synthetic(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    score = 0.7 * f1 + 0.5 * f2
    label = (score > np.median(score)).astype(int)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "f1": f1,
            "f2": f2,
            "label": label,
        }
    )


class WalkForwardCVPropertiesTests(unittest.TestCase):
    """Pure-function properties: no train/val leakage, no overlapping vals."""

    def test_no_leakage_per_fold(self) -> None:
        df = _synthetic(n=1500)
        folds = walk_forward_cv(df, window_size=500, step_size=100)
        self.assertGreater(len(folds), 0)
        for i, (train_df, val_df) in enumerate(folds):
            last_train = pd.to_datetime(train_df["timestamp"].iloc[-1])
            first_val = pd.to_datetime(val_df["timestamp"].iloc[0])
            self.assertLess(
                last_train,
                first_val,
                msg=f"fold {i}: train extends past val start",
            )

    def test_val_slices_do_not_overlap(self) -> None:
        df = _synthetic(n=1500)
        folds = walk_forward_cv(df, window_size=500, step_size=200)
        seen_ts: set = set()
        for _train_df, val_df in folds:
            ts = set(val_df["timestamp"].tolist())
            overlap = ts & seen_ts
            self.assertFalse(
                overlap,
                msg=f"validation slices overlap on {len(overlap)} timestamps",
            )
            seen_ts |= ts

    def test_each_fold_train_is_anchored_at_zero(self) -> None:
        # Anchored rolling: every fold's training set starts at the very
        # first row. Train length grows by step_size each fold.
        df = _synthetic(n=1200)
        folds = walk_forward_cv(df, window_size=400, step_size=100)
        first_ts = df["timestamp"].iloc[0]
        for train_df, _ in folds:
            self.assertEqual(train_df["timestamp"].iloc[0], first_ts)

    def test_train_lengths_increase_monotonically(self) -> None:
        df = _synthetic(n=1200)
        folds = walk_forward_cv(df, window_size=400, step_size=100)
        train_lens = [len(t) for t, _ in folds]
        for a, b in zip(train_lens, train_lens[1:]):
            self.assertLess(a, b)

    def test_returns_empty_when_dataset_smaller_than_window(self) -> None:
        df = _synthetic(n=200)
        folds = walk_forward_cv(df, window_size=500, step_size=100)
        self.assertEqual(folds, [])

    def test_returns_empty_when_window_equals_n(self) -> None:
        df = _synthetic(n=100)
        # window_size == n leaves no rows to validate on.
        folds = walk_forward_cv(df, window_size=100, step_size=10)
        self.assertEqual(folds, [])

    def test_invalid_arg_combinations_raise(self) -> None:
        df = _synthetic(n=200)
        with self.assertRaises(ValueError):
            walk_forward_cv(df, window_size=0, step_size=10)
        with self.assertRaises(ValueError):
            walk_forward_cv(df, window_size=10, step_size=0)
        with self.assertRaises(ValueError):
            walk_forward_cv(df, window_size=-5, step_size=10)


class TrainWithWalkForwardCVTests(unittest.TestCase):
    """End-to-end: fit + calibrate on each fold, aggregate metrics."""

    def test_aggregates_metrics_across_folds(self) -> None:
        df = _synthetic(n=1500)
        summary = train_with_walk_forward_cv(
            df,
            window_size=500,
            step_size=300,
            xgb_kwargs={"n_estimators": 25, "max_depth": 3},
        )
        self.assertIsInstance(summary, WalkForwardSummary)
        self.assertGreater(summary.n_folds, 1)
        self.assertEqual(len(summary.per_fold), summary.n_folds)
        # Mean metrics include canonical names.
        for key in ("brier", "log_loss", "accuracy"):
            self.assertIn(key, summary.mean_metrics)
        self.assertEqual(summary.feature_count, 2)
        self.assertEqual(summary.rows_total, 1500)

    def test_each_fold_metric_dict_carries_timestamp_bounds(self) -> None:
        df = _synthetic(n=1200)
        summary = train_with_walk_forward_cv(
            df,
            window_size=400,
            step_size=200,
            xgb_kwargs={"n_estimators": 20, "max_depth": 3},
        )
        for fold in summary.per_fold:
            for key in (
                "train_first_ts",
                "train_last_ts",
                "val_first_ts",
                "val_last_ts",
                "rows_train",
                "rows_val",
            ):
                self.assertIn(key, fold)
            self.assertLess(fold["train_last_ts"], fold["val_first_ts"])

    def test_raises_when_dataset_too_small(self) -> None:
        df = _synthetic(n=200)
        with self.assertRaises(ValueError):
            train_with_walk_forward_cv(
                df,
                window_size=500,
                step_size=100,
                xgb_kwargs={"n_estimators": 10, "max_depth": 2},
            )

    def test_raises_when_label_column_missing(self) -> None:
        df = _synthetic(n=600).drop(columns=["label"])
        with self.assertRaises(ValueError):
            train_with_walk_forward_cv(df, window_size=200, step_size=50)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
