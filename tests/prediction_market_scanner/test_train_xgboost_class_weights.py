"""Class-weighting tests for ``crypto_training.train_xgboost`` (Lane B P1 #15).

We don't always have a 50/50 label split. ``_compute_class_weights``
emits ``scale_pos_weight = count_neg / count_pos`` so XGBoost penalises
the majority class proportionally. These tests verify:

  * the formula matches the standard xgboost convention,
  * the booster actually receives the kwarg through ``train()``,
  * meta.json carries the class distribution + weight payload for
    later audit.
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

import numpy as np
import pandas as pd

from crypto_training.train_xgboost import (
    _class_distribution,
    _compute_class_weights,
    train,
)


def _imbalanced_dataset(
    n: int = 800, *, pos_frac: float = 0.4, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    # Force exact pos_frac.
    n_pos = int(round(n * pos_frac))
    label = np.zeros(n, dtype=int)
    label[:n_pos] = 1
    rng.shuffle(label)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    return pd.DataFrame(
        {"timestamp": timestamps, "f1": f1, "f2": f2, "label": label}
    )


class ComputeClassWeightsTests(unittest.TestCase):
    def test_60_40_split_gives_scale_pos_weight_1_5(self) -> None:
        # 60 negatives, 40 positives -> 60 / 40 == 1.5.
        y = np.concatenate([np.zeros(60, dtype=int), np.ones(40, dtype=int)])
        w = _compute_class_weights(y)
        self.assertIn("scale_pos_weight", w)
        self.assertAlmostEqual(w["scale_pos_weight"], 1.5, places=6)

    def test_balanced_split_gives_scale_pos_weight_1(self) -> None:
        y = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
        w = _compute_class_weights(y)
        self.assertAlmostEqual(w["scale_pos_weight"], 1.0, places=6)

    def test_extreme_imbalance(self) -> None:
        y = np.concatenate([np.zeros(990, dtype=int), np.ones(10, dtype=int)])
        w = _compute_class_weights(y)
        self.assertAlmostEqual(w["scale_pos_weight"], 99.0, places=6)

    def test_single_class_returns_empty(self) -> None:
        y = np.zeros(50, dtype=int)
        self.assertEqual(_compute_class_weights(y), {})

    def test_multiclass_emits_per_class_weights(self) -> None:
        # 3-class with counts (50, 100, 150). Balanced weight per class:
        # n / (n_classes * count_k).
        y = np.concatenate(
            [
                np.zeros(50, dtype=int),
                np.ones(100, dtype=int),
                np.full(150, 2, dtype=int),
            ]
        )
        w = _compute_class_weights(y)
        self.assertIn("sample_weight_0", w)
        self.assertIn("sample_weight_1", w)
        self.assertIn("sample_weight_2", w)
        n = 300
        self.assertAlmostEqual(w["sample_weight_0"], n / (3 * 50), places=6)
        self.assertAlmostEqual(w["sample_weight_1"], n / (3 * 100), places=6)
        self.assertAlmostEqual(w["sample_weight_2"], n / (3 * 150), places=6)


class ClassDistributionTests(unittest.TestCase):
    def test_basic_counts(self) -> None:
        y = np.array([0, 0, 0, 1, 1])
        self.assertEqual(_class_distribution(y), {"0": 3, "1": 2})

    def test_empty_array(self) -> None:
        self.assertEqual(_class_distribution(np.array([], dtype=int)), {})


class TrainPropagatesClassWeightsTests(unittest.TestCase):
    """End-to-end: a 60/40 dataset hands ``scale_pos_weight≈1.5`` to xgboost."""

    def test_class_weight_lands_in_booster_and_meta(self) -> None:
        df = _imbalanced_dataset(n=800, pos_frac=0.4)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            out_dir = Path(td) / "model"
            summary = train(
                dataset_path=ds_path,
                output_dir=out_dir,
                val_frac=0.2,
                test_frac=0.2,
                xgb_kwargs={"n_estimators": 20, "max_depth": 3},
            )
            meta = json.loads((out_dir / "meta.json").read_text())
            # Class weights persisted.
            self.assertIn("class_weights", meta)
            self.assertIn("scale_pos_weight", meta["class_weights"])
            # Pos fraction in the *training* slice will land near 0.4
            # (split is sequential, so class balance is preserved up to
            # sampling noise). Allow generous slack.
            spw = meta["class_weights"]["scale_pos_weight"]
            self.assertGreater(spw, 1.0)
            self.assertLess(spw, 3.0)
            # Booster received the kwarg.
            self.assertIn("scale_pos_weight", meta["xgb_kwargs"])
            self.assertAlmostEqual(
                meta["xgb_kwargs"]["scale_pos_weight"], spw, places=6
            )
            # Class distribution audited per split.
            for split in ("train", "val", "test"):
                self.assertIn(split, meta["class_distribution"])

            # The summary doesn't carry class_weights directly, but the
            # train run should still complete without throwing.
            self.assertGreater(summary.rows_train, 0)

    def test_explicit_scale_pos_weight_wins(self) -> None:
        df = _imbalanced_dataset(n=800, pos_frac=0.4)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            out_dir = Path(td) / "model"
            train(
                dataset_path=ds_path,
                output_dir=out_dir,
                val_frac=0.2,
                test_frac=0.2,
                xgb_kwargs={
                    "n_estimators": 15,
                    "max_depth": 3,
                    "scale_pos_weight": 4.2,
                },
            )
            meta = json.loads((out_dir / "meta.json").read_text())
            # Explicit override should land in xgb_kwargs untouched.
            self.assertAlmostEqual(
                meta["xgb_kwargs"]["scale_pos_weight"], 4.2, places=6
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
