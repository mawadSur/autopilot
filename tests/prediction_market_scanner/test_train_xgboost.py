"""Tests for ``calibration_agent.train_xgboost``."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from calibration_agent.build_dataset import OUTPUT_COLUMNS
from calibration_agent.ml_service import ALL_FEATURE_COLUMNS
from calibration_agent.train_xgboost import (
    MIN_TOTAL_ROWS,
    _time_based_split,
    train,
)


def _synthetic_dataset(
    n_rows: int = 60,
    *,
    seed: int = 0,
    shuffle_labels: bool = False,
) -> pd.DataFrame:
    """Build a tiny synthetic dataset with a real signal in news_sentiment_score.

    Positive ``news_sentiment_score`` predicts ``market_outcome=1``; the implied
    probability is set to 0.5 with mild noise so it lacks the signal.
    """

    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        sentiment = rng.uniform(-100, 100)
        # Strong logistic-style relationship; sentiment > 0 ⇒ outcome usually 1.
        prob_true = 1.0 / (1.0 + np.exp(-sentiment / 10.0))
        outcome = int(rng.uniform() < prob_true)
        captured = base_time + timedelta(hours=i)
        row: Dict[str, Any] = {col: 0.0 for col in ALL_FEATURE_COLUMNS}
        row.update(
            {
                "trade_id": f"t{i:03d}",
                "captured_at_utc": captured.isoformat(),
                "settled_at": (captured + timedelta(days=1)).isoformat(),
                "implied_prob": float(0.5 + rng.uniform(-0.02, 0.02)),
                "spread": 0.02,
                "volume_24h": 1000.0,
                "news_sentiment_score": float(sentiment),
                "market_outcome": outcome,
                "final_outcome": outcome,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    if shuffle_labels:
        shuffled = df["market_outcome"].sample(frac=1.0, random_state=seed + 1).to_numpy()
        df["market_outcome"] = shuffled
        df["final_outcome"] = shuffled
    return df


def _write_dataset(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


class TrainXGBoostTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        self.dataset_path = self.tmp_path / "dataset.parquet"
        _write_dataset(_synthetic_dataset(), self.dataset_path)
        self.model_path = self.tmp_path / "model.joblib"

    def test_train_writes_model_and_metadata(self) -> None:
        summary = train(self.dataset_path, output_path=self.model_path)

        self.assertTrue(self.model_path.exists())
        meta_path = self.model_path.with_suffix(self.model_path.suffix + ".meta.json")
        self.assertTrue(meta_path.exists())

        for key in (
            "model_path",
            "meta_path",
            "training_range",
            "n_samples",
            "calibration_method",
            "xgb_kwargs",
            "random_state",
            "feature_columns",
            "test_metrics",
            "baseline_metrics",
            "trained_at_utc",
        ):
            self.assertIn(key, summary, msg=f"missing summary key: {key}")

        self.assertEqual(summary["feature_columns"], list(ALL_FEATURE_COLUMNS))
        for split in ("train", "val", "test", "total"):
            self.assertGreater(summary["n_samples"][split], 0)

        # Persisted metadata mirrors the returned summary.
        meta_on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta_on_disk["feature_columns"], list(ALL_FEATURE_COLUMNS))

    def test_loaded_model_can_predict_proba(self) -> None:
        train(self.dataset_path, output_path=self.model_path)

        model = joblib.load(self.model_path)
        df = pd.read_parquet(self.dataset_path)
        x = df[list(ALL_FEATURE_COLUMNS)].astype(float).head(3)
        proba = model.predict_proba(x)
        self.assertEqual(np.asarray(proba).shape, (3, 2))
        for value in np.asarray(proba)[:, 1]:
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0)

    def test_time_based_split_preserves_chronology(self) -> None:
        df = _synthetic_dataset(n_rows=60)
        train_df, val_df, test_df = _time_based_split(
            df, val_fraction=0.2, test_fraction=0.1
        )

        last_train = pd.to_datetime(train_df["captured_at_utc"].iloc[-1])
        first_val = pd.to_datetime(val_df["captured_at_utc"].iloc[0])
        last_val = pd.to_datetime(val_df["captured_at_utc"].iloc[-1])
        first_test = pd.to_datetime(test_df["captured_at_utc"].iloc[0])

        self.assertLessEqual(last_train, first_val)
        self.assertLessEqual(last_val, first_test)
        self.assertEqual(
            len(train_df) + len(val_df) + len(test_df), len(df)
        )

    def test_too_few_rows_raises(self) -> None:
        small = _synthetic_dataset(n_rows=MIN_TOTAL_ROWS - 1)
        small_path = self.tmp_path / "small.parquet"
        _write_dataset(small, small_path)

        with self.assertRaises(ValueError):
            train(small_path, output_path=self.tmp_path / "junk.joblib")

    def test_xgb_kwargs_propagate_to_metadata(self) -> None:
        summary = train(
            self.dataset_path,
            output_path=self.model_path,
            n_estimators=50,
            max_depth=3,
        )
        self.assertEqual(summary["xgb_kwargs"]["n_estimators"], 50)
        self.assertEqual(summary["xgb_kwargs"]["max_depth"], 3)


if __name__ == "__main__":
    unittest.main()
