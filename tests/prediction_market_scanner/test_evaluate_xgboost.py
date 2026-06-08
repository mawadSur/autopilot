"""Tests for ``calibration_agent.evaluate_xgboost``."""

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
from calibration_agent.evaluate_xgboost import (
    BRIER_MAX,
    LOGLOSS_MAX,
    SLOPE_MAX,
    SLOPE_MIN,
    _evaluate_criteria,
    _reliability_slope,
    evaluate,
)
from calibration_agent.ml_service import ALL_FEATURE_COLUMNS
from calibration_agent.train_xgboost import train


def _synthetic_dataset(
    n_rows: int = 200,
    *,
    seed: int = 0,
    shuffle_labels: bool = False,
) -> pd.DataFrame:
    """Same generator as the train tests, with a slightly larger default n.

    A larger n_rows is needed here so the gate-passing assertions are stable
    on the small sample sizes; the relationship between sentiment and outcome
    is still linear-logistic and easily learnable.
    """

    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        sentiment = rng.uniform(-100, 100)
        prob_true = 1.0 / (1.0 + np.exp(-sentiment / 10.0))
        outcome = int(rng.uniform() < prob_true)
        captured = base_time + timedelta(hours=i)
        row: Dict[str, Any] = {col: 0.0 for col in ALL_FEATURE_COLUMNS}
        row.update(
            {
                "trade_id": f"t{i:04d}",
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
        shuffled = (
            df["market_outcome"].sample(frac=1.0, random_state=seed + 1).to_numpy()
        )
        df["market_outcome"] = shuffled
        df["final_outcome"] = shuffled
    return df


def _write_dataset(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


class EvaluateXGBoostTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        # Real-signal dataset.
        self.dataset_path = self.tmp_path / "dataset.parquet"
        _write_dataset(_synthetic_dataset(), self.dataset_path)

        # Model trained against the real-signal dataset.
        self.model_path = self.tmp_path / "model.joblib"
        train(self.dataset_path, output_path=self.model_path)

    def test_real_signal_model_passes_gate(self) -> None:
        result = evaluate(self.model_path, self.dataset_path)

        self.assertTrue(result["pass"], msg=f"unexpected fail: {result}")
        self.assertLess(result["model_metrics"]["brier"], BRIER_MAX)
        self.assertLess(result["model_metrics"]["log_loss"], LOGLOSS_MAX)
        self.assertLess(
            result["model_metrics"]["brier"], result["baseline_metrics"]["brier"]
        )
        self.assertGreaterEqual(
            result["model_metrics"]["reliability_slope"], SLOPE_MIN
        )
        self.assertLessEqual(
            result["model_metrics"]["reliability_slope"], SLOPE_MAX
        )

    def test_evaluate_result_has_expected_shape(self) -> None:
        result = evaluate(self.model_path, self.dataset_path)

        for key in (
            "pass",
            "model_metrics",
            "baseline_metrics",
            "criteria",
            "thresholds",
            "evaluated_at",
            "model_path",
            "dataset_path",
        ):
            self.assertIn(key, result, msg=f"missing key: {key}")

        for k in (
            "brier_lt_0_20",
            "logloss_lt_0_55",
            "slope_in_band",
            "beats_baseline_brier",
        ):
            self.assertIn(k, result["criteria"])
            self.assertIsInstance(result["criteria"][k], bool)

    def test_garbage_model_fails_gate(self) -> None:
        # Train a model on label-shuffled data: no signal ⇒ Brier ≥ 0.25.
        garbage_dataset = self.tmp_path / "garbage.parquet"
        _write_dataset(
            _synthetic_dataset(shuffle_labels=True), garbage_dataset
        )
        garbage_model = self.tmp_path / "garbage.joblib"
        train(garbage_dataset, output_path=garbage_model)

        # Evaluate the garbage model against the *real-signal* dataset so the
        # mismatch is unambiguous.
        result = evaluate(garbage_model, self.dataset_path)

        self.assertFalse(result["pass"])

    def test_report_path_writes_json(self) -> None:
        from calibration_agent.evaluate_xgboost import main as eval_main

        report_path = self.tmp_path / "report.json"
        rc = eval_main(
            [
                str(self.model_path),
                str(self.dataset_path),
                "--report-path",
                str(report_path),
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue(report_path.exists())

        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("pass", loaded)
        self.assertIn("criteria", loaded)

    def test_missing_model_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate(self.tmp_path / "nope.joblib", self.dataset_path)

    def test_missing_dataset_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate(self.model_path, self.tmp_path / "nope.parquet")

    def test_dataset_missing_features_raises(self) -> None:
        df = pd.read_parquet(self.dataset_path).drop(columns=["news_sentiment_score"])
        broken_path = self.tmp_path / "broken.parquet"
        df.to_parquet(broken_path, index=False)
        with self.assertRaises(ValueError):
            evaluate(self.model_path, broken_path)


class CriteriaBoundaryTests(unittest.TestCase):
    """Boundary unit tests for ``_evaluate_criteria`` (no model required)."""

    def _make_metrics(
        self,
        *,
        brier: float,
        log_loss: float,
        slope: float,
    ) -> Dict[str, float]:
        return {
            "brier": brier,
            "log_loss": log_loss,
            "accuracy": 0.5,
            "auc": 0.5,
            "reliability_slope": slope,
        }

    def test_all_pass_at_boundaries(self) -> None:
        # Brier just under 0.20, logloss just under 0.55, slope inside band,
        # baseline strictly worse on Brier ⇒ all four flags True.
        model = self._make_metrics(brier=0.199, log_loss=0.549, slope=1.0)
        baseline = self._make_metrics(brier=0.25, log_loss=0.6, slope=1.0)
        crit = _evaluate_criteria(model, baseline)
        self.assertTrue(all(crit.values()))

    def test_brier_just_over_fails(self) -> None:
        model = self._make_metrics(brier=0.201, log_loss=0.4, slope=1.0)
        baseline = self._make_metrics(brier=0.25, log_loss=0.6, slope=1.0)
        crit = _evaluate_criteria(model, baseline)
        self.assertFalse(crit["brier_lt_0_20"])
        self.assertFalse(all(crit.values()))

    def test_logloss_just_over_fails(self) -> None:
        model = self._make_metrics(brier=0.1, log_loss=0.551, slope=1.0)
        baseline = self._make_metrics(brier=0.25, log_loss=0.6, slope=1.0)
        crit = _evaluate_criteria(model, baseline)
        self.assertFalse(crit["logloss_lt_0_55"])

    def test_slope_outside_band_fails(self) -> None:
        below = _evaluate_criteria(
            self._make_metrics(brier=0.1, log_loss=0.4, slope=0.79),
            self._make_metrics(brier=0.25, log_loss=0.6, slope=1.0),
        )
        above = _evaluate_criteria(
            self._make_metrics(brier=0.1, log_loss=0.4, slope=1.21),
            self._make_metrics(brier=0.25, log_loss=0.6, slope=1.0),
        )
        self.assertFalse(below["slope_in_band"])
        self.assertFalse(above["slope_in_band"])

    def test_does_not_beat_baseline_fails(self) -> None:
        model = self._make_metrics(brier=0.18, log_loss=0.4, slope=1.0)
        baseline = self._make_metrics(brier=0.18, log_loss=0.4, slope=1.0)
        crit = _evaluate_criteria(model, baseline)
        self.assertFalse(crit["beats_baseline_brier"])

    def test_nan_slope_fails(self) -> None:
        model = self._make_metrics(brier=0.1, log_loss=0.4, slope=float("nan"))
        baseline = self._make_metrics(brier=0.25, log_loss=0.6, slope=1.0)
        crit = _evaluate_criteria(model, baseline)
        self.assertFalse(crit["slope_in_band"])


class ReliabilitySlopeTests(unittest.TestCase):
    def test_perfect_calibration_slope_one(self) -> None:
        # Construct a long ascending probability vector and matching outcomes
        # so empirical-frequency-by-decile lines up with predicted-prob midpoint.
        rng = np.random.default_rng(0)
        n = 5000
        probs = rng.uniform(0.0, 1.0, size=n)
        outcomes = (rng.uniform(size=n) < probs).astype(int)
        slope = _reliability_slope(outcomes, probs)
        self.assertAlmostEqual(slope, 1.0, delta=0.15)

    def test_constant_predictions_returns_nan(self) -> None:
        n = 100
        probs = np.full(n, 0.5)
        outcomes = np.zeros(n, dtype=int)
        slope = _reliability_slope(outcomes, probs)
        self.assertTrue(np.isnan(slope))


if __name__ == "__main__":
    unittest.main()
