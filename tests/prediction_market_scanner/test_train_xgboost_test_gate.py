"""Unit tests for the test-split gate in train_xgboost.py.

The gate checks that the val-chosen optimal_threshold achieves a
minimum win rate AND minimum trade count on the held-out test split.
If either floor is missed, ``optimal_threshold`` is set to null and
``threshold_status`` is set to ``"test_gate_failed"``.

Tests:

  * ``_compute_test_gate`` returns correct (win_rate, n_trades) for
    various thresholds and label distributions.
  * ``_compute_test_gate`` returns (0.0, 0) when no probas exceed threshold.
  * ``train()`` writes new fields: ``test_winrate_at_optimal_threshold``,
    ``test_ntrades_at_optimal_threshold``, ``test_gate_reason``.
  * Gate passes (strong signal): ``optimal_threshold`` is a float,
    ``threshold_status == "ok"``.
  * Gate fails (winrate floor): ``optimal_threshold`` is null,
    ``threshold_status == "test_gate_failed"``, reason string present.
  * Gate fails (ntrades floor): same null/status/reason outcome.
  * Custom min_test_winrate / min_test_ntrades kwargs thread through train().
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from crypto_training.train_xgboost import _compute_test_gate, train


# ---------------------------------------------------------------------------
# _compute_test_gate unit tests
# ---------------------------------------------------------------------------


class ComputeTestGateTests(unittest.TestCase):
    def test_all_wins_above_threshold(self) -> None:
        y = np.ones(50, dtype=int)
        p = np.full(50, 0.8)
        wr, nt = _compute_test_gate(y, p, optimal_threshold=0.7)
        self.assertAlmostEqual(wr, 1.0)
        self.assertEqual(nt, 50)

    def test_half_wins(self) -> None:
        y = np.array([1, 0] * 20, dtype=int)
        p = np.full(40, 0.6)
        wr, nt = _compute_test_gate(y, p, optimal_threshold=0.5)
        self.assertAlmostEqual(wr, 0.5)
        self.assertEqual(nt, 40)

    def test_no_trades_above_threshold(self) -> None:
        y = np.ones(20, dtype=int)
        p = np.full(20, 0.3)
        wr, nt = _compute_test_gate(y, p, optimal_threshold=0.8)
        self.assertAlmostEqual(wr, 0.0)
        self.assertEqual(nt, 0)

    def test_partial_trigger(self) -> None:
        # Only rows where p >= 0.6 count; first 10 are wins, last 10 are losses.
        y = np.array([1] * 10 + [0] * 10, dtype=int)
        p = np.array([0.7] * 10 + [0.4] * 10)
        wr, nt = _compute_test_gate(y, p, optimal_threshold=0.6)
        self.assertAlmostEqual(wr, 1.0)
        self.assertEqual(nt, 10)


# ---------------------------------------------------------------------------
# Helpers shared by end-to-end gate tests
# ---------------------------------------------------------------------------


def _make_strong_signal_dataset(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Bimodal proba distribution that should produce a clear separating threshold."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    score = 1.5 * f1 + 0.6 * f2
    label = (score > np.percentile(score, 40)).astype(int)
    ts = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    return pd.DataFrame({"timestamp": ts, "f1": f1, "f2": f2, "label": label})


def _make_anti_predictive_dataset(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Random labels — model can't beat coin-flip, win rate ~50% ± noise."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    label = rng.binomial(1, 0.5, size=n)
    ts = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    return pd.DataFrame({"timestamp": ts, "f1": f1, "f2": f2, "label": label})


# ---------------------------------------------------------------------------
# train() integration tests
# ---------------------------------------------------------------------------


class TrainTestGateMetaFieldsTests(unittest.TestCase):
    """New meta fields exist regardless of gate outcome."""

    def test_meta_has_new_gate_fields(self) -> None:
        df = _make_strong_signal_dataset()
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.csv"
            df.to_csv(ds, index=False)
            out = Path(td) / "model"
            train(
                dataset_path=ds,
                output_dir=out,
                val_frac=0.15,
                test_frac=0.15,
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
            )
            meta = json.loads((out / "meta.json").read_text())

        for key in (
            "test_winrate_at_optimal_threshold",
            "test_ntrades_at_optimal_threshold",
            "test_gate_reason",
        ):
            self.assertIn(key, meta, f"meta.json missing key {key!r}")


class TrainGatePassesTests(unittest.TestCase):
    """Strong signal dataset: gate should pass with very low winrate floor."""

    def test_gate_pass_writes_optimal_threshold(self) -> None:
        df = _make_strong_signal_dataset()
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.csv"
            df.to_csv(ds, index=False)
            out = Path(td) / "model"
            train(
                dataset_path=ds,
                output_dir=out,
                val_frac=0.15,
                test_frac=0.15,
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
                # Floor well below 50% so almost any signal passes.
                min_test_winrate=0.10,
                min_test_ntrades=1,
            )
            meta = json.loads((out / "meta.json").read_text())

        self.assertIsNotNone(meta["optimal_threshold"])
        self.assertEqual(meta["threshold_status"], "ok")
        self.assertIsNone(meta["test_gate_reason"])
        # Sanity: reported test winrate matches the gate condition we set.
        self.assertGreaterEqual(
            meta["test_winrate_at_optimal_threshold"], 0.10
        )


class TrainGateFailsWinrateTests(unittest.TestCase):
    """Gate fails when min_test_winrate is impossibly high (1.0)."""

    def test_gate_fail_nulls_optimal_threshold(self) -> None:
        df = _make_strong_signal_dataset()
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.csv"
            df.to_csv(ds, index=False)
            out = Path(td) / "model"
            train(
                dataset_path=ds,
                output_dir=out,
                val_frac=0.15,
                test_frac=0.15,
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
                min_test_winrate=1.0,   # impossible: requires 100% win rate
                min_test_ntrades=1,
            )
            meta = json.loads((out / "meta.json").read_text())

        self.assertIsNone(meta["optimal_threshold"])
        self.assertEqual(meta["threshold_status"], "test_gate_failed")
        self.assertIsNotNone(meta["test_gate_reason"])
        self.assertIn("test_winrate", meta["test_gate_reason"])


class TrainGateFailsNtradesTests(unittest.TestCase):
    """Gate fails when min_test_ntrades is impossibly large."""

    def test_gate_fail_ntrades_floor_nulls_threshold(self) -> None:
        df = _make_strong_signal_dataset()
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.csv"
            df.to_csv(ds, index=False)
            out = Path(td) / "model"
            train(
                dataset_path=ds,
                output_dir=out,
                val_frac=0.15,
                test_frac=0.15,
                xgb_kwargs={"n_estimators": 30, "max_depth": 3},
                min_test_winrate=0.0,
                min_test_ntrades=99_999,  # impossible: more trades than rows
            )
            meta = json.loads((out / "meta.json").read_text())

        self.assertIsNone(meta["optimal_threshold"])
        self.assertEqual(meta["threshold_status"], "test_gate_failed")
        self.assertIsNotNone(meta["test_gate_reason"])
        self.assertIn("test_ntrades", meta["test_gate_reason"])


class TrainGateWinrateFieldsAreSaneTests(unittest.TestCase):
    """test_winrate_at_optimal_threshold is in [0, 1] for any outcome."""

    def test_winrate_field_bounds(self) -> None:
        df = _make_anti_predictive_dataset()
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.csv"
            df.to_csv(ds, index=False)
            out = Path(td) / "model"
            train(
                dataset_path=ds,
                output_dir=out,
                val_frac=0.15,
                test_frac=0.15,
                xgb_kwargs={"n_estimators": 20, "max_depth": 3},
            )
            meta = json.loads((out / "meta.json").read_text())

        wr = meta["test_winrate_at_optimal_threshold"]
        nt = meta["test_ntrades_at_optimal_threshold"]
        if wr is not None:
            self.assertGreaterEqual(wr, 0.0)
            self.assertLessEqual(wr, 1.0)
        if nt is not None:
            self.assertGreaterEqual(nt, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
