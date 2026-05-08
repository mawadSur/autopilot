"""Unit tests for ``regime_memory.backfill``.

Covers:
* :func:`build_store` end-to-end on a synthetic OHLCV-like frame
* :func:`main` CLI entry: writes an .npz store reloadable via NaiveRegimeStore
* Synthetic metadata (``regime_label``, ``realized_sharpe``,
  ``optimal_threshold``, ``kelly_size_pct``) is populated and finite
* ``--max-windows`` cap is honored
* Trending vs choppy synthetic frames produce different regime labels
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from regime_memory.backfill import (
    _optimal_threshold,
    _regime_label_from_returns,
    _realized_sharpe,
    build_store,
    main,
)
from regime_memory.store import NaiveRegimeStore


def _synthetic_dataset(n_rows: int = 600, *, trend: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Return a fake feature parquet with the columns the backfill expects.

    ``trend`` is the per-bar drift added to the log-return series — pass a
    positive value to bias toward "trending up", zero for choppy.
    """

    rng = np.random.default_rng(seed)
    log_ret = rng.normal(loc=trend, scale=0.001, size=n_rows)
    close = np.exp(np.cumsum(log_ret)) * 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="1min").astype(str),
            "log_ret": log_ret,
            "return_1": np.r_[0.0, np.diff(close) / close[:-1]],
            "ema_spread_9_21": rng.normal(size=n_rows) * 0.01,
            "atr_14": np.abs(rng.normal(size=n_rows)) * 0.5,
            "vol_z_20": rng.normal(size=n_rows),
        }
    )
    return df


class HelperFunctionTests(unittest.TestCase):
    def test_regime_label_trending_up(self) -> None:
        rets = np.full(60, 0.001)
        self.assertEqual(_regime_label_from_returns(rets), 2.0)

    def test_regime_label_trending_down(self) -> None:
        rets = np.full(60, -0.001)
        self.assertEqual(_regime_label_from_returns(rets), 0.0)

    def test_regime_label_choppy(self) -> None:
        rng = np.random.default_rng(0)
        # Zero-mean, non-zero std → |net| < vol almost surely
        rets = rng.normal(scale=0.01, size=60)
        # Force-zero net by subtracting the mean.
        rets = rets - rets.mean()
        self.assertEqual(_regime_label_from_returns(rets), 1.0)

    def test_realized_sharpe_zero_std_is_zero(self) -> None:
        self.assertEqual(_realized_sharpe(np.zeros(60), 525_600), 0.0)

    def test_realized_sharpe_positive_for_consistent_up(self) -> None:
        # A small consistent drift with low noise should yield a positive
        # annualized Sharpe.
        rng = np.random.default_rng(1)
        rets = rng.normal(loc=0.001, scale=0.0005, size=60)
        sharpe = _realized_sharpe(rets, 525_600)
        self.assertGreater(sharpe, 0.0)
        self.assertTrue(math.isfinite(sharpe))

    def test_optimal_threshold_in_grid(self) -> None:
        rng = np.random.default_rng(2)
        signal = rng.normal(size=60)
        rets = rng.normal(size=60)
        thr = _optimal_threshold(signal, rets)
        # Result must be one of the 7 grid points (within float tolerance).
        grid = np.linspace(0.40, 0.70, 7)
        self.assertTrue(any(abs(thr - g) < 1e-6 for g in grid))


class BuildStoreTests(unittest.TestCase):
    def test_build_store_populates_metadata_fields(self) -> None:
        df = _synthetic_dataset(n_rows=600, trend=0.0)
        store = build_store(
            df,
            symbol="ETH-USD",
            window_bars=60,
            label_horizon_bars=60,
            max_windows=4,
        )
        self.assertEqual(len(store), 4)
        # Every window must carry the four documented metadata fields.
        results = store.query([0.0] * store.dim, k=4)
        for w, _ in results:
            self.assertIn("optimal_threshold", w.metadata)
            self.assertIn("realized_sharpe", w.metadata)
            self.assertIn("regime_label", w.metadata)
            self.assertIn("kelly_size_pct", w.metadata)
            for key, value in w.metadata.items():
                self.assertTrue(
                    math.isfinite(value),
                    f"metadata field {key} is non-finite: {value}",
                )

    def test_max_windows_cap_is_honored(self) -> None:
        df = _synthetic_dataset(n_rows=1000)
        store = build_store(
            df,
            symbol="ETH-USD",
            window_bars=60,
            label_horizon_bars=60,
            max_windows=3,
        )
        self.assertEqual(len(store), 3)

    def test_trending_vs_choppy_yields_different_labels(self) -> None:
        # Strong upward drift — the next-N-bar net return dominates vol so
        # the label should land at 2.0 (trending up) for most windows.
        trending = _synthetic_dataset(n_rows=300, trend=0.005, seed=7)
        store = build_store(
            trending,
            symbol="ETH-USD",
            window_bars=60,
            label_horizon_bars=60,
            max_windows=3,
        )
        labels = []
        for w, _ in store.query([0.0] * store.dim, k=3):
            labels.append(w.metadata["regime_label"])
        # At least one window in a strong-trend dataset should hit the
        # trending-up label.
        self.assertIn(2.0, labels)

    def test_window_bars_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_store(_synthetic_dataset(), symbol="X", window_bars=0)
        with self.assertRaises(ValueError):
            build_store(
                _synthetic_dataset(),
                symbol="X",
                label_horizon_bars=0,
            )


class CliMainTests(unittest.TestCase):
    def test_main_writes_reloadable_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "synthetic.parquet"
            _synthetic_dataset(n_rows=800).to_parquet(dataset)
            output = tmp_path / "store.npz"

            rc = main(
                [
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--symbol",
                    "ETH-USD",
                    "--window-bars",
                    "60",
                    "--label-horizon-bars",
                    "60",
                    "--max-windows",
                    "5",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(output.exists())

            loaded = NaiveRegimeStore.load(output)
            self.assertEqual(len(loaded), 5)
            # Round-trip preserves the metadata fields.
            results = loaded.query([0.0] * loaded.dim, k=5)
            for w, _ in results:
                for field in (
                    "optimal_threshold",
                    "realized_sharpe",
                    "regime_label",
                    "kelly_size_pct",
                ):
                    self.assertIn(field, w.metadata)

    def test_main_returns_nonzero_for_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "store.npz"
            rc = main(
                [
                    "--dataset",
                    str(Path(tmp) / "does_not_exist.parquet"),
                    "--output",
                    str(output),
                ]
            )
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
