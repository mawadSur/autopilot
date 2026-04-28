"""Tests for src/predictor.py.

Two layers:
  * Pure decision-mapping tests that don't touch torch.
  * One end-to-end test that loads the real model_sanity bundle, feeds a
    synthetic OHLCV history through a fake exchange, and asserts the
    predictor returns a well-shaped (side, confidence) tuple.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np

# macOS xgboost+sklearn libomp conflict -- harmless flag, set before any
# torch/sklearn import via the predictor.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class _FakeExchange:
    """Returns canned OHLCV candles. Records calls for assertion."""

    def __init__(self, candles: List[Dict[str, Any]]) -> None:
        self._candles = candles
        self.calls: List[Dict[str, Any]] = []

    def fetch_recent_candles(
        self, symbol: str, *, granularity: str = "ONE_MINUTE", limit: int = 350
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {"symbol": symbol, "granularity": granularity, "limit": limit}
        )
        return list(self._candles)


def _synthetic_candles(n: int = 500, start_price: float = 2000.0) -> List[Dict[str, Any]]:
    """Generate ``n`` synthetic 1m OHLCV bars ending at "now" (UTC).

    Uses a deterministic random walk so feature compute produces non-NaN values.
    """
    rng = np.random.default_rng(seed=42)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows: List[Dict[str, Any]] = []
    price = start_price
    for i in range(n):
        ts = end - timedelta(minutes=(n - 1 - i))
        ret = float(rng.normal(0.0, 0.0008))
        new_price = price * (1.0 + ret)
        high = max(price, new_price) * (1.0 + abs(rng.normal(0.0, 0.0003)))
        low = min(price, new_price) * (1.0 - abs(rng.normal(0.0, 0.0003)))
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(price),
                "high": float(high),
                "low": float(low),
                "close": float(new_price),
                "volume": float(abs(rng.normal(50.0, 10.0))),
            }
        )
        price = new_price
    return rows


# ---------------------------------------------------------------------------
# Pure decision-mapping tests (no model load)
# ---------------------------------------------------------------------------


class DecisionMappingTests(unittest.TestCase):
    """Test ``_probs_to_decision`` in isolation by stubbing __init__."""

    def _make_predictor(
        self, *, thr_long: float = 0.55, thr_short: float = 0.60, num_classes: int = 3
    ):
        from predictor import LegacyTransformerPredictor

        # Build a partially-initialised instance so we can call the pure method
        # without touching torch / load_model_bundle.
        p = LegacyTransformerPredictor.__new__(LegacyTransformerPredictor)
        p.thr_long = thr_long
        p.thr_short = thr_short
        p.num_classes = num_classes
        p.margin = 0.0
        return p

    def test_long_above_thr_returns_buy_with_long_prob(self) -> None:
        p = self._make_predictor()
        side, conf = p._probs_to_decision(np.array([0.10, 0.20, 0.70]))
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.70, places=6)

    def test_short_above_thr_returns_sell_with_short_prob(self) -> None:
        p = self._make_predictor()
        side, conf = p._probs_to_decision(np.array([0.65, 0.20, 0.15]))
        self.assertEqual(side, "sell")
        self.assertAlmostEqual(conf, 0.65, places=6)

    def test_below_thresholds_returns_neutral(self) -> None:
        p = self._make_predictor()
        side, conf = p._probs_to_decision(np.array([0.40, 0.30, 0.30]))
        # Neutral: long not above 0.55 and short not above 0.60.
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.5, places=6)

    def test_dominant_long_breaks_tie_against_short(self) -> None:
        # p_long == p_short, both above threshold -> long-dominant rule wins.
        p = self._make_predictor(thr_long=0.4, thr_short=0.4)
        side, conf = p._probs_to_decision(np.array([0.45, 0.10, 0.45]))
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.45, places=6)

    def test_two_class_input(self) -> None:
        p = self._make_predictor()
        side, conf = p._probs_to_decision(np.array([0.20, 0.80]))
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.80, places=6)

    def test_binary_head_input(self) -> None:
        p = self._make_predictor(num_classes=1)
        side, conf = p._probs_to_decision(np.array([0.70]))
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.70, places=6)

    def test_empty_input_returns_neutral(self) -> None:
        p = self._make_predictor()
        side, conf = p._probs_to_decision(np.array([]))
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.5, places=6)


# ---------------------------------------------------------------------------
# build_default_predict_fn tests
# ---------------------------------------------------------------------------


class BuildDefaultTests(unittest.TestCase):
    def test_returns_none_when_dir_missing(self) -> None:
        from predictor import build_default_predict_fn

        with mock.patch.dict(os.environ, {"LEGACY_MODEL_DIR": "/no/such/dir/xyz"}):
            self.assertIsNone(build_default_predict_fn(exchange=None))

    def test_returns_none_when_dir_blank(self) -> None:
        from predictor import build_default_predict_fn

        with mock.patch.dict(os.environ, {"LEGACY_MODEL_DIR": "   "}):
            self.assertIsNone(build_default_predict_fn(exchange=None))


# ---------------------------------------------------------------------------
# Integration: load real model_sanity, run end-to-end inference
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    Path("model_sanity/model.pt").exists(),
    "model_sanity bundle not present in repo root",
)
class LegacyTransformerIntegrationTests(unittest.TestCase):
    """Heavy: loads the real model + scaler. Requires torch."""

    def test_predictor_loads_and_returns_well_shaped_decision(self) -> None:
        from predictor import LegacyTransformerPredictor

        candles = _synthetic_candles(n=500)
        ex = _FakeExchange(candles)
        predictor = LegacyTransformerPredictor(
            model_dir="model_sanity",
            exchange=ex,
            warmup_bars=350,
        )

        # Fake ticker (predictor doesn't actually use the ticker fields right
        # now -- candles drive features).
        fake_ticker = mock.MagicMock(
            symbol="ETH/USD", bid=2000.0, ask=2000.5, last=2000.25
        )
        side, conf = predictor("ETH/USD", fake_ticker)

        self.assertIn(side, ("buy", "sell"))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        # The predictor pulled candles at least once.
        self.assertGreaterEqual(len(ex.calls), 1)

    def test_predictor_returns_neutral_on_insufficient_history(self) -> None:
        from predictor import LegacyTransformerPredictor

        # Far less than max(window, 240) bars -> warmup, neutral.
        candles = _synthetic_candles(n=50)
        ex = _FakeExchange(candles)
        predictor = LegacyTransformerPredictor(
            model_dir="model_sanity",
            exchange=ex,
            warmup_bars=350,
        )
        fake_ticker = mock.MagicMock(symbol="ETH/USD", bid=2000.0, ask=2000.5, last=2000.25)
        side, conf = predictor("ETH/USD", fake_ticker)
        self.assertEqual((side, conf), ("buy", 0.5))


class XGBoostPredictorBuildFallbackTests(unittest.TestCase):
    """``build_default_predict_fn`` priority: crypto > legacy > placeholder."""

    def test_falls_back_to_legacy_when_crypto_dir_unset(self) -> None:
        from predictor import build_default_predict_fn

        with mock.patch.dict(
            os.environ,
            {"CRYPTO_MODEL_DIR": "", "LEGACY_MODEL_DIR": "/no/such/dir/zzz"},
            clear=False,
        ):
            self.assertIsNone(build_default_predict_fn(exchange=None))

    def test_falls_back_to_legacy_when_crypto_dir_missing(self) -> None:
        from predictor import build_default_predict_fn

        with mock.patch.dict(
            os.environ,
            {
                "CRYPTO_MODEL_DIR": "/no/such/crypto/dir",
                "LEGACY_MODEL_DIR": "/no/such/legacy/dir",
            },
            clear=False,
        ):
            self.assertIsNone(build_default_predict_fn(exchange=None))


@unittest.skipUnless(
    Path("model_sanity/model.pt").exists(),
    "model_sanity bundle not present in repo root",
)
class XGBoostPredictorIntegrationTests(unittest.TestCase):
    """Train a tiny XGBoost model in a tempdir, wrap it, run end-to-end."""

    def _make_synth_dataset(self, n: int = 500) -> "pd.DataFrame":
        import pandas as pd

        rng = np.random.default_rng(seed=11)
        # Build features that match what compute_features actually produces
        # so meta.feature_cols can reference them and the predictor's
        # candle->feature path will recover them from real candles.
        # We just need *some* known names. Pick a handful of common ones.
        f = {
            "return_5": rng.normal(0, 0.001, size=n),
            "rv_60": rng.uniform(0.0001, 0.005, size=n),
            "tod_sin": rng.uniform(-1, 1, size=n),
        }
        score = (
            0.6 * f["return_5"] / 0.001
            + 0.3 * (f["rv_60"] - 0.0025) / 0.001
            + 0.1 * f["tod_sin"]
        )
        labels = (score > np.median(score)).astype(int)
        timestamps = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
        return pd.DataFrame({"timestamp": timestamps, **f, "label": labels})

    def _train_tiny_model(self, out_dir: Path) -> None:
        import tempfile

        from crypto_training.train_xgboost import train

        df = self._make_synth_dataset(n=500)
        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "ds.csv"
            df.to_csv(ds_path, index=False)
            train(
                dataset_path=ds_path,
                output_dir=out_dir,
                val_frac=0.2,
                test_frac=0.2,
                xgb_kwargs={"n_estimators": 20, "max_depth": 3},
            )

    def test_xgb_predictor_loads_meta_and_returns_well_shaped_decision(self) -> None:
        from predictor import XGBoostPredictor

        candles = _synthetic_candles(n=400)
        ex = _FakeExchange(candles)
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model_xgb"
            self._train_tiny_model(model_dir)
            predictor = XGBoostPredictor(
                model_dir=str(model_dir),
                exchange=ex,
                thr_long=0.5,
                warmup_bars=350,
            )
            fake_ticker = mock.MagicMock(symbol="ETH/USD")
            side, conf = predictor("ETH/USD", fake_ticker)
        self.assertIn(side, ("buy", "sell"))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertGreaterEqual(len(ex.calls), 1)

    def test_xgb_predictor_returns_neutral_below_thr_long(self) -> None:
        from predictor import XGBoostPredictor

        candles = _synthetic_candles(n=400)
        ex = _FakeExchange(candles)
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model_xgb"
            self._train_tiny_model(model_dir)
            # Set thr_long very high so almost no real prediction can clear
            # it -- we expect the safe-default neutral.
            predictor = XGBoostPredictor(
                model_dir=str(model_dir),
                exchange=ex,
                thr_long=0.99,
                warmup_bars=350,
            )
            fake_ticker = mock.MagicMock(symbol="ETH/USD")
            side, conf = predictor("ETH/USD", fake_ticker)
        self.assertEqual((side, conf), ("buy", 0.5))

    def test_xgb_predictor_returns_neutral_on_insufficient_history(self) -> None:
        from predictor import XGBoostPredictor

        candles = _synthetic_candles(n=50)
        ex = _FakeExchange(candles)
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model_xgb"
            self._train_tiny_model(model_dir)
            predictor = XGBoostPredictor(
                model_dir=str(model_dir), exchange=ex, thr_long=0.5
            )
            fake_ticker = mock.MagicMock(symbol="ETH/USD")
            side, conf = predictor("ETH/USD", fake_ticker)
        self.assertEqual((side, conf), ("buy", 0.5))

    def test_xgb_predictor_raises_when_meta_missing(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "missing"
            model_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                XGBoostPredictor(model_dir=str(model_dir), exchange=None)


if __name__ == "__main__":
    unittest.main()
