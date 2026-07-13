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
        # Confidence must be 0.0 (not 0.5) so any reasonable supervisor
        # --min-confidence filters neutrals. With the old 0.5 sentinel,
        # lowering --min-confidence below 0.5 to let real low-prob triggers
        # through caused every neutral tick to fire a trade.
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.0, places=6)

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

    def test_neutral_sentinel_is_below_any_sensible_min_confidence(self) -> None:
        # Regression contract: the predictor's neutral signal must be lower
        # than any --min-confidence an operator would reasonably pass to the
        # supervisor. 0.0 ensures the supervisor filters neutrals regardless
        # of the chosen entry threshold.
        from predictor import _NEUTRAL_RESULT
        side, conf = _NEUTRAL_RESULT
        self.assertEqual(side, "buy")
        self.assertEqual(conf, 0.0)

    def test_empty_input_returns_neutral(self) -> None:
        p = self._make_predictor()
        side, conf = p._probs_to_decision(np.array([]))
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.0, places=6)


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

    def test_raises_when_crypto_dir_missing(self) -> None:
        # Per Lane B P0 #5: a configured-but-missing CRYPTO_MODEL_DIR is a
        # human error (typo, leftover env var, dataset moved). Silently
        # falling back hides the mistake; raise loudly instead.
        from predictor import build_default_predict_fn

        with mock.patch.dict(
            os.environ,
            {
                "CRYPTO_MODEL_DIR": "/no/such/crypto/dir",
                "LEGACY_MODEL_DIR": "/no/such/legacy/dir",
            },
            clear=False,
        ):
            with self.assertRaises(FileNotFoundError):
                build_default_predict_fn(exchange=None)


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
            # it -- we expect the safe-default neutral. A *valid* prediction
            # below threshold returns _NEUTRAL_RESULT (0.0), the sentinel that
            # any sensible --min-confidence filters out (see b4e31c4). This is
            # distinct from the warmup/error path, which returns 0.5.
            predictor = XGBoostPredictor(
                model_dir=str(model_dir),
                exchange=ex,
                thr_long=0.99,
                warmup_bars=350,
            )
            fake_ticker = mock.MagicMock(symbol="ETH/USD")
            side, conf = predictor("ETH/USD", fake_ticker)
        self.assertEqual((side, conf), ("buy", 0.0))

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


class MultiSymbolMapParseTests(unittest.TestCase):
    """``_parse_crypto_model_map`` quote-handling + malformed-entry fallback."""

    def test_parse_simple(self) -> None:
        from predictor import _parse_crypto_model_map

        m = _parse_crypto_model_map(
            "ETH/USD=model_crypto/eth_usd_v1,BTC/USD=model_crypto/btc_usd_v1"
        )
        self.assertEqual(set(m.keys()), {"ETH/USD", "BTC/USD"})
        self.assertEqual(m["ETH/USD"], ("model_crypto/eth_usd_v1", None))

    def test_parse_with_thresholds(self) -> None:
        from predictor import _parse_crypto_model_map

        m = _parse_crypto_model_map(
            "ETH/USD=model_crypto/eth_usd_v1:0.50,BTC/USD=model_crypto/btc_usd_v1:0.30"
        )
        self.assertAlmostEqual(m["ETH/USD"][1], 0.50)
        self.assertAlmostEqual(m["BTC/USD"][1], 0.30)

    def test_parse_skips_malformed_entries(self) -> None:
        from predictor import _parse_crypto_model_map

        m = _parse_crypto_model_map(
            "ETH/USD=ok_dir,GARBAGE,=missing_sym,BTC/USD="
        )
        # Only ETH/USD survives; the rest are malformed.
        self.assertEqual(list(m.keys()), ["ETH/USD"])

    def test_parse_empty(self) -> None:
        from predictor import _parse_crypto_model_map

        self.assertEqual(_parse_crypto_model_map(""), {})
        self.assertEqual(_parse_crypto_model_map("  "), {})


class EntryFilterTests(unittest.TestCase):
    """Verify CRYPTO_ENTRY_FILTER env parsing + filter rejection path."""

    def test_unset_env_returns_no_filter(self) -> None:
        from predictor import _build_entry_filter_from_env
        os.environ.pop("CRYPTO_ENTRY_FILTER", None)
        fn, name = _build_entry_filter_from_env()
        self.assertIsNone(fn)
        self.assertEqual(name, "")

    def test_none_value_returns_no_filter(self) -> None:
        from predictor import _build_entry_filter_from_env
        os.environ["CRYPTO_ENTRY_FILTER"] = "none"
        try:
            fn, name = _build_entry_filter_from_env()
            self.assertIsNone(fn)
            self.assertEqual(name, "")
        finally:
            os.environ.pop("CRYPTO_ENTRY_FILTER")

    def test_vol_proxy_filter_built(self) -> None:
        from predictor import _build_entry_filter_from_env
        os.environ["CRYPTO_ENTRY_FILTER"] = "vol_proxy:1.5"
        try:
            fn, name = _build_entry_filter_from_env()
            self.assertIsNotNone(fn)
            self.assertEqual(name, "vol_proxy(1.5)")
            # Below-MA bar (vol_log=0 -> expm1=0 < 100 * 1.5) rejected.
            self.assertFalse(fn({"vol_log": 0.0, "vol_ma_20": 100.0}))
            # Above-MA bar accepted.
            self.assertTrue(fn({"vol_log": 7.0, "vol_ma_20": 100.0}))  # expm1(7)~1095
        finally:
            os.environ.pop("CRYPTO_ENTRY_FILTER")

    def test_combined_filter_anded(self) -> None:
        from predictor import _build_entry_filter_from_env
        os.environ["CRYPTO_ENTRY_FILTER"] = "vol_proxy:1.5,atr:80"
        try:
            fn, name = _build_entry_filter_from_env()
            self.assertIsNotNone(fn)
            self.assertIn("vol_proxy", name)
            self.assertIn("atr", name)
        finally:
            os.environ.pop("CRYPTO_ENTRY_FILTER")

    def test_unknown_gate_warned_and_dropped(self) -> None:
        from predictor import _build_entry_filter_from_env
        os.environ["CRYPTO_ENTRY_FILTER"] = "nonsense:42"
        try:
            fn, name = _build_entry_filter_from_env()
            self.assertIsNone(fn)
            self.assertEqual(name, "")
        finally:
            os.environ.pop("CRYPTO_ENTRY_FILTER")


class MultiSymbolPredictorRoutingTests(unittest.TestCase):
    """Test that the multi-symbol predictor routes to the right per-symbol model."""

    def test_routes_to_correct_predictor_per_symbol(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor

        # Two stub predictors that record which symbol they were called with.
        class _StubPredictor:
            def __init__(self, label: str) -> None:
                self.label = label
                self.calls: List[str] = []

            def __call__(self, symbol, ticker):
                self.calls.append(symbol)
                return ("buy", 0.99 if self.label == "eth" else 0.42)

        eth_stub = _StubPredictor("eth")
        btc_stub = _StubPredictor("btc")
        multi = MultiSymbolXGBoostPredictor(
            model_map={"ETH/USD": eth_stub, "BTC/USD": btc_stub}
        )
        # ETH call goes to eth_stub.
        side_eth, conf_eth = multi("ETH/USD", None)
        self.assertEqual((side_eth, conf_eth), ("buy", 0.99))
        self.assertEqual(eth_stub.calls, ["ETH/USD"])
        self.assertEqual(btc_stub.calls, [])
        # BTC call goes to btc_stub.
        multi("BTC/USD", None)
        self.assertEqual(btc_stub.calls, ["BTC/USD"])

    def test_unknown_symbol_returns_neutral(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor

        multi = MultiSymbolXGBoostPredictor(model_map={"ETH/USD": object()})
        side, conf = multi("DOGE/USD", None)
        self.assertEqual((side, conf), ("buy", 0.0))

    def test_empty_map_raises(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor

        with self.assertRaises(ValueError):
            MultiSymbolXGBoostPredictor(model_map={})

    def test_call_mirrors_child_kelly_and_regime_caches(self) -> None:
        """The supervisor reads ``model_predict_fn._last_resolved_kelly_pct``
        + ``..._last_resolved_regime_label`` directly. With multi-symbol mode
        those caches live on the per-symbol child predictor. ``__call__``
        must mirror them onto self after each predict so the supervisor's
        getattr resolves to a real value, not the always-None default.
        """
        from predictor import MultiSymbolXGBoostPredictor

        class _StubWithCache:
            def __init__(self, kelly: Optional[float], label: Optional[str]) -> None:
                self._last_resolved_kelly_pct = kelly
                self._last_resolved_regime_label = label

            def __call__(self, symbol, ticker):
                return ("buy", 0.7)

        eth_stub = _StubWithCache(kelly=0.08, label="trend_up")
        btc_stub = _StubWithCache(kelly=None, label=None)
        multi = MultiSymbolXGBoostPredictor(
            model_map={"ETH/USD": eth_stub, "BTC/USD": btc_stub}
        )
        # Pre-call: caches are the safe sentinel.
        self.assertIsNone(multi._last_resolved_kelly_pct)
        self.assertIsNone(multi._last_resolved_regime_label)
        # After predicting ETH, multi mirrors the ETH child's values.
        multi("ETH/USD", None)
        self.assertEqual(multi._last_resolved_kelly_pct, 0.08)
        self.assertEqual(multi._last_resolved_regime_label, "trend_up")
        # After predicting BTC (which has None caches), multi mirrors None.
        # This is the "supervisor reads after the most recent predict"
        # semantics — per-symbol state isolation.
        multi("BTC/USD", None)
        self.assertIsNone(multi._last_resolved_kelly_pct)
        self.assertIsNone(multi._last_resolved_regime_label)

    def test_unknown_symbol_resets_caches(self) -> None:
        """An unknown symbol call returns the neutral result; it should also
        clear the caches so the supervisor doesn't reuse a stale value from
        the previously-called symbol.
        """
        from predictor import MultiSymbolXGBoostPredictor

        class _StubWithCache:
            def __init__(self, kelly: Optional[float]) -> None:
                self._last_resolved_kelly_pct = kelly
                self._last_resolved_regime_label = "chop"

            def __call__(self, symbol, ticker):
                return ("buy", 0.6)

        eth_stub = _StubWithCache(kelly=0.04)
        multi = MultiSymbolXGBoostPredictor(model_map={"ETH/USD": eth_stub})
        # Prime the multi caches with ETH's values.
        multi("ETH/USD", None)
        self.assertEqual(multi._last_resolved_kelly_pct, 0.04)
        # Unknown symbol — must clear, not leave ETH's stale cache visible.
        multi("DOGE/USD", None)
        self.assertIsNone(multi._last_resolved_kelly_pct)
        self.assertIsNone(multi._last_resolved_regime_label)


class PredictorResultDataclassTests(unittest.TestCase):
    """``PredictorResult`` shape + defaults (Lane B / A1 gap closure)."""

    def test_defaults_have_no_rich_fields(self) -> None:
        from predictor import PredictorResult

        r = PredictorResult(side="buy", confidence=0.5)
        self.assertEqual(r.side, "buy")
        self.assertAlmostEqual(r.confidence, 0.5)
        self.assertIsNone(r.feature_buffer)
        self.assertIsNone(r.model_probs)
        self.assertEqual(r.extras, {})

    def test_can_be_constructed_with_rich_fields(self) -> None:
        from predictor import PredictorResult

        r = PredictorResult(
            side="sell",
            confidence=0.71,
            feature_buffer={"return_5": 0.001, "rv_60": 0.002},
            model_probs={"long": 0.29, "short": 0.71},
        )
        self.assertEqual(r.feature_buffer["return_5"], 0.001)
        self.assertAlmostEqual(r.model_probs["short"], 0.71)


class XGBoostPredictFullTests(unittest.TestCase):
    """``XGBoostPredictor.predict_full`` returns the rich result shape."""

    def _make_synth_dataset(self, n: int = 500) -> "pd.DataFrame":
        import pandas as pd

        rng = np.random.default_rng(seed=11)
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

    def test_predict_full_returns_predictor_result_with_rich_fields(self) -> None:
        from predictor import PredictorResult, XGBoostPredictor

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
            result = predictor.predict_full("ETH/USD", fake_ticker)

        self.assertIsInstance(result, PredictorResult)
        self.assertIn(result.side, ("buy", "sell"))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        # Rich fields populated for a real inference (not warmup).
        self.assertIsNotNone(result.feature_buffer)
        self.assertIsNotNone(result.model_probs)
        # feature_buffer carries every meta.feature_cols entry.
        for col in predictor.feature_cols:
            self.assertIn(col, result.feature_buffer)
        # model_probs has both long + short keys (binary head).
        self.assertIn("long", result.model_probs)
        self.assertIn("short", result.model_probs)
        # Probs sum to ~1 (within float slop).
        self.assertAlmostEqual(
            result.model_probs["long"] + result.model_probs["short"],
            1.0,
            places=4,
        )

    def test_predict_full_warmup_returns_neutral_with_no_rich_fields(self) -> None:
        from predictor import XGBoostPredictor

        candles = _synthetic_candles(n=50)  # below 240-bar warmup
        ex = _FakeExchange(candles)
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model_xgb"
            self._train_tiny_model(model_dir)
            predictor = XGBoostPredictor(
                model_dir=str(model_dir), exchange=ex, thr_long=0.5
            )
            fake_ticker = mock.MagicMock(symbol="ETH/USD")
            result = predictor.predict_full("ETH/USD", fake_ticker)

        # Same neutral semantics as ``__call__``: ("buy", 0.5) with no rich
        # data so A1's checks know to skip rather than analyse stale data.
        self.assertEqual((result.side, result.confidence), ("buy", 0.5))
        self.assertIsNone(result.feature_buffer)
        self.assertIsNone(result.model_probs)

    def test_call_still_returns_two_tuple_unchanged(self) -> None:
        # Critical backward-compat assertion: ``__call__`` must keep the
        # legacy 2-tuple return so the supervisor + 700+ tests that
        # destructure ``side, conf = predictor(...)`` are unchanged by
        # the predict_full extension.
        from predictor import XGBoostPredictor

        candles = _synthetic_candles(n=400)
        ex = _FakeExchange(candles)
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model_xgb"
            self._train_tiny_model(model_dir)
            predictor = XGBoostPredictor(
                model_dir=str(model_dir), exchange=ex, thr_long=0.5
            )
            fake_ticker = mock.MagicMock(symbol="ETH/USD")
            out = predictor("ETH/USD", fake_ticker)

        # Tuple length 2, not a PredictorResult.
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        side, conf = out  # destructure works -- this is the contract
        self.assertIn(side, ("buy", "sell"))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_model_meta_accessor_returns_loaded_meta(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model_xgb"
            self._train_tiny_model(model_dir)
            predictor = XGBoostPredictor(
                model_dir=str(model_dir), exchange=None, thr_long=0.5
            )
            meta = predictor.model_meta

        self.assertIsInstance(meta, dict)
        # Keys we actually persist today -- ensures A1 can read them.
        self.assertIn("feature_cols", meta)
        self.assertIn("optimal_threshold", meta)
        # Mutating the returned dict must NOT mutate the predictor's
        # internal meta (cheap defence against accidental shared state).
        meta["feature_cols"] = ["bogus"]
        self.assertNotEqual(predictor.meta["feature_cols"], ["bogus"])


class MultiSymbolPredictFullTests(unittest.TestCase):
    """Routing of ``predict_full`` + ``model_meta_for`` in the multi-symbol predictor."""

    def test_routes_predict_full_to_per_symbol_predictor(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor, PredictorResult

        class _RichStub:
            def __init__(self, label: str) -> None:
                self.label = label
                self.full_calls: List[str] = []
                self.feature_cols: List[str] = ["f1"]
                self.model = object()
                self.thr_long = 0.5

            def __call__(self, symbol, ticker):
                # 2-tuple legacy surface (unused in this test).
                return ("buy", 0.5)

            def predict_full(self, symbol, ticker):
                self.full_calls.append(symbol)
                return PredictorResult(
                    side="buy",
                    confidence=0.71 if self.label == "eth" else 0.33,
                    feature_buffer={"f1": 1.0},
                    model_probs={"long": 0.71, "short": 0.29},
                )

        eth = _RichStub("eth")
        btc = _RichStub("btc")
        multi = MultiSymbolXGBoostPredictor(
            model_map={"ETH/USD": eth, "BTC/USD": btc}
        )
        result = multi.predict_full("ETH/USD", None)
        self.assertIsInstance(result, PredictorResult)
        self.assertAlmostEqual(result.confidence, 0.71)
        self.assertEqual(result.feature_buffer, {"f1": 1.0})
        self.assertEqual(eth.full_calls, ["ETH/USD"])
        self.assertEqual(btc.full_calls, [])

    def test_predict_full_falls_back_for_legacy_stub(self) -> None:
        # An older stub that only implements ``__call__`` (no
        # ``predict_full``) should still work through the multi-symbol
        # routing -- we project the 2-tuple back to PredictorResult.
        from predictor import MultiSymbolXGBoostPredictor, PredictorResult

        class _LegacyStub:
            def __init__(self) -> None:
                self.feature_cols: List[str] = ["f1"]
                self.model = object()
                self.thr_long = 0.5

            def __call__(self, symbol, ticker):
                return ("sell", 0.62)

        multi = MultiSymbolXGBoostPredictor(model_map={"ETH/USD": _LegacyStub()})
        result = multi.predict_full("ETH/USD", None)
        self.assertIsInstance(result, PredictorResult)
        self.assertEqual(result.side, "sell")
        self.assertAlmostEqual(result.confidence, 0.62)
        self.assertIsNone(result.feature_buffer)
        self.assertIsNone(result.model_probs)

    def test_predict_full_unknown_symbol_returns_neutral_rich_result(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor, PredictorResult

        class _Stub:
            feature_cols = ["f1"]
            model = object()
            thr_long = 0.5

            def __call__(self, symbol, ticker):
                return ("buy", 0.5)

        multi = MultiSymbolXGBoostPredictor(model_map={"ETH/USD": _Stub()})
        result = multi.predict_full("DOGE/USD", None)
        self.assertIsInstance(result, PredictorResult)
        self.assertEqual((result.side, result.confidence), ("buy", 0.5))
        self.assertIsNone(result.feature_buffer)
        self.assertIsNone(result.model_probs)

    def test_model_meta_for_returns_per_symbol_meta(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor

        class _Stub:
            feature_cols = ["f1"]
            model = object()
            thr_long = 0.5
            model_meta = {"feature_cols": ["f1"], "optimal_threshold": 0.55}

            def __call__(self, symbol, ticker):
                return ("buy", 0.5)

        multi = MultiSymbolXGBoostPredictor(model_map={"ETH/USD": _Stub()})
        meta = multi.model_meta_for("ETH/USD")
        self.assertEqual(meta["feature_cols"], ["f1"])
        self.assertAlmostEqual(meta["optimal_threshold"], 0.55)
        # Unknown symbol -> empty dict (not None) so callers can ``meta.get(...)``.
        self.assertEqual(multi.model_meta_for("DOGE/USD"), {})


class XGBoostBackwardCompatibilityTests(unittest.TestCase):
    """Hard guarantees that the rich-result extension didn't regress legacy paths."""

    def test_call_signature_is_two_tuple_with_invalid_proba(self) -> None:
        # The NaN/inf guard inside ``_predict_with_features`` must still
        # route to the neutral 2-tuple via ``__call__`` (existing test
        # ``test_xgb_invalid_proba_routes_to_neutral`` covers __call__;
        # add a parallel one for predict_full so the rich path matches).
        from predictor import PredictorResult, XGBoostPredictor

        # Synthesise a minimal predictor with a stubbed model that returns
        # NaN proba so we exercise the guard without doing a full train.
        p = XGBoostPredictor.__new__(XGBoostPredictor)
        p.model_dir = "/tmp/stub"
        p.feature_cols = ["return_5", "rv_60", "tod_sin"]
        p.thr_long = 0.5
        p._buffers = {}
        p._last_seeded_minute = {}
        p._lock = __import__("threading").Lock()
        # __init__ skipped via __new__; explicitly null the regime fields so
        # _resolve_threshold's `lookup is None` short-circuit is exercised.
        p.regime_lookup = None
        p._last_resolved_kelly_pct = None

        class _BadModel:
            def predict_proba(self, X):
                return np.array([[0.0, float("nan")]])

        p.model = _BadModel()
        p.scaler = None

        # Seed a 300-bar buffer so the warmup gate passes. Also pin the
        # last_seeded_minute so _refresh_buffer is a no-op (the stub
        # predictor has no real exchange).
        import pandas as pd

        rows = _synthetic_candles(n=300)
        p._buffers["ETH/USD"] = pd.DataFrame(rows)
        p._last_seeded_minute["ETH/USD"] = int(
            datetime.now(timezone.utc).timestamp() // 60
        )

        # __call__ must still return a 2-tuple equal to neutral.
        side, conf = p("ETH/USD", mock.MagicMock())
        self.assertEqual((side, conf), ("buy", 0.5))

        # predict_full must still return a PredictorResult with the same
        # neutral semantics + no rich fields (consistent with warmup).
        result = p.predict_full("ETH/USD", mock.MagicMock())
        self.assertIsInstance(result, PredictorResult)
        self.assertEqual((result.side, result.confidence), ("buy", 0.5))
        self.assertIsNone(result.feature_buffer)
        self.assertIsNone(result.model_probs)


class _StubRegimeLookup:
    """Minimal stub matching :class:`regime_memory.lookup.RegimeLookup`.

    Tests inject this in place of the real RegimeLookup so we never have
    to encode features or load a FAISS store. ``resolved_payload`` is the
    dict returned verbatim from :meth:`resolve_params`; ``raise_on_resolve``
    raises a synthetic exception to exercise the graceful-degradation
    branch in :meth:`XGBoostPredictor._resolve_threshold`.
    """

    def __init__(
        self,
        *,
        resolved_payload: Dict[str, float] | None = None,
        raise_on_resolve: bool = False,
    ) -> None:
        self.resolved_payload = resolved_payload or {}
        self.raise_on_resolve = raise_on_resolve
        self.calls: List[Dict[str, Any]] = []

    def resolve_params(
        self, features: Any, *, k: int = 10, window_size: int = 60
    ) -> Dict[str, float]:
        self.calls.append({"k": k, "window_size": window_size})
        if self.raise_on_resolve:
            raise RuntimeError("simulated regime store failure")
        return dict(self.resolved_payload)


class XGBoostRegimeLookupResolveTests(unittest.TestCase):
    """``XGBoostPredictor._resolve_threshold`` integration with RegimeLookup.

    These tests bypass __init__ via __new__ so we don't have to load a
    real model bundle. ``regime_lookup`` is injected directly so we can
    verify the confidence gate (>= 0.5), the static fallback, and the
    graceful-degradation branch in isolation.
    """

    def _bare_predictor(
        self,
        *,
        thr_long: float = 0.55,
        regime_lookup: Any = None,
    ):
        from predictor import XGBoostPredictor

        p = XGBoostPredictor.__new__(XGBoostPredictor)
        p.thr_long = thr_long
        p.regime_lookup = regime_lookup
        p._last_resolved_kelly_pct = None
        return p

    def test_no_lookup_returns_static_threshold(self) -> None:
        """Lookup is None → static thr_long is used (regression baseline)."""
        p = self._bare_predictor(thr_long=0.55, regime_lookup=None)
        thr = p._resolve_threshold(feature_window=object())
        self.assertAlmostEqual(thr, 0.55)
        self.assertIsNone(p._last_resolved_kelly_pct)

    def test_high_confidence_overrides_threshold(self) -> None:
        """confidence=0.7 → resolved optimal_threshold replaces thr_long."""
        lookup = _StubRegimeLookup(
            resolved_payload={
                "_regime_confidence": 0.7,
                "optimal_threshold": 0.62,
                "kelly_size_pct": 0.18,
            }
        )
        p = self._bare_predictor(thr_long=0.55, regime_lookup=lookup)
        thr = p._resolve_threshold(feature_window=object())
        self.assertAlmostEqual(thr, 0.62)
        self.assertAlmostEqual(p._last_resolved_kelly_pct, 0.18)
        # Lookup was consulted exactly once with the documented k.
        self.assertEqual(len(lookup.calls), 1)
        self.assertEqual(lookup.calls[0]["k"], 10)

    def test_low_confidence_falls_back_to_static(self) -> None:
        """confidence=0.3 → static thr_long; cached kelly is cleared."""
        lookup = _StubRegimeLookup(
            resolved_payload={
                "_regime_confidence": 0.3,
                "optimal_threshold": 0.62,
                "kelly_size_pct": 0.18,
            }
        )
        p = self._bare_predictor(thr_long=0.55, regime_lookup=lookup)
        # Pre-seed a stale kelly to verify we clear it when confidence
        # falls below the gate.
        p._last_resolved_kelly_pct = 0.99
        thr = p._resolve_threshold(feature_window=object())
        self.assertAlmostEqual(thr, 0.55)
        self.assertIsNone(p._last_resolved_kelly_pct)

    def test_lookup_raise_falls_back_to_static(self) -> None:
        """Mid-prediction RegimeLookup exception → static threshold used."""
        lookup = _StubRegimeLookup(raise_on_resolve=True)
        p = self._bare_predictor(thr_long=0.55, regime_lookup=lookup)
        thr = p._resolve_threshold(feature_window=object())
        self.assertAlmostEqual(thr, 0.55)
        self.assertIsNone(p._last_resolved_kelly_pct)

    def test_confidence_exactly_at_threshold_uses_resolved(self) -> None:
        """confidence=0.5 (boundary) → still uses resolved threshold."""
        lookup = _StubRegimeLookup(
            resolved_payload={
                "_regime_confidence": 0.5,
                "optimal_threshold": 0.61,
                "kelly_size_pct": 0.05,
            }
        )
        p = self._bare_predictor(thr_long=0.55, regime_lookup=lookup)
        thr = p._resolve_threshold(feature_window=object())
        self.assertAlmostEqual(thr, 0.61)


class XGBoostRegimeLookupInitTests(unittest.TestCase):
    """``_maybe_init_regime_lookup`` env-var gating + load-failure tolerance."""

    def test_unset_env_var_leaves_lookup_none(self) -> None:
        """REGIME_STORE_PATH unset → no lookup is constructed."""
        from predictor import XGBoostPredictor

        p = XGBoostPredictor.__new__(XGBoostPredictor)
        p.feature_cols = ["return_5", "rv_60"]
        p.thr_long = 0.55
        p.regime_lookup = None
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REGIME_STORE_PATH", None)
            p._maybe_init_regime_lookup()
        self.assertIsNone(p.regime_lookup)

    def test_missing_path_logs_and_skips(self) -> None:
        """Env points at a non-existent file → lookup stays None, no crash."""
        from predictor import XGBoostPredictor

        p = XGBoostPredictor.__new__(XGBoostPredictor)
        p.feature_cols = ["return_5", "rv_60"]
        p.thr_long = 0.55
        p.regime_lookup = None
        with mock.patch.dict(
            os.environ,
            {"REGIME_STORE_PATH": "/nonexistent/path/to/store.npz"},
        ):
            p._maybe_init_regime_lookup()
        self.assertIsNone(p.regime_lookup)

    def test_load_failure_falls_back_to_none(self) -> None:
        """An import or load exception is swallowed; lookup stays None."""
        from predictor import XGBoostPredictor

        p = XGBoostPredictor.__new__(XGBoostPredictor)
        p.feature_cols = ["return_5", "rv_60"]
        p.thr_long = 0.55
        p.regime_lookup = None
        # Create a path-exists-but-malformed file so the loader raises.
        with tempfile.TemporaryDirectory() as td:
            stub = Path(td) / "broken.npz"
            stub.write_bytes(b"not a real npz")
            with mock.patch.dict(
                os.environ,
                {"REGIME_STORE_PATH": str(stub)},
            ):
                p._maybe_init_regime_lookup()
        self.assertIsNone(p.regime_lookup)


class MultiSymbolPerSymbolRegimePathTests(unittest.TestCase):
    """``MultiSymbolXGBoostPredictor`` honours per-symbol REGIME_STORE_PATH_*."""

    def test_per_symbol_env_var_triggers_lookup_init(self) -> None:
        """REGIME_STORE_PATH_ETH_USD set → matching predictor's helper runs."""
        from predictor import MultiSymbolXGBoostPredictor, XGBoostPredictor

        # Build two bare predictors so we can verify the helper is called
        # only on the symbol that has a per-symbol env var.
        eth = XGBoostPredictor.__new__(XGBoostPredictor)
        eth.feature_cols = ["f1"]
        eth.thr_long = 0.5
        eth.model = object()
        eth.regime_lookup = None
        eth._init_calls = 0  # type: ignore[attr-defined]

        def _eth_helper() -> None:
            eth._init_calls += 1  # type: ignore[attr-defined]

        eth._maybe_init_regime_lookup = _eth_helper  # type: ignore[method-assign]

        btc = XGBoostPredictor.__new__(XGBoostPredictor)
        btc.feature_cols = ["f1"]
        btc.thr_long = 0.5
        btc.model = object()
        btc.regime_lookup = None
        btc._init_calls = 0  # type: ignore[attr-defined]

        def _btc_helper() -> None:
            btc._init_calls += 1  # type: ignore[attr-defined]

        btc._maybe_init_regime_lookup = _btc_helper  # type: ignore[method-assign]

        with mock.patch.dict(
            os.environ,
            {"REGIME_STORE_PATH_ETH_USD": "/tmp/eth.npz"},
            clear=False,
        ):
            os.environ.pop("REGIME_STORE_PATH_BTC_USD", None)
            MultiSymbolXGBoostPredictor(
                model_map={"ETH/USD": eth, "BTC/USD": btc}
            )
        # ETH had a per-symbol var → helper was re-invoked. BTC had none →
        # helper was NOT invoked again (its global-init from __init__ stays).
        self.assertEqual(eth._init_calls, 1)
        self.assertEqual(btc._init_calls, 0)

    def test_symbol_env_token_normalises_separators(self) -> None:
        """``ETH/USD`` and ``ETH-USD`` normalise to ``ETH_USD`` consistently."""
        from predictor import MultiSymbolXGBoostPredictor

        self.assertEqual(
            MultiSymbolXGBoostPredictor._symbol_env_token("ETH/USD"), "ETH_USD"
        )
        self.assertEqual(
            MultiSymbolXGBoostPredictor._symbol_env_token("eth-usd"), "ETH_USD"
        )
        self.assertEqual(
            MultiSymbolXGBoostPredictor._symbol_env_token("BTC.USDT"), "BTC_USDT"
        )


if __name__ == "__main__":
    unittest.main()
