"""Per-symbol model isolation + scaler order assertion (Lane B P0 #5).

These tests cover the fail-loud guarantees:

* a scaler whose ``feature_names_in_`` doesn't match ``meta.feature_cols``
  raises at predictor construction (silently scaling features in the wrong
  order produces probabilities that look fine but are wrong),
* a missing model directory raises ``FileNotFoundError`` rather than
  passing through a partially-initialised object,
* a meta.json that's missing key fields raises rather than silently
  loading a half-broken bundle,
* ``MultiSymbolXGBoostPredictor`` validates each per-symbol predictor at
  construction so a bad bundle fails fast at startup, not on first call.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, List
from unittest import mock

# libomp dance must be set BEFORE numpy/sklearn/xgboost get imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np


def _train_and_persist_tiny_bundle(out_dir: Path) -> List[str]:
    """Train a tiny XGBoost model + write a model.joblib + meta.json.

    Returns the feature column ordering used. Tests can then drop a
    deliberately-mis-ordered scaler.joblib next to it and assert the
    predictor refuses to load.
    """
    import pandas as pd

    from crypto_training.train_xgboost import train

    rng = np.random.default_rng(42)
    n = 600
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    f3 = rng.normal(0, 1, size=n)
    score = 0.6 * f1 + 0.3 * f2
    label = (score > np.median(score)).astype(int)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=n, freq="1min"
            ).astype(str),
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "label": label,
        }
    )
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
    # Mirror what train() actually wrote -- read the meta back.
    meta = json.loads((out_dir / "meta.json").read_text())
    return list(meta["feature_cols"])


class ScalerOrderAssertionTests(unittest.TestCase):
    """Predictor must refuse a scaler whose column order is wrong."""

    def test_scaler_with_mismatched_column_order_raises(self) -> None:
        from predictor import XGBoostPredictor
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            cols = _train_and_persist_tiny_bundle(model_dir)
            # Fit a scaler on the wrong column order.
            wrong_order = list(reversed(cols))
            X = pd.DataFrame(
                np.random.default_rng(0).normal(0, 1, size=(50, len(cols))),
                columns=wrong_order,
            )
            sc = StandardScaler().fit(X)
            joblib.dump(sc, model_dir / "scaler.joblib")
            with self.assertRaises(ValueError) as ctx:
                XGBoostPredictor(
                    model_dir=str(model_dir), exchange=None, thr_long=0.5
                )
            self.assertIn("scaler", str(ctx.exception).lower())

    def test_scaler_without_feature_names_in_raises(self) -> None:
        from predictor import XGBoostPredictor
        from sklearn.preprocessing import MinMaxScaler

        # Use a real sklearn transformer but fit it on a numpy array (not a
        # DataFrame) -- that path doesn't populate ``feature_names_in_`` so
        # the predictor can't verify column order. Must reject it.
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            _train_and_persist_tiny_bundle(model_dir)
            sc = MinMaxScaler().fit(np.random.rand(50, 3))
            self.assertFalse(hasattr(sc, "feature_names_in_"))
            joblib.dump(sc, model_dir / "scaler.joblib")
            with self.assertRaises(ValueError) as ctx:
                XGBoostPredictor(
                    model_dir=str(model_dir), exchange=None, thr_long=0.5
                )
            self.assertIn("feature_names_in_", str(ctx.exception))

    def test_scaler_with_correct_column_order_loads_cleanly(self) -> None:
        # Sanity: the validation isn't over-eager -- a correctly-fit scaler
        # passes through.
        from predictor import XGBoostPredictor
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            cols = _train_and_persist_tiny_bundle(model_dir)
            X = pd.DataFrame(
                np.random.default_rng(0).normal(0, 1, size=(50, len(cols))),
                columns=cols,
            )
            sc = StandardScaler().fit(X)
            joblib.dump(sc, model_dir / "scaler.joblib")
            p = XGBoostPredictor(
                model_dir=str(model_dir), exchange=None, thr_long=0.5
            )
            self.assertIsNotNone(p.scaler)
            self.assertEqual(list(p.scaler.feature_names_in_), cols)


class MissingModelDirTests(unittest.TestCase):
    """Constructor + ``build_default_predict_fn`` must raise on missing dirs."""

    def test_xgb_predictor_raises_when_dir_missing(self) -> None:
        from predictor import XGBoostPredictor

        with self.assertRaises(FileNotFoundError):
            XGBoostPredictor(
                model_dir="/no/such/xgb/bundle/dir/abc",
                exchange=None,
                thr_long=0.5,
            )

    def test_xgb_predictor_raises_when_meta_missing(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "empty"
            model_dir.mkdir()
            # Dir exists, but no meta.json + model.joblib inside.
            with self.assertRaises(FileNotFoundError):
                XGBoostPredictor(
                    model_dir=str(model_dir), exchange=None, thr_long=0.5
                )

    def test_xgb_predictor_raises_when_meta_has_no_feature_cols(self) -> None:
        from predictor import XGBoostPredictor

        # A real bundle minus feature_cols should fail loud.
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            _train_and_persist_tiny_bundle(model_dir)
            # Strip feature_cols from meta on disk.
            meta_path = model_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["feature_cols"] = []
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                XGBoostPredictor(
                    model_dir=str(model_dir), exchange=None, thr_long=0.5
                )

    def test_build_default_raises_when_crypto_dir_missing(self) -> None:
        from predictor import build_default_predict_fn

        with mock.patch.dict(
            os.environ,
            {
                "CRYPTO_MODEL_DIR": "/no/such/path/xyz/abc",
                "CRYPTO_MODEL_MAP": "",
                "LEGACY_MODEL_DIR": "",
            },
            clear=False,
        ):
            with self.assertRaises(FileNotFoundError):
                build_default_predict_fn(exchange=None)

    def test_build_default_raises_when_map_entry_missing(self) -> None:
        from predictor import build_default_predict_fn

        with mock.patch.dict(
            os.environ,
            {
                "CRYPTO_MODEL_MAP": "ETH/USD=/no/such/eth/dir",
                "CRYPTO_MODEL_DIR": "",
                "LEGACY_MODEL_DIR": "",
            },
            clear=False,
        ):
            with self.assertRaises(FileNotFoundError):
                build_default_predict_fn(exchange=None)


class MultiSymbolValidationTests(unittest.TestCase):
    """``MultiSymbolXGBoostPredictor`` validates every entry at construction."""

    def test_none_value_raises(self) -> None:
        from predictor import MultiSymbolXGBoostPredictor

        with self.assertRaises(ValueError) as ctx:
            MultiSymbolXGBoostPredictor(
                model_map={"ETH/USD": None}  # type: ignore[dict-item]
            )
        self.assertIn("ETH/USD", str(ctx.exception))


class ThresholdPrecedenceTests(unittest.TestCase):
    """Threshold precedence: explicit > meta.optimal_threshold > 0.5 default."""

    def test_default_threshold_used_when_no_meta_no_explicit(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            _train_and_persist_tiny_bundle(model_dir)
            # train() may have populated optimal_threshold via the
            # Sharpe sweep (Lane B P1 #12). Strip it to test the pure
            # "no meta override" fallback path.
            meta_path = model_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["optimal_threshold"] = None
            meta_path.write_text(json.dumps(meta))
            p = XGBoostPredictor(model_dir=str(model_dir), exchange=None)
            self.assertAlmostEqual(p.thr_long, 0.5, places=6)
            self.assertEqual(p._thr_source, "default")

    def test_meta_optimal_threshold_used_when_no_explicit(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            _train_and_persist_tiny_bundle(model_dir)
            # Inject optimal_threshold into meta.json.
            meta_path = model_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["optimal_threshold"] = 0.42
            meta_path.write_text(json.dumps(meta))
            p = XGBoostPredictor(model_dir=str(model_dir), exchange=None)
            self.assertAlmostEqual(p.thr_long, 0.42, places=6)
            self.assertEqual(p._thr_source, "meta.optimal_threshold")

    def test_explicit_threshold_wins_over_meta(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            _train_and_persist_tiny_bundle(model_dir)
            meta_path = model_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["optimal_threshold"] = 0.42
            meta_path.write_text(json.dumps(meta))
            p = XGBoostPredictor(
                model_dir=str(model_dir), exchange=None, thr_long=0.77
            )
            self.assertAlmostEqual(p.thr_long, 0.77, places=6)
            self.assertEqual(p._thr_source, "explicit")


class ModelMetaAccessorTests(unittest.TestCase):
    """``XGBoostPredictor.model_meta`` exposes the meta.json dict for A1."""

    def test_meta_accessor_returns_a_dict_copy(self) -> None:
        from predictor import XGBoostPredictor

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "bundle"
            cols = _train_and_persist_tiny_bundle(model_dir)
            p = XGBoostPredictor(
                model_dir=str(model_dir), exchange=None, thr_long=0.5
            )
            meta = p.model_meta

        self.assertIsInstance(meta, dict)
        # Must include the keys A1 actually reads -- the consumer surface
        # is not allowed to silently drop these.
        self.assertEqual(meta["feature_cols"], cols)
        self.assertIn("optimal_threshold", meta)
        # The accessor returns a copy: mutating it does NOT corrupt the
        # predictor's loaded meta.
        meta["feature_cols"] = ["bogus"]
        self.assertNotEqual(p.meta["feature_cols"], ["bogus"])


class PredictFullBackwardCompatTests(unittest.TestCase):
    """Top-level guarantee: ``__call__`` 2-tuple unchanged after Lane B extension.

    A1 + future snapshot writers can adopt ``predict_full``; existing 2-tuple
    callers must keep working. We assert this here by stubbing the model so
    the test runs without the heavy training fixture cost.
    """

    def test_call_returns_two_tuple_for_xgb_predictor(self) -> None:
        from predictor import XGBoostPredictor

        # Stub a predictor with a fake model that always returns 0.7
        # P(long) so __call__ should fire ("buy", 0.7) at thr=0.5.
        p = XGBoostPredictor.__new__(XGBoostPredictor)
        p.model_dir = "/tmp/stub_compat"
        p.feature_cols = ["f1", "f2", "f3"]
        p.thr_long = 0.5
        p._buffers = {}
        p._last_seeded_minute = {}
        p._lock = __import__("threading").Lock()
        p.scaler = None

        class _StubModel:
            def predict_proba(self, X):
                return np.array([[0.3, 0.7]])

        p.model = _StubModel()

        # Pin buffer + minute so _refresh_buffer is a no-op without an
        # exchange. Buffer must hold valid 1m OHLCV columns so
        # compute_features doesn't crash; we use a synthetic random walk.
        import pandas as pd

        from datetime import datetime, timedelta, timezone as tz

        rng = np.random.default_rng(0)
        n = 260
        rows: List[Any] = []
        end = datetime.now(tz.utc).replace(second=0, microsecond=0)
        price = 1000.0
        for i in range(n):
            ts = end - timedelta(minutes=(n - 1 - i))
            ret = float(rng.normal(0, 0.0008))
            new_price = price * (1.0 + ret)
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": price,
                    "high": max(price, new_price) * 1.0001,
                    "low": min(price, new_price) * 0.9999,
                    "close": new_price,
                    "volume": 50.0,
                }
            )
            price = new_price
        p._buffers["ETH/USD"] = pd.DataFrame(rows)
        p._last_seeded_minute["ETH/USD"] = int(
            datetime.now(tz.utc).timestamp() // 60
        )

        # The compute_features path expects feature columns we don't have
        # in our stub feature_cols list, so the predictor will skip with
        # "missing feature cols" and route to neutral. Assert that path
        # also returns a clean 2-tuple.
        out = p("ETH/USD", None)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        side, conf = out
        self.assertIn(side, ("buy", "sell"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
