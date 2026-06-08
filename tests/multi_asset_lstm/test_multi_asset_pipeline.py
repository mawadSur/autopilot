"""Tests for the pooled multi-asset LSTM pipeline.

Run with::

    PYTHONPATH=src ./.venv/bin/python -m pytest tests/multi_asset/ -q

Focus is on the correctness-critical invariants:
  * source normalization -> canonical schema (sorted, de-duped, UTC ISO ts)
  * scale-invariant feature selection (no raw price/size leaks into the pool)
  * sequence builder leakage guards (no cross-asset windows; data holes split;
    weekend gaps in daily series do NOT shatter windows)
  * pooled model forward shape + asset embedding
  * full train -> save -> load -> predict round-trip on synthetic data
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from multi_asset import sources
from multi_asset.build_pooled_dataset import (
    SCALE_INVARIANT_FEATURES,
    assemble_pooled,
    build_for_instrument,
)
from multi_asset.model import PooledLSTMClassifier
from multi_asset.sequences import build_asset_vocab, build_sequences
from multi_asset.universe import DEFAULT_UNIVERSE, Instrument, Universe, safe_symbol


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #
def synth_ohlcv(n: int, *, start: str = "2022-01-01", freq: str = "1D", seed: int = 0) -> pd.DataFrame:
    """Random-walk OHLCV in the canonical schema."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rets = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1e3, 1e4, n)
    return pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(
                r"(\d{2})(\d{2})$", r"\1:\2", regex=True
            ),
            "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        }
    )


def synth_pooled_for_sequences(asset_ids, n_per_asset=50, *, freq="1D", feature_cols=("f0", "f1")):
    """A minimal pooled-shaped frame for sequence tests (bypasses compute_features)."""
    rng = np.random.default_rng(7)
    rows = []
    ts = pd.date_range("2022-01-01", periods=n_per_asset, freq=freq, tz="UTC")
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    for aid in asset_ids:
        for i in range(n_per_asset):
            row = {"timestamp": ts_str[i], "asset_id": aid, "asset_class": aid.split(":")[0],
                   "label": int(rng.integers(0, 2))}
            for c in feature_cols:
                row[c] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class SourceNormalizeTests(unittest.TestCase):
    def test_ccxt_rows_to_schema_sorted_deduped(self):
        # ms timestamps out of order, with a duplicate.
        rows = [
            [1_700_000_120_000, 2, 3, 1, 2.5, 10],
            [1_700_000_000_000, 1, 2, 0.5, 1.5, 5],
            [1_700_000_060_000, 1.5, 2.5, 1, 2, 7],
            [1_700_000_060_000, 9, 9, 9, 9, 99],  # dup ts -> last wins
        ]
        df = sources.normalize_ccxt_ohlcv(rows)
        self.assertEqual(list(df.columns), sources.OHLCV_COLUMNS)
        self.assertEqual(len(df), 3)  # dup collapsed
        # ascending by time
        ks = pd.to_datetime(df["timestamp"], utc=True)
        self.assertTrue(ks.is_monotonic_increasing)
        # last-wins on the duplicate minute
        dup_close = df.loc[ks == ks.iloc[1], "close"].iloc[0]
        self.assertEqual(dup_close, 9.0)

    def test_yfinance_frame_to_schema(self):
        idx = pd.date_range("2022-01-03", periods=3, freq="1D", tz="UTC")
        raw = pd.DataFrame(
            {"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1, 2],
             "Close": [1.5, 2.5, 3.5], "Volume": [10, 20, 30]}, index=idx,
        )
        raw.index.name = "Date"
        df = sources.normalize_yfinance(raw)
        self.assertEqual(list(df.columns), sources.OHLCV_COLUMNS)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["close"].tolist(), [1.5, 2.5, 3.5])

    def test_empty_inputs_return_empty_schema(self):
        self.assertEqual(len(sources.normalize_ccxt_ohlcv([])), 0)
        self.assertEqual(list(sources.normalize_ccxt_ohlcv([]).columns), sources.OHLCV_COLUMNS)

    def test_get_source_factory(self):
        self.assertEqual(sources.get_source("crypto").asset_class, "crypto")
        self.assertEqual(sources.get_source("stock").asset_class, "stock")
        with self.assertRaises(ValueError):
            sources.get_source("forex")


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
class UniverseTests(unittest.TestCase):
    def test_asset_id_and_safe_symbol(self):
        self.assertEqual(safe_symbol("BTC/USD"), "BTC-USD")
        self.assertEqual(Instrument("BTC/USD", "crypto").asset_id, "crypto:BTC-USD")
        self.assertEqual(Instrument("AAPL", "stock").asset_id, "stock:AAPL")

    def test_data_dir_tree(self):
        c = Instrument("ETH/USD", "crypto").data_dir(Path("data"), "1d")
        s = Instrument("SPY", "stock").data_dir(Path("data"), "1d")
        self.assertEqual(c, Path("data/crypto/ETH-USD/1d"))
        self.assertEqual(s, Path("data/stocks/SPY/1d"))

    def test_json_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.json"
            DEFAULT_UNIVERSE.to_json(p)
            back = Universe.from_json(p)
            self.assertEqual(back.granularity, DEFAULT_UNIVERSE.granularity)
            self.assertEqual(back.asset_ids(), DEFAULT_UNIVERSE.asset_ids())


# --------------------------------------------------------------------------- #
# Scale-invariant feature contract
# --------------------------------------------------------------------------- #
class ScaleInvariantFeatureTests(unittest.TestCase):
    def test_no_raw_scale_features_leak_in(self):
        banned = {"open", "high", "low", "close", "volume", "vwap_roll_50",
                  "ema_9", "ema_21", "ema_50", "atr_14", "bb_mid_20",
                  "bb_upper_20", "bb_lower_20", "vol_ma_20", "vol_log",
                  "macd", "macd_signal", "macd_hist", "best_bid", "best_ask", "mid"}
        leaked = banned.intersection(SCALE_INVARIANT_FEATURES)
        self.assertEqual(leaked, set(), f"raw-scale features leaked into the pool: {leaked}")

    def test_features_exist_in_canonical_list(self):
        from utils import FEATURE_COLUMNS_PROFITABLE
        for c in SCALE_INVARIANT_FEATURES:
            self.assertIn(c, FEATURE_COLUMNS_PROFITABLE)


# --------------------------------------------------------------------------- #
# Pooled dataset builder
# --------------------------------------------------------------------------- #
class BuildPooledDatasetTests(unittest.TestCase):
    def test_build_for_instrument_shapes_and_label(self):
        df = synth_ohlcv(400, seed=1)
        out = build_for_instrument(
            df, asset_id="crypto:BTC-USD", asset_class="crypto",
            horizon_bars=1, vol_normalize_k=0.5, warmup_bars=60,
            feature_cols=SCALE_INVARIANT_FEATURES,
        )
        self.assertIn("label", out.columns)
        self.assertIn("asset_id", out.columns)
        self.assertTrue(set(out["label"].unique()).issubset({0, 1}))
        # warmup + trailing horizon dropped -> fewer than input rows
        self.assertLess(len(out), 400)
        self.assertGreater(len(out), 0)
        # only the requested features (plus the 4 meta cols) survive
        non_meta = [c for c in out.columns if c not in ("timestamp", "asset_id", "asset_class", "label")]
        self.assertTrue(set(non_meta).issubset(set(SCALE_INVARIANT_FEATURES)))

    def test_assemble_pooled_global_time_sort_and_asset_ids(self):
        insts = {
            "crypto:BTC-USD": Instrument("BTC/USD", "crypto"),
            "stock:AAPL": Instrument("AAPL", "stock"),
        }
        frames = {
            "crypto:BTC-USD": synth_ohlcv(300, seed=2),
            "stock:AAPL": synth_ohlcv(300, seed=3),
        }
        pooled, summary = assemble_pooled(frames, instruments=insts, horizon_bars=1, warmup_bars=60)
        self.assertEqual(summary.n_assets, 2)
        self.assertEqual(set(pooled["asset_id"].unique()), {"crypto:BTC-USD", "stock:AAPL"})
        # globally time-sorted
        ks = pd.to_datetime(pooled["timestamp"], utc=True)
        self.assertTrue(ks.is_monotonic_increasing)
        self.assertTrue(0.0 <= summary.label_positive_rate <= 1.0)


# --------------------------------------------------------------------------- #
# Sequence builder leakage guards
# --------------------------------------------------------------------------- #
class SequenceLeakageTests(unittest.TestCase):
    def test_no_cross_asset_windows(self):
        df = synth_pooled_for_sequences(["crypto:BTC-USD", "stock:AAPL"], n_per_asset=40)
        vocab = build_asset_vocab(df)
        batch = build_sequences(df, feature_cols=["f0", "f1"], asset_to_idx=vocab, window=10)
        # every window's asset_idx is a valid single asset; total count is
        # (40-9) windows * 2 assets when contiguous.
        self.assertEqual(len(batch), (40 - 9) * 2)
        self.assertEqual(set(batch.asset_idx.tolist()), set(vocab.values()))

    def test_label_is_last_bar(self):
        df = synth_pooled_for_sequences(["crypto:BTC-USD"], n_per_asset=20)
        df = df.sort_values("timestamp").reset_index(drop=True)
        vocab = build_asset_vocab(df)
        batch = build_sequences(df, feature_cols=["f0", "f1"], asset_to_idx=vocab, window=5)
        # first window covers rows 0..4 -> label should equal df.label[4]
        self.assertEqual(batch.y[0], int(df["label"].iloc[4]))

    def test_data_hole_splits_windows(self):
        # 1-minute series with a multi-hour hole in the middle.
        n = 60
        ts = list(pd.date_range("2022-01-01", periods=n, freq="1min", tz="UTC"))
        # punch a 5-hour hole after row 30
        ts = ts[:30] + [t + pd.Timedelta(hours=5) for t in ts[30:]]
        df = pd.DataFrame({
            "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S%z") for t in ts],
            "asset_id": "crypto:BTC-USD", "asset_class": "crypto",
            "label": np.random.default_rng(0).integers(0, 2, n),
            "f0": np.random.default_rng(1).normal(size=n),
        })
        vocab = build_asset_vocab(df)
        win = 10
        batch = build_sequences(df, feature_cols=["f0"], asset_to_idx=vocab, window=win, gap_multiplier=5.0)
        # Without the hole there'd be n-win+1 = 51 windows. The hole (≫5× the
        # 60s median) must remove every window straddling it (those ending at
        # rows 30..38, i.e. 9 windows).
        self.assertEqual(len(batch), (n - win + 1) - 9)

    def test_weekend_gaps_do_not_shatter_daily(self):
        # Business-day daily bars => Fri->Mon ~3x median, must NOT be split.
        ts = pd.bdate_range("2022-01-03", periods=40, tz="UTC")
        df = pd.DataFrame({
            "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S%z") for t in ts],
            "asset_id": "stock:AAPL", "asset_class": "stock",
            "label": np.random.default_rng(0).integers(0, 2, 40),
            "f0": np.random.default_rng(1).normal(size=40),
        })
        vocab = build_asset_vocab(df)
        batch = build_sequences(df, feature_cols=["f0"], asset_to_idx=vocab, window=10, gap_multiplier=5.0)
        # No windows dropped despite the weekend gaps.
        self.assertEqual(len(batch), 40 - 10 + 1)


# --------------------------------------------------------------------------- #
# Pooled model
# --------------------------------------------------------------------------- #
class PooledModelTests(unittest.TestCase):
    def test_forward_shape(self):
        import torch
        m = PooledLSTMClassifier(n_features=12, n_assets=5, embed_dim=4,
                                 hidden_size=16, num_layers=1, num_classes=2)
        out = m(torch.randn(7, 9, 12), torch.randint(0, 5, (7,)))
        self.assertEqual(tuple(out.shape), (7, 2))

    def test_from_config_round_trip(self):
        m = PooledLSTMClassifier(n_features=8, n_assets=3, embed_dim=2,
                                 hidden_size=8, num_layers=1, num_classes=2)
        m2 = PooledLSTMClassifier.from_config(m.config)
        self.assertEqual(m.config, m2.config)


# --------------------------------------------------------------------------- #
# Full train -> predict round-trip
# --------------------------------------------------------------------------- #
class TrainPredictRoundTripTests(unittest.TestCase):
    def test_end_to_end(self):
        import tempfile
        from multi_asset.build_pooled_dataset import assemble_pooled
        from multi_asset.predictor import PooledLSTMPredictor
        from multi_asset.train_pooled_lstm import train

        insts = {
            "crypto:BTC-USD": Instrument("BTC/USD", "crypto"),
            "stock:AAPL": Instrument("AAPL", "stock"),
        }
        frames = {
            "crypto:BTC-USD": synth_ohlcv(500, seed=10),
            "stock:AAPL": synth_ohlcv(500, seed=11),
        }
        pooled, _ = assemble_pooled(frames, instruments=insts, horizon_bars=1, warmup_bars=60)

        with tempfile.TemporaryDirectory() as d:
            ds = Path(d) / "pooled.csv"
            pooled.to_csv(ds, index=False)
            out = Path(d) / "model"
            summary = train(
                dataset_path=ds, output_dir=out, window=8, embed_dim=4,
                hidden_size=16, num_layers=1, epochs=2, batch_size=64,
                patience=2, min_test_winrate=0.0, min_test_ntrades=1,
                force_cpu=True, verbose=False,
            )
            self.assertEqual(summary.n_assets, 2)
            self.assertTrue((out / "model.pt").exists())
            self.assertTrue((out / "meta.json").exists())
            self.assertTrue((out / "scaler.joblib").exists())

            pred = PooledLSTMPredictor(str(out), device="cpu")
            self.assertEqual(set(pred.known_assets()), {"crypto:BTC-USD", "stock:AAPL"})

            fresh = synth_ohlcv(400, seed=99)
            res = pred.predict("crypto:BTC-USD", fresh)
            self.assertIn(res.side, ("long", "flat"))
            self.assertTrue(0.0 <= res.confidence <= 1.0)
            self.assertEqual(res.used_bars, 8)

            with self.assertRaises(KeyError):
                pred.predict("crypto:DOGE-USD", fresh)


if __name__ == "__main__":
    unittest.main()
