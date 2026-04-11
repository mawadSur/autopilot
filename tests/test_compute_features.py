import unittest
import numpy as np
import pandas as pd

from utils import compute_features, FEATURE_COLUMNS
from history_coindesk import l2_snapshots_to_minute_features


class FeatureColumnsTest(unittest.TestCase):
    def test_compute_features_matches_feature_columns(self):
        n = 300
        rng = np.random.default_rng(0)
        base = np.linspace(100.0, 101.0, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
            "open": base,
            "high": base * 1.001,
            "low": base * 0.999,
            "close": base * 1.0005,
            "volume": rng.lognormal(mean=2.0, sigma=0.3, size=n),
        })

        engineered = compute_features(df)

        self.assertEqual(
            FEATURE_COLUMNS,
            list(engineered.columns),
            "compute_features must return columns exactly matching FEATURE_COLUMNS",
        )


def _build_liquidity_df(n: int = 80) -> pd.DataFrame:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
        "open": np.full(n, 99.7),
        "high": np.full(n, 100.2),
        "low": np.full(n, 99.4),
        "close": np.full(n, 99.8),
        "volume": np.full(n, 10.0),
    })

    high_sweep_idx = 30
    df.loc[high_sweep_idx, ["open", "high", "low", "close"]] = [99.9, 101.2, 99.7, 100.1]

    spike_idx = 40
    df.loc[spike_idx, "volume"] = 250.0
    df.loc[spike_idx, ["open", "high", "low", "close"]] = [110.0, 111.0, 109.0, 110.5]

    low_sweep_idx = 55
    df.loc[low_sweep_idx, ["open", "high", "low", "close"]] = [99.6, 100.0, 98.7, 99.7]

    book = {
        "bids": [[99.0, 3.0], [98.5, 1.0]],
        "asks": [[100.0, 5.0], [100.5, 2.0]],
    }
    df["bids"] = [book["bids"]] * n
    df["asks"] = [book["asks"]] * n
    return df


def _build_golden_pocket_long_df(n: int = 140) -> pd.DataFrame:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
        "open": np.full(n, 110.0),
        "high": np.full(n, 110.4),
        "low": np.full(n, 109.6),
        "close": np.full(n, 110.0),
        "volume": np.full(n, 10.0),
    })
    df.loc[5, ["open", "high", "low", "close"]] = [100.2, 100.5, 100.0, 100.3]
    df.loc[100, ["open", "high", "low", "close"]] = [119.5, 120.0, 119.0, 119.6]
    df.loc[110, ["open", "high", "low", "close"]] = [107.0, 107.6, 106.9, 107.2]
    df.loc[111, ["open", "high", "low", "close"]] = [109.0, 109.4, 108.8, 109.2]
    return df


def _build_golden_pocket_short_df(n: int = 140) -> pd.DataFrame:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-02-01", periods=n, freq="min", tz="UTC"),
        "open": np.full(n, 110.0),
        "high": np.full(n, 110.4),
        "low": np.full(n, 109.6),
        "close": np.full(n, 110.0),
        "volume": np.full(n, 10.0),
    })
    df.loc[5, ["open", "high", "low", "close"]] = [119.7, 120.0, 119.4, 119.8]
    df.loc[100, ["open", "high", "low", "close"]] = [100.6, 101.0, 100.0, 100.4]
    df.loc[110, ["open", "high", "low", "close"]] = [112.6, 113.0, 112.4, 112.8]
    df.loc[111, ["open", "high", "low", "close"]] = [110.8, 111.1, 110.6, 110.9]
    return df


def test_compute_features_detects_liquidated_sweeps():
    feats = compute_features(_build_liquidity_df())

    assert feats.loc[30, "liq_sweep_high"] == 1.0
    assert feats.loc[30, "liq_sweep_high_strength"] > 0.0
    assert feats.loc[55, "liq_sweep_low"] == 1.0
    assert feats.loc[55, "liq_sweep_low_strength"] > 0.0


def test_compute_features_resets_anchored_vwap_on_volume_spike():
    raw = _build_liquidity_df()
    feats = compute_features(raw)
    spike_idx = 40
    expected_hlc3 = float((raw.loc[spike_idx, "high"] + raw.loc[spike_idx, "low"] + raw.loc[spike_idx, "close"]) / 3.0)

    assert feats.loc[spike_idx, "avwap_spike_age"] == 0.0
    assert feats.loc[spike_idx + 1, "avwap_spike_age"] == 1.0
    assert abs(float(feats.loc[spike_idx, "avwap_spike"]) - expected_hlc3) < 1e-9


def test_compute_features_trades_relative_to_book_profile():
    feats = compute_features(_build_liquidity_df())

    assert float(feats.loc[10, "book_poc"]) == 100.0
    assert float(feats.loc[10, "book_va_low"]) == 99.0
    assert float(feats.loc[10, "book_va_high"]) == 100.0
    assert feats.loc[10, "book_in_value_area"] == 1.0
    assert feats.loc[10, "book_above_va"] == 0.0
    assert feats.loc[10, "book_below_va"] == 0.0


def test_compute_features_detects_fibonacci_golden_pocket_for_long_and_short_swings():
    long_feats = compute_features(_build_golden_pocket_long_df())
    short_feats = compute_features(_build_golden_pocket_short_df())

    assert long_feats.loc[110, "in_golden_pocket"] == 1.0
    assert float(long_feats.loc[110, "dist_to_golden_pocket"]) == 0.0
    assert long_feats.loc[111, "in_golden_pocket"] == 0.0
    assert float(long_feats.loc[111, "dist_to_golden_pocket"]) > 0.0

    assert short_feats.loc[110, "in_golden_pocket"] == 1.0
    assert float(short_feats.loc[110, "dist_to_golden_pocket"]) == 0.0
    assert short_feats.loc[111, "in_golden_pocket"] == 0.0
    assert float(short_feats.loc[111, "dist_to_golden_pocket"]) > 0.0


def test_compute_features_handles_flat_golden_pocket_range():
    n = 140
    flat = pd.DataFrame({
        "timestamp": pd.date_range("2024-03-01", periods=n, freq="min", tz="UTC"),
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 5.0),
    })

    feats = compute_features(flat)

    assert feats.loc[130, "in_golden_pocket"] == 0.0
    assert float(feats.loc[130, "dist_to_golden_pocket"]) == 0.0


def test_l2_snapshot_profile_emits_poc_and_value_area():
    df_snap = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01T00:00:05Z")],
        "bids": [[[99.0, 3.0], [98.5, 1.0]]],
        "asks": [[[100.0, 5.0], [100.5, 2.0]]],
    })

    out = l2_snapshots_to_minute_features(df_snap)

    assert "book_poc" in out.columns
    assert "book_va_low" in out.columns
    assert "book_va_high" in out.columns
    assert float(out.loc[0, "book_poc"]) == 100.0
    assert float(out.loc[0, "book_va_low"]) == 99.0
    assert float(out.loc[0, "book_va_high"]) == 100.0


if __name__ == "__main__":
    unittest.main()
