#!/usr/bin/env python3
"""
Small unit checks for feature engineering:
1) Required columns exist
2) TF features are shifted (no lookahead)
3) Works on OHLCV-only input
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from utils import compute_features, FEATURE_COLUMNS_PROFITABLE


def _build_ohlcv(n: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    close = np.arange(100.0, 100.0 + n, dtype=float)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 10.0),
    })
    return df


def check_required_columns():
    df = _build_ohlcv()
    feats = compute_features(df)
    missing = [c for c in FEATURE_COLUMNS_PROFITABLE if c not in feats.columns]
    assert not missing, f"Missing FEATURE_COLUMNS_PROFITABLE: {missing}"


def check_tf_shift():
    df = _build_ohlcv(180)
    feats = compute_features(df)

    # Build expected tf5_log_ret_1 with explicit shift
    base = df.set_index("timestamp")[["open", "high", "low", "close"]]
    tf = base.resample("5min", label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna(subset=["close"])
    tf_log_ret = np.log(tf["close"]).diff()
    tf_feat = pd.DataFrame({
        "tf_ts": tf.index,
        "tf5_log_ret_1": tf_log_ret.shift(1),
    })
    expected = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        tf_feat.sort_values("tf_ts"),
        left_on="timestamp",
        right_on="tf_ts",
        direction="backward",
    )["tf5_log_ret_1"].to_numpy()

    actual = feats["tf5_log_ret_1"].to_numpy()
    mask = ~np.isnan(expected)
    assert mask.any(), "Expected tf5_log_ret_1 should have some non-NaN values"
    max_diff = np.nanmax(np.abs(actual[mask] - expected[mask]))
    assert max_diff < 1e-8, f"TF shift mismatch: max_diff={max_diff}"


def check_tf_idempotent():
    df = _build_ohlcv(180)
    first = compute_features(df)
    second = compute_features(first)
    # No tf*_x/_y artifacts
    bad_cols = [c for c in second.columns if (
        (c.startswith("tf5_") or c.startswith("tf15_") or c.startswith("tf60_")) and
        (c.endswith("_x") or c.endswith("_y"))
    )]
    assert not bad_cols, f"Found tf merge artifacts: {bad_cols}"
    # Canonical tf columns exist
    for c in ("tf5_log_ret_1", "tf15_log_ret_1", "tf60_log_ret_1"):
        assert c in second.columns, f"Missing canonical TF column: {c}"


def main():
    check_required_columns()
    check_tf_shift()
    check_tf_idempotent()
    print("✅ Feature checks passed.")


if __name__ == "__main__":
    main()
