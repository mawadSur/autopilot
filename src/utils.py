#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — shared helpers for data IO, feature/window building, model/meta loading, and formatting.

Centralizes functionality used by:
- backtest.py
- paper_trade.py
- inference.py
- live_trader.py (if needed)

This keeps the codebase consistent and avoids duplication errors.
"""

from __future__ import annotations

import glob
import logging
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns
 
# Import robust model helpers (aliased to avoid clashing with utils.load_meta)
try:
    from models import (
        build_model_from_meta as _build_model_from_meta,
        load_meta as _load_meta_model,
        load_scaler as _load_scaler,
        load_checkpoint_state as _load_checkpoint_state,
        load_state_dict_flexible as _load_state_dict_flexible,
        PROFIT_MODEL_VERSION as _PROFIT_MODEL_VERSION,
        resolve_path as _resolve_path,
    )
except Exception:
    _build_model_from_meta = _load_meta_model = _load_scaler = _resolve_path = None
    _load_checkpoint_state = _load_state_dict_flexible = _PROFIT_MODEL_VERSION = None

import numpy as np
import pandas as pd
import torch

# TA-Lib is optional. Current feature engineering is implemented with pandas/numpy,
# so training/inference should not hard-fail when TA-Lib is unavailable.
try:
    import talib as ta  # type: ignore  # pragma: no cover
except Exception:  # pragma: no cover
    ta = None

# Optional scaler persistence
try:
    import joblib
except Exception:
    joblib = None

# Optional dotenv
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    def _load_dotenv(*_args, **_kwargs):
        return False


# ---------------------------
# Defaults shared across scripts
# ---------------------------

PRICE_CANDIDATES = ["close", "adj_close", "adj close", "close_price", "price", "last", "mid", "c"]

DEFAULT_COLS_6 = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_COLS_7 = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


# ---------------------------
# Env / seed utilities
# ---------------------------

def load_dotenv() -> None:
    """Load environment variables from a local .env if present."""
    try:
        _load_dotenv()
    except Exception:
        pass


def set_seed(seed: int) -> None:
    """Reproducible seeding for numpy/torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# CSV header repair & loading
# ---------------------------

def _columns_look_headerless(cols: List[str]) -> bool:
    lowers = [str(c).strip().lower() for c in cols]
    if any(k in lowers for k in ["open", "high", "low", "close", "volume", "timestamp", "time", "c", "o", "h", "l", "v"]):
        return False
    numeric_like = 0
    for c in cols:
        s = str(c).strip().replace(".", "", 1).replace("-", "", 1)
        if s.isdigit():
            numeric_like += 1
    return numeric_like >= max(3, len(cols) // 2)


def _apply_default_headers(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[1]
    if n == 6:
        df.columns = DEFAULT_COLS_6
    elif n == 7:
        df.columns = DEFAULT_COLS_7
    else:
        df.columns = [f"col{i}" for i in range(n)]
    # === NEW: Multi-Timeframe Feature Engineering ===
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df.set_index("timestamp", inplace=True)

            ema_15m = df["close"].resample("15T").last().ewm(span=12, adjust=False).mean()
            rsi_15m = df["close"].resample("15T").last().diff().ewm(alpha=1/14, adjust=False).mean()
            ema_1h = df["close"].resample("1H").last().ewm(span=12, adjust=False).mean()
            rsi_1h = df["close"].resample("1H").last().diff().ewm(alpha=1/14, adjust=False).mean()

            df["ema_15m"] = ema_15m
            df["rsi_15m"] = rsi_15m
            df["ema_1h"] = ema_1h
            df["rsi_1h"] = rsi_1h

            df.fillna(method="ffill", inplace=True)
            df.reset_index(inplace=True)

            df["price_vs_ema15m"] = (df["close"] / (df["ema_15m"] + 1e-12)).fillna(1.0)
            df["price_vs_ema1h"] = (df["close"] / (df["ema_1h"] + 1e-12)).fillna(1.0)
        except Exception as e:
            print(f"[WARN] Failed to compute multi-timeframe features: {e}")
            if isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)

    return df


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Fix headerless CSVs by assigning default OHLCV columns when needed."""
    if _columns_look_headerless(list(df.columns)):
        df = _apply_default_headers(df)
    return df


def list_csvs_sorted(path: str) -> List[str]:
    """Return sorted list of CSVs if directory, else a single CSV path."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return files
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.splitext(path)[1].lower() != ".csv":
        raise ValueError(f"Only .csv supported; got {path}")
    return [path]


def iter_csv_chunks_with_fix(path: str, chunksize: int) -> Generator[pd.DataFrame, None, None]:
    """Yield normalized CSV chunks for very large files."""
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield normalize_headers(chunk)


def read_csv_concat_sorted(data_dir: str) -> pd.DataFrame:
    """Read all CSVs (or one CSV) and return a single concatenated, normalized DataFrame."""
    p = Path(data_dir)
    files: List[str]
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
    else:
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file or directory, got: {data_dir}")
        files = [str(p)]
    parts = []
    for f in files:
        parts.append(normalize_headers(pd.read_csv(f)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------
# Feature / price helpers
# ---------------------------

def resolve_price_col(columns: List[str], preferred: Optional[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in columns}
    if preferred:
        if preferred in columns:
            return preferred
        if preferred.lower() in lower_map:
            return lower_map[preferred.lower()]
    for cand in PRICE_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]
    return None


def infer_feature_cols(sample_df: pd.DataFrame, feature_cols: Optional[List[str]], label_col: Optional[str], price_col: Optional[str]) -> List[str]:
    """Infer numeric feature columns if not explicitly provided."""
    if feature_cols:
        return list(feature_cols)
    drop = {c for c in [label_col, price_col, "timestamp", "time"] if c is not None}
    numeric = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in drop]
    if not feats:
        raise ValueError("Could not infer numeric feature columns from the sample.")
    return feats


# ---------------------------
# Feature engineering (copied from train_model.py)
# ---------------------------

ROLL_WINDOW = 20

# ---------------------------
# Shared defaults / feature list
# ---------------------------

DEFAULT_SEQ_LENS = [60, 90, 120]
DEFAULT_SEQ_LEN = 90

FEATURE_COLUMNS_PROFITABLE = [
    "open", "high", "low", "close", "volume",
    "return_1", "return_5", "return_15",
    "log_ret", "zret_20", "zret_60",
    "range_pct", "body_to_range_ratio", "body_to_range", "abs_body_to_range",
    "upper_wick_ratio", "lower_wick_ratio",
    "ema_9", "ema_21", "ema_50", "ema_spread_9_21", "ema_spread_21_50",
    "macd", "macd_signal", "macd_hist",
    "atr_14", "atrp_14", "ret_std_30",
    "bb_mid_20", "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_pctb_20",
    "vol_log", "vol_ma_20", "vol_z_20",
    "price_pos_donchian20", "vwap_roll_50",
    # Bucket 2 raw microstructure inputs (optional)
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
    "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    "trade_count", "buy_count", "sell_count",
    "taker_buy_volume_base", "taker_sell_volume_base",
    "taker_buy_volume_quote", "taker_sell_volume_quote",
    "volume_quote",
    # Bucket 2 derived features
    "mid", "spread_abs", "spread_pct", "microprice", "l1_imbalance", "mid_log_ret", "spread_z_60",
    "l2_imbalance_5", "l2_imbalance_10", "l2_imbalance_20",
    "depth_ratio_5", "depth_ratio_10", "depth_ratio_20", "book_pressure_5",
    "total_taker_vol_base", "ofi_base", "ofi_ratio", "buy_sell_count_imb",
    "avg_trade_size_base", "avg_trade_size_quote",
    "ofi_over_depth_10", "spread_times_imbalance",
    # Time-of-day features
    "minute_of_day", "tod_sin", "tod_cos",
    "day_of_week", "dow_sin", "dow_cos",
    # Volatility regime
    "rv_5", "rv_15", "rv_60", "rv_240", "vol_of_vol_60", "range_ma_20",
    # Gap / micro-momentum
    "open_to_prev_close", "close_to_prev_close", "hlc3", "close_pos_in_range",
    # Stationary ratios / zscores
    "close_over_ema_9", "close_over_ema_21", "close_over_ema_50", "close_over_vwap_50",
    "price_z_60", "price_z_240",
    # Multi-timeframe (5m/15m/60m)
    "tf5_log_ret_1", "tf5_rv_20", "tf5_atrp_14", "tf5_ema_spread",
    "tf15_log_ret_1", "tf15_rv_20", "tf15_atrp_14", "tf15_ema_spread",
    "tf60_log_ret_1", "tf60_rv_20", "tf60_atrp_14", "tf60_ema_spread",
    "adx",
]

class ProfitOptimizedFeatureEngineer:
    """
    Profit-obsessed feature selector.

    - Computes forward returns
    - Ranks features by correlation to 5-min forward return
    - Drops low-corr features and those that increase walk-forward drawdown
    """

    def __init__(
        self,
        *,
        horizon_bars: int = 5,
        min_abs_corr: float = 0.02,
        walk_folds: int = 3,
        drawdown_tolerance: float = 1e-6,
    ) -> None:
        self.horizon_bars = int(max(1, horizon_bars))
        self.min_abs_corr = float(min_abs_corr)
        self.walk_folds = int(max(2, walk_folds))
        self.drawdown_tolerance = float(drawdown_tolerance)

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if equity.size < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.maximum(1e-12, peak)
        return float(abs(np.min(dd)))

    def compute_forward_returns(self, df: pd.DataFrame, price_col: str) -> pd.Series:
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' missing from DataFrame")
        prices = df[price_col].astype(float)
        fwd = prices.pct_change(self.horizon_bars).shift(-self.horizon_bars)
        return fwd

    def rank_by_correlation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        price_col: str,
    ) -> Dict[str, float]:
        fwd = self.compute_forward_returns(df, price_col)
        corr = {}
        for c in feature_cols:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            corr_val = v.corr(fwd)
            if corr_val is None or not np.isfinite(corr_val):
                continue
            corr[c] = float(corr_val)
        return corr

    def _walk_forward_indices(self, n: int) -> List[Tuple[int, int]]:
        fold = max(1, n // self.walk_folds)
        idx = []
        start = 0
        while start < n:
            end = min(n, start + fold)
            if end - start >= 10:
                idx.append((start, end))
            start = end
        return idx if idx else [(0, n)]

    def _portfolio_returns(
        self,
        features: np.ndarray,
        corrs: np.ndarray,
        fwd: np.ndarray,
    ) -> np.ndarray:
        pos = np.sign(features * corrs.reshape(1, -1))
        avg_pos = np.nanmean(pos, axis=1)
        return avg_pos * fwd

    def filter_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        *,
        price_col: str = "close",
    ) -> List[str]:
        corr = self.rank_by_correlation(df, feature_cols, price_col)
        if not corr:
            return list(feature_cols)

        ranked = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
        candidates = [f for f, c in ranked if abs(c) >= self.min_abs_corr]
        if not candidates:
            # Keep top 20 by correlation if all are below threshold
            candidates = [f for f, _ in ranked[:20]]

        fwd = self.compute_forward_returns(df, price_col).to_numpy(dtype=np.float32, copy=False)
        feat_mat = df[candidates].to_numpy(dtype=np.float32, copy=False)
        corr_arr = np.array([corr.get(c, 0.0) for c in candidates], dtype=np.float32)

        # Clean NaNs
        valid_mask = np.isfinite(fwd)
        feat_mat = feat_mat[valid_mask]
        fwd = fwd[valid_mask]
        feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)

        if feat_mat.shape[0] < 1000 or feat_mat.shape[1] < 2:
            return list(candidates)

        splits = self._walk_forward_indices(feat_mat.shape[0])
        baseline_dds: List[float] = []
        for a, b in splits:
            r = self._portfolio_returns(feat_mat[a:b], corr_arr, fwd[a:b])
            equity = np.cumprod(1.0 + r)
            baseline_dds.append(self._max_drawdown(equity))
        baseline_dd = float(np.mean(baseline_dds)) if baseline_dds else 0.0

        # Remove features that worsen drawdown
        kept = []
        for j, feat in enumerate(candidates):
            if feat_mat.shape[1] <= 1:
                kept.append(feat)
                continue
            mat_minus = np.delete(feat_mat, j, axis=1)
            corr_minus = np.delete(corr_arr, j)
            dd_list = []
            for a, b in splits:
                r = self._portfolio_returns(mat_minus[a:b], corr_minus, fwd[a:b])
                equity = np.cumprod(1.0 + r)
                dd_list.append(self._max_drawdown(equity))
            dd_minus = float(np.mean(dd_list)) if dd_list else 0.0
            if dd_minus + self.drawdown_tolerance < baseline_dd:
                # Removing this feature improves drawdown => feature increases drawdown
                continue
            kept.append(feat)

        # Rank by correlation
        kept_sorted = sorted(kept, key=lambda f: abs(corr.get(f, 0.0)), reverse=True)
        return kept_sorted


def align_feature_columns(meta_feature_cols: Optional[List[str]], *, expected_size: Optional[int] = None) -> List[str]:
    """
    Strict alignment: require metadata features to match expected_size.
    """
    expected = expected_size if (expected_size is not None and expected_size > 0) else len(FEATURE_COLUMNS_PROFITABLE)
    meta_feature_cols = list(meta_feature_cols or [])
    if len(meta_feature_cols) != expected:
        raise ValueError(
            "Feature count mismatch. Retrain required — delete old checkpoint and retrain."
        )
    return meta_feature_cols


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering using only OHLCV.

    Outputs a compact, high-signal set that aligns with FEATURE_COLUMNS_PROFITABLE (single source of truth).
    Keeps the original OHLCV columns; adds engineered columns; fills early NaNs.
    """

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    df = df.copy()

    # ---------- Price / return ----------
    df["return_1"] = df["close"].pct_change().fillna(0.0)
    df["return_5"] = df["close"].pct_change(5).fillna(0.0)
    df["return_15"] = df["close"].pct_change(15).fillna(0.0)
    # Log return (more stationary); optionally clip to tame outliers
    df["log_ret"] = np.log(df["close"]).diff().fillna(0.0)
    # Uncomment to clip extreme moves (e.g., +/-5%)
    # df["log_ret"] = df["log_ret"].clip(-0.05, 0.05)

    # Rolling z-scores of log returns (short- and mid-horizon)
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0, 1e-12)
        return (series - mean) / std

    df["zret_20"] = _zscore(df["log_ret"], 20).fillna(0.0)
    df["zret_60"] = _zscore(df["log_ret"], 60).fillna(0.0)

    # Volatility regime (log-return based)
    df["rv_5"] = df["log_ret"].rolling(5, min_periods=1).std().fillna(0.0)
    df["rv_15"] = df["log_ret"].rolling(15, min_periods=1).std().fillna(0.0)
    df["rv_60"] = df["log_ret"].rolling(60, min_periods=1).std().fillna(0.0)
    df["rv_240"] = df["log_ret"].rolling(240, min_periods=1).std().fillna(0.0)
    df["vol_of_vol_60"] = df["rv_15"].rolling(60, min_periods=1).std().fillna(0.0)

    # ---------- Candle structure ----------
    range_raw = (df["high"] - df["low"]).replace(0, 1e-12)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, 1e-12)
    df["body_to_range_ratio"] = (df["close"] - df["open"]) / range_raw
    # Normalized imbalance of the candle body; absolute version for magnitude
    df["body_to_range"] = (df["close"] - df["open"]) / range_raw
    df["abs_body_to_range"] = df["body_to_range"].abs()
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / range_raw
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / range_raw
    df["range_ma_20"] = df["range_pct"].rolling(20, min_periods=1).mean().fillna(0.0)

    # Gap / micro-momentum
    prev_close = df["close"].shift(1)
    df["open_to_prev_close"] = (df["open"] / prev_close.replace(0, 1e-12)) - 1.0
    df["close_to_prev_close"] = (df["close"] / prev_close.replace(0, 1e-12)) - 1.0
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["close_pos_in_range"] = (df["close"] - df["low"]) / ((df["high"] - df["low"]) + 1e-12)

    # ---------- Trend / momentum ----------
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_spread_9_21"] = (df["ema_9"] - df["ema_21"]) / df["close"].replace(0, 1e-12)
    df["ema_spread_21_50"] = (df["ema_21"] - df["ema_50"]) / df["close"].replace(0, 1e-12)
    # Stationary ratios vs EMA
    df["close_over_ema_9"] = (df["close"] / df["ema_9"].replace(0, 1e-12)) - 1.0
    df["close_over_ema_21"] = (df["close"] / df["ema_21"].replace(0, 1e-12)) - 1.0
    df["close_over_ema_50"] = (df["close"] / df["ema_50"].replace(0, 1e-12)) - 1.0

    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_line - macd_signal

    # ---------- Volatility ----------
    true_range = (df["high"] - df["low"]).abs().combine(
        (df["high"] - df["close"].shift(1)).abs(), max
    ).combine(
        (df["low"] - df["close"].shift(1)).abs(), max
    )
    df["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    df["atrp_14"] = df["atr_14"] / df["close"].replace(0, 1e-12)
    df["ret_std_30"] = df["return_1"].rolling(30, min_periods=1).std().fillna(0.0)
    # Wilder-style ADX (falls back to exponentially weighted smoothing)
    adx_period = 14
    eps = 1e-12
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    alpha = 1.0 / adx_period
    tr_smooth = true_range.replace(np.nan, 0.0).ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.replace(np.nan, 0.0).ewm(alpha=alpha, adjust=False).mean() / (tr_smooth + eps)
    minus_di = 100.0 * minus_dm.replace(np.nan, 0.0).ewm(alpha=alpha, adjust=False).mean() / (tr_smooth + eps)
    dx = (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + eps) * 100.0
    df["adx"] = dx.replace(np.nan, 0.0).ewm(alpha=alpha, adjust=False).mean().fillna(0.0)

    # ---------- Bollinger / bands ----------
    bb_mid = df["close"].rolling(20, min_periods=1).mean()
    bb_std = df["close"].rolling(20, min_periods=1).std().fillna(0.0)
    df["bb_mid_20"] = bb_mid
    df["bb_upper_20"] = bb_mid + 2 * bb_std
    df["bb_lower_20"] = bb_mid - 2 * bb_std
    df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / bb_mid.replace(0, 1e-12)
    df["bb_pctb_20"] = (df["close"] - df["bb_lower_20"]) / (
        (df["bb_upper_20"] - df["bb_lower_20"]).replace(0, 1e-12)
    )

    # ---------- Volume ----------
    df["vol_log"] = np.log1p(df["volume"])
    vol_ma = df["volume"].rolling(20, min_periods=1).mean()
    vol_std = df["volume"].rolling(20, min_periods=1).std().replace(0, 1e-12)
    df["vol_ma_20"] = vol_ma
    df["vol_z_20"] = (df["volume"] - vol_ma) / vol_std

    # ---------- Positioning ----------
    roll_max = df["close"].rolling(20, min_periods=1).max()
    roll_min = df["close"].rolling(20, min_periods=1).min()
    df["price_pos_donchian20"] = (df["close"] - roll_min) / (roll_max - roll_min + 1e-12)

    # Rolling VWAP proxy over 50 bars
    pv = (df["close"] * df["volume"]).rolling(50, min_periods=1).sum()
    v = df["volume"].rolling(50, min_periods=1).sum().replace(0, 1e-12)
    df["vwap_roll_50"] = pv / v
    df["close_over_vwap_50"] = (df["close"] / df["vwap_roll_50"].replace(0, 1e-12)) - 1.0

    # Price z-scores (log price)
    log_close = np.log(df["close"].replace(0, 1e-12))
    df["price_z_60"] = _zscore(log_close, 60).fillna(0.0)
    df["price_z_240"] = _zscore(log_close, 240).fillna(0.0)

    # Time-of-day features (UTC) if timestamp present
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            # Drop any prior TF columns to make compute_features idempotent
            drop_cols = [c for c in df.columns if (
                c.startswith("tf5_") or c.startswith("tf15_") or c.startswith("tf60_") or c == "tf_ts" or
                c.startswith("tf5_") and (c.endswith("_x") or c.endswith("_y")) or
                c.startswith("tf15_") and (c.endswith("_x") or c.endswith("_y")) or
                c.startswith("tf60_") and (c.endswith("_x") or c.endswith("_y"))
            )]
            if drop_cols:
                df = df.drop(columns=drop_cols, errors="ignore")
            minute_of_day = (ts.dt.hour * 60 + ts.dt.minute).astype("float64")
            df["minute_of_day"] = minute_of_day
            df["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
            df["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
            dow = ts.dt.dayofweek.astype("float64")
            df["day_of_week"] = dow
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

            # Multi-timeframe context without lookahead
            df_mt = df.copy()
            df_mt["_ts"] = ts
            df_mt = df_mt.sort_values("_ts")

            def _tf_features(rule: str, prefix: str) -> pd.DataFrame:
                base = df_mt.set_index("_ts")[["open", "high", "low", "close"]]
                tf = base.resample(rule, label="right", closed="right").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                }).dropna(subset=["close"])
                tf_log_ret = np.log(tf["close"].replace(0, 1e-12)).diff()
                tf_rv_20 = tf_log_ret.rolling(20, min_periods=1).std()
                tr = (tf["high"] - tf["low"]).abs().combine(
                    (tf["high"] - tf["close"].shift(1)).abs(), max
                ).combine(
                    (tf["low"] - tf["close"].shift(1)).abs(), max
                )
                tf_atr = tr.rolling(14, min_periods=1).mean()
                tf_atrp = tf_atr / tf["close"].replace(0, 1e-12)
                tf_ema9 = tf["close"].ewm(span=9, adjust=False).mean()
                tf_ema21 = tf["close"].ewm(span=21, adjust=False).mean()
                tf_ema_spread = (tf_ema9 - tf_ema21) / tf["close"].replace(0, 1e-12)

                feat = pd.DataFrame({
                    "tf_ts": tf.index,
                    f"{prefix}_log_ret_1": tf_log_ret,
                    f"{prefix}_rv_20": tf_rv_20,
                    f"{prefix}_atrp_14": tf_atrp,
                    f"{prefix}_ema_spread": tf_ema_spread,
                })
                # Shift by 1 TF bar to avoid lookahead
                feat[[f"{prefix}_log_ret_1", f"{prefix}_rv_20", f"{prefix}_atrp_14", f"{prefix}_ema_spread"]] = (
                    feat[[f"{prefix}_log_ret_1", f"{prefix}_rv_20", f"{prefix}_atrp_14", f"{prefix}_ema_spread"]].shift(1)
                )
                return feat.dropna(subset=["tf_ts"])

            tf5 = _tf_features("5min", "tf5")
            tf15 = _tf_features("15min", "tf15")
            tf60 = _tf_features("60min", "tf60")

            for tf_df in (tf5, tf15, tf60):
                if tf_df.empty:
                    continue
                df_mt = pd.merge_asof(
                    df_mt,
                    tf_df.sort_values("tf_ts"),
                    left_on="_ts",
                    right_on="tf_ts",
                    direction="backward",
                )
                if "tf_ts" in df_mt.columns:
                    df_mt = df_mt.drop(columns=["tf_ts"])

            # Restore to original order and copy merged tf columns back
            df_mt = df_mt.sort_index()
            for col in df_mt.columns:
                if col.startswith("tf5_") or col.startswith("tf15_") or col.startswith("tf60_"):
                    df[col] = df_mt[col]

    # ---------- Bucket 2 microstructure (optional inputs) ----------
    eps = 1e-12
    bucket2_raw = [
        "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
        "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
        "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
        "trade_count", "buy_count", "sell_count",
        "taker_buy_volume_base", "taker_sell_volume_base",
        "taker_buy_volume_quote", "taker_sell_volume_quote",
        "volume_quote",
    ]
    for c in bucket2_raw:
        if c not in df.columns:
            df[c] = np.nan

    # Fill sizes/volumes/counts with 0.0 when missing
    size_like = [
        "bid_size_l1", "ask_size_l1",
        "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
        "taker_buy_volume_base", "taker_sell_volume_base",
        "taker_buy_volume_quote", "taker_sell_volume_quote",
        "trade_count", "buy_count", "sell_count",
        "volume_quote",
    ]
    for c in size_like:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Forward-fill price-like columns within the same day if timestamp exists
    price_like = [
        "best_bid", "best_ask",
        "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    ]
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            day = ts.dt.date
            for c in price_like:
                if c in df.columns:
                    df[c] = df[c].groupby(day).ffill()

    # Derived top-of-book features (if available)
    if "best_bid" in df.columns and "best_ask" in df.columns:
        df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
        df["spread_abs"] = df["best_ask"] - df["best_bid"]
        df["spread_pct"] = df["spread_abs"] / (df["mid"] + eps)
        if "bid_size_l1" in df.columns and "ask_size_l1" in df.columns:
            denom = df["bid_size_l1"] + df["ask_size_l1"] + eps
            df["microprice"] = (df["best_ask"] * df["bid_size_l1"] + df["best_bid"] * df["ask_size_l1"]) / denom
            df["l1_imbalance"] = (df["bid_size_l1"] - df["ask_size_l1"]) / denom
        df["mid_log_ret"] = np.log(df["mid"].replace(0, eps)).diff()
        df["spread_z_60"] = _zscore(df["spread_pct"], 60)

    # Derived L2 depth features (if available)
    if "bid_depth_5" in df.columns and "ask_depth_5" in df.columns:
        df["l2_imbalance_5"] = (df["bid_depth_5"] - df["ask_depth_5"]) / (df["bid_depth_5"] + df["ask_depth_5"] + eps)
        df["depth_ratio_5"] = df["bid_depth_5"] / (df["ask_depth_5"] + eps)
    if "bid_depth_10" in df.columns and "ask_depth_10" in df.columns:
        df["l2_imbalance_10"] = (df["bid_depth_10"] - df["ask_depth_10"]) / (df["bid_depth_10"] + df["ask_depth_10"] + eps)
        df["depth_ratio_10"] = df["bid_depth_10"] / (df["ask_depth_10"] + eps)
    if "bid_depth_20" in df.columns and "ask_depth_20" in df.columns:
        df["l2_imbalance_20"] = (df["bid_depth_20"] - df["ask_depth_20"]) / (df["bid_depth_20"] + df["ask_depth_20"] + eps)
        df["depth_ratio_20"] = df["bid_depth_20"] / (df["ask_depth_20"] + eps)

    if "microprice" in df.columns and "mid" in df.columns and "spread_abs" in df.columns:
        df["book_pressure_5"] = (df["microprice"] - df["mid"]) / (df["spread_abs"] + eps)

    # Derived trade-flow features (if available)
    if "taker_buy_volume_base" in df.columns and "taker_sell_volume_base" in df.columns:
        df["total_taker_vol_base"] = df["taker_buy_volume_base"] + df["taker_sell_volume_base"]
        df["ofi_base"] = df["taker_buy_volume_base"] - df["taker_sell_volume_base"]
        df["ofi_ratio"] = df["ofi_base"] / (df["total_taker_vol_base"] + eps)
        if "trade_count" in df.columns:
            df["buy_sell_count_imb"] = (df["buy_count"] - df["sell_count"]) / (df["trade_count"] + eps)
            df["avg_trade_size_base"] = df["total_taker_vol_base"] / (df["trade_count"] + eps)
            df["avg_trade_size_quote"] = (
                (df["taker_buy_volume_quote"] + df["taker_sell_volume_quote"]) / (df["trade_count"] + eps)
            )

    # Cross features (book + trades)
    if "ofi_base" in df.columns and "bid_depth_10" in df.columns and "ask_depth_10" in df.columns:
        df["ofi_over_depth_10"] = df["ofi_base"] / (df["bid_depth_10"] + df["ask_depth_10"] + eps)
    if "spread_pct" in df.columns and "l1_imbalance" in df.columns:
        df["spread_times_imbalance"] = df["spread_pct"] * df["l1_imbalance"]

    # Fill any remaining gaps created by rolling windows.
    # NOTE: avoid backfilling (bfill) because it can leak future info into the past.
    df.ffill(inplace=True)

    # Ensure all FEATURE_COLUMNS_PROFITABLE exist. For OHLCV-only inputs, many optional
    # microstructure columns will be missing — default them to 0.0 to prevent NaNs.
    for col in FEATURE_COLUMNS_PROFITABLE:
        if col not in df.columns:
            df[col] = 0.0

    missing_after = [c for c in FEATURE_COLUMNS_PROFITABLE if c not in df.columns]
    if missing_after:
        raise ValueError(f"compute_features missing engineered columns: {missing_after}")

    # Final safety: eliminate any NaN/Inf in model features (prevents NaN losses).
    df = df[FEATURE_COLUMNS_PROFITABLE].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df



# ---------------------------
# Windows
# ---------------------------

def build_windows_from_flat(features: np.ndarray, seq_len: int) -> np.ndarray:
    """Create overlapping [N-seq_len+1, seq_len, F] windows from [N, F] flat features."""
    N, F = features.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=np.float32)
    stride0, stride1 = features.strides
    shape = (N - seq_len + 1, seq_len, F)
    strides = (stride0, stride0, stride1)
    return np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides).copy()


def build_windows(features: np.ndarray, seq_len: int) -> np.ndarray:
    """Alias used by some scripts."""
    return build_windows_from_flat(features, seq_len)


# ---------------------------
# Formatting helpers
# ---------------------------

def fmt_money(x, currency: str = "$") -> str:
    """Format a number as money."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1e12:
        return f"{currency}{x:.3e}"
    return f"{currency}{x:,.2f}"


def fmt_pct(x) -> str:
    """Format a fraction as percent with two decimals."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return str(x)


# ---------------------------
# Meta / model loading
# ---------------------------

def load_meta(model_dir_or_path: str) -> Dict:
    """
    Load model_meta.json from either a directory (e.g. 'model')
    or a direct file path (e.g. 'model/model_meta.json').
    """
    p = Path(model_dir_or_path)
    meta_path = p if p.is_file() else (p / "model_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found at {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def update_strategy_registry(model_dir: Path, profitability_score: float, *, min_score: float = 2.0) -> None:
    registry_path = model_dir / "strategy_registry.json"
    try:
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"strategies": []}
    except Exception:
        registry = {"strategies": []}
    strategies = registry.get("strategies", []) if isinstance(registry, dict) else []
    strategies = [s for s in strategies if isinstance(s, dict)]

    if profitability_score <= min_score:
        registry["strategies"] = [s for s in strategies if s.get("model_dir") != str(model_dir)]
    else:
        strategies = [s for s in strategies if s.get("model_dir") != str(model_dir)]
        strategies.append({
            "model_dir": str(model_dir),
            "profitability_score": float(profitability_score),
            "updated": datetime.utcnow().isoformat(),
        })
        registry["strategies"] = strategies

    tmp_path = registry_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    os.replace(tmp_path, registry_path)


def load_model_with_metadata(model_dir: Path):
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found in {model_dir}")

    meta_obj = _load_meta_model(str(meta_path))
    meta = meta_obj.to_dict() if hasattr(meta_obj, "to_dict") else dict(meta_obj)

    weights_path = _resolve_path(str(model_dir), meta.get("model_state_path", "model.pt"))
    state_dict, ckpt_meta = _load_checkpoint_state(weights_path)

    feature_count = int(ckpt_meta.get("feature_count", 0))
    feature_names = list(ckpt_meta.get("feature_names") or [])
    version = str(ckpt_meta.get("version", "")).strip()
    train_date = ckpt_meta.get("train_date")
    if not feature_count or not feature_names:
        raise ValueError("Checkpoint metadata invalid. Retrain required — delete old checkpoint and retrain.")
    if version != _PROFIT_MODEL_VERSION:
        raise ValueError(
            f"Model version mismatch (checkpoint={version or 'unknown'}, expected={_PROFIT_MODEL_VERSION}). "
            "Retrain required — delete old checkpoint and retrain."
        )

    meta["feature_count"] = feature_count
    meta["feature_names"] = feature_names
    meta["version"] = version
    meta["train_date"] = train_date

    meta["input_size"] = feature_count
    meta["feature_cols"] = list(feature_names)

    model = _build_model_from_meta(meta)
    incompatible = _load_state_dict_flexible(model, state_dict)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    ok_keys = [k for k in state_dict.keys() if k not in unexpected_keys]
    logging.info(
        "Loaded state_dict from %s with %d keys (%d OK, %d missing, %d unexpected).",
        weights_path,
        len(state_dict),
        len(ok_keys),
        len(missing_keys),
        len(unexpected_keys),
    )
    return model, meta, feature_count


def load_model_bundle(model_dir: str):
    """
    Loads a complete model bundle (model, scaler, meta) using helpers from models.py.
    Returns (model.eval(), scaler_or_None, meta_dict).
    """
    model_dir_path = Path(model_dir)
    meta_path = model_dir_path / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found in {model_dir}")

    model, meta, _ = load_model_with_metadata(model_dir_path)
    # Load the scaler if one is specified
    scaler = None
    if meta.get("feature_scaling", True) and meta.get("scaler_path", None):
        scaler_path = _resolve_path(model_dir, meta["scaler_path"])
        scaler = _load_scaler(scaler_path)
        feature_cols = list(meta.get("feature_cols") or FEATURE_COLUMNS_PROFITABLE)
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(feature_cols):
            raise ValueError(
                "Scaler feature count mismatch. Retrain required — delete old checkpoint and retrain."
            )
        if scaler is not None:
            scaler.feature_names_in_ = list(feature_cols)
    return model.eval(), scaler, meta


# ---------------------------
# Simple binary label helper (optional)
# ---------------------------

def binary_label_next_bar_up(closes: np.ndarray) -> np.ndarray:
    """
    y[i] = 1 if close[i] > close[i-1], else 0. First element is 0 by construction.
    """
    y = np.zeros_like(closes, dtype=np.int64)
    y[1:] = (closes[1:] > closes[:-1]).astype(np.int64)
    return y


# ---------------------------
# Live Signal Generator (SageMaker endpoint)
# ---------------------------

class SignalGenerator:
    """Streaming feature buffer and SageMaker inference helper.

    - Maintains a deque of the last `window_size` engineered feature rows.
    - Accepts raw kline dicts and computes features using train_model._compute_features.
    - When the buffer is full, invokes a SageMaker endpoint and returns a signal.

    Returns dicts shaped like:
      {"confidence": float, "probability": float, "signal": int, "threshold": float}
    where signal = 1 when confidence >= threshold else 0.
    """

    def __init__(self, endpoint_name: str, window_size: int = 192):
        if not endpoint_name:
            raise ValueError("endpoint_name is required")
        self.endpoint_name = endpoint_name
        self.window_size = int(window_size)

        # Raw history used for feature computation context (prefill from trade.py)
        self.history = deque(maxlen=self.window_size)
        self.history_size = self.window_size

        # Engineered features buffer (the model window)
        self._feat_buf = deque(maxlen=self.window_size)

        # Determined lazily on first call based on available engineered columns
        self._feature_cols: Optional[List[str]] = None

        # Confidence threshold fallback (endpoint may return its own)
        self.threshold = float(os.getenv("BUY_THRESHOLD", "0.75"))

    def _ensure_feature_cols(self, engineered_df: pd.DataFrame) -> List[str]:
        """Pick an ordered feature list aligned with training.

        Uses FEATURE_COLUMNS_PROFITABLE as the canonical list.
        """
        if self._feature_cols is not None:
            return self._feature_cols

        available = set(engineered_df.columns.tolist())
        cols = [c for c in FEATURE_COLUMNS_PROFITABLE if c in available]
        if not cols:
            # As a last resort, take numeric columns (best effort)
            cols = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
        self._feature_cols = cols
        return self._feature_cols

    def get_signal(self, kline_data: Dict) -> Dict[str, float]:
        """Ingest one kline dict, update buffer, and infer.

        kline_data keys typically: {date, open, high, low, close, volume}
        """
        # Append raw bar to history for context
        self.history.append(kline_data)

        # Build a small DataFrame from recent bars and engineer features
        df_raw = pd.DataFrame(list(self.history))
        # Use the centralized feature engineering
        df_feat = compute_features(df_raw.copy())
        feat_cols = self._ensure_feature_cols(df_feat)

        # Extract the newest engineered row; coerce to float32 for model
        latest_row = df_feat.iloc[-1:][feat_cols].astype(np.float32, copy=False)
        # Store as 1D np.ndarray
        self._feat_buf.append(latest_row.to_numpy(dtype=np.float32, copy=False).ravel())

        # Not enough rows yet to form a window
        if len(self._feat_buf) < self.window_size:
            return {"probability": 0.0, "confidence": 0.0, "signal": 0, "threshold": self.threshold}

        # Form [T, F] matrix for exactly one window (T=window_size)
        X = np.stack(list(self._feat_buf), axis=0)  # [T, F]

        # Send as CSV so server can reorder/select features based on its meta
        df_send = pd.DataFrame(X, columns=feat_cols)
        csv_payload = df_send.to_csv(index=False)

        # Lazy imports
        import json as _json  # noqa: F401
        try:
            import boto3  # type: ignore
        except Exception as e:
            raise RuntimeError(f"boto3 is required for SageMaker invocation: {e}")

        rt = boto3.client("sagemaker-runtime")
        resp = rt.invoke_endpoint(EndpointName=self.endpoint_name, ContentType="text/csv", Body=csv_payload)
        body = resp["Body"].read().decode("utf-8")

        try:
            out = _json.loads(body)
        except Exception as e:
            raise RuntimeError(f"Failed to parse endpoint response: {e}; body={body[:200]}...")

        probs = out.get("probs") or []
        threshold = float(out.get("threshold", self.threshold))
        prob = float(probs[-1]) if probs else 0.0
        signal = 1 if prob >= threshold else 0

        # Keep both keys for compatibility with callers
        return {"probability": prob, "confidence": prob, "signal": signal, "threshold": threshold}


__all__ = [
    # defaults
    "DEFAULT_SEQ_LENS", "DEFAULT_SEQ_LEN", "FEATURE_COLUMNS_PROFITABLE",
    "align_feature_columns",
    "PRICE_CANDIDATES",
    "ProfitOptimizedFeatureEngineer",
    # env / device / seed
    "load_dotenv", "set_seed", "get_device_str",
    # csv utils
    "normalize_headers", "list_csvs_sorted", "iter_csv_chunks_with_fix", "read_csv_concat_sorted",
    # feature / price
    "resolve_price_col", "infer_feature_cols", "compute_features",
    # windows
    "build_windows_from_flat", "build_windows",
    # formatting
    "fmt_money", "fmt_pct",
    # meta/model
    "load_meta", "load_model_bundle", "SignalGenerator",
    # labels
    "binary_label_next_bar_up",
]
