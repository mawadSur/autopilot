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
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
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

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_selection import mutual_info_regression
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None
    mutual_info_regression = None

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

REQUIRED_RAW_COLUMNS = ["open", "high", "low", "close", "volume"]

OPTIONAL_MICROSTRUCTURE_RAW_COLUMNS = [
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
    "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    "book_poc", "book_va_low", "book_va_high",
    "trade_count", "buy_count", "sell_count",
    "taker_buy_volume_base", "taker_sell_volume_base",
    "taker_buy_volume_quote", "taker_sell_volume_quote",
    "volume_quote",
]

OPTIONAL_MICROSTRUCTURE_SIZE_COLUMNS = [
    "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
    "taker_buy_volume_base", "taker_sell_volume_base",
    "taker_buy_volume_quote", "taker_sell_volume_quote",
    "trade_count", "buy_count", "sell_count",
    "volume_quote",
]

OPTIONAL_MICROSTRUCTURE_PRICE_COLUMNS = [
    "best_bid", "best_ask",
    "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    "book_poc", "book_va_low", "book_va_high",
]


def ensure_optional_microstructure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure optional bucket-2 raw columns exist with stable defaults."""
    for col in OPTIONAL_MICROSTRUCTURE_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in OPTIONAL_MICROSTRUCTURE_SIZE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in OPTIONAL_MICROSTRUCTURE_PRICE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            day = ts.dt.date
            for col in OPTIONAL_MICROSTRUCTURE_PRICE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].groupby(day).ffill()
    return df


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
    "liq_sweep_high", "liq_sweep_low", "liq_sweep_high_strength", "liq_sweep_low_strength",
    "avwap_spike", "avwap_cycle",
    "close_over_avwap_spike", "close_over_avwap_cycle",
    "avwap_spike_age", "avwap_cycle_age",
    "in_golden_pocket", "dist_to_golden_pocket",
    # Bucket 2 raw microstructure inputs (optional)
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
    "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    "book_poc", "book_va_low", "book_va_high",
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
    "close_over_book_poc", "book_poc_distance_atr",
    "book_value_area_width", "book_in_value_area", "book_above_va", "book_below_va",
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

FEATURE_COLUMNS = FEATURE_COLUMNS_PROFITABLE


class LiquidityEngineer:
    """Derive sweep, anchored VWAP, and order-book profile context."""

    def __init__(
        self,
        *,
        sweep_lookback: int = 20,
        volume_anchor_window: int = 50,
        volume_anchor_z: float = 2.0,
        cycle_lookback: int = 50,
        golden_pocket_lookback: int = 120,
        value_area_pct: float = 0.70,
    ) -> None:
        self.sweep_lookback = int(max(3, sweep_lookback))
        self.volume_anchor_window = int(max(10, volume_anchor_window))
        self.volume_anchor_z = float(volume_anchor_z)
        self.cycle_lookback = int(max(10, cycle_lookback))
        self.golden_pocket_lookback = int(max(20, golden_pocket_lookback))
        self.value_area_pct = float(min(max(value_area_pct, 0.50), 0.95))

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std().replace(0.0, np.nan)
        return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _bars_since(mask: pd.Series) -> pd.Series:
        values = mask.fillna(False).astype(bool).to_numpy()
        out = np.zeros(len(values), dtype=np.float64)
        last_hit = 0
        seen = False
        for i, hit in enumerate(values):
            if hit:
                last_hit = i
                seen = True
                out[i] = 0.0
            elif seen:
                out[i] = float(i - last_hit)
            else:
                out[i] = float(i)
        return pd.Series(out, index=mask.index)

    @staticmethod
    def _rolling_extrema_with_index(
        series: pd.Series,
        window: int,
        *,
        mode: str,
    ) -> Tuple[pd.Series, pd.Series]:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        n = len(values)
        out_values = np.full(n, np.nan, dtype=np.float64)
        out_indices = np.full(n, -1, dtype=np.int64)
        dq: deque[int] = deque()
        is_max = mode == "max"
        if mode not in {"max", "min"}:
            raise ValueError(f"Unsupported rolling extrema mode: {mode}")

        fill_value = -np.inf if is_max else np.inf
        clean = np.where(np.isfinite(values), values, fill_value)

        for i in range(n):
            start = i - window + 1
            while dq and dq[0] < start:
                dq.popleft()
            if is_max:
                while dq and clean[i] >= clean[dq[-1]]:
                    dq.pop()
            else:
                while dq and clean[i] <= clean[dq[-1]]:
                    dq.pop()
            dq.append(i)
            best_idx = dq[0]
            out_indices[i] = best_idx
            out_values[i] = clean[best_idx]

        out_values = np.where(np.isfinite(out_values), out_values, np.nan)
        return pd.Series(out_values, index=series.index), pd.Series(out_indices, index=series.index, dtype=np.int64)

    @staticmethod
    def _anchored_vwap(price: pd.Series, volume: pd.Series, anchor_mask: pd.Series) -> pd.Series:
        px = pd.to_numeric(price, errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        vol = pd.to_numeric(volume, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64, copy=False)
        anchors = anchor_mask.fillna(False).astype(bool).to_numpy(copy=True)
        out = np.zeros(len(px), dtype=np.float64)
        if len(px) == 0:
            return pd.Series(dtype=np.float64)
        anchors[0] = True

        cum_pv = np.cumsum(px * vol)
        cum_v = np.cumsum(vol)
        anchor_idx = 0

        for i in range(len(px)):
            if anchors[i]:
                anchor_idx = i
            prev = anchor_idx - 1
            base_pv = cum_pv[prev] if prev >= 0 else 0.0
            base_v = cum_v[prev] if prev >= 0 else 0.0
            denom = cum_v[i] - base_v
            out[i] = (cum_pv[i] - base_pv) / max(denom, 1e-12)

        return pd.Series(out, index=price.index)

    @staticmethod
    def _parse_book_level(level: Any) -> Optional[Tuple[float, float]]:
        price = size = None
        if isinstance(level, dict):
            price = level.get("price", level.get("p", level.get("rate", level.get("px"))))
            size = level.get("size", level.get("s", level.get("qty", level.get("q", level.get("amount", level.get("volume"))))))
        elif isinstance(level, Sequence) and not isinstance(level, (str, bytes, bytearray)) and len(level) >= 2:
            price, size = level[0], level[1]
        if price is None or size is None:
            return None
        try:
            price_f = float(price)
            size_f = float(size)
        except Exception:
            return None
        if not math.isfinite(price_f) or not math.isfinite(size_f) or size_f <= 0.0:
            return None
        return price_f, size_f

    @classmethod
    def _levels(cls, levels: Any, side: str) -> List[Tuple[float, float]]:
        parsed: List[Tuple[float, float]] = []
        if isinstance(levels, Sequence) and not isinstance(levels, (str, bytes, bytearray)):
            for level in levels:
                value = cls._parse_book_level(level)
                if value is not None:
                    parsed.append(value)
        parsed.sort(key=lambda x: x[0], reverse=(side == "bid"))
        return parsed

    def _profile_from_levels(
        self,
        bids: Sequence[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]],
    ) -> Dict[str, float]:
        book: Dict[float, float] = {}
        for price, size in list(bids) + list(asks):
            if size <= 0.0:
                continue
            book[float(price)] = book.get(float(price), 0.0) + float(size)

        if not book:
            return {"book_poc": np.nan, "book_va_low": np.nan, "book_va_high": np.nan}

        prices = np.array(sorted(book.keys()), dtype=np.float64)
        sizes = np.array([book[p] for p in prices], dtype=np.float64)
        total = float(sizes.sum())
        if total <= 0.0:
            return {"book_poc": np.nan, "book_va_low": np.nan, "book_va_high": np.nan}

        poc_idx = int(np.argmax(sizes))
        left = right = poc_idx
        covered = float(sizes[poc_idx])
        target = total * self.value_area_pct

        while covered < target and (left > 0 or right < len(prices) - 1):
            left_size = sizes[left - 1] if left > 0 else -1.0
            right_size = sizes[right + 1] if right < len(prices) - 1 else -1.0
            if right_size > left_size and right < len(prices) - 1:
                right += 1
                covered += float(sizes[right])
            elif left > 0:
                left -= 1
                covered += float(sizes[left])
            elif right < len(prices) - 1:
                right += 1
                covered += float(sizes[right])
            else:
                break

        return {
            "book_poc": float(prices[poc_idx]),
            "book_va_low": float(prices[left]),
            "book_va_high": float(prices[right]),
        }

    def add_liquidated_sweeps(self, df: pd.DataFrame) -> None:
        eps = 1e-12
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        atr = pd.to_numeric(df.get("atr_14", (high - low).rolling(14, min_periods=1).mean()), errors="coerce").fillna(0.0)

        prior_high = high.rolling(self.sweep_lookback, min_periods=2).max().shift(1)
        prior_low = low.rolling(self.sweep_lookback, min_periods=2).min().shift(1)
        candle_range = (high - low).abs().replace(0.0, eps)
        upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0.0)
        lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low).clip(lower=0.0)

        sweep_high = (high > prior_high) & (close < prior_high)
        sweep_low = (low < prior_low) & (close > prior_low)

        high_extension = (high - prior_high).clip(lower=0.0)
        low_extension = (prior_low - low).clip(lower=0.0)
        high_reclaim = (prior_high - close).clip(lower=0.0)
        low_reclaim = (close - prior_low).clip(lower=0.0)

        df["liq_sweep_high"] = sweep_high.astype(float)
        df["liq_sweep_low"] = sweep_low.astype(float)
        df["liq_sweep_high_strength"] = np.where(
            sweep_high,
            (high_extension + high_reclaim + upper_wick) / (atr + candle_range + eps),
            0.0,
        )
        df["liq_sweep_low_strength"] = np.where(
            sweep_low,
            (low_extension + low_reclaim + lower_wick) / (atr + candle_range + eps),
            0.0,
        )

    def add_anchored_vwap(self, df: pd.DataFrame) -> None:
        eps = 1e-12
        close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
        high = pd.to_numeric(df["high"], errors="coerce").ffill().fillna(close)
        low = pd.to_numeric(df["low"], errors="coerce").ffill().fillna(close)
        hlc3 = pd.to_numeric(df.get("hlc3", (high + low + close) / 3.0), errors="coerce").ffill().fillna(close)
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

        vol_z = self._zscore(volume, self.volume_anchor_window)
        prev_spike_high = volume.rolling(self.volume_anchor_window, min_periods=2).max().shift(1)
        spike_anchor = (vol_z >= self.volume_anchor_z) & (volume >= prev_spike_high.fillna(0.0))

        prior_cycle_high = high.rolling(self.cycle_lookback, min_periods=2).max().shift(1)
        prior_cycle_low = low.rolling(self.cycle_lookback, min_periods=2).min().shift(1)
        cycle_anchor = (high > prior_cycle_high) | (low < prior_cycle_low)

        spike_anchor = spike_anchor.fillna(False)
        cycle_anchor = cycle_anchor.fillna(False)

        df["avwap_spike"] = self._anchored_vwap(hlc3, volume, spike_anchor)
        df["avwap_cycle"] = self._anchored_vwap(hlc3, volume, cycle_anchor)
        df["close_over_avwap_spike"] = (close / df["avwap_spike"].replace(0.0, eps)) - 1.0
        df["close_over_avwap_cycle"] = (close / df["avwap_cycle"].replace(0.0, eps)) - 1.0
        df["avwap_spike_age"] = self._bars_since(spike_anchor | pd.Series(np.arange(len(df)) == 0, index=df.index))
        df["avwap_cycle_age"] = self._bars_since(cycle_anchor | pd.Series(np.arange(len(df)) == 0, index=df.index))

    def add_fibonacci_golden_pocket(self, df: pd.DataFrame) -> None:
        eps = 1e-12
        close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
        high = pd.to_numeric(df["high"], errors="coerce").ffill().fillna(close)
        low = pd.to_numeric(df["low"], errors="coerce").ffill().fillna(close)
        atr = pd.to_numeric(df.get("atr_14", (high - low).rolling(14, min_periods=1).mean()), errors="coerce").fillna(0.0)

        swing_high, swing_high_idx = self._rolling_extrema_with_index(
            high,
            self.golden_pocket_lookback,
            mode="max",
        )
        swing_low, swing_low_idx = self._rolling_extrema_with_index(
            low,
            self.golden_pocket_lookback,
            mode="min",
        )
        swing_range = (swing_high - swing_low).clip(lower=0.0)
        valid_range = swing_range > eps

        long_gp_low = swing_high - (0.66 * swing_range)
        long_gp_high = swing_high - (0.618 * swing_range)
        short_gp_low = swing_low + (0.618 * swing_range)
        short_gp_high = swing_low + (0.66 * swing_range)

        upswing_active = valid_range & (swing_high_idx > swing_low_idx)
        downswing_active = valid_range & (swing_low_idx > swing_high_idx)

        gp_low = pd.Series(np.nan, index=df.index, dtype=np.float64)
        gp_high = pd.Series(np.nan, index=df.index, dtype=np.float64)
        gp_low = gp_low.mask(upswing_active, long_gp_low)
        gp_high = gp_high.mask(upswing_active, long_gp_high)
        gp_low = gp_low.mask(downswing_active, short_gp_low)
        gp_high = gp_high.mask(downswing_active, short_gp_high)

        in_pocket = valid_range & gp_low.notna() & gp_high.notna() & (close >= gp_low) & (close <= gp_high)
        gap_below = (gp_low - close).clip(lower=0.0)
        gap_above = (close - gp_high).clip(lower=0.0)
        dist = (gap_below + gap_above).where(~in_pocket, 0.0)
        dist = dist.where(valid_range & gp_low.notna() & gp_high.notna(), 0.0)

        df["in_golden_pocket"] = in_pocket.astype(float)
        df["dist_to_golden_pocket"] = dist / (atr + eps)

    def add_volume_profile_context(self, df: pd.DataFrame) -> None:
        eps = 1e-12
        profile_cols = ("book_poc", "book_va_low", "book_va_high")
        if "bids" in df.columns and "asks" in df.columns:
            profiles = []
            for _, row in df.iterrows():
                bids = self._levels(row.get("bids", []), "bid")
                asks = self._levels(row.get("asks", []), "ask")
                profiles.append(self._profile_from_levels(bids, asks))
            profile_df = pd.DataFrame(profiles, index=df.index)
            for col in profile_cols:
                existing = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
                df[col] = existing.where(existing.notna(), profile_df[col])
        else:
            for col in profile_cols:
                if col not in df.columns:
                    df[col] = np.nan

        close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
        atr = pd.to_numeric(df.get("atr_14", 0.0), errors="coerce").fillna(0.0)
        poc = pd.to_numeric(df["book_poc"], errors="coerce")
        va_low = pd.to_numeric(df["book_va_low"], errors="coerce")
        va_high = pd.to_numeric(df["book_va_high"], errors="coerce")
        va_width = (va_high - va_low).clip(lower=0.0)

        df["close_over_book_poc"] = (close / poc.replace(0.0, eps)) - 1.0
        df["book_poc_distance_atr"] = (close - poc) / (atr + eps)
        df["book_value_area_width"] = va_width / close.replace(0.0, eps)
        df["book_in_value_area"] = ((close >= va_low) & (close <= va_high)).astype(float)
        df["book_above_va"] = (close > va_high).astype(float)
        df["book_below_va"] = (close < va_low).astype(float)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.add_liquidated_sweeps(df)
        self.add_anchored_vwap(df)
        self.add_fibonacci_golden_pocket(df)
        self.add_volume_profile_context(df)
        return df

class ProfitOptimizedFeatureEngineer:
    """
    Profit-obsessed feature selector.

    - Computes forward returns
    - Ranks features with nonlinear dependency scores
    - Uses mutual information plus gradient-boosted tree importance
    - Drops weak-signal features and those that increase walk-forward drawdown
    """

    def __init__(
        self,
        *,
        horizon_bars: int = 5,
        min_score: float = 0.02,
        min_abs_corr: Optional[float] = None,
        walk_folds: int = 3,
        drawdown_tolerance: float = 1e-6,
        mi_neighbors: int = 5,
        tree_sample_rows: int = 25_000,
        tree_estimators: int = 125,
        random_state: int = 42,
        ranking_method: str = "combined",
    ) -> None:
        self.horizon_bars = int(max(1, horizon_bars))
        if min_abs_corr is not None:
            min_score = float(min_abs_corr)
        self.min_score = float(max(0.0, min_score))
        self.walk_folds = int(max(2, walk_folds))
        self.drawdown_tolerance = float(drawdown_tolerance)
        self.mi_neighbors = int(max(1, mi_neighbors))
        self.tree_sample_rows = int(max(1_000, tree_sample_rows))
        self.tree_estimators = int(max(25, tree_estimators))
        self.random_state = int(random_state)
        ranking_method = str(ranking_method or "combined").strip().lower()
        if ranking_method in {"mi", "mutual_info"}:
            ranking_method = "mutual_information"
        if ranking_method not in {"combined", "mutual_information", "tree_importance"}:
            raise ValueError(f"Unsupported ranking_method: {ranking_method}")
        self.ranking_method = ranking_method
        self.selection_summary_: Dict[str, Any] = {}

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

    def _prepare_selection_arrays(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        price_col: str,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            return [], np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

        feat_df = df[available].apply(pd.to_numeric, errors="coerce")
        fwd = self.compute_forward_returns(df, price_col).to_numpy(dtype=np.float32, copy=False)
        feat_mat = feat_df.to_numpy(dtype=np.float32, copy=False)
        valid_mask = np.isfinite(fwd)
        if feat_mat.size:
            valid_mask &= np.any(np.isfinite(feat_mat), axis=1)
        feat_mat = feat_mat[valid_mask]
        fwd = fwd[valid_mask]
        feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
        return available, feat_mat, fwd

    @staticmethod
    def _normalize_scores(scores: Dict[str, float], ordered_features: List[str]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = np.array([max(0.0, float(scores.get(f, 0.0))) for f in ordered_features], dtype=np.float32)
        peak = float(np.max(vals)) if vals.size else 0.0
        if peak <= 0.0:
            return {f: 0.0 for f in ordered_features}
        return {f: float(max(0.0, scores.get(f, 0.0)) / peak) for f in ordered_features}

    @staticmethod
    def _discrete_feature_mask(feat_mat: np.ndarray) -> np.ndarray:
        if feat_mat.ndim != 2 or feat_mat.shape[1] == 0:
            return np.zeros((0,), dtype=bool)
        out = np.zeros(feat_mat.shape[1], dtype=bool)
        for j in range(feat_mat.shape[1]):
            col = feat_mat[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            uniques = np.unique(col)
            if uniques.size <= 12 and np.all(np.isclose(uniques, np.round(uniques), atol=1e-6)):
                out[j] = True
        return out

    def rank_by_mutual_information(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        price_col: str,
    ) -> Dict[str, float]:
        if mutual_info_regression is None:
            return {}

        available, feat_mat, fwd = self._prepare_selection_arrays(df, feature_cols, price_col)
        if not available or feat_mat.shape[0] < 32:
            return {}

        discrete_mask = self._discrete_feature_mask(feat_mat)
        n_neighbors = min(self.mi_neighbors, max(1, feat_mat.shape[0] - 1))
        scores = mutual_info_regression(
            feat_mat,
            fwd,
            discrete_features=discrete_mask,
            n_neighbors=n_neighbors,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return {
            feature: float(max(0.0, score))
            for feature, score in zip(available, scores)
            if np.isfinite(score)
        }

    def rank_by_tree_importance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        price_col: str,
    ) -> Dict[str, float]:
        if GradientBoostingRegressor is None:
            return {}

        available, feat_mat, fwd = self._prepare_selection_arrays(df, feature_cols, price_col)
        if not available or feat_mat.shape[0] < 256:
            return {}

        if feat_mat.shape[0] > self.tree_sample_rows:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(feat_mat.shape[0], size=self.tree_sample_rows, replace=False))
            feat_mat = feat_mat[idx]
            fwd = fwd[idx]

        model = GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            n_estimators=self.tree_estimators,
            max_depth=3,
            subsample=0.7,
            random_state=self.random_state,
        )
        model.fit(feat_mat, fwd)
        scores = np.asarray(getattr(model, "feature_importances_", np.zeros(len(available))), dtype=np.float32)
        return {
            feature: float(max(0.0, score))
            for feature, score in zip(available, scores)
            if np.isfinite(score)
        }

    def rank_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        price_col: str,
    ) -> Dict[str, float]:
        mi_scores = self.rank_by_mutual_information(df, feature_cols, price_col)
        tree_scores = self.rank_by_tree_importance(df, feature_cols, price_col)
        ordered = [c for c in feature_cols if c in df.columns]
        if not ordered:
            return {}

        mi_norm = self._normalize_scores(mi_scores, ordered)
        tree_norm = self._normalize_scores(tree_scores, ordered)
        use_tree = any(v > 0.0 for v in tree_norm.values())
        scores: Dict[str, float] = {}
        ranking_label = self.ranking_method
        if self.ranking_method == "mutual_information":
            scores = {feature: mi_norm.get(feature, 0.0) for feature in ordered}
        elif self.ranking_method == "tree_importance":
            if use_tree:
                scores = {feature: tree_norm.get(feature, 0.0) for feature in ordered}
            else:
                scores = {feature: mi_norm.get(feature, 0.0) for feature in ordered}
                ranking_label = "tree_importance_fallback_mutual_information"
        else:
            ranking_label = "mutual_information+gradient_boosting_importance"
            for feature in ordered:
                mi_score = mi_norm.get(feature, 0.0)
                if use_tree:
                    scores[feature] = 0.65 * mi_score + 0.35 * tree_norm.get(feature, 0.0)
                else:
                    scores[feature] = mi_score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.selection_summary_ = {
            "ranking_method": ranking_label,
            "combined_top": ranked[:10],
            "mi_top": sorted(mi_scores.items(), key=lambda item: item[1], reverse=True)[:10],
            "tree_top": sorted(tree_scores.items(), key=lambda item: item[1], reverse=True)[:10],
        }
        return scores

    @staticmethod
    def _feature_direction(values: np.ndarray, fwd: np.ndarray) -> float:
        if values.size < 32 or fwd.size != values.size:
            return 0.0

        lo_q, hi_q = np.quantile(values, [0.2, 0.8])
        lo_mask = values <= lo_q
        hi_mask = values >= hi_q
        if int(lo_mask.sum()) >= 8 and int(hi_mask.sum()) >= 8:
            edge = float(np.nanmean(fwd[hi_mask]) - np.nanmean(fwd[lo_mask]))
            if np.isfinite(edge) and abs(edge) > 1e-12:
                return float(np.sign(edge))

        corr = np.corrcoef(values, fwd)[0, 1] if values.size >= 2 else np.nan
        if np.isfinite(corr) and abs(corr) > 1e-12:
            return float(np.sign(corr))
        return 0.0

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
        directions: np.ndarray,
        fwd: np.ndarray,
    ) -> np.ndarray:
        if features.size == 0 or directions.size == 0:
            return np.zeros_like(fwd)
        medians = np.nanmedian(features, axis=0, keepdims=True)
        centered = features - medians
        active = np.abs(directions) > 0.0
        if not np.any(active):
            return np.zeros_like(fwd)
        pos = np.sign(centered[:, active] * directions[active].reshape(1, -1))
        avg_pos = np.nanmean(pos, axis=1)
        return avg_pos * fwd

    def filter_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        *,
        price_col: str = "close",
    ) -> List[str]:
        scores = self.rank_features(df, feature_cols, price_col)
        if not scores:
            return list(feature_cols)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [f for f, score in ranked if score >= self.min_score]
        if not candidates:
            candidates = [f for f, _ in ranked[:20]]

        _, feat_mat, fwd = self._prepare_selection_arrays(df, candidates, price_col)

        if feat_mat.shape[0] < 1000 or feat_mat.shape[1] < 2:
            self.selection_summary_["selected_features"] = list(candidates)
            return list(candidates)

        directions = np.array(
            [self._feature_direction(feat_mat[:, j], fwd) for j in range(feat_mat.shape[1])],
            dtype=np.float32,
        )
        splits = self._walk_forward_indices(feat_mat.shape[0])
        baseline_dds: List[float] = []
        for a, b in splits:
            r = self._portfolio_returns(feat_mat[a:b], directions, fwd[a:b])
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
            dir_minus = np.delete(directions, j)
            dd_list = []
            for a, b in splits:
                r = self._portfolio_returns(mat_minus[a:b], dir_minus, fwd[a:b])
                equity = np.cumprod(1.0 + r)
                dd_list.append(self._max_drawdown(equity))
            dd_minus = float(np.mean(dd_list)) if dd_list else 0.0
            if dd_minus + self.drawdown_tolerance < baseline_dd:
                continue
            kept.append(feat)

        kept_sorted = sorted(kept, key=lambda f: scores.get(f, 0.0), reverse=True)
        self.selection_summary_["selected_features"] = list(kept_sorted)
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


def _feature_row_value(feature_row: Optional[Any], key: str, default: float = np.nan) -> float:
    if feature_row is None:
        return float(default)
    value: Any = default
    if hasattr(feature_row, "get"):
        value = feature_row.get(key, default)
    else:
        try:
            value = feature_row[key]
        except Exception:
            value = default
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def apply_confluence_filter(sig: int, feature_row: Optional[Any], *, avwap_cycle_band: float = 0.001) -> int:
    """
    Hard-gate class signals with liquidity / AVWAP confluence.

    Signal convention: 0=short, 1=hold, 2=long.
    """
    sig = int(sig)
    if sig not in (0, 2) or feature_row is None:
        return sig

    close_over_avwap_cycle = _feature_row_value(feature_row, "close_over_avwap_cycle", np.nan)
    near_cycle_avwap = bool(
        np.isfinite(close_over_avwap_cycle) and abs(close_over_avwap_cycle) <= float(avwap_cycle_band)
    )

    if sig == 2:
        liq_sweep_low = _feature_row_value(feature_row, "liq_sweep_low", 0.0) > 0.5
        return 2 if (liq_sweep_low or near_cycle_avwap) else 1

    liq_sweep_high = _feature_row_value(feature_row, "liq_sweep_high", 0.0) > 0.5
    return 0 if (liq_sweep_high or near_cycle_avwap) else 1


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering using only OHLCV.

    Outputs a compact, high-signal set that aligns with FEATURE_COLUMNS_PROFITABLE (single source of truth).
    Keeps the original OHLCV columns; adds engineered columns; fills early NaNs.
    """

    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")

    df = df.copy()
    df = ensure_optional_microstructure_columns(df)

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

    # ---------- Liquidity context ----------
    LiquidityEngineer().transform(df)

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
    "DEFAULT_SEQ_LENS", "DEFAULT_SEQ_LEN", "FEATURE_COLUMNS", "FEATURE_COLUMNS_PROFITABLE",
    "REQUIRED_RAW_COLUMNS",
    "OPTIONAL_MICROSTRUCTURE_RAW_COLUMNS",
    "OPTIONAL_MICROSTRUCTURE_SIZE_COLUMNS",
    "OPTIONAL_MICROSTRUCTURE_PRICE_COLUMNS",
    "ensure_optional_microstructure_columns",
    "align_feature_columns",
    "apply_confluence_filter",
    "PRICE_CANDIDATES",
    "LiquidityEngineer",
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
