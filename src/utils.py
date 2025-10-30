#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — shared helpers for data IO, feature/window building, model/meta loading, and formatting.

Centralizes functionality used by:
- aws_train_model.py
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
        load_model_state as _load_model_state,
        load_scaler as _load_scaler,
        resolve_path as _resolve_path,
    )
except Exception:
    _build_model_from_meta = _load_meta_model = _load_model_state = _load_scaler = _resolve_path = None

import numpy as np
import pandas as pd
import torch
import subprocess
import sys

try:
    import talib as ta
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
    import talib as ta

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
ATR_ALPHA = 1 / 14
RSI_ALPHA = 1 / 14

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full training feature set used by train_model.py."""
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")
    
   
    df["return"] = df["close"].pct_change().fillna(1.0)
    df["Range"] = (df["high"] / df["low"]) - 1
    # df["Volatility"] = df['return'].rolling(window=ROLL_WINDOW).std()

    # # Volume-Based Indicators
    # df['OBV'] = ta.OBV(df['close'], df['volume'])
    # df['ADL'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])


    # # Momentum-Based Indicators
    # df['Stoch_Oscillator'] = ta.STOCH(df['high'], df['low'], df['close'])[0]

    # df['PSAR'] = ta.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

    # Remove rows containing inf or nan values
    df.dropna(inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    
    
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


def load_model_bundle(model_dir: str):
    """
    Loads a complete model bundle (model, scaler, meta) using helpers from models.py.
    Returns (model.eval(), scaler_or_None, meta_dict).
    """
    model_dir_path = Path(model_dir)
    meta_path = model_dir_path / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found in {model_dir}")

    meta = _load_meta_model(str(meta_path))
    # Build model architecture from metadata
    model = _build_model_from_meta(meta)
    # Load the trained weights into the model (with graceful fallback)
    weights_path = _resolve_path(model_dir, getattr(meta, 'model_state_path', 'model.pt'))
    try:
        _load_model_state(model, weights_path)
    except RuntimeError as e:
        # Dimension mismatch is the common case; try fallback checkpoint if provided
        fallback_name = getattr(meta, 'last_model_state_path', None) or 'model_last.pt'
        fallback_path = _resolve_path(model_dir, fallback_name)
        try:
            _load_model_state(model, fallback_path)
            logging.warning(
                "Primary weights '%s' incompatible with model (likely meta mismatch). "
                "Loaded fallback '%s' instead.",
                weights_path,
                fallback_path,
            )
        except Exception:
            raise
    # Load the scaler if one is specified
    scaler = None
    if getattr(meta, 'feature_scaling', True) and getattr(meta, 'scaler_path', None):
        scaler_path = _resolve_path(model_dir, meta.scaler_path)
        scaler = _load_scaler(scaler_path)
    return model.eval(), scaler, meta.to_dict()


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

        Prefers train_model.FEATURES; falls back to DEFAULT_FEATURE_COLS.
        """
        if self._feature_cols is not None:
            return self._feature_cols

        try:
            from train_model import FEATURES as TRAIN_FEATURES  # type: ignore
        except Exception:
            TRAIN_FEATURES = []

        available = set(engineered_df.columns.tolist())
        cols = [c for c in TRAIN_FEATURES if c in available]
        if len(cols) < 4:
            cols = [c for c in DEFAULT_FEATURE_COLS if c in available]
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
    "PRICE_CANDIDATES",
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
