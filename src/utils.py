# utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Iterator
import glob, os
import numpy as np
import pandas as pd
import json

# ---------- Feature spec ----------
@dataclass(frozen=True)
class FeatureSpec:
    feature_cols: List[str]
    window_size: int = 150

DEFAULT_FEATURE_COLS: List[str] = [
    "open","high","low","close",
    "body","range","upper_wick","lower_wick","return",
    "sma_ratio","ema_20","macd","rsi_14","vol_change",
    "atr","price_vs_hourly_trend","bb_width",
]
DEFAULT_FEATURE_SPEC = FeatureSpec(DEFAULT_FEATURE_COLS, 150)


# ---------- Meta ----------
def load_meta(path: str = "model_meta.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "feature_cols": DEFAULT_FEATURE_COLS,
            "window_size": 150,
            "buy_threshold": 0.5,
            "sell_threshold": 0.5,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": False,
        }
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.setdefault("feature_cols", DEFAULT_FEATURE_COLS)
    meta.setdefault("window_size", 150)
    meta.setdefault("buy_threshold", 0.5)
    meta.setdefault("sell_threshold", 0.5)
    return meta


# ---------- IO: stream OHLCV files ----------
def load_ohlc_chunks(
    folder: str,
    pattern: str = "*.csv",
    required_cols: Tuple[str, ...] = ("open","high","low","close","volume"),
) -> Iterator[pd.DataFrame]:
    """
    Yields OHLCV dataframes (monthly or otherwise) with a tz-aware DatetimeIndex.
    Accepts files that either have a 'timestamp'/'date' column, or already have an index.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    for p in paths:
        df = pd.read_csv(p)
        # normalize time index
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif "date" in df.columns:
            ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
        else:
            # assume index is a timestamp-like column name saved by to_csv(index=True)
            idx_name = df.columns[0]
            ts = pd.to_datetime(df[idx_name], utc=True, errors="coerce")
        df["__ts__"] = ts
        df = df.dropna(subset=["__ts__"]).set_index("__ts__").sort_index()

        # column subset + dtype
        missing = set(required_cols) - set(df.columns)
        if missing:
            # try lower/upper case accidental variants
            for m in list(missing):
                if m.title() in df.columns:  # e.g. Open
                    df[m] = pd.to_numeric(df[m.title()], errors="coerce")
                    missing.remove(m)
            if missing:
                raise ValueError(f"{p} is missing columns: {missing}")

        for c in required_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # keep only what's needed (plus 'close' is used later)
        yield df[["open","high","low","close","volume"]].dropna().sort_index()


# ---------- TA helpers ----------
def _ensure_dt_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    else:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
    return out.sort_index()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ---------- Feature pipeline (single source of truth) ----------
def build_features(raw_df: pd.DataFrame, *, add_hourly_trend: bool = True,
                   compat_inf_to_zero: bool = False, keep_helper_cols: bool = False) -> pd.DataFrame:
    df = _ensure_dt_index_utc(raw_df).copy()

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_wick"] = df[["close","open"]].min(axis=1) - df["low"]

    df["return"] = df["close"].pct_change()
    df["vol_change"] = df["volume"].pct_change()

    df["sma_10"] = df["close"].rolling(10, min_periods=10).mean()
    df["sma_50"] = df["close"].rolling(50, min_periods=50).mean()
    df["sma_ratio"] = (df["sma_10"]/df["sma_50"]) - 1.0

    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()

    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df, 14)

    if add_hourly_trend:
        hourly = df["close"].resample("1h").mean()
        hourly_ema20 = hourly.ewm(span=20, adjust=False).mean()
        df["hourly_ema_20"] = hourly_ema20.reindex(df.index, method="ffill")
        denom = df["hourly_ema_20"].replace(0, np.nan)
        df["price_vs_hourly_trend"] = (df["close"] - df["hourly_ema_20"]) / denom
    else:
        df["price_vs_hourly_trend"] = np.nan

    bb_std = df["close"].rolling(20, min_periods=20).std()
    bb_mid = df["close"].rolling(20, min_periods=20).mean()
    df["bb_width"] = (4.0 * bb_std) / (bb_mid.replace(0, np.nan))

    df.replace([np.inf, -np.inf], 0 if compat_inf_to_zero else np.nan, inplace=True)
    df.dropna(inplace=True)

    if not keep_helper_cols:
        for c in ("sma_10","sma_50","hourly_ema_20"):
            if c in df.columns: df.drop(columns=c, inplace=True)
    return df


def make_model_window(
    feat_df: pd.DataFrame,
    spec: FeatureSpec,
    scaler=None,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    missing = [c for c in spec.feature_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    if len(feat_df) < spec.window_size:
        price = float(feat_df["close"].iloc[-1]) if len(feat_df) else None
        return None, price
    mat = feat_df[spec.feature_cols].tail(spec.window_size).to_numpy(dtype=np.float32)
    if scaler is not None:
        mat = scaler.transform(mat)
    X = np.expand_dims(mat, axis=0)
    return X, float(feat_df["close"].iloc[-1])


def proba_to_signal(p: float, buy_threshold: float, sell_threshold: float) -> tuple[str, float]:
    if p >= buy_threshold: return "BUY", float(p)
    if p <= sell_threshold: return "SELL", float(1.0 - p)
    mid = 0.5 * (buy_threshold + sell_threshold)
    span = max(buy_threshold - mid, 1e-9)
    conf = 1.0 - (abs(p - mid) / span)
    return "HOLD", float(max(0.0, min(1.0, conf)))
