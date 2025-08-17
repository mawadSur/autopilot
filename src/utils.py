# utils.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────────
# Defaults that are safe across train/backtest/inference
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_FEATURE_COLS: List[str] = [
    "open", "high", "low", "close", "body", "range", "upper_wick", "lower_wick",
    "return", "sma_ratio", "ema_20", "macd", "rsi_14", "vol_change", "atr",
    "price_vs_hourly_trend", "bb_width",
]
DEFAULT_WINDOW_SIZE: int = 150


# ────────────────────────────────────────────────────────────────────────────────
# Dataclass used by backtesting & training
# ────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureSpec:
    feature_cols: List[str]
    window_size: int = DEFAULT_WINDOW_SIZE
    buy_threshold: float = 0.5
    sell_threshold: float = 0.5
    label_def: str = "next_bar_up"
    scaler_type: str = "standard"

    @staticmethod
    def from_meta(meta: Dict[str, Any]) -> "FeatureSpec":
        # accept either "feature_cols" or legacy "features"
        feature_cols = meta.get("feature_cols", meta.get("features", DEFAULT_FEATURE_COLS))
        if not isinstance(feature_cols, list) or not all(isinstance(x, str) for x in feature_cols):
            feature_cols = DEFAULT_FEATURE_COLS

        return FeatureSpec(
            feature_cols=list(feature_cols),
            window_size=int(meta.get("window_size", DEFAULT_WINDOW_SIZE)),
            buy_threshold=float(meta.get("buy_threshold", 0.5)),
            sell_threshold=float(meta.get("sell_threshold", 0.5)),
            label_def=str(meta.get("label_def", "next_bar_up")),
            scaler_type=str(meta.get("scaler_type", "standard")),
        )


# ────────────────────────────────────────────────────────────────────────────────
# Meta helpers
# ────────────────────────────────────────────────────────────────────────────────
def _resolve_meta_path(explicit: Optional[str] = None) -> Optional[Path]:
    """
    Resolve a model_meta.json location with the following preference:
      1) explicit path argument
      2) env MODEL_META_PATH
      3) ./model_meta.json (cwd)
      4) $SM_MODEL_DIR/model_meta.json (SageMaker)
    """
    candidates: List[Optional[str]] = [
        explicit,
        os.getenv("MODEL_META_PATH"),
        os.path.join(os.getcwd(), "model_meta.json"),
        os.path.join(os.getenv("SM_MODEL_DIR", "/opt/ml/model"), "model_meta.json"),
    ]
    for c in [Path(p) for p in candidates if p]:
        if c.exists():
            return c
    return None


def load_meta(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model_meta.json (tolerant to missing keys) and return a dict.
    Always includes 'feature_cols' and 'window_size' at minimum.
    """
    meta_path = _resolve_meta_path(path)
    base: Dict[str, Any] = {
        "feature_cols": DEFAULT_FEATURE_COLS.copy(),
        "window_size": DEFAULT_WINDOW_SIZE,
        "buy_threshold": 0.5,
        "sell_threshold": 0.5,
        "label_def": "next_bar_up",
        "scaler_type": "standard",
    }

    if meta_path is None:
        return base

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        if not isinstance(raw, dict):
            return base

        # Normalize keys and merge
        if "feature_cols" not in raw and "features" in raw:
            raw["feature_cols"] = raw["features"]

        out = {**base, **raw}

        # Coerce types
        if not isinstance(out.get("feature_cols"), list) or not all(
            isinstance(x, str) for x in out["feature_cols"]
        ):
            out["feature_cols"] = DEFAULT_FEATURE_COLS.copy()
        out["window_size"] = int(out.get("window_size", DEFAULT_WINDOW_SIZE))
        out["buy_threshold"] = float(out.get("buy_threshold", 0.5))
        out["sell_threshold"] = float(out.get("sell_threshold", 0.5))
        out["label_def"] = str(out.get("label_def", "next_bar_up"))
        out["scaler_type"] = str(out.get("scaler_type", "standard"))
        return out
    except Exception:
        # On any error, fall back to safe defaults
        return base


def load_feature_spec(path: Optional[str] = None) -> FeatureSpec:
    """Convenience: directly get a FeatureSpec from model_meta.json."""
    return FeatureSpec.from_meta(load_meta(path))


# ────────────────────────────────────────────────────────────────────────────────
# CSV / OHLCV loading (robust) + chunk iterator expected by backtests
# ────────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return str(s).strip().lower()

_BINANCE_12 = {0: "ts", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"}

def _map_ohlcv_columns(df: pd.DataFrame, path: str) -> pd.DataFrame:
    cols = list(df.columns)
    norm = [_norm(c) for c in cols]
    out = df.copy()

    if {"open", "high", "low", "close", "volume"}.issubset(set(norm)):
        rename = {}
        for c in cols:
            n = _norm(c)
            if n in {"open", "high", "low", "close", "volume"}:
                rename[c] = n
            elif n in {"date", "time", "timestamp"}:
                rename[c] = "ts"
        out = out.rename(columns=rename)
    else:
        if len(cols) >= 6:
            rename = {c: _BINANCE_12.get(i, f"c{i}") for i, c in enumerate(cols)}
            out = out.rename(columns=rename)
        else:
            raise ValueError(f"Unrecognized CSV schema: {path}")

    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], unit="ms", errors="coerce").fillna(method="ffill")
    else:
        out["ts"] = pd.date_range(start=pd.Timestamp.utcnow(), periods=len(out), freq="min")

    out = out.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).set_index("ts").sort_index()
    return out[["open", "high", "low", "close", "volume"]]

def _read_csv_robust(path: str) -> pd.DataFrame:
    # Try headered
    try:
        df = pd.read_csv(path)
        if all(str(c).replace(".", "", 1).isdigit() for c in df.columns):
            raise ValueError("numeric headers -> headerless")
        return _map_ohlcv_columns(df, path)
    except Exception:
        pass
    # Headerless
    probe = pd.read_csv(path, header=None, nrows=1)
    n = probe.shape[1]
    if n >= 6:
        df = pd.read_csv(path, header=None)
        df.columns = list(range(n))
        return _map_ohlcv_columns(df, path)
    raise ValueError(f"Could not parse CSV: {path}")

def load_ohlc_chunks(path_or_glob: str, *, recursive: bool = True) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Yield (key, df) for each CSV under the given file/dir/glob.
    - key: a short identifier (filename without extension)
    - df : DataFrame indexed by datetime with columns open/high/low/close/volume
    """
    import glob as _glob
    p = Path(path_or_glob)
    files: List[str] = []

    if p.exists():
        if p.is_dir():
            pattern = str(p / ("**/*.csv" if recursive else "*.csv"))
            files = sorted(_glob.glob(pattern, recursive=recursive))
        else:
            files = [str(p)]
    else:
        files = sorted(_glob.glob(path_or_glob, recursive=True))

    for f in files:
        key = Path(f).stem
        df = _read_csv_robust(f)
        yield key, df


# ────────────────────────────────────────────────────────────────────────────────
# Feature engineering shared by train/backtest/inference
# ────────────────────────────────────────────────────────────────────────────────
def _ema(a: pd.Series, span: int) -> pd.Series:
    return a.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=period).mean()
    loss = down.rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def build_features(df: pd.DataFrame, compat_inf_to_zero: bool = False) -> pd.DataFrame:
    """
    Expect df columns: ['open','high','low','close','volume'] at minimum.
    Produces a superset that includes DEFAULT_FEATURE_COLS.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"build_features expected columns {sorted(required)}, missing {missing}")

    out = df.copy().sort_index()

    # Candle anatomy
    out["body"] = out["close"] - out["open"]
    out["range"] = out["high"] - out["low"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["return"] = out["close"].pct_change().fillna(0.0)

    # Trend & momentum
    sma20 = out["close"].rolling(20, min_periods=1).mean()
    out["sma_ratio"] = (out["close"] / (sma20 + 1e-12)).astype(np.float32)
    out["ema_20"] = _ema(out["close"], 20)

    ema12 = _ema(out["close"], 12)
    ema26 = _ema(out["close"], 26)
    out["macd"] = ema12 - ema26

    out["rsi_14"] = _rsi(out["close"], 14)

    # Volatility / volume
    out["vol_change"] = (
        out["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14, min_periods=1).mean()

    hourly = out["close"].rolling(60, min_periods=1).mean()
    out["price_vs_hourly_trend"] = out["close"] / (hourly + 1e-12)

    std20 = out["close"].rolling(20, min_periods=1).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_width"] = (upper - lower) / (sma20 + 1e-12)

    if compat_inf_to_zero:
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Ensure numeric dtype
    for c in out.columns:
        if c not in ("open", "high", "low", "close", "volume"):
            out[c] = out[c].astype("float32", copy=False)
    return out


def find_eth_1m_data() -> str:
    """
    Return the eth_1m_data directory that sits one level above /src.
    Only this path is considered.
    """
    p = Path(__file__).resolve().parent.parent / "eth_1m_data"
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(
            f"Expected ETH 1m data directory at: {p}\n"
            "Please ensure the folder exists or pass --data explicitly."
        )
    return str(p)