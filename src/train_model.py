#!/usr/bin/env python3
"""
Memory-safe LSTM trainer with streaming windows and triple-barrier labels.

Key points
- Uses centralized feature engineering from utils.compute_features (includes ADX).
- Reads existing model_meta.json (if present) to lock features/window/hidden sizes.
- Saves best and last checkpoints, scaler, and an updated model_meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from collections import deque
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import compute_features, normalize_headers
from models import (
    ModelMeta,
    TransformerClassifier,
    build_model_from_meta as build_model_from_meta_core,
)
from utils import FEATURE_COLUMNS, DEFAULT_SEQ_LENS
try:
    from streamer import KlineStreamer
except ModuleNotFoundError:  # when executed from src/ without repo root on sys.path
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from streamer import KlineStreamer
from logging_utils import logger, setup_logging

# Simple CSV lister reused from KlineStreamer
def _list_csvs(path: str) -> List[Path]:
    p = Path(path)
    if not p.exists():
        alt = Path(__file__).resolve().parent.parent / path
        if alt.exists():
            p = alt
    return KlineStreamer._list_csvs(p)


_RAW_SCHEMA_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
    "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
    "trade_count", "buy_count", "sell_count",
    "taker_buy_volume_base", "taker_sell_volume_base",
    "taker_buy_volume_quote", "taker_sell_volume_quote",
    "volume_quote",
]


def validate_raw_schema(files: Sequence[Union[str, Path]], sample_rows: int = 2000) -> Dict[str, List[str]]:
    """Validate sampled raw CSV schema and report missing raw columns per file."""
    missing_by_file: Dict[str, List[str]] = {}
    for fp in files:
        path = Path(fp)
        try:
            sample = pd.read_csv(path, nrows=sample_rows)
        except Exception as e:
            print(f"[schema] ERROR reading {path}: {e}")
            missing_by_file[str(path)] = list(_RAW_SCHEMA_COLUMNS)
            continue
        sample = normalize_headers(sample)
        missing = [c for c in _RAW_SCHEMA_COLUMNS if c not in sample.columns]
        if missing:
            print(f"[schema] {path} missing {len(missing)} raw cols: {missing}")
            missing_by_file[str(path)] = missing

    if not missing_by_file:
        print("[schema] All sampled files include required raw columns.")
    return missing_by_file


def _drop_features_for_missing_raw(
    desired_features: List[str],
    missing_raw_anywhere: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Drop features whose raw dependencies are missing in sampled files."""
    missing_set = set(missing_raw_anywhere)
    if not missing_set:
        return list(desired_features), []

    deps: Dict[str, set[str]] = {
        # raw passthrough features
        "best_bid": {"best_bid"},
        "best_ask": {"best_ask"},
        "bid_size_l1": {"bid_size_l1"},
        "ask_size_l1": {"ask_size_l1"},
        "bid_depth_5": {"bid_depth_5"},
        "ask_depth_5": {"ask_depth_5"},
        "bid_depth_10": {"bid_depth_10"},
        "ask_depth_10": {"ask_depth_10"},
        "bid_depth_20": {"bid_depth_20"},
        "ask_depth_20": {"ask_depth_20"},
        "vwap_bid_5": {"vwap_bid_5"},
        "vwap_ask_5": {"vwap_ask_5"},
        "vwap_bid_10": {"vwap_bid_10"},
        "vwap_ask_10": {"vwap_ask_10"},
        "vwap_bid_20": {"vwap_bid_20"},
        "vwap_ask_20": {"vwap_ask_20"},
        "trade_count": {"trade_count"},
        "buy_count": {"buy_count"},
        "sell_count": {"sell_count"},
        "taker_buy_volume_base": {"taker_buy_volume_base"},
        "taker_sell_volume_base": {"taker_sell_volume_base"},
        "taker_buy_volume_quote": {"taker_buy_volume_quote"},
        "taker_sell_volume_quote": {"taker_sell_volume_quote"},
        "volume_quote": {"volume_quote"},
        # derived microstructure features
        "mid": {"best_bid", "best_ask"},
        "spread_abs": {"best_bid", "best_ask"},
        "spread_pct": {"best_bid", "best_ask"},
        "microprice": {"best_bid", "best_ask", "bid_size_l1", "ask_size_l1"},
        "l1_imbalance": {"bid_size_l1", "ask_size_l1"},
        "mid_log_ret": {"best_bid", "best_ask"},
        "spread_z_60": {"best_bid", "best_ask"},
        "l2_imbalance_5": {"bid_depth_5", "ask_depth_5"},
        "l2_imbalance_10": {"bid_depth_10", "ask_depth_10"},
        "l2_imbalance_20": {"bid_depth_20", "ask_depth_20"},
        "depth_ratio_5": {"bid_depth_5", "ask_depth_5"},
        "depth_ratio_10": {"bid_depth_10", "ask_depth_10"},
        "depth_ratio_20": {"bid_depth_20", "ask_depth_20"},
        "book_pressure_5": {"bid_depth_5", "ask_depth_5"},
        "total_taker_vol_base": {"taker_buy_volume_base", "taker_sell_volume_base"},
        "ofi_base": {"taker_buy_volume_base", "taker_sell_volume_base"},
        "ofi_ratio": {"taker_buy_volume_base", "taker_sell_volume_base"},
        "buy_sell_count_imb": {"buy_count", "sell_count"},
        "avg_trade_size_base": {"trade_count", "volume"},
        "avg_trade_size_quote": {"trade_count", "volume_quote"},
        "ofi_over_depth_10": {"taker_buy_volume_base", "taker_sell_volume_base", "bid_depth_10", "ask_depth_10"},
        "spread_times_imbalance": {"best_bid", "best_ask", "bid_size_l1", "ask_size_l1"},
    }

    dropped = [f for f in desired_features if f in deps and not deps[f].isdisjoint(missing_set)]
    filtered = [f for f in desired_features if f not in dropped]
    return filtered, dropped


# ----------------------------
# Repro & Device
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def get_device(preferred: Optional[str] = None, force_cpu_on_mac: bool = True) -> torch.device:
    pref = (preferred or "").strip().lower()
    is_mac = platform.system() == "Darwin"

    if pref:
        if pref == "cuda":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if pref == "mps":
            if is_mac and force_cpu_on_mac:
                return torch.device("cpu")
            return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and not (is_mac and force_cpu_on_mac):
        return torch.device("mps")
    return torch.device("cpu")


def log_epoch_summary(
    *,
    fold_idx: int,
    total_folds: int,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float],
    val_acc: float,
    macro_f1: float,
    class_names: List[str],
    precisions: List[float],
    recalls: List[float],
    pred_dist: Dict[int, float],
    expected_value: Optional[float] = None,
) -> None:
    """Log a compact epoch summary with per-class diagnostics."""
    precision_lines = [f"  {name}: {float(value):.4f}" for name, value in zip(class_names, precisions)]
    recall_lines = [f"  {name}: {float(value):.4f}" for name, value in zip(class_names, recalls)]
    pred_lines = [f"  {name}: {float(pred_dist.get(i, 0.0)) * 100.0:.4f}%" for i, name in enumerate(class_names)]

    lines = [
        "",
        "=" * 60,
        f"FOLD {fold_idx}/{total_folds} | EPOCH {epoch}/{total_epochs}",
        "-" * 60,
        f"Train Loss: {float(train_loss):.4f}",
        f"Val Loss:   {float(val_loss):.4f}" if val_loss is not None else "Val Loss:   n/a",
        f"Val Acc:    {float(val_acc):.4f}",
        f"Macro F1:   {float(macro_f1):.4f}",
        "",
        "Precision:",
        *precision_lines,
        "",
        "Recall:",
        *recall_lines,
        "",
        "Prediction Distribution:",
        *pred_lines,
        "",
    ]
    if expected_value is not None:
        lines.append(f"Expected Value per Trade: {float(expected_value):.4f}")
    lines.append("=" * 60)
    logger.info("\n".join(lines))


def compute_expected_value(
    *,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    take_profit: float,
    stop_loss: float,
    trading_fee: float,
) -> float:
    """Estimate expected value per executed trade from triple-barrier outcomes.

    Class mapping is assumed to be:
    - 0: short_win
    - 1: timeout
    - 2: long_win
    """
    preds = np.asarray(predictions, dtype=np.int64).reshape(-1)
    targs = np.asarray(targets, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities) if probabilities is not None else np.zeros((0, 0), dtype=np.float32)

    if preds.size == 0 or targs.size == 0:
        return 0.0

    n = min(preds.size, targs.size)
    preds = preds[:n]
    targs = targs[:n]
    if probs.ndim >= 2 and probs.shape[0] >= n:
        probs = probs[:n]

    # Execute only directional signals (short/long), skip timeout predictions.
    exec_mask = (preds == 0) | (preds == 2)
    if not np.any(exec_mask):
        return 0.0

    p = preds[exec_mask]
    y = targs[exec_mask]
    gross = np.zeros_like(p, dtype=np.float64)

    long_mask = p == 2
    short_mask = p == 0

    # Long trade outcomes
    gross[long_mask & (y == 2)] = float(take_profit)
    gross[long_mask & (y == 0)] = -float(stop_loss)
    # Short trade outcomes
    gross[short_mask & (y == 0)] = float(take_profit)
    gross[short_mask & (y == 2)] = -float(stop_loss)
    # Timeout outcomes remain 0.0 gross

    net = gross - float(trading_fee)
    return float(np.mean(net)) if net.size else 0.0


# ----------------------------
# Feature Names (single source of truth)
# ----------------------------
ALL_FEATURES = FEATURE_COLUMNS



def _ensure_meta(meta: Union[ModelMeta, Dict[str, Any]]) -> ModelMeta:
    if isinstance(meta, ModelMeta):
        return meta
    return ModelMeta.from_dict(meta)


def build_model_from_meta(meta: Union[ModelMeta, Dict[str, Any]]) -> nn.Module:
    meta_obj = _ensure_meta(meta)
    model_type = str(getattr(meta_obj, "model_type", "lstm_classifier")).lower()
    if model_type in {"transformer", "transformer_classifier"}:
        num_heads = getattr(meta_obj, "num_heads", 4) or 4
        return TransformerClassifier(
            input_size=meta_obj.input_size,
            hidden_size=meta_obj.hidden_size,
            num_layers=meta_obj.num_layers,
            num_heads=int(num_heads),
            dropout=meta_obj.dropout,
            num_classes=meta_obj.num_classes,
        )
    return build_model_from_meta_core(meta_obj)


# ----------------------------
# Triple barrier labels
# ----------------------------
def apply_triple_barrier_labels_dynamic(
    df: pd.DataFrame,
    price_col: str,
    atr_col: str,
    fee_pct: float,
    slippage_pct: float,
    cost_mult: float,
    k_tp: float,
    k_sl: float,
    time_limit: int,
    min_atrp: float = 0.0005,
    min_barrier_pct: float = 0.0,
) -> np.ndarray:
    """Dynamic triple-barrier labels using ATR% + costs, with conservative ordering.

    Labels: {-1,0,1} => {short_win, timeout, long_win}
    Conservative rule: within each future bar, check SL before TP.
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}' in DataFrame")
    for c in ("high", "low"):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' for triple barrier labels")

    # Ensure ATR exists (no lookahead)
    if atr_col not in df.columns:
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df[price_col].astype(float)
        tr = (highs - lows).abs().combine(
            (highs - closes.shift(1)).abs(), max
        ).combine(
            (lows - closes.shift(1)).abs(), max
        )
        df[atr_col] = tr.rolling(14, min_periods=1).mean()

    n = len(df)
    prices = df[price_col].to_numpy(dtype=float, copy=False)
    highs = df["high"].to_numpy(dtype=float, copy=False)
    lows = df["low"].to_numpy(dtype=float, copy=False)
    atr = df[atr_col].to_numpy(dtype=float, copy=False)

    labels = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels

    time_limit = max(1, int(time_limit))
    round_trip_cost = 2.0 * (float(fee_pct) + float(slippage_pct))
    minimum_edge = float(cost_mult) * round_trip_cost

    for i in range(n):
        entry = prices[i]
        if not np.isfinite(entry) or entry <= 0:
            labels[i] = 0
            continue
        atrp = atr[i] / entry if np.isfinite(atr[i]) and atr[i] > 0 else 0.0
        tp_pct_i = max(minimum_edge, float(k_tp) * atrp)
        sl_pct_i = max(minimum_edge, float(k_sl) * atrp)
        # Widen/no-trade rule for low-volatility bars or too-tight barriers
        if (atrp < float(min_atrp)) or (tp_pct_i <= minimum_edge + float(min_barrier_pct)):
            labels[i] = 0
            continue
        tp_price = entry * (1.0 + tp_pct_i)
        sl_price = entry * (1.0 - sl_pct_i)

        out = 0
        max_j = min(time_limit, n - i - 1)
        for j in range(1, max_j + 1):
            hi = highs[i + j]
            lo = lows[i + j]
            # Conservative: SL first if both touched in same bar
            if lo <= sl_price:
                out = -1
                break
            if hi >= tp_price:
                out = 1
                break
        labels[i] = out

    return labels


def compute_dynamic_barrier_pcts(
    df: pd.DataFrame,
    price_col: str,
    atr_col: str,
    fee_pct: float,
    slippage_pct: float,
    cost_mult: float,
    k_tp: float,
    k_sl: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-row tp/sl pct based on ATR% and round-trip cost."""
    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}' in DataFrame")
    if atr_col not in df.columns:
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df[price_col].astype(float)
        tr = (highs - lows).abs().combine(
            (highs - closes.shift(1)).abs(), max
        ).combine(
            (lows - closes.shift(1)).abs(), max
        )
        df[atr_col] = tr.rolling(14, min_periods=1).mean()

    prices = df[price_col].to_numpy(dtype=float, copy=False)
    atr = df[atr_col].to_numpy(dtype=float, copy=False)
    round_trip_cost = 2.0 * (float(fee_pct) + float(slippage_pct))
    minimum_edge = float(cost_mult) * round_trip_cost

    tp_pct = np.zeros_like(prices, dtype=float)
    sl_pct = np.zeros_like(prices, dtype=float)
    for i in range(len(prices)):
        entry = prices[i]
        if not np.isfinite(entry) or entry <= 0:
            tp_pct[i] = minimum_edge
            sl_pct[i] = minimum_edge
            continue
        atrp = atr[i] / entry if np.isfinite(atr[i]) and atr[i] > 0 else 0.0
        tp_pct[i] = max(minimum_edge, float(k_tp) * atrp)
        sl_pct[i] = max(minimum_edge, float(k_sl) * atrp)
    return tp_pct, sl_pct


# ----------------------------
# Streaming window dataset
# ----------------------------
class StreamWindowDataset(IterableDataset):
    """Yields (window, label) pairs lazily from a stream of feature frames.

    We keep a deque of the last `window_size` feature rows to form the next sample.
    """

    def __init__(
        self,
        files: List[Path],
        feature_cols: List[str],
        price_col: str,
        window_size: int,
        atr_col: str,
        fee_pct: float,
        slippage_pct: float,
        cost_mult: float,
        k_tp: float,
        k_sl: float,
        tp_pct: float,
        sl_pct: float,
        time_limit: int,
        min_atrp: float,
        min_barrier_pct: float,
        chunksize: int = 500_000,
        overlap: int = 256,
        scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.files = files
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.atr_col = atr_col
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.cost_mult = float(cost_mult)
        self.k_tp = float(k_tp)
        self.k_sl = float(k_sl)
        self.window = int(window_size)
        self.tp_pct = float(tp_pct)
        self.sl_pct = float(sl_pct)
        self.time_limit = int(time_limit)
        self.min_atrp = float(min_atrp)
        self.min_barrier_pct = float(min_barrier_pct)
        self.chunksize = int(chunksize)
        self.overlap = max(int(overlap), self.window + self.time_limit + 1)
        self.scaler = scaler

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files = self.files
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = self.files[worker_id::num_workers]
        print(f"[data] worker_id={worker_id} num_workers={num_workers} files={len(files)}")

        def _label_fn(df: pd.DataFrame):
            labels_raw = apply_triple_barrier_labels_dynamic(
                df,
                price_col=self.price_col,
                atr_col=self.atr_col,
                fee_pct=self.fee_pct,
                slippage_pct=self.slippage_pct,
                cost_mult=self.cost_mult,
                k_tp=self.k_tp,
                k_sl=self.k_sl,
                time_limit=self.time_limit,
                min_atrp=self.min_atrp,
                min_barrier_pct=self.min_barrier_pct,
            )
            return (labels_raw + 1).astype(np.int64, copy=False)

        streamer = KlineStreamer(
            files,
            window_size=self.window,
            feature_cols=self.feature_cols,
            chunksize=self.chunksize,
            overlap_rows=self.overlap,
            label_fn=_label_fn,
        )

        for window, label, _meta in streamer:
            if label is None:
                continue
            Xw = window
            if self.scaler is not None:
                Xw = self.scaler.transform(Xw).astype(np.float32, copy=False)
                Xw = np.clip(Xw, -10.0, 10.0)
            yield torch.from_numpy(Xw).float(), torch.tensor(int(label), dtype=torch.long)


class StreamWindowDatasetReg(IterableDataset):
    """Streaming windows for regression.

    Label is the percentage return over the prediction horizon.
    """
    def __init__(
        self,
        files: List[Path],
        feature_cols: List[str],
        price_col: str,
        window_size: int,
        horizon_bars: int = 3,
        chunksize: int = 500_000,
        overlap: int = 256,
        scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.files = files
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.window = int(window_size)
        self.horizon = int(max(1, horizon_bars))
        self.chunksize = int(chunksize)
        self.overlap = max(int(overlap), self.window + self.horizon + 1)
        self.scaler = scaler

    def __iter__(self):
        if self.scaler is None:
            raise ValueError("Scaler must be provided and pre-fitted for regression streaming dataset")

        worker_info = get_worker_info()
        if worker_info is None:
            files = self.files
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = self.files[worker_id::num_workers]
        print(f"[data] worker_id={worker_id} num_workers={num_workers} files={len(files)}")

        def _label_fn(df: pd.DataFrame):
            prices = df[self.price_col].to_numpy(dtype=np.float32, copy=False)
            labels = np.full(len(df), np.nan, dtype=np.float32)
            for i in range(len(df)):
                j = i + self.horizon
                if j >= len(df):
                    continue
                base_price = float(prices[i])
                future_price = float(prices[j])
                if abs(base_price) < 1e-8:
                    continue
                labels[i] = (future_price / base_price) - 1.0
            return labels

        streamer = KlineStreamer(
            files,
            window_size=self.window,
            feature_cols=self.feature_cols,
            chunksize=self.chunksize,
            overlap_rows=self.overlap,
            label_fn=_label_fn,
        )

        for window, label, _meta in streamer:
            if label is None or not np.isfinite(label):
                continue
            Xw = self.scaler.transform(window).astype(np.float32, copy=False)
            Xw = np.clip(Xw, -10.0, 10.0)
            yield torch.from_numpy(Xw).float(), torch.tensor(float(label), dtype=torch.float32)


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    data_path: str
    output_dir: str
    meta_path: str
    window_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_frac: float
    accumulate: int
    seed: int
    price_col: str
    tp_pct: float
    sl_pct: float
    time_limit: int
    fee_pct: float
    slippage_pct: float
    cost_mult: float
    k_tp: float
    k_sl: float
    atr_col: str
    amp: bool
    workers: int
    chunksize: int
    task: str
    horizon: int
    model_type: Optional[str] = None
    max_folds: Optional[int] = None
    feature_cols: Optional[List[str]] = None
    disable_scaling: bool = False
    device: str = "auto"
    force_cpu_on_mac: bool = True
    grad_clip_norm: Optional[float] = None
    detect_anomaly: bool = False
    balanced_val_eval: bool = False
    balanced_val_samples: int = 50_000
    early_stop_patience: int = 10
    early_stop_min_epochs: int = 5
    early_stop_metric: str = "macro_f1"
    min_atrp: float = 0.0005
    min_barrier_pct: float = 0.0
    # Loss/Calibration
    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    calibrate_temp: bool = True
    optimize_thresholds: bool = True


# ----------------------------
# Training
# ----------------------------
def _split_stream(files: List[Path], val_frac: float) -> Tuple[List[Path], List[Path]]:
    if len(files) == 1:
        return files, files
    k = max(1, int(round(len(files) * (1.0 - val_frac))))
    return files[:k], files[k:]

def fit_scaler(train_files, feature_cols, chunksize=500_000, max_rows=2_000_000):
    scaler = StandardScaler()
    samples = []
    total_rows = 0
    for f in train_files:
        for chunk in pd.read_csv(f, chunksize=chunksize):
            chunk = normalize_headers(chunk)
            chunk = compute_features(chunk)
            # Forward-fill NaNs for time-series data (no bfill to avoid leakage)
            chunk[feature_cols] = chunk[feature_cols].ffill().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            sample = chunk[feature_cols].astype(np.float32, copy=False).to_numpy()
            samples.append(sample)
            total_rows += len(sample)
            if total_rows >= max_rows:
                break
        if total_rows >= max_rows:
            break
    if not samples:
        raise ValueError("No valid samples for scaler fitting")
    all_samples = np.vstack(samples)[:max_rows]
    scaler.fit(all_samples)
    # Debug output
    print(f"Total rows processed: {total_rows}")
    print("Valid features:", feature_cols)

    return scaler, feature_cols


def train(cfg: TrainConfig):
    setup_logging(level="INFO", serialize=False, patch_print=False)
    if "seq_60" in str(cfg.output_dir) and cfg.window_size < 30:
        raise ValueError(
            f"Config mismatch: output_dir implies seq_60 but window_size={cfg.window_size}. "
            "Set window_size=60 (or update output_dir)."
        )
    start_time = time.time()
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = get_device(getattr(cfg, "device", None), getattr(cfg, "force_cpu_on_mac", True))
    if device.type == "mps":
        print("[device] MPS selected. AMP will be disabled for stability.")
    # AMP is only reliable/beneficial on CUDA; disable for MPS/CPU to avoid NaNs.
    amp_enabled = bool(getattr(cfg, "amp", False) and device.type == "cuda")
    if getattr(cfg, "grad_clip_norm", None) is None:
        grad_clip_norm = 0.5 if device.type == "cuda" else 0.25
    else:
        grad_clip_norm = float(cfg.grad_clip_norm)

    files = _list_csvs(cfg.data_path)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under data_path={cfg.data_path}. "
            "Check the path and naming pattern."
        )

    def _count_rows_fast(path: str, max_rows: int = 2000) -> int:
        # Read a small chunk to confirm file isn't empty / malformed
        try:
            df = pd.read_csv(path, nrows=max_rows)
            return len(df)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {path} error={e}")

    sample_paths = [files[0], files[-1]] if len(files) > 1 else [files[0]]
    for p in sample_paths:
        n = _count_rows_fast(str(p))
        if n < (cfg.window_size + 5):
            raise RuntimeError(
                f"CSV too short for windowing: {p} rows_read={n} "
                f"required_min={cfg.window_size + 5}"
            )

    schema_missing = validate_raw_schema(files)
    missing_raw_anywhere = sorted({col for cols in schema_missing.values() for col in cols})

    abs_files = [str(p.resolve()) for p in files]
    print("[data] file_paths:")
    for p in abs_files:
        print(f"  {p}")
    print(f"[data] file_count={len(abs_files)}")
    print(f"[data] first3={[Path(p).name for p in abs_files[:3]]}")
    print(f"[data] last3={[Path(p).name for p in abs_files[-3:]]}")

    # Load meta to lock features/window/model dims (if present)
    meta_existing = {}
    meta_path = Path(cfg.meta_path)
    if meta_path.exists():
        try:
            meta_existing = json.loads(meta_path.read_text())
        except Exception:
            meta_existing = {}
    if meta_existing.get("file_manifest"):
        if list(meta_existing.get("file_manifest")) != abs_files:
            raise ValueError("File manifest mismatch vs model_meta.json; refusing to resume with different data files.")

    model_type_override = None
    if getattr(cfg, "model_type", None):
        model_type_override = str(cfg.model_type).strip().lower()
    meta_model_type = str(meta_existing.get("model_type", "")).strip().lower() or None
    selected_model_type = model_type_override or meta_model_type or "transformer"

    # Determine feature set
    if cfg.feature_cols:
        desired_features = list(cfg.feature_cols)
    else:
        desired_features = list(meta_existing.get("feature_cols", ALL_FEATURES))
    desired_features, dropped_features = _drop_features_for_missing_raw(desired_features, missing_raw_anywhere)
    if dropped_features:
        print(
            f"[schema] Dropping {len(dropped_features)} features due to missing raw inputs "
            f"in sampled files: {dropped_features[:20]}"
        )
    if "desired_features" not in locals() or not desired_features:
        raise ValueError("desired_features is empty; cannot determine feature set")

    # Build peek streamer without feature filtering so we can discover all engineered columns.
    streamer_peek = KlineStreamer(
        files,
        window_size=cfg.window_size,
        feature_cols=None,  # IMPORTANT: let streamer emit all engineered columns
        chunksize=min(cfg.chunksize, 200_000),
        overlap_rows=cfg.window_size + 5,
    )
    peek = None
    for _frame in streamer_peek._iter_feature_frames():
        peek = _frame
        break

    if peek is None:
        raise RuntimeError(
            "KlineStreamer produced zero feature frames during peek. "
            f"files={len(files)} window_size={cfg.window_size} chunksize={cfg.chunksize} "
            f"feature_cols={getattr(streamer_peek, 'feature_cols', None)}. "
            "Likely causes: empty/invalid CSVs, too few rows for window, parsing failure, "
            "or streamer filtering dropped all rows."
        )
    available = set(peek.columns.tolist())
    feature_cols = [c for c in desired_features if c in available]
    if not feature_cols:
        raise RuntimeError(
            "After peeking, zero feature_cols matched available columns. "
            f"desired_features={desired_features[:30]}... "
            f"available={sorted(list(available))[:30]}..."
        )
    # Ensure ADX is part of training features if available
    if "adx" in available and "adx" not in feature_cols:
        feature_cols.append("adx")
    if len(feature_cols) < 4:
        raise ValueError(f"Too few features after engineering. Wanted={desired_features}, available={sorted(available)}")
    assert isinstance(feature_cols, list) and len(feature_cols) > 0, "feature_cols must be a non-empty list"

    # Feature health diagnostics (concise)
    try:
        raw_feat = peek[feature_cols].replace([np.inf, -np.inf], np.nan)
        nan_ratio = raw_feat.isna().mean().sort_values(ascending=False)
        filled = raw_feat.ffill().fillna(0.0)
        zero_ratio = (filled == 0).mean().sort_values(ascending=False)
        stats = filled.agg(["min", "max", "mean", "std"])

        top_nan = nan_ratio.head(20).index.tolist()
        top_zero = zero_ratio.head(20).index.tolist()

        def _print_table(title: str, cols: List[str]):
            rows = []
            for c in cols:
                rows.append(
                    (
                        c,
                        float(nan_ratio.get(c, 0.0)),
                        float(zero_ratio.get(c, 0.0)),
                        float(stats.at["mean", c]),
                        float(stats.at["std", c]),
                        float(stats.at["min", c]),
                        float(stats.at["max", c]),
                    )
                )
            print(title)
            print("feature | nan% | zero% | mean | std | min | max")
            for r in rows:
                print(
                    f"{r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.4g} | {r[4]:.4g} | {r[5]:.4g} | {r[6]:.4g}"
                )

        _print_table("[sanity] Top 20 NaN ratio features:", top_nan)
        _print_table("[sanity] Top 20 Zero ratio features:", top_zero)
    except Exception as e:
        print(f"[sanity] Feature health diagnostics failed: {e}")

    # Walk-forward folds: train on months 1..k, validate on month k+1
    folds: List[Tuple[List[Path], List[Path]]] = []
    if len(files) >= 2:
        for k in range(1, len(files)):
            folds.append((files[:k], [files[k]]))
    else:
        folds.append((files, files))

    if cfg.max_folds is not None:
        max_folds = max(1, int(cfg.max_folds))
        folds = folds[:max_folds]

    from sklearn.preprocessing import StandardScaler

    fold_logs: List[dict] = []
    best_thresholds: Optional[dict] = None
    last_fold_best_state = None
    last_fold_last_state = None
    last_fold_scaler = None
    scaler: Optional[StandardScaler] = None
    temperature_final: float = 1.0
    for fold_idx, (train_files, val_files) in enumerate(folds, start=1):
        if getattr(cfg, "detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
        best_thresholds_fold = None
        fold_outdir = Path(cfg.output_dir) / f"fold_{fold_idx}"
        fold_outdir.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_outdir / "metrics.json"
        epoch_metrics: List[dict] = []
        balanced_eval_indices = None
        best_val_acc = 0.0
        best_macro_f1 = 0.0
        best_ev = 0.0
        best_epoch_for_fold = 0

        def seed_worker(worker_id: int):
            seed = int(cfg.seed) + int(worker_id) + (int(fold_idx) * 1000)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        g = torch.Generator()
        g.manual_seed(int(cfg.seed) + int(fold_idx))

        def collate_batch(batch):
            xb, yb = zip(*batch)  # xb: [B,T,F]
            xb = torch.stack(list(xb), dim=0)
            if isinstance(yb[0], torch.Tensor):
                yb = torch.stack(list(yb), dim=0)
            else:
                yb = torch.tensor(yb)
            return xb, yb

        assert isinstance(feature_cols, list) and len(feature_cols) > 0, "feature_cols must be a non-empty list"
        if getattr(cfg, 'task', 'classification') == 'regression':
            scaler, valid_cols = fit_scaler(train_files, feature_cols, chunksize=cfg.chunksize)
            if valid_cols != feature_cols:
                raise ValueError(f"Scaler fitted on subset of features; mismatch: expected {feature_cols}, got {valid_cols}")
            train_ds = StreamWindowDatasetReg(
                train_files, feature_cols, cfg.price_col, cfg.window_size,
                horizon_bars=getattr(cfg, 'horizon', 3), chunksize=cfg.chunksize,
                scaler=scaler
            )
            val_ds = StreamWindowDatasetReg(
                val_files, feature_cols, cfg.price_col, cfg.window_size,
                horizon_bars=getattr(cfg, 'horizon', 3), chunksize=cfg.chunksize,
                scaler=scaler
            )
        else:
            scaler = None
            if not getattr(cfg, "disable_scaling", False):
                scaler, valid_cols = fit_scaler(train_files, feature_cols, chunksize=cfg.chunksize)
                if valid_cols != feature_cols:
                    raise ValueError(f"Scaler fitted on subset of features; mismatch: expected {feature_cols}, got {valid_cols}")
            train_ds = StreamWindowDataset(
                train_files, feature_cols, cfg.price_col, cfg.window_size,
                cfg.atr_col, cfg.fee_pct, cfg.slippage_pct, cfg.cost_mult, cfg.k_tp, cfg.k_sl,
                cfg.tp_pct, cfg.sl_pct, cfg.time_limit, cfg.min_atrp, cfg.min_barrier_pct,
                chunksize=cfg.chunksize,
                scaler=scaler,
            )
            val_ds = StreamWindowDataset(
                val_files, feature_cols, cfg.price_col, cfg.window_size,
                cfg.atr_col, cfg.fee_pct, cfg.slippage_pct, cfg.cost_mult, cfg.k_tp, cfg.k_sl,
                cfg.tp_pct, cfg.sl_pct, cfg.time_limit, cfg.min_atrp, cfg.min_barrier_pct,
                chunksize=cfg.chunksize,
                scaler=scaler,
            )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, num_workers=cfg.workers,
            pin_memory=(device.type == "cuda"), collate_fn=collate_batch,
            worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, num_workers=max(0, cfg.workers // 2),
            pin_memory=(device.type == "cuda"), collate_fn=collate_batch,
            worker_init_fn=seed_worker, generator=g,
        )
        # Estimate label distribution from actual streamed windows (not raw peek)
        if getattr(cfg, "task", "classification") != "regression":
            def _estimate_dist(loader, max_batches=500):
                counts = np.zeros(3, dtype=np.int64)
                seen = 0
                for b_idx, (_, yb) in enumerate(loader):
                    if b_idx >= max_batches:
                        break
                    y_np = yb.detach().cpu().numpy().astype(np.int64).ravel()
                    for i in range(3):
                        counts[i] += int((y_np == i).sum())
                    seen += len(y_np)
                pct = counts / max(1, counts.sum())
                return counts, pct, seen

            train_counts_est, train_pct_est, train_seen = _estimate_dist(train_loader, max_batches=500)
            val_counts_est, val_pct_est, val_seen = _estimate_dist(val_loader, max_batches=500)
            class_names = ["short_win", "timeout", "long_win"]
            print(
                f"[label_dist_est] train_samples={train_seen} "
                f"{dict(zip(class_names, train_counts_est.tolist()))} "
                f"({dict(zip(class_names, np.round(train_pct_est, 4).tolist()))})"
            )
            print(
                f"[label_dist_est] val_samples={val_seen} "
                f"{dict(zip(class_names, val_counts_est.tolist()))} "
                f"({dict(zip(class_names, np.round(val_pct_est, 4).tolist()))})"
            )

        # Build model from meta helper
        model_meta_dict = dict(meta_existing)
        model_meta_dict.update({
            "input_size": len(feature_cols),
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "bidirectional": cfg.bidirectional,
            "num_classes": (1 if getattr(cfg, 'task', 'classification') == 'regression' else 3),
            "model_type": selected_model_type,
        })
        if selected_model_type in {"transformer", "transformer_classifier"}:
            model_meta_dict.setdefault("num_heads", meta_existing.get("num_heads", 4))
        else:
            model_meta_dict.pop("num_heads", None)
        model = build_model_from_meta(model_meta_dict).to(device)
        # Class weights + train label distribution (classification only)
        class_weights = None
        train_label_counts = None
        if getattr(cfg, 'task', 'classification') != 'regression' and cfg.use_class_weights:
            # Use label distribution from actual streamed windows (estimate)
            counts = train_counts_est.astype(np.float32) if 'train_counts_est' in locals() else None
            if counts is None or counts.sum() <= 0:
                counts = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / np.sqrt(counts)
            cw = (inv / inv.sum()) * 3.0
            class_weights = torch.tensor(cw, device=device, dtype=torch.float32)
            train_label_counts = counts.astype(int)

        # Loss: focal or cross entropy
        class FocalLoss(nn.Module):
            def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
                super().__init__()
                self.gamma = gamma
                self.weight = weight
            def forward(self, logits, target):
                ce = nn.functional.cross_entropy(logits, target, reduction='none', weight=self.weight)
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** self.gamma) * ce
                return loss.mean()

        if getattr(cfg, 'task', 'classification') == 'regression':
            # Huber loss (SmoothL1) with tighter delta to reduce sensitivity to outliers
            criterion = nn.HuberLoss(delta=0.1)
        else:
            if cfg.use_focal_loss:
                criterion = FocalLoss(gamma=float(cfg.focal_gamma), weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        # Cosine annealing with warm restarts (no plateau dependency)
        t0 = max(5, cfg.epochs // 4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=1, eta_min=cfg.learning_rate * 0.1)
        # Use CUDA AMP scaler (torch.amp.GradScaler unavailable in older wheels)
        scaler_obj = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val = -1.0
        best_state = None
        best_mae = float("inf")
        best_directional = 0.0
        collapse_flat_epochs = 0
        collapse_best_val_acc = -float("inf")
        early_stop_counter = 0
        early_stop_patience = int(getattr(cfg, "early_stop_patience", 10))
        early_stop_min_epochs = int(getattr(cfg, "early_stop_min_epochs", 5))
        best_epoch = 0
        long_recall_bad_streak = 0
        warmup_steps = max(50, int(cfg.batch_size))
        global_step = 0
        steps_per_epoch = None

        for epoch in range(1, cfg.epochs + 1):
            epoch_start_time = time.time()
            model.train()
            running = 0.0
            step = 0
            opt_steps = 0
            skipped_nonfinite_train = 0
            nan_skip_count = 0
            nan_skip_limit = 20
            printed_input_stats = False
            optimizer.zero_grad(set_to_none=True)

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                finite_mask = torch.isfinite(xb).all(dim=(1, 2))
                if torch.is_floating_point(yb):
                    finite_mask = finite_mask & torch.isfinite(yb)
                if not finite_mask.all():
                    skipped_nonfinite_train += int((~finite_mask).sum().item())
                    if not finite_mask.any():
                        continue
                    xb = xb[finite_mask]
                    yb = yb[finite_mask]
                if not printed_input_stats:
                    with torch.no_grad():
                        xb_stats = xb.detach()
                        zero_pct = (xb_stats == 0).float().mean().item()
                        mean_val = xb_stats.mean().item()
                        std_val = xb_stats.std().item()
                    logger.info(f"  input_stats: mean={mean_val:.4f} std={std_val:.4f} zero_pct={zero_pct:.4f}")
                    printed_input_stats = True
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(xb)
                    if getattr(cfg, 'task', 'classification') == 'regression':
                        logits = logits.squeeze(-1)

                # Compute loss in FP32 for numerical stability (avoids FP16 overflow in CE)
                if getattr(cfg, 'task', 'classification') == 'regression':
                    loss = criterion(logits.float(), yb.float())
                else:
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                    logits = logits.clamp(-20.0, 20.0)
                    loss = criterion(logits.float(), yb)
                if not torch.isfinite(loss):
                    with torch.no_grad():
                        xb_stats = xb.detach()
                        log_stats = logits.detach()
                        logger.warning(
                            f"[nan] fold={fold_idx} epoch={epoch} step={step} "
                            f"xb(min={xb_stats.min().item():.4g}, max={xb_stats.max().item():.4g}, "
                            f"mean={xb_stats.mean().item():.4g}, std={xb_stats.std().item():.4g}) "
                            f"logits(min={log_stats.min().item():.4g}, max={log_stats.max().item():.4g}, "
                            f"mean={log_stats.mean().item():.4g}, std={log_stats.std().item():.4g})"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    # Reduce LR on NaN/Inf to recover
                    for pg in optimizer.param_groups:
                        pg["lr"] = max(pg["lr"] * 0.5, 1e-7)
                    nan_skip_count += 1
                    if nan_skip_count > nan_skip_limit:
                        raise FloatingPointError(
                            f"Non-finite loss persisted (nan_skip_count={nan_skip_count}) "
                            f"at fold={fold_idx} epoch={epoch}."
                        )
                    continue
                if scaler_obj.is_enabled():
                    scaler_obj.scale(loss / cfg.accumulate).backward()
                else:
                    (loss / cfg.accumulate).backward()

                if (step + 1) % cfg.accumulate == 0:
                    if scaler_obj.is_enabled():
                        scaler_obj.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                        scaler_obj.step(optimizer)
                        scaler_obj.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    # Warmup then cosine schedule (avoid len() on IterableDataset)
                    opt_steps += 1
                    global_step += 1
                    if global_step <= warmup_steps:
                        warmup_lr = cfg.learning_rate * (global_step / max(1, warmup_steps))
                        for pg in optimizer.param_groups:
                            pg["lr"] = warmup_lr
                    elif steps_per_epoch is not None:
                        progress = (epoch - 1) + (opt_steps / max(1, steps_per_epoch))
                        scheduler.step(progress)
                    

                running += loss.item()
                step += 1

            # Flush remainder grads if batches not divisible by accumulate
            if step % cfg.accumulate != 0:
                if scaler_obj.is_enabled():
                    scaler_obj.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler_obj.step(optimizer)
                    scaler_obj.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                global_step += 1
                if global_step <= warmup_steps:
                    warmup_lr = cfg.learning_rate * (global_step / max(1, warmup_steps))
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                elif steps_per_epoch is not None:
                    progress = (epoch - 1) + (opt_steps / max(1, steps_per_epoch))
                    scheduler.step(progress)

            # Set steps_per_epoch after first pass (IterableDataset has no __len__)
            if steps_per_epoch is None:
                steps_per_epoch = max(1, opt_steps)
            train_loss_epoch = running / max(1, step)
            if not np.isfinite(train_loss_epoch):
                logger.warning("⚠ WARNING: Model Collapse Detected")
                logger.warning("Reason: nan loss")
                epoch_duration = time.time() - epoch_start_time
                logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
                break

            # Validation
            model.eval()
            if getattr(cfg, 'task', 'classification') == 'regression':
                abs_err_sum = 0.0
                mape_sum = 0.0
                base_abs_err_sum = 0.0
                base_mape_sum = 0.0
                base_dir_matches = 0
                directional_matches = 0
                total = 0
                skipped_nonfinite_val = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True).float()
                        finite_mask = torch.isfinite(xb).all(dim=(1, 2))
                        if torch.is_floating_point(yb):
                            finite_mask = finite_mask & torch.isfinite(yb)
                        if not finite_mask.all():
                            skipped_nonfinite_val += int((~finite_mask).sum().item())
                            if not finite_mask.any():
                                continue
                            xb = xb[finite_mask]
                            yb = yb[finite_mask]
                        with torch.autocast(device_type=device.type, enabled=amp_enabled):
                            pred = model(xb).squeeze(-1)
                        diff = pred - yb
                        abs_err_sum += torch.abs(diff).sum().item()
                        denom = torch.clamp(torch.abs(yb), min=1e-8)
                        mape_sum += (torch.abs(diff) / denom).sum().item()
                        directional_matches += (torch.sign(pred) == torch.sign(yb)).sum().item()

                        # Baseline: predict zero return (no-move)
                        base_pred = torch.zeros_like(yb)
                        base_diff = base_pred - yb
                        base_abs_err_sum += torch.abs(base_diff).sum().item()
                        base_mape_sum += (torch.abs(base_diff) / denom).sum().item()
                        base_dir_matches += (torch.sign(base_pred) == torch.sign(yb)).sum().item()
                        total += yb.numel()
                val_mae = abs_err_sum / max(1, total)
                val_mape = mape_sum / max(1, total)
                directional_acc = directional_matches / max(1, total)
                base_mae = base_abs_err_sum / max(1, total)
                base_mape = base_mape_sum / max(1, total)
                base_hit_rate = base_dir_matches / max(1, total)
                
                score = -val_mae
                if score > best_val or best_state is None:
                    best_val = score
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_mae = val_mae
                    best_directional = directional_acc
                else:
                    if epoch >= early_stop_min_epochs:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stop_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            epoch_duration = time.time() - epoch_start_time
                            logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
                            break

                logger.info(
                    f"Fold {fold_idx}/{len(folds)} Epoch {epoch}/{cfg.epochs} - "
                    f"train_loss={train_loss_epoch:.4f} train_loss_ = {running:.4f} "
                    f"val_mae={val_mae:.6f} val_mape={val_mape:.6f} hit_rate={directional_acc:.4f} "
                    f"base_mae={base_mae:.6f} base_mape={base_mape:.6f} base_hit_rate={base_hit_rate:.4f}"
                )
                logger.info(f"  skipped_nonfinite: train={skipped_nonfinite_train} val={skipped_nonfinite_val}")
                if directional_acc > collapse_best_val_acc + 1e-12:
                    collapse_best_val_acc = float(directional_acc)
                    collapse_flat_epochs = 0
                else:
                    collapse_flat_epochs += 1
                if collapse_flat_epochs >= 10:
                    logger.warning("⚠ WARNING: Model Collapse Detected")
                    logger.warning("Reason: no improvement")
                    epoch_duration = time.time() - epoch_start_time
                    logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
                    break
                if directional_acc >= best_val_acc:
                    best_val_acc = float(directional_acc)
                    best_epoch_for_fold = epoch
                epoch_duration = time.time() - epoch_start_time
                logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
            else:
                val_loss_sum = 0.0
                val_loss_count = 0
                all_targets: List[int] = []
                all_preds: List[int] = []
                all_probabilities: List[np.ndarray] = []
                skipped_nonfinite_val = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        finite_mask = torch.isfinite(xb).all(dim=(1, 2))
                        if torch.is_floating_point(yb):
                            finite_mask = finite_mask & torch.isfinite(yb)
                        if not finite_mask.all():
                            skipped_nonfinite_val += int((~finite_mask).sum().item())
                            if not finite_mask.any():
                                continue
                            xb = xb[finite_mask]
                            yb = yb[finite_mask]
                        with torch.autocast(device_type=device.type, enabled=amp_enabled):
                            logits = model(xb)
                        pred = logits.argmax(dim=-1)
                        probs = torch.softmax(logits, dim=-1)
                        batch_val_loss = criterion(logits.float(), yb).item()
                        val_loss_sum += batch_val_loss * yb.numel()
                        val_loss_count += yb.numel()
                        all_targets.extend(yb.detach().cpu().numpy().tolist())
                        all_preds.extend(pred.detach().cpu().numpy().tolist())
                        all_probabilities.append(probs.detach().cpu().numpy())
                val_loss = val_loss_sum / max(1, val_loss_count)

                class_names = ["short_win", "timeout", "long_win"]
                if all_preds:
                    precision_raw, recall_raw, f1_raw, _ = precision_recall_fscore_support(
                        all_targets,
                        all_preds,
                        average=None,
                        zero_division=0,
                    )
                    val_acc = float(accuracy_score(all_targets, all_preds))
                    unique, counts = np.unique(all_preds, return_counts=True)
                    pred_dist = {
                        int(k): float(v)
                        for k, v in zip(unique.tolist(), (counts / len(all_preds)).tolist())
                    }

                    labels_present = np.unique(np.concatenate([
                        np.asarray(all_targets, dtype=np.int64),
                        np.asarray(all_preds, dtype=np.int64),
                    ]))
                    precision_full = np.zeros(3, dtype=np.float64)
                    recall_full = np.zeros(3, dtype=np.float64)
                    f1_full = np.zeros(3, dtype=np.float64)
                    for idx, label in enumerate(labels_present.tolist()):
                        if 0 <= int(label) < 3 and idx < len(precision_raw):
                            precision_full[int(label)] = float(precision_raw[idx])
                            recall_full[int(label)] = float(recall_raw[idx])
                            f1_full[int(label)] = float(f1_raw[idx])
                    precisions = precision_full.tolist()
                    recalls = recall_full.tolist()
                    macro_f1 = float(f1_raw.mean()) if len(f1_raw) else 0.0
                else:
                    val_acc = 0.0
                    macro_f1 = 0.0
                    precisions = [0.0, 0.0, 0.0]
                    recalls = [0.0, 0.0, 0.0]
                    pred_dist = {0: 0.0, 1: 0.0, 2: 0.0}

                # Class distribution
                val_counts = np.bincount(np.asarray(all_targets, dtype=np.int64), minlength=3) if all_targets else np.zeros(3, dtype=int)
                val_pct = val_counts / max(1, val_counts.sum())
                if train_label_counts is None:
                    train_label_counts = val_counts
                train_pct = train_label_counts / max(1, train_label_counts.sum())
                total_support = int(val_counts.sum())
                baseline_acc = float(val_counts.max() / max(1, total_support))
                balanced_acc = float(np.mean(recalls)) if recalls else 0.0

                cur_metric = macro_f1
                if str(getattr(cfg, "early_stop_metric", "macro_f1")) == "balanced_acc":
                    cur_metric = balanced_acc

                if cur_metric > best_val:
                    best_val = cur_metric
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    if epoch >= early_stop_min_epochs:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stop_patience:
                            logger.info(
                                f"[early_stop] fold={fold_idx} best_epoch={best_epoch} "
                                f"best_{cfg.early_stop_metric}={best_val:.4f}"
                            )
                            epoch_duration = time.time() - epoch_start_time
                            logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
                            break

                # Prediction distribution (what model predicts)
                pred_counts = np.bincount(np.asarray(all_preds, dtype=np.int64), minlength=3) if all_preds else np.zeros(3, dtype=int)
                pred_pct = pred_counts / max(1, pred_counts.sum())
                max_pred_share = float(pred_pct.max()) if pred_pct.size else 0.0
                if max_pred_share > 0.90:
                    logger.warning("⚠ WARNING: Model Collapse Detected")
                    logger.warning("Reason: imbalance")
                    epoch_duration = time.time() - epoch_start_time
                    logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
                    break

                # Calibration metrics
                probs_np = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.zeros((0, 3), dtype=np.float32)
                if len(all_targets) > 0 and probs_np.shape[0] == len(all_targets):
                    probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=1.0, neginf=0.0)
                    probs_np = np.clip(probs_np, 1e-12, 1.0)
                    y_true_np = np.asarray(all_targets, dtype=np.int64)
                    true_prob = probs_np[np.arange(len(y_true_np)), y_true_np]
                    log_loss = float(-np.mean(np.log(true_prob)))
                    y_onehot = np.eye(3, dtype=np.float32)[y_true_np]
                    brier = float(np.mean(np.sum((probs_np - y_onehot) ** 2, axis=1)))
                    # ECE (10 bins)
                    conf = probs_np.max(axis=1)
                    pred_cls = probs_np.argmax(axis=1)
                    bins = np.linspace(0.0, 1.0, 11)
                    ece = 0.0
                    for i in range(len(bins) - 1):
                        lo, hi = bins[i], bins[i + 1]
                        mask = (conf >= lo) & (conf < hi)
                        if not np.any(mask):
                            continue
                        acc = np.mean(pred_cls[mask] == y_true_np[mask])
                        avg_conf = float(np.mean(conf[mask]))
                        ece += float(np.mean(mask)) * abs(acc - avg_conf)
                    ece = float(ece)
                else:
                    log_loss = float("nan")
                    brier = float("nan")
                    ece = float("nan")

                # Win-class combined metrics: (short_win + long_win) vs timeout
                y_true_win = np.asarray([1 if t in (0, 2) else 0 for t in all_targets], dtype=np.int64)
                y_pred_win = np.asarray([1 if p in (0, 2) else 0 for p in all_preds], dtype=np.int64)
                tp_win = int(((y_true_win == 1) & (y_pred_win == 1)).sum())
                fp_win = int(((y_true_win == 0) & (y_pred_win == 1)).sum())
                fn_win = int(((y_true_win == 1) & (y_pred_win == 0)).sum())
                prec_win = tp_win / max(1, tp_win + fp_win)
                rec_win = tp_win / max(1, tp_win + fn_win)
                f1_win = (2 * prec_win * rec_win) / max(1e-12, (prec_win + rec_win))

                # Recall guardrail for long_win
                long_recall = recalls[2] if len(recalls) > 2 else 0.0
                if long_recall < 0.02:
                    long_recall_bad_streak += 1
                else:
                    long_recall_bad_streak = 0
                if long_recall_bad_streak >= 5:
                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["lr"] * 0.5
                    logger.info(
                        f"[guardrail] long_win recall < 0.02 for 5 epochs, "
                        f"reducing LR to {optimizer.param_groups[0]['lr']:.6g}"
                    )
                    long_recall_bad_streak = 0

                expected_value = compute_expected_value(
                    probabilities=probs_np,
                    predictions=np.asarray(all_preds, dtype=np.int64),
                    targets=np.asarray(all_targets, dtype=np.int64),
                    take_profit=float(cfg.tp_pct),
                    stop_loss=float(cfg.sl_pct),
                    trading_fee=float(2.0 * (cfg.fee_pct + cfg.slippage_pct)),
                )
                ev_for_tracking = float(expected_value)
                if val_acc >= best_val_acc:
                    best_val_acc = float(val_acc)
                    best_epoch_for_fold = epoch
                if macro_f1 >= best_macro_f1:
                    best_macro_f1 = float(macro_f1)
                if ev_for_tracking >= best_ev:
                    best_ev = ev_for_tracking
                log_epoch_summary(
                    fold_idx=fold_idx,
                    total_folds=len(folds),
                    epoch=epoch,
                    total_epochs=cfg.epochs,
                    train_loss=train_loss_epoch,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    macro_f1=macro_f1,
                    class_names=class_names,
                    precisions=precisions,
                    recalls=recalls,
                    pred_dist=pred_dist,
                    expected_value=expected_value,
                )
                if val_acc > collapse_best_val_acc + 1e-12:
                    collapse_best_val_acc = float(val_acc)
                    collapse_flat_epochs = 0
                else:
                    collapse_flat_epochs += 1
                if collapse_flat_epochs >= 10:
                    logger.warning("⚠ WARNING: Model Collapse Detected")
                    logger.warning("Reason: no improvement")
                    epoch_duration = time.time() - epoch_start_time
                    logger.info(f"Epoch Duration: {epoch_duration:.2f}s")
                    break

                # Optional balanced validation reporting (fixed sample)
                if getattr(cfg, "balanced_val_eval", False):
                    y_true_arr = np.asarray(all_targets, dtype=np.int64)
                    y_pred_arr = np.asarray(all_preds, dtype=np.int64)
                    if balanced_eval_indices is None:
                        rng = np.random.default_rng(int(cfg.seed) + int(fold_idx) * 1000)
                        per_class = max(1, int(cfg.balanced_val_samples) // 3)
                        idxs = [np.where(y_true_arr == i)[0] for i in range(3)]
                        min_count = min(len(i) for i in idxs) if all(len(i) > 0 for i in idxs) else 0
                        if min_count > 0:
                            take = min(per_class, min_count)
                            sel = np.concatenate([rng.choice(i, size=take, replace=False) for i in idxs])
                            rng.shuffle(sel)
                            balanced_eval_indices = sel
                        else:
                            balanced_eval_indices = np.array([], dtype=int)

                    if balanced_eval_indices.size > 0:
                        yb_true = y_true_arr[balanced_eval_indices]
                        yb_pred = y_pred_arr[balanced_eval_indices]
                        cm_b = np.zeros((3, 3), dtype=int)
                        for t, p in zip(yb_true, yb_pred):
                            if 0 <= t < 3 and 0 <= p < 3:
                                cm_b[t, p] += 1
                        precisions_b = []
                        recalls_b = []
                        f1s_b = []
                        for i in range(3):
                            tp = cm_b[i, i]
                            prec = tp / max(1, cm_b[:, i].sum())
                            rec = tp / max(1, cm_b[i, :].sum())
                            f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
                            precisions_b.append(prec)
                            recalls_b.append(rec)
                            f1s_b.append(f1)
                        macro_f1_b = float(np.mean(f1s_b))
                        acc_b = float((yb_true == yb_pred).mean())
                        bal_acc_b = float(np.mean(recalls_b))
                        logger.info(
                            f"  balanced_val: n={len(yb_true)} acc={acc_b:.4f} "
                            f"macro_f1={macro_f1_b:.4f} balanced_acc={bal_acc_b:.4f}"
                        )
                    else:
                        logger.info("  balanced_val: insufficient class support for balanced sampling")

                # Persist per-epoch metrics for this fold
                epoch_metrics.append({
                    "epoch": epoch,
                    "val_acc": float(val_acc),
                    "macro_f1": float(macro_f1),
                    "balanced_acc": float(balanced_acc),
                    "baseline_acc": float(baseline_acc),
                    "log_loss": float(log_loss),
                    "brier": float(brier),
                    "ece": float(ece),
                    "precision": dict(zip(class_names, np.round(precisions, 4).tolist())),
                    "recall": dict(zip(class_names, np.round(recalls, 4).tolist())),
                    "support": dict(zip(class_names, val_counts.tolist())),
                    "pred_dist": dict(zip(class_names, np.round(pred_pct, 4).tolist())),
                    "win_precision": float(prec_win),
                    "win_recall": float(rec_win),
                    "win_f1": float(f1_win),
                    "expected_value": float(expected_value),
                })
                metrics_path.write_text(json.dumps({"fold": fold_idx, "epochs": epoch_metrics}, indent=2))
                epoch_duration = time.time() - epoch_start_time
                logger.info(f"Epoch Duration: {epoch_duration:.2f}s")

        # Track last fold artifacts
        last_fold_best_state = best_state
        last_fold_last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        last_fold_scaler = scaler
        logger.info("#" * 60)
        logger.info(f"FOLD {fold_idx} COMPLETE")
        logger.info(f"Best Epoch: {best_epoch_for_fold}")
        logger.info(f"Best Val Acc: {best_val_acc:.4f}")
        logger.info(f"Best Macro F1: {best_macro_f1:.4f}")
        logger.info(f"Best Expected Value: {best_ev:.4f}")
        logger.info("#" * 60)
        if getattr(cfg, 'task', 'classification') == 'regression':
            best_mae_value = best_mae if np.isfinite(best_mae) else 0.0
            fold_logs.append({
                "fold": fold_idx,
                "val_acc": float(best_val),
                "val_mae": float(best_mae_value),
                "directional_acc": float(best_directional),
                "val_mape": float(val_mape) if 'val_mape' in locals() else 0.0,
                "base_mae": float(base_mae) if 'base_mae' in locals() else 0.0,
                "base_mape": float(base_mape) if 'base_mape' in locals() else 0.0,
                "base_hit_rate": float(base_hit_rate) if 'base_hit_rate' in locals() else 0.0,
            })
        else:
            fold_logs.append({
                "fold": fold_idx,
                "val_acc": float(val_acc) if 'val_acc' in locals() else 0.0,
                "val_macro_f1": float(best_macro_f1),
                "precision": dict(zip(class_names, np.round(precisions, 4).tolist())) if 'precisions' in locals() else {},
                "recall": dict(zip(class_names, np.round(recalls, 4).tolist())) if 'recalls' in locals() else {},
                "pred_dist": dict(zip(class_names, np.round(pred_pct, 4).tolist())) if 'pred_pct' in locals() else {},
            })

        # Optional threshold optimizer (classification only)
        if cfg.optimize_thresholds and getattr(cfg, 'task', 'classification') != 'regression':
            if best_state is not None:
                model.load_state_dict(best_state)
            best_thresholds_fold = optimize_thresholds_on_val(
                model=model,
                scaler=scaler,
                feature_cols=feature_cols,
                val_files=val_files,
                window_size=cfg.window_size,
                price_col=cfg.price_col,
                atr_col=cfg.atr_col,
                fee_pct=cfg.fee_pct,
                slippage_pct=cfg.slippage_pct,
                cost_mult=cfg.cost_mult,
                k_tp=cfg.k_tp,
                k_sl=cfg.k_sl,
                time_limit=cfg.time_limit,
                min_atrp=cfg.min_atrp,
                min_barrier_pct=cfg.min_barrier_pct,
                device=device,
            )

        # Temperature calibration on this fold's validation set (optional)
        if cfg.calibrate_temp and getattr(cfg, 'task', 'classification') != 'regression':
            print("[calibrate] Fitting temperature on validation set ...")
            model_cal = build_model_from_meta(model_meta_dict).to(device)
            if best_state is not None:
                model_cal.load_state_dict(best_state)
            model_cal.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model_cal(xb)
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(yb.detach().cpu())
            if all_logits:
                logits_cat = torch.cat(all_logits, dim=0)
                labels_cat = torch.cat(all_labels, dim=0)
                T = torch.ones(1, requires_grad=True)
                opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
                def _closure():
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(logits_cat / T.clamp(min=1e-3), labels_cat)
                    loss.backward()
                    return loss
                for _ in range(10):
                    opt.step(_closure)
                temperature_final = float(T.detach().clamp(min=1e-3).item())
                if not np.isfinite(temperature_final):
                    temperature_final = 1.0
                    print("[calibrate] Temperature is non-finite; falling back to 1.0")
                else:
                    print(f"[calibrate] Temperature = {temperature_final:.4f}")

        # Save per-fold artifacts
        fold_outdir = Path(cfg.output_dir) / f"fold_{fold_idx}"
        fold_outdir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = fold_outdir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"file_manifest_fold{fold_idx}.json").write_text(
            json.dumps({"files": abs_files, "count": len(abs_files)}, indent=2)
        )
        fold_model_path = fold_outdir / "model.pt"
        fold_last_path = fold_outdir / "model_last.pt"
        fold_scaler_path = fold_outdir / "scaler.pkl"
        fold_meta_path = fold_outdir / "model_meta.json"
        fold_metrics_path = fold_outdir / "metrics.json"

        fold_last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        fold_best_state = best_state if best_state is not None else fold_last_state
        torch.save(fold_last_state, fold_last_path)
        torch.save(fold_best_state, fold_model_path)
        if scaler is not None:
            joblib.dump(scaler, fold_scaler_path)

        feature_cols_meta = list(feature_cols)
        if "adx" not in feature_cols_meta:
            feature_cols_meta.append("adx")

        model_type = selected_model_type or "transformer"
        task = getattr(cfg, 'task', 'classification')
        num_classes = 1 if task == 'regression' else 3
        label_def = 'return_regression' if task == 'regression' else 'triple_barrier_{-1,0,1}'
        notes = meta_existing.get("notes") or ("Return regression with walk-forward validation." if task == 'regression' else "Triple-barrier classification with walk-forward validation.")
        fold_meta = dict(meta_existing)
        fold_meta.update({
            "model_type": model_type,
            "framework": "pytorch",
            "feature_scaling": True,
            "scaler_type": "standard",
            "feature_cols": feature_cols_meta,
            "label_def": label_def,
            "num_classes": num_classes,
            "task": task,
            "price_col": cfg.price_col,
            "window_size": cfg.window_size,
            "tp_pct": cfg.tp_pct,
            "sl_pct": cfg.sl_pct,
            "time_limit": cfg.time_limit,
            "fee_pct": cfg.fee_pct,
            "slippage_pct": cfg.slippage_pct,
            "round_trip_cost": 2.0 * (cfg.fee_pct + cfg.slippage_pct),
            "cost_mult": cfg.cost_mult,
            "k_tp": cfg.k_tp,
            "k_sl": cfg.k_sl,
            "min_atrp": cfg.min_atrp,
            "min_barrier_pct": cfg.min_barrier_pct,
            "atr_col": cfg.atr_col,
            "input_size": len(feature_cols),
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "bidirectional": cfg.bidirectional,
            "model_state_path": "model.pt",
            "last_model_state_path": "model_last.pt",
            "scaler_path": "scaler.pkl",
            "notes": notes,
            "temperature": float(temperature_final),
            "optimized_thresholds": best_thresholds_fold or {},
            "fold": fold_idx,
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
            "file_manifest": abs_files,
        })
        if model_type.lower() in {"transformer", "transformer_classifier"}:
            fold_meta["num_heads"] = int(fold_meta.get("num_heads", meta_existing.get("num_heads", 4)))
        else:
            fold_meta.pop("num_heads", None)
        fold_meta_path.write_text(json.dumps(fold_meta, indent=2))

        fold_metrics = dict(fold_logs[-1]) if fold_logs else {}
        fold_metrics.update({
            "temperature": float(temperature_final),
            "optimized_thresholds": best_thresholds_fold or {},
            "fold": fold_idx,
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
        })
        fold_metrics_path.write_text(json.dumps(fold_metrics, indent=2))

        # keep last fold thresholds for aggregate output
        best_thresholds = best_thresholds_fold

    # Save artifacts
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.pt"
    last_path = outdir / "model_last.pt"
    meta_path = outdir / "model_meta.json"
    scaler_path = outdir / "scaler.joblib"

    if last_fold_last_state is None:
        raise RuntimeError("Training did not produce any model state.")
    torch.save(last_fold_last_state, last_path)
    torch.save(last_fold_best_state if last_fold_best_state is not None else last_fold_last_state, model_path)
    if last_fold_scaler is not None:
        joblib.dump(last_fold_scaler, scaler_path)

    # Ensure ADX present in meta feature list
    feature_cols_meta = list(feature_cols)
    if "adx" not in feature_cols_meta:
        feature_cols_meta.append("adx")

    agg_val = float(np.mean([f["val_acc"] for f in fold_logs])) if fold_logs else 0.0

    meta = dict(meta_existing)
    model_type = selected_model_type or "transformer"
    task = getattr(cfg, 'task', 'classification')
    num_classes = 1 if task == 'regression' else 3
    label_def = 'return_regression' if task == 'regression' else 'triple_barrier_{-1,0,1}'
    notes = meta.get("notes") or ("Return regression with walk-forward validation." if task == 'regression' else "Triple-barrier classification with walk-forward validation.")
    meta.update({
        "model_type": model_type,
        "framework": "pytorch",
        "feature_scaling": True,
        "scaler_type": "standard",
        "feature_cols": feature_cols_meta,
        "label_def": label_def,
        "num_classes": num_classes,
        "task": task,
        "price_col": cfg.price_col,
        "window_size": cfg.window_size,
        "tp_pct": cfg.tp_pct,
        "sl_pct": cfg.sl_pct,
        "time_limit": cfg.time_limit,
        "fee_pct": cfg.fee_pct,
        "slippage_pct": cfg.slippage_pct,
        "round_trip_cost": 2.0 * (cfg.fee_pct + cfg.slippage_pct),
        "cost_mult": cfg.cost_mult,
        "k_tp": cfg.k_tp,
        "k_sl": cfg.k_sl,
        "min_atrp": cfg.min_atrp,
        "min_barrier_pct": cfg.min_barrier_pct,
        "atr_col": cfg.atr_col,
        "input_size": len(feature_cols),  # model input matches training features
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "bidirectional": cfg.bidirectional,
        "model_state_path": "model.pt",
        "last_model_state_path": "model_last.pt",
        "scaler_path": "scaler.joblib",
        "notes": notes,
        "temperature": float(temperature_final),
        "file_manifest": abs_files,
    })
    if model_type.lower() in {"transformer", "transformer_classifier"}:
        meta["num_heads"] = int(meta.get("num_heads", meta_existing.get("num_heads", 4)))
    else:
        meta.pop("num_heads", None)
    meta_path.write_text(json.dumps(meta, indent=2))

    if fold_logs:
        best_fold_idx = int(np.argmax([f.get("val_acc", 0.0) for f in fold_logs]))
    summary = {
        "total_folds": len(folds),
        "epochs_per_fold": cfg.epochs,
        "final_aggregate_val_acc": float(agg_val),
        "best_fold": int(best_fold_idx) if 'best_fold_idx' in locals() else None,
        "feature_cols_used": feature_cols_meta,
        "model_type": model_type,
        "task": task,
        "total_training_time_seconds": round(time.time() - start_time, 1),
    }
    (outdir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    if best_thresholds:
        (outdir / "best_live_config.json").write_text(json.dumps({
            "live_config": best_thresholds
        }, indent=2))

    logger.info("✅ TRAINING FINISHED SUCCESSFULLY")
    logger.info(f"   Model saved → {model_path}")
    logger.info(f"   Best aggregate val_acc = {agg_val:.4f}")
    logger.info(f"   Artifacts in: {outdir.resolve()}")

    return agg_val


def _stream_rows(
    files: Sequence[Union[str, Path]],
    *,
    chunksize: int,
    overlap: int,
    burn_in: int = 0,
) -> Iterator[pd.DataFrame]:
    """
    Stream raw rows from one or more CSVs in chunks, carrying overlap rows between chunks.
    burn_in: number of initial rows to skip globally (useful for indicator warmup).
    """
    if not files:
        return
        yield  # make it a generator type even if empty

    carry: Optional[pd.DataFrame] = None
    skipped = 0

    for fp in files:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"_stream_rows missing file: {path}")

        for chunk in pd.read_csv(path, chunksize=chunksize):
            # Ensure consistent columns / header casing if your pipeline expects it
            try:
                chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            except Exception:
                pass

            df = pd.concat([carry, chunk], ignore_index=True) if carry is not None else chunk

            # burn-in skip (global)
            if burn_in and skipped < burn_in:
                take = min(len(df), burn_in - skipped)
                df = df.iloc[take:].reset_index(drop=True)
                skipped += take
                if len(df) == 0:
                    # keep carry small, continue reading
                    carry = None
                    continue

            # Keep overlap for next chunk/file
            if overlap and len(df) > overlap:
                out = df.iloc[:-overlap].reset_index(drop=True)
                carry = df.iloc[-overlap:].reset_index(drop=True)
            else:
                out = df.reset_index(drop=True)
                carry = df.reset_index(drop=True) if overlap else None

            if len(out) > 0:
                yield out

    # NOTE: do not yield carry at end; most label/feature functions need forward context and/or overlap discipline.


def optimize_thresholds_on_val(
    *,
    model: nn.Module,
    scaler: Optional[StandardScaler],
    feature_cols: List[str],
    val_files: List[Path],
    window_size: int,
    price_col: str,
    atr_col: str,
    fee_pct: float,
    slippage_pct: float,
    cost_mult: float,
    k_tp: float,
    k_sl: float,
    time_limit: int,
    min_atrp: float = 0.0005,
    min_barrier_pct: float = 0.0,
    device,
) -> Dict[str, float]:
    """Simple threshold optimizer using validation fold.

    Uses dynamic barriers to estimate per-trade PnL from labels.
    """
    model.eval()
    round_trip_cost = 2.0 * (fee_pct + slippage_pct)
    thr_grid = np.round(np.arange(0.55, 0.91, 0.05), 2)

    # Collect probs/labels/pct arrays
    probs_all: List[np.ndarray] = []
    labels_all: List[int] = []
    tp_all: List[float] = []
    sl_all: List[float] = []

    buf: Deque[np.ndarray] = deque(maxlen=window_size)
    X_batch: List[np.ndarray] = []
    meta_batch: List[Tuple[int, float, float]] = []

    def _flush():
        if not X_batch:
            return
        X = np.stack(X_batch, axis=0).astype(np.float32)
        if scaler is not None:
            B, T, Fdim = X.shape
            X = scaler.transform(X.reshape(B * T, Fdim)).reshape(B, T, Fdim)
        xb = torch.from_numpy(X).to(device)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for p, (lab, tp, sl) in zip(probs, meta_batch):
            probs_all.append(p)
            labels_all.append(lab)
            tp_all.append(tp)
            sl_all.append(sl)
        X_batch.clear()
        meta_batch.clear()

    burn_in = max(window_size, 240, time_limit + 5)
    missing_warned = False
    for df in _stream_rows(val_files, chunksize=200_000, overlap=window_size + 5, burn_in=burn_in):
        # --- Ensure DF has engineered features like training ---
        df = normalize_headers(df)

        # compute_features should add return_*, ema_*, bb_*, etc.
        # It should be safe to call even if some raw columns are missing.
        df = compute_features(df)

        # Sanity: required cols for labeling must exist
        if price_col not in df.columns:
            raise RuntimeError(f"[optimize_thresholds_on_val] price_col '{price_col}' missing after compute_features(). cols={list(df.columns)[:30]}")
        if atr_col not in df.columns:
            raise RuntimeError(f"[optimize_thresholds_on_val] atr_col '{atr_col}' missing after compute_features(). cols={list(df.columns)[:30]}")

        labels_raw = apply_triple_barrier_labels_dynamic(
            df,
            price_col=price_col,
            atr_col=atr_col,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            cost_mult=cost_mult,
            k_tp=k_tp,
            k_sl=k_sl,
            time_limit=time_limit,
            min_atrp=min_atrp,
            min_barrier_pct=min_barrier_pct,
        )
        labels = labels_raw.astype(int)
        tp_pct, sl_pct = compute_dynamic_barrier_pcts(
            df,
            price_col=price_col,
            atr_col=atr_col,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            cost_mult=cost_mult,
            k_tp=k_tp,
            k_sl=k_sl,
        )

        # IMPORTANT: do not df[feature_cols] directly (may be missing)
        # Reindex guarantees exact feature order + fills missing columns with NaN.
        feat_df = (
            df.reindex(columns=feature_cols)
              .ffill()
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
              .astype(np.float32, copy=False)
        )

        missing = [c for c in feature_cols if c not in df.columns]
        if missing and not missing_warned:
            logger.warning(
                f"[optimize_thresholds_on_val] Missing {len(missing)}/{len(feature_cols)} features in val data. "
                f"Filling with 0. Example={missing[:10]}"
            )
            missing_warned = True

        feats = feat_df.to_numpy(dtype=np.float32, copy=False)
        if scaler is not None:
            feats = scaler.transform(feats).astype(np.float32, copy=False)
            feats = np.clip(feats, -10.0, 10.0)
        for i in range(len(df)):
            buf.append(feats[i])
            if len(buf) < window_size:
                continue
            if i >= len(df) - time_limit:
                continue
            X_batch.append(np.stack(list(buf), axis=0))
            meta_batch.append((labels[i], float(tp_pct[i]), float(sl_pct[i])))
            if len(X_batch) >= 512:
                _flush()
        _flush()

    if not probs_all:
        return {"thr_long": 0.75, "thr_short": 0.75}

    probs_arr = np.asarray(probs_all)
    labels_arr = np.asarray(labels_all)
    tp_arr = np.asarray(tp_all)
    sl_arr = np.asarray(sl_all)

    best = {"thr_long": 0.75, "thr_short": 0.75, "score": -1e9}
    for tl in thr_grid:
        for ts in thr_grid:
            equity = 1.0
            for p, lab, tp, sl in zip(probs_arr, labels_arr, tp_arr, sl_arr):
                p_short, p_hold, p_long = p[0], p[1], p[2]
                sig = 0
                if p_long >= tl and p_long >= p_short:
                    sig = 1
                elif p_short >= ts and p_short >= p_long:
                    sig = -1
                if sig == 0:
                    continue
                if sig == 1:
                    if lab == 1:
                        ret = tp - round_trip_cost
                    elif lab == -1:
                        ret = -sl - round_trip_cost
                    else:
                        ret = -round_trip_cost
                else:
                    if lab == -1:
                        ret = tp - round_trip_cost
                    elif lab == 1:
                        ret = -sl - round_trip_cost
                    else:
                        ret = -round_trip_cost
                equity *= (1.0 + ret)
            if equity > best["score"]:
                best = {"thr_long": float(tl), "thr_short": float(ts), "score": float(equity)}
    return {"thr_long": best["thr_long"], "thr_short": best["thr_short"], "score": best["score"]}


# ----------------------------
# CLI
# ----------------------------
def env_default(key: str, fallback: str) -> str:
    v = os.environ.get(key)
    return v if v else fallback


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memory-safe streaming trainer for LSTM classifier (meta-aware).")
    p.add_argument("--data-path", type=str, default=env_default("SM_CHANNEL_TRAIN", "eth_1m_data"))
    p.add_argument("--output-dir", type=str, default=env_default("SM_MODEL_DIR", "./model"))
    p.add_argument("--meta-path", type=str, default="model/model_meta.json")

    # Model / data (these are fallback defaults; meta can override)
    p.add_argument("--window-size", type=int, default=None,
                   help="Single sequence length. If not set, uses --seq-lens or DEFAULT_SEQ_LENS.")
    p.add_argument("--seq-lens", type=str, default=None,
                   help="Comma/space-separated sequence lengths (e.g., '60,90,120').")
    p.add_argument("--hidden-size", type=int, default=128) 
    p.add_argument("--num-layers", type=int, default=2) 
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--model-type", choices=["lstm_classifier", "lstm_attention", "transformer"], default="lstm_attention")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=7e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--disable-scaling", type=str2bool, default=False)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--price-col", type=str, default="close")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--force-cpu-on-mac", type=str2bool, default=True)
    p.add_argument("--grad-clip-norm", type=float, default=None, help="Override grad clip norm (default 0.5 CUDA, 0.25 CPU/MPS)")
    p.add_argument("--detect-anomaly", type=str2bool, default=False, help="Enable autograd anomaly detection for debug runs")
    p.add_argument("--balanced-val-eval", type=str2bool, default=False, help="Report metrics on balanced validation sample")
    p.add_argument("--balanced-val-samples", type=int, default=50_000, help="Total windows to sample for balanced val eval")
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-epochs", type=int, default=5)
    p.add_argument("--early-stop-metric", choices=["macro_f1", "balanced_acc"], default="macro_f1")
    p.add_argument("--max-folds", type=int, default=None, help="Limit walk-forward folds for faster runs")
    # Triple barrier params (classification)
    p.add_argument("--tp-pct", type=float, default=0.01, help="Legacy TP pct (fallback for debugging)")
    p.add_argument("--sl-pct", type=float, default=0.005, help="Legacy SL pct (fallback for debugging)")
    p.add_argument("--time-limit", type=int, default=5, help="Bars ahead to evaluate outcome (default 5)")
    p.add_argument("--fee-pct", type=float, default=0.0001, help="Per-side fee used in labeling")
    p.add_argument("--slippage-pct", type=float, default=0.0001, help="Per-side slippage used in labeling")
    p.add_argument("--cost-mult", type=float, default=1.5, help="Minimum edge multiplier on round-trip cost")
    p.add_argument("--k-tp", type=float, default=1.2, help="ATR multiplier for TP barrier")
    p.add_argument("--k-sl", type=float, default=1.0, help="ATR multiplier for SL barrier")
    p.add_argument("--atr-col", type=str, default="atr_14", help="ATR column for dynamic barriers")
    p.add_argument("--min-atrp", type=float, default=0.0005, help="Min ATR%% to label (below => timeout)")
    p.add_argument("--min-barrier-pct", type=float, default=0.0, help="Extra pct margin above cost floor to label")
    # Task toggle & regression horizon
    p.add_argument("--task", choices=["classification","regression"], default="classification")
    p.add_argument("--feature-cols", nargs="*", default=None,
                   help="Optional explicit feature list; defaults to FEATURE_COLUMNS")
    p.add_argument("--horizon", type=int, default=3, help="Bars ahead to predict for regression (3=3 minutes)")
    # Loss / calibration
    p.add_argument("--use-class-weights", type=str2bool, default=True)
    p.add_argument("--use-focal-loss", type=str2bool, default=True)
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Recommended sweep: 1.0, 1.5, 2.0, 2.5")
    p.add_argument("--calibrate-temp", type=str2bool, default=True)
    p.add_argument("--optimize-thresholds", type=str2bool, default=True)
    return p


def main():
    args = build_parser().parse_args()

    def _parse_seq_lens(val):
        if val is None:
            return None
        parts = str(val).replace(",", " ").split()
        return [int(p) for p in parts if p.strip()]

    seq_lens = _parse_seq_lens(args.seq_lens)
    if seq_lens is None:
        if args.window_size is not None:
            seq_lens = [int(args.window_size)]
        else:
            seq_lens = list(DEFAULT_SEQ_LENS)

    for seq_len in seq_lens:
        out_dir = Path(args.output_dir)
        if len(seq_lens) > 1:
            out_dir = out_dir / f"seq_{seq_len}"

        # If meta-path is inside output dir, ensure parent exists
        mp = Path(args.meta_path)
        if not mp.is_absolute():
            mp = out_dir / mp
        mp.parent.mkdir(parents=True, exist_ok=True)

        cfg = TrainConfig(
            data_path=args.data_path,
            output_dir=str(out_dir),
            meta_path=str(mp),
            window_size=int(seq_len),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            disable_scaling=args.disable_scaling,
            val_frac=args.val_frac,
            accumulate=args.accumulate,
            seed=args.seed,
            price_col=args.price_col,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            time_limit=args.time_limit,
            fee_pct=args.fee_pct,
            slippage_pct=args.slippage_pct,
            cost_mult=args.cost_mult,
            k_tp=args.k_tp,
            k_sl=args.k_sl,
            atr_col=args.atr_col,
            min_atrp=args.min_atrp,
            min_barrier_pct=args.min_barrier_pct,
            amp=args.amp,
            workers=args.workers,
            chunksize=args.chunksize,
            max_folds=args.max_folds,
            device=args.device,
            force_cpu_on_mac=args.force_cpu_on_mac,
            grad_clip_norm=args.grad_clip_norm,
            detect_anomaly=args.detect_anomaly,
            balanced_val_eval=args.balanced_val_eval,
            balanced_val_samples=args.balanced_val_samples,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_epochs=args.early_stop_min_epochs,
            early_stop_metric=args.early_stop_metric,
            task=args.task,
            horizon=args.horizon,
            model_type=args.model_type,
            use_class_weights=args.use_class_weights,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            calibrate_temp=args.calibrate_temp,
            optimize_thresholds=args.optimize_thresholds,
            feature_cols=args.feature_cols,
        )
        try:
            train(cfg)
        except Exception:
            logger.exception("Unhandled exception during training")
            raise

if __name__ == "__main__":
    main()
