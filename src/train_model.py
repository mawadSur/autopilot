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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import compute_features
from models import (
    ModelMeta,
    TransformerClassifier,
    build_model_from_meta as build_model_from_meta_core,
)
from utils import FEATURE_COLUMNS, DEFAULT_SEQ_LENS


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
# CSV / streaming helpers
# ----------------------------
def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).lower() for c in df.columns]
    has_any = any(c in cols for c in ["open", "high", "low", "close", "volume", "timestamp", "time"]) 
    if not has_any and df.shape[1] >= 6:
        df = df.copy()
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"][:df.shape[1]]
    return df


def _list_csvs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        # to work on specific range. 
        # files = sorted(
        #     f for f in p.glob("*.csv")
        #     if "ethusdt_1m_" in f.name and "2023-09" <= f.stem.split("_")[-1] <= "2024-12"
        # )
        if not files:
            raise FileNotFoundError(f"No CSVs found in directory: {path}")
        return files
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file or directory, got: {path}")
    return [p]


def _stream_rows(files: List[Path], chunksize: int = 500_000, overlap: int = 256) -> Iterable[pd.DataFrame]:
    tail: Optional[pd.DataFrame] = None
    for f in files:
        for chunk in pd.read_csv(f, chunksize=chunksize):
            chunk = _normalize_headers(chunk)
            if tail is not None:
                chunk = pd.concat([tail, chunk], ignore_index=True)
            chunk = compute_features(chunk)
            if len(chunk) > overlap:
                yield chunk.iloc[overlap:].reset_index(drop=True)
                tail = chunk.iloc[-overlap:].reset_index(drop=True)
            else:
                tail = chunk
    # no final yield


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
        chunksize: int = 500_000,
        overlap: int = 256,
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
        self.chunksize = int(chunksize)
        self.overlap = max(int(overlap), self.window + self.time_limit + 1)

    def __iter__(self):
        buf: Deque[np.ndarray] = deque(maxlen=self.window)
        for df in _stream_rows(self.files, chunksize=self.chunksize, overlap=self.overlap):
            feat_df = df[self.feature_cols].astype(np.float32, copy=False)
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
            )
            labels = (labels_raw + 1).astype(np.int64, copy=False)  # {-1,0,1} → {0,1,2}

            feats = feat_df.to_numpy(dtype=np.float32, copy=False)
            n = len(df)
            for i in range(n):
                buf.append(feats[i])
                if len(buf) < self.window:
                    continue
                if i >= n - self.time_limit:
                    continue
                Xw = np.stack(list(buf), axis=0)  # [T, F]
                y = int(labels[i])
                yield torch.from_numpy(Xw).float(), torch.tensor(y, dtype=torch.long)


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

        buf: Deque[np.ndarray] = deque(maxlen=self.window)
        for df in _stream_rows(self.files, chunksize=self.chunksize, overlap=self.overlap):
            feat_df = df[self.feature_cols].astype(np.float32, copy=False)
            feats = feat_df.to_numpy(dtype=np.float32, copy=False)
            prices = df[self.price_col].to_numpy(dtype=np.float32, copy=False)
            feats = self.scaler.transform(feats)
            n = len(df)
            buf.clear()
            for i in range(n):
                buf.append(feats[i])
                if len(buf) < self.window:
                    continue
                j = i + self.horizon
                if j >= n:
                    continue
                Xw = np.stack(list(buf), axis=0)
                base_price = float(prices[i])
                future_price = float(prices[j])
                if abs(base_price) < 1e-8:
                    continue
                ret = (future_price / base_price) - 1.0
                yield torch.from_numpy(Xw).float(), torch.tensor(ret, dtype=torch.float32)


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
    # Loss/Calibration
    use_class_weights: bool = True
    use_focal_loss: bool = False
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
            chunk = _normalize_headers(chunk)
            chunk = compute_features(chunk)
            # Forward-fill NaNs for time-series data
            chunk[feature_cols] = chunk[feature_cols].fillna(method='ffill').fillna(0)
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
    print("Scaler means:", scaler.mean_)
    print("Scaler stds:", scaler.scale_)
    print("Valid features:", feature_cols)

    return scaler, feature_cols


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = get_device()

    files = _list_csvs(cfg.data_path)

    # Load meta to lock features/window/model dims (if present)
    meta_existing = {}
    meta_path = Path(cfg.meta_path)
    if meta_path.exists():
        try:
            meta_existing = json.loads(meta_path.read_text())
        except Exception:
            meta_existing = {}

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
    peek = next(_stream_rows(files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
    available = set(peek.columns.tolist())
    feature_cols = [c for c in desired_features if c in available]
    # Ensure ADX is part of training features if available
    if "adx" in available and "adx" not in feature_cols:
        feature_cols.append("adx")
    if len(feature_cols) < 4:
        raise ValueError(f"Too few features after engineering. Wanted={desired_features}, available={sorted(available)}")

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

        def collate_batch(batch):
            xb, yb = zip(*batch)  # xb: [B,T,F]
            xb = torch.stack(list(xb), dim=0)
            if isinstance(yb[0], torch.Tensor):
                yb = torch.stack(list(yb), dim=0)
            else:
                yb = torch.tensor(yb)
            return xb, yb

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
            train_ds = StreamWindowDataset(
                train_files, feature_cols, cfg.price_col, cfg.window_size,
                cfg.atr_col, cfg.fee_pct, cfg.slippage_pct, cfg.cost_mult, cfg.k_tp, cfg.k_sl,
                cfg.tp_pct, cfg.sl_pct, cfg.time_limit, chunksize=cfg.chunksize,
            )
            val_ds = StreamWindowDataset(
                val_files, feature_cols, cfg.price_col, cfg.window_size,
                cfg.atr_col, cfg.fee_pct, cfg.slippage_pct, cfg.cost_mult, cfg.k_tp, cfg.k_sl,
                cfg.tp_pct, cfg.sl_pct, cfg.time_limit, chunksize=cfg.chunksize,
            )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, num_workers=cfg.workers,
            pin_memory=(device.type == "cuda"), collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, num_workers=max(0, cfg.workers // 2),
            pin_memory=(device.type == "cuda"), collate_fn=collate_batch,
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
            # We use config flag later when choosing criterion
            train_peek = next(_stream_rows(train_files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
            labels_peek = apply_triple_barrier_labels_dynamic(
                train_peek,
                price_col=cfg.price_col,
                atr_col=cfg.atr_col,
                fee_pct=cfg.fee_pct,
                slippage_pct=cfg.slippage_pct,
                cost_mult=cfg.cost_mult,
                k_tp=cfg.k_tp,
                k_sl=cfg.k_sl,
                time_limit=cfg.time_limit,
            )
            labels_peek = (labels_peek + 1).clip(0, 2)
            counts = np.bincount(labels_peek.astype(np.int64), minlength=3).astype(np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
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
        scaler_obj = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        best_val = -1.0
        best_state = None
        best_mae = float("inf")
        best_directional = 0.0
        early_stop_counter = 0
        early_stop_patience = 10
        early_stop_min_epochs = 5
        warmup_steps = max(50, int(cfg.batch_size))
        global_step = 0
        steps_per_epoch = None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            step = 0
            opt_steps = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                    logits = model(xb)
            
                    if getattr(cfg, 'task', 'classification') == 'regression':
                        logits = logits.squeeze(-1)
             
                    loss = criterion(logits, yb)
                if scaler_obj.is_enabled():
                    scaler_obj.scale(loss / cfg.accumulate).backward()
                else:
                    (loss / cfg.accumulate).backward()

                if (step + 1) % cfg.accumulate == 0:
                    if scaler_obj.is_enabled():
                        scaler_obj.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        scaler_obj.step(optimizer)
                        scaler_obj.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

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

            # Set steps_per_epoch after first pass (IterableDataset has no __len__)
            if steps_per_epoch is None:
                steps_per_epoch = max(1, opt_steps)

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
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True).float()
                        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
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
                            print(f"Early stopping at epoch {epoch}")
                            break

                print(
                    f"Fold {fold_idx}/{len(folds)} Epoch {epoch}/{cfg.epochs} - "
                    f"train_loss={running/max(1, step):.4f} train_loss_ = {running:.4f} "
                    f"val_mae={val_mae:.6f} val_mape={val_mape:.6f} hit_rate={directional_acc:.4f} "
                    f"base_mae={base_mae:.6f} base_mape={base_mape:.6f} base_hit_rate={base_hit_rate:.4f}"
                )
            else:
                correct = total = 0
                y_true_all: List[int] = []
                y_pred_all: List[int] = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                            logits = model(xb)
                        pred = logits.argmax(dim=-1)
                        correct += (pred == yb).sum().item()
                        total += yb.numel()
                        y_true_all.extend(yb.detach().cpu().numpy().tolist())
                        y_pred_all.extend(pred.detach().cpu().numpy().tolist())
                val_acc = correct / max(1, total)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                # Confusion matrix + per-class precision/recall
                cm = np.zeros((3, 3), dtype=int)
                for t, p in zip(y_true_all, y_pred_all):
                    if 0 <= t < 3 and 0 <= p < 3:
                        cm[t, p] += 1
                class_names = ["short_win", "timeout", "long_win"]
                precisions = []
                recalls = []
                for i in range(3):
                    tp = cm[i, i]
                    prec = tp / max(1, cm[:, i].sum())
                    rec = tp / max(1, cm[i, :].sum())
                    precisions.append(prec)
                    recalls.append(rec)

                # Class distribution
                val_counts = cm.sum(axis=1)
                val_pct = val_counts / max(1, val_counts.sum())
                if train_label_counts is None:
                    train_label_counts = val_counts
                train_pct = train_label_counts / max(1, train_label_counts.sum())

                print(
                    f"Fold {fold_idx}/{len(folds)} Epoch {epoch}/{cfg.epochs} - "
                    f"train_loss={running/max(1, step):.4f} val_acc={val_acc:.4f}"
                )
                print(
                    f"  train_dist: {dict(zip(class_names, train_label_counts.tolist()))} "
                    f"({dict(zip(class_names, np.round(train_pct, 4).tolist()))})"
                )
                print(
                    f"  val_dist:   {dict(zip(class_names, val_counts.tolist()))} "
                    f"({dict(zip(class_names, np.round(val_pct, 4).tolist()))})"
                )
                print(f"  precision: {dict(zip(class_names, np.round(precisions, 4).tolist()))}")
                print(f"  recall:    {dict(zip(class_names, np.round(recalls, 4).tolist()))}")
                print(f"  conf_mat:\n{cm}")

        # Track last fold artifacts
        last_fold_best_state = best_state
        last_fold_last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        last_fold_scaler = scaler
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
            fold_logs.append({"fold": fold_idx, "val_acc": float(best_val)})

        # Optional threshold optimizer (classification only)
        if cfg.optimize_thresholds and getattr(cfg, 'task', 'classification') != 'regression':
            if best_state is not None:
                model.load_state_dict(best_state)
            best_thresholds = optimize_thresholds_on_val(
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
                print(f"[calibrate] Temperature = {temperature_final:.4f}")

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
    })
    if model_type.lower() in {"transformer", "transformer_classifier"}:
        meta["num_heads"] = int(meta.get("num_heads", meta_existing.get("num_heads", 4)))
    else:
        meta.pop("num_heads", None)
    meta_path.write_text(json.dumps(meta, indent=2))

    (outdir / "training_summary.json").write_text(json.dumps({
        "walk_forward_folds": len(folds),
        "folds": fold_logs,
        "val_acc_aggregate": agg_val,
        "feature_cols": feature_cols_meta,
        "num_params": sum(p.numel() for p in model.parameters()),
        "config": vars(cfg) | {"meta_path": str(meta_path)},
    }, indent=2))

    if best_thresholds:
        (outdir / "best_live_config.json").write_text(json.dumps({
            "live_config": best_thresholds
        }, indent=2))

    print(f"Saved: {model_path}, {last_path}, scaler={'True' if last_fold_scaler is not None else 'False'}, meta={meta_path}")

    return agg_val


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

    for df in _stream_rows(val_files, chunksize=200_000, overlap=window_size + 5):
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
        feat_df = df[feature_cols].astype(np.float32, copy=False)
        feats = feat_df.to_numpy(dtype=np.float32, copy=False)
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
    p.add_argument("--model-type", choices=["lstm_classifier", "lstm_attention", "lstm_regressor", "transformer"], default="lstm_classifier")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64) 
    p.add_argument("--learning-rate", type=float, default=1e-4) 
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--price-col", type=str, default="close")
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
    # Task toggle & regression horizon
    p.add_argument("--task", choices=["classification","regression"], default="classification")
    p.add_argument("--feature-cols", nargs="*", default=None,
                   help="Optional explicit feature list; defaults to FEATURE_COLUMNS")
    p.add_argument("--horizon", type=int, default=3, help="Bars ahead to predict for regression (3=3 minutes)")
    # Loss / calibration
    p.add_argument("--use-class-weights", type=str2bool, default=True)
    p.add_argument("--use-focal-loss", type=str2bool, default=False)
    p.add_argument("--focal-gamma", type=float, default=2.0)
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
            amp=args.amp,
            workers=args.workers,
            chunksize=args.chunksize,
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
        train(cfg)

if __name__ == "__main__":
    main()

