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
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import compute_features
from models import (
    ModelMeta,
    TransformerClassifier,
    build_model_from_meta as build_model_from_meta_core,
)


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
# Feature Names (superset)
# ----------------------------
ALL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'ATR_6', 'EMA_6', 
    'RSI_6', 'VWAP_6', 'ROC_6', 'KC_upper_6', 'KC_middle_6', 'Donchian_upper_6', 
    'Donchian_lower_6', 'MACD_6', 'MACD_signal_6', 'BB_upper_6', 'BB_middle_6', 
    'BB_lower_6', 'EWO_6', 'ATR_8', 'EMA_8', 'RSI_8', 'VWAP_8', 'ROC_8', 'KC_upper_8', 
    'KC_middle_8', 'Donchian_upper_8', 'Donchian_lower_8', 'MACD_8', 'MACD_signal_8',
    'BB_upper_8', 'BB_middle_8', 'BB_lower_8', 'EWO_8', 'ATR_10', 'EMA_10', 'RSI_10', 
    'VWAP_10', 'ROC_10', 'KC_upper_10', 'KC_middle_10', 'Donchian_upper_10', 'Donchian_lower_10', 
    'MACD_10', 'MACD_signal_10', 'BB_upper_10', 'BB_middle_10', 'BB_lower_10', 'EWO_10', 'ATR_12', 
    'EMA_12', 'RSI_12', 'VWAP_12', 'ROC_12', 'KC_upper_12', 'KC_middle_12', 'Donchian_upper_12', 
    'Donchian_lower_12', 'MACD_12', 'MACD_signal_12', 'BB_upper_12', 'BB_middle_12', 'BB_lower_12', 
    'EWO_12', 'ATR_14', 'EMA_14', 'RSI_14', 'VWAP_14', 'ROC_14', 'KC_upper_14', 'KC_middle_14', 'Donchian_upper_14', 
    'Donchian_lower_14', 'MACD_14', 'MACD_signal_14', 'BB_upper_14', 'BB_middle_14', 'BB_lower_14', 'EWO_14', 
    'ATR_16', 'EMA_16', 'RSI_16', 'VWAP_16', 'ROC_16', 'KC_upper_16', 'KC_middle_16', 'Donchian_upper_16', 
    'Donchian_lower_16', 'MACD_16', 'MACD_signal_16', 'BB_upper_16', 'BB_middle_16', 'BB_lower_16', 'EWO_16', 
    'ATR_18', 'EMA_18', 'RSI_18', 'VWAP_18', 'ROC_18', 'KC_upper_18', 'KC_middle_18', 'Donchian_upper_18', 'Donchian_lower_18', 
    'MACD_18', 'MACD_signal_18', 'BB_upper_18', 'BB_middle_18', 'BB_lower_18', 'EWO_18', 'ATR_22', 'EMA_22', 'RSI_22', 'VWAP_22', 
    'ROC_22', 'KC_upper_22', 'KC_middle_22', 'Donchian_upper_22', 'Donchian_lower_22', 'MACD_22', 'MACD_signal_22', 'BB_upper_22', 
    'BB_middle_22', 'BB_lower_22', 'EWO_22', 'ATR_26', 'EMA_26', 'RSI_26', 'VWAP_26', 'ROC_26', 'KC_upper_26', 'KC_middle_26', 
    'Donchian_upper_26', 'Donchian_lower_26', 'MACD_26', 'MACD_signal_26', 'BB_upper_26', 'BB_middle_26', 'BB_lower_26', 'EWO_26', 
    'ATR_33', 'EMA_33', 'RSI_33', 'VWAP_33', 'ROC_33', 'KC_upper_33', 'KC_middle_33', 'Donchian_upper_33', 'Donchian_lower_33', 'MACD_33', 
    'MACD_signal_33', 'BB_upper_33', 'BB_middle_33', 'BB_lower_33', 'EWO_33', 'ATR_44', 'EMA_44', 'RSI_44', 'VWAP_44', 'ROC_44', 'KC_upper_44', 
    'KC_middle_44', 'Donchian_upper_44', 'Donchian_lower_44', 'MACD_44', 'MACD_signal_44', 'BB_upper_44', 'BB_middle_44', 'BB_lower_44', 'EWO_44', 
    'ATR_55', 'EMA_55', 'RSI_55', 'VWAP_55', 'ROC_55', 'KC_upper_55', 'KC_middle_55', 'Donchian_upper_55', 'Donchian_lower_55', 'MACD_55', 
    'MACD_signal_55', 'BB_upper_55', 'BB_middle_55', 'BB_lower_55', 'EWO_55', 'return', 'Range', 'Volatility', 'OBV', 'ADL', 'Stoch_Oscillator', 'PSAR'
]



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
def apply_triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str,
    tp_pct: float,
    sl_pct: float,
    time_limit: int,
) -> np.ndarray:
    """Triple-barrier labels per row using future highs/lows up to time_limit bars.

    - If future high first crosses price[i] * (1 + tp_pct) → +1 (win)
    - If future low first crosses price[i] * (1 - sl_pct) → -1 (loss)
    - If neither within time_limit → 0 (timeout)
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}' in DataFrame")
    for c in ("high", "low"):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' for triple barrier labels")

    n = len(df)
    prices = df[price_col].to_numpy(dtype=float, copy=False)
    highs = df["high"].to_numpy(dtype=float, copy=False)
    lows = df["low"].to_numpy(dtype=float, copy=False)

    labels = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels

    time_limit = max(1, int(time_limit))
    tp_mult = 1.0 + float(tp_pct)
    sl_mult = 1.0 - float(sl_pct)

    for i in range(n):
        entry = prices[i]
        if not np.isfinite(entry):
            labels[i] = 0
            continue
        tp_price = entry * tp_mult
        sl_price = entry * sl_mult

        out = 0
        max_j = min(time_limit, n - i - 1)
        for j in range(1, max_j + 1):
            hi = highs[i + j]
            lo = lows[i + j]
            if hi >= tp_price:
                out = 1
                break
            if lo <= sl_price:
                out = -1
                break
        labels[i] = out

    return labels


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
            labels_raw = apply_triple_barrier_labels(df, self.price_col, self.tp_pct, self.sl_pct, self.time_limit)
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
        buf: Deque[np.ndarray] = deque(maxlen=self.window)
        for df in _stream_rows(self.files, chunksize=self.chunksize, overlap=self.overlap):
            feat_df = df[self.feature_cols].astype(np.float32, copy=False)
            feats = feat_df.to_numpy(dtype=np.float32, copy=False)
            prices = df[self.price_col].to_numpy(dtype=np.float32, copy=False)
            feats = self.scaler.fit_transform(feats) if self.scaler else feats
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
    amp: bool
    workers: int
    chunksize: int
    task: str
    horizon: int
    model_type: Optional[str] = None
    max_folds: Optional[int] = None
    # Loss/Calibration
    use_class_weights: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    calibrate_temp: bool = True


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
    valid_cols = feature_cols.copy()
    for f in train_files:
        for chunk in pd.read_csv(f, chunksize=chunksize):
            chunk = _normalize_headers(chunk)
            chunk = compute_features(chunk)
            # Forward-fill NaNs for time-series data
            chunk[feature_cols] = chunk[feature_cols].fillna(method='ffill').fillna(0)
            # Check variances
            variances = chunk[feature_cols].var()
            valid_cols = [col for col in valid_cols if variances.get(col, 0) > 1e-3]  # Stricter threshold
            if not valid_cols:
                print(f"Warning: No valid features in chunk from {f}")
                continue
            sample = chunk[valid_cols].astype(np.float32, copy=False).to_numpy()
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
    print("Valid features:", valid_cols)
    print("Feature variances:", variances[valid_cols].to_dict() if variances is not None else {})

    return scaler


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
    last_fold_best_state = None
    last_fold_last_state = None
    last_fold_scaler = None
    scaler = StandardScaler()
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
            train_ds = StreamWindowDataset(
                train_files, feature_cols, cfg.price_col, cfg.window_size,
                cfg.tp_pct, cfg.sl_pct, cfg.time_limit, chunksize=cfg.chunksize,
            )
            val_ds = StreamWindowDataset(
                val_files, feature_cols, cfg.price_col, cfg.window_size,
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
        # Class weights (classification only)
        class_weights = None
        if getattr(cfg, 'task', 'classification') != 'regression' and cfg.use_class_weights:
            # We use config flag later when choosing criterion
            train_peek = next(_stream_rows(train_files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
            labels_peek = apply_triple_barrier_labels(train_peek, cfg.price_col, cfg.tp_pct, cfg.sl_pct, cfg.time_limit)
            labels_peek = (labels_peek + 1).clip(0, 2)
            counts = np.bincount(labels_peek.astype(np.int64), minlength=3).astype(np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
            cw = (inv / inv.sum()) * 3.0
            class_weights = torch.tensor(cw, device=device, dtype=torch.float32)

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
            criterion = nn.SmoothL1Loss()
        else:
            if cfg.use_focal_loss:
                criterion = FocalLoss(gamma=float(cfg.focal_gamma), weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scaler_obj = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        best_val = -1.0
        best_state = None
        best_mae = float("inf")
        best_directional = 0.0
        early_stop_counter = 0
        early_stop_patience = 15

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            step = 0

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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler_obj.step(optimizer)
                        scaler_obj.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    

                running += loss.item()
                step += 1

            # Validation
            model.eval()
            if getattr(cfg, 'task', 'classification') == 'regression':
                abs_err_sum = 0.0
                directional_matches = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True).float()
                        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                            pred = model(xb).squeeze(-1)
                        abs_err_sum += torch.abs(pred - yb).sum().item()
                        directional_matches += (torch.sign(pred) == torch.sign(yb)).sum().item()
                        total += yb.numel()
                val_mae = abs_err_sum / max(1, total)
                directional_acc = directional_matches / max(1, total)
                
                # Scheduler step
                scheduler.step(val_mae)
            
                score = -val_mae
                if score > best_val or best_state is None:
                    best_val = score
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_mae = val_mae
                    best_directional = directional_acc
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                print(f"Fold {fold_idx}/{len(folds)} Epoch {epoch}/{cfg.epochs} - train_loss={running/max(1, step):.4f} train_loss_ = {running:.4f} val_mae={val_mae:.6f} dir_acc={directional_acc:.4f}")
            else:
                correct = total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                            logits = model(xb)
                        pred = logits.argmax(dim=-1)
                        correct += (pred == yb).sum().item()
                        total += yb.numel()
                val_acc = correct / max(1, total)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                print(f"Fold {fold_idx}/{len(folds)} Epoch {epoch}/{cfg.epochs} - train_loss={running/max(1, step):.4f} val_acc={val_acc:.4f}")

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
            })
        else:
            fold_logs.append({"fold": fold_idx, "val_acc": float(best_val)})

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

    print(f"Saved: {model_path}, {last_path}, scaler={'True' if last_fold_scaler is not None else 'False'}, meta={meta_path}")

    return agg_val


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
    p.add_argument("--window-size", type=int, default=240)
    p.add_argument("--hidden-size", type=int, default=128) 
    p.add_argument("--num-layers", type=int, default=2) 
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--model-type", choices=["lstm_classifier", "lstm_attention", "lstm_regressor", "transformer"], default="lstm_regressor")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32) 
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
    p.add_argument("--tp-pct", type=float, default=0.01, help="Take-profit percent e.g., 0.7%")
    p.add_argument("--sl-pct", type=float, default=0.005, help="Stop-loss percent e.g., 0.3%")
    p.add_argument("--time-limit", type=int, default=30, help="Bars ahead to evaluate outcome")
    # Task toggle & regression horizon
    p.add_argument("--task", choices=["classification","regression"], default="regression")
    p.add_argument("--horizon", type=int, default=3, help="Bars ahead to predict for regression (3=3 minutes)")
    # Loss / calibration
    p.add_argument("--use-class-weights", type=str2bool, default=True)
    p.add_argument("--use-focal-loss", type=str2bool, default=True)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--calibrate-temp", type=str2bool, default=True)
    return p


def main():
    args = build_parser().parse_args()

    # If meta-path is inside output dir, ensure parent exists
    mp = Path(args.meta_path)
    if not mp.is_absolute():
        mp = Path(args.output_dir) / mp
    mp.parent.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        meta_path=str(mp),
        window_size=args.window_size,
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
    )
    train(cfg)

if __name__ == "__main__":
    main()

