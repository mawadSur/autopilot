#!/usr/bin/env python3
"""
Memory-safe LSTM trainer with streaming windows.

Key changes:
- Uses model_meta.json (if present) to lock feature_cols, window_size, and core dims.
- Computes ALL features in meta by name (including vol_change, price_vs_hourly_trend).
- Saves best and last checkpoints along with scaler and updated meta.

Usage examples
--------------
python train_model.py --data-path eth_1m_data --output-dir model
python train_model.py --data-path eth_1m_data --output-dir model --batch-size 512 --epochs 40 --accumulate 2 --amp 1
python train_model.py --data-path eth_1m_2024-03.csv --output-dir model --window-size 192
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from collections import deque
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from utils import compute_features


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
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return",
    "sma_ratio",
    "ema_20",
    "macd",
    "rsi_14",
    "vol_change",
    "atr",
    "price_vs_hourly_trend",
    "bb_width",
    # Added features
    "vol_20", "vol_50", "vol_100",
    "vwap_ratio",
    "obv",
    "roc_14",
]


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # OHLC sanity
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")
=======
def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure basic columns exist
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")
>>>>>>> 8d962b1 (Updated launch_sagemaker_job and train_model)

    # Candlestick geometry
    df["body"] = df["close"] - df["open"]
    rng = (df["high"] - df["low"])
    df["range"] = rng.replace(0, 1e-12)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1))
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"])
    df["return"] = df["close"].pct_change().fillna(0.0)

    # SMA ratio (20)
    sma = df["close"].rolling(ROLL_WINDOW).mean()
    df["sma_ratio"] = (df["close"] / (sma + 1e-12)).fillna(1.0)

    # EMA(20)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI(14) with EMA smoothing
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=RSI_ALPHA, adjust=False).mean()
    roll_down = down.ewm(alpha=RSI_ALPHA, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # Volume percent change
    if "volume" in df.columns:
        df["vol_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        df["vol_change"] = 0.0

    # ATR(14) (EMA of True Range)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=ATR_ALPHA, adjust=False).mean().fillna(tr.mean())

    # MACD(12,26) - signal(9)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = (macd - signal).fillna(0.0)

<<<<<<< HEAD
    # Bollinger Band width (20, 2σ)
    std_20 = df["close"].rolling(ROLL_WINDOW).std()
    upper = sma + 2 * std_20
    lower = sma - 2 * std_20
    df["bb_width"] = ((upper - lower) / (sma + 1e-12)).fillna(0.0)

    # Volatility features: rolling std normalized by rolling mean
    for w in (20, 50, 100):
        roll_std = df["close"].rolling(w).std()
        roll_mean = df["close"].rolling(w).mean()
        df[f"vol_{w}"] = (roll_std / (roll_mean + 1e-12)).fillna(0.0)

    # Ensure 'volume' column exists for VWAP/OBV calculations
    if "volume" not in df.columns:
        df["volume"] = 1.0
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # VWAP ratio: daily VWAP if date/time available, else rolling approx (1440 bars)
    ts_col = None
    for cand in ("timestamp", "time", "date"):
        if cand in df.columns:
            ts_col = cand
            break
    vwap = None
    if ts_col is not None:
        try:
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            if ts.notna().any():
                day = ts.dt.floor("D")
                pv_sum = (df["close"] * vol).groupby(day).transform("sum")
                v_sum = vol.groupby(day).transform("sum")
                vwap = pv_sum / (v_sum + 1e-12)
        except Exception:
            vwap = None
    if vwap is None:
        # Fallback: approximate daily with 1440-bar rolling VWAP
        win = 1440
        pv_roll = (df["close"] * vol).rolling(win, min_periods=1).sum()
        v_roll = vol.rolling(win, min_periods=1).sum()
        vwap = pv_roll / (v_roll + 1e-12)
    df["vwap_ratio"] = (df["close"] / (vwap + 1e-12)).fillna(1.0)

    # On-Balance Volume (OBV)
    price_diff = df["close"].diff()
    dir_sign = (price_diff > 0).astype(int) - (price_diff < 0).astype(int)
    df["obv"] = (dir_sign * vol).cumsum().astype(float)

    # Rate of Change (ROC 14) as fractional change
    df["roc_14"] = df["close"].pct_change(periods=14).fillna(0.0)

    return df
=======
    # Hourly trend ratio (1m data → 60)
    hourly = df["close"].ewm(span=60, adjust=False).mean()
    df["price_vs_hourly_trend"] = (df["close"] / (hourly + 1e-12)).fillna(1.0)

    # Bollinger Band width (20, 2σ)
    std_20 = df["close"].rolling(ROLL_WINDOW).std()
    upper = sma + 2 * std_20
    lower = sma - 2 * std_20
    df["bb_width"] = ((upper - lower) / (sma + 1e-12)).fillna(0.0)

    return df
>>>>>>> 8d962b1 (Updated launch_sagemaker_job and train_model)

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).lower() for c in df.columns]
    has_any = any(c in cols for c in ["open","high","low","close","volume","timestamp","time"])
    if not has_any and df.shape[1] >= 6:
        df = df.copy()
        df.columns = ["timestamp","open","high","low","close","volume"][:df.shape[1]]
    return df

def _list_csvs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
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

def apply_triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str,
    tp_pct: float,
    sl_pct: float,
    time_limit: int,
) -> np.ndarray:
    """
    Triple-barrier labels per row using future highs/lows up to `time_limit` bars.

    - If future high first crosses price[i] * (1 + tp_pct) → +1 (win)
    - If future low first crosses price[i] * (1 - sl_pct) → -1 (loss)
    - If neither within time_limit → 0 (timeout)

    Returns an np.ndarray of shape [len(df)] with values in {-1, 0, 1}.
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

    # Ensure sane params
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

        out = 0  # default timeout
        # Scan forward up to time_limit bars (or to end of df)
        max_j = min(time_limit, n - i - 1)
        for j in range(1, max_j + 1):
            hi = highs[i + j]
            lo = lows[i + j]
            # If both thresholds are crossed on the same bar, treat as TP first.
            if hi >= tp_price:
                out = 1
                break
            if lo <= sl_price:
                out = -1
                break
        labels[i] = out

    return labels

<<<<<<< HEAD

class StreamWindowDataset(IterableDataset):
    """
    Yields (window, label) pairs lazily from a stream of feature frames.
    We keep a deque of the last `window_size` feature rows to form the next sample.
    """
    def __init__(self,
                 files: List[Path],
                 feature_cols: List[str],
                 price_col: str,
                 window_size: int,
                 tp_pct: float,
                 sl_pct: float,
                 time_limit: int,
                 chunksize: int = 500_000,
                 overlap: int = 256):
        super().__init__()
        self.files = files
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.window = window_size
        self.tp_pct = float(tp_pct)
        self.sl_pct = float(sl_pct)
        self.time_limit = int(time_limit)
        self.chunksize = chunksize
        # Ensure enough history for features (window_size) and future for labels (time_limit)
        self.overlap = max(overlap, window_size + self.time_limit + 1)

    def __iter__(self):
        buf: Deque[np.ndarray] = deque(maxlen=self.window)
        for df in _stream_rows(self.files, chunksize=self.chunksize, overlap=self.overlap):
            # Select features and label
            feat_df = df[self.feature_cols].astype(np.float32, copy=False)
            labels_raw = apply_triple_barrier_labels(
                df, self.price_col, self.tp_pct, self.sl_pct, self.time_limit
            )
            # Map {-1,0,1} → {0,1,2} for CE loss
            labels = (labels_raw + 1).astype(np.int64, copy=False)

            feats = feat_df.to_numpy(dtype=np.float32, copy=False)
            prices = df[self.price_col].to_numpy(dtype=np.float32, copy=False)

            n = len(df)
            for i in range(n):
                # push current feature row
                buf_feats.append(feats[i])
                if len(buf_feats) < self.window:
                    continue
                # Only yield if we still have at least `time_limit` future bars available
                # within this df chunk to determine labels reliably.
                if i >= n - self.time_limit:
                    continue
                y = int(labels[i])
                yield torch.from_numpy(Xw).float(), torch.tensor(y, dtype=torch.long)

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

class LSTMClassifier(nn.Module):
=======
class LSTMClassifier(nn.Module):
>>>>>>> 8d962b1 (Updated launch_sagemaker_job and train_model)
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d = 2 if bidirectional else 1
<<<<<<< HEAD
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * d),
            nn.Linear(hidden_size * d, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, 3),  # 3 classes: {-1,0,1} → {0,1,2}
        )
=======
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * d),
            nn.Linear(hidden_size * d, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, 2),
            nn.Linear(hidden_size // 2, 2),
        )
>>>>>>> 8d962b1 (Updated launch_sagemaker_job and train_model)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return self.head(h)

def _split_stream(files: List[Path], val_frac: float) -> Tuple[List[Path], List[Path]]:
    if len(files) == 1:
        return files, files
    k = max(1, int(round(len(files) * (1.0 - val_frac))))
    return files[:k], files[k:]

def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = get_device()

    files = _list_csvs(cfg.data_path)

    # --- Load meta if present to lock features/window/model dims ---
    meta_existing = {}
    meta_path = Path(cfg.meta_path)
    if meta_path.exists():
        try:
            meta_existing = json.loads(meta_path.read_text())
        except Exception:
            meta_existing = {}

    # Determine feature set
    desired_features = meta_existing.get("feature_cols", ALL_FEATURES)
    # Stream a peek to ensure features exist and build scaler
    peek = next(_stream_rows(files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
    available = set(peek.columns.tolist())
    feature_cols = [c for c in desired_features if c in available]
    if len(feature_cols) < 4:
        raise ValueError(f"Too few features after engineering. Wanted={desired_features}, available={sorted(available)}")

    # Walk-forward folds: train on months 1..k, validate on month k+1
    n_files = len(files)
    folds: List[Tuple[List[Path], List[Path]]] = []
    if n_files >= 2:
        for k in range(1, n_files):
            folds.append((files[:k], [files[k]]))
    else:
        # Fallback: single-file split treated as one fold
        folds.append((files, files))

    from sklearn.preprocessing import StandardScaler

    agg_acc = 0.0
    fold_logs = []

    last_fold_best_state = None
    last_fold_last_state = None
    last_fold_scaler = None

    for fold_idx, (train_files, val_files) in enumerate(folds, start=1):
        # Fit scaler on training sample (small peek) for this fold
        train_peek = next(_stream_rows(train_files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
        scaler = StandardScaler()
        sample = train_peek[feature_cols].astype(np.float32, copy=False).to_numpy()[:200_000]
        scaler.fit(sample)

        def collate_batch(batch):
            xb, yb = zip(*batch)
            xb = torch.stack(list(xb), dim=0)  # [B,T,F]
            yb = torch.stack(list(yb), dim=0)
            # scale features per window with fitted scaler (vectorized)
            B, T, F = xb.shape
            xflat = xb.reshape(B*T, F).numpy()
            xflat = scaler.transform(xflat).astype(np.float32, copy=False)
            xb2 = torch.from_numpy(xflat).view(B, T, F)
            return xb2, yb

        # Build loaders for this fold
        train_ds = StreamWindowDataset(
            train_files,
            feature_cols,
            cfg.price_col,
            cfg.window_size,
            cfg.tp_pct,
            cfg.sl_pct,
            cfg.time_limit,
            chunksize=cfg.chunksize,
        )
        val_ds = StreamWindowDataset(
            val_files,
            feature_cols,
            cfg.price_col,
            cfg.window_size,
            cfg.tp_pct,
            cfg.sl_pct,
            cfg.time_limit,
            chunksize=cfg.chunksize,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            num_workers=max(0, cfg.workers // 2),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_batch
        )

        # Fresh model per fold
        model = LSTMClassifier(
            input_size=len(feature_cols),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        scaler_obj = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        best_val = -1.0
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            step = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                    logits = model(xb)
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
                    optimizer.zero_grad(set_to_none=True)

                running += loss.item()
                step += 1

            # Validation (streamed) for this fold
            model.eval()
            correct = 0
            total = 0
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

            print(
                f"Fold {fold_idx}/{len(folds)} Epoch {epoch}/{cfg.epochs} - "
                f"train_loss={running/max(1, step):.4f} val_acc={val_acc:.4f}"
            )

        # Track aggregate and fold logs
        agg_acc += max(best_val, 0.0)
        fold_logs.append({
            "fold": fold_idx,
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
            "val_acc": best_val,
        })
        last_fold_best_state = best_state
        last_fold_last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        last_fold_scaler = scaler

    agg_val = agg_acc / max(1, len(folds))

    # Save artifacts from final fold
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.pt"
    last_path  = outdir / "model_last.pt"
    meta_path = outdir / "model_meta.json"
    scaler_path = outdir / "scaler.joblib"

    torch.save(last_fold_last_state, last_path)
    torch.save(last_fold_best_state if last_fold_best_state is not None else last_fold_last_state, model_path)

    if last_fold_scaler is not None:
        joblib.dump(last_fold_scaler, scaler_path)

    # Write meta that exactly matches the checkpoint
    meta = dict(meta_existing)  # start from any existing settings
    meta.update({
        "model_type": "lstm_classifier",
        "framework": "pytorch",
        "feature_scaling": True,
        "scaler_type": "standard",
        "feature_cols": feature_cols,
        "label_def": "triple_barrier",
        "num_classes": 3,
        "price_col": cfg.price_col,
        "tp_pct": cfg.tp_pct,
        "sl_pct": cfg.sl_pct,
        "time_limit": cfg.time_limit,
        "window_size": cfg.window_size,
        "input_size": len(feature_cols),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "buy_threshold": meta.get("buy_threshold", 0.60),
        "sell_threshold": meta.get("sell_threshold", 0.60),
        "tx_cost": meta.get("tx_cost", 0.0008),
        "model_state_path": "model.pt",
        "last_model_state_path": "model_last.pt",
        "scaler_path": "scaler.joblib",
        "notes": "Triple-barrier classification with walk-forward validation.",
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    (outdir / "training_summary.json").write_text(json.dumps({
        "walk_forward_folds": len(folds),
        "folds": fold_logs,
        "val_acc_aggregate": agg_val,
        "feature_cols": feature_cols,
        "num_params": sum(p.numel() for p in model.parameters()),
        "config": vars(cfg) | {"meta_path": str(meta_path)},
    }, indent=2))

    print(
        f"Saved: {model_path}, {last_path}, scaler=True, meta={meta_path}. "
        f"Walk-forward folds={len(folds)} agg_val_acc={agg_val:.4f}"
    )
=======
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = get_device()

    files = _list_csvs(cfg.data_path)

    # --- Load meta if present to lock features/window/model dims ---
    meta_existing = {}
    meta_path = Path(cfg.meta_path)
    if meta_path.exists():
        try:
            meta_existing = json.loads(meta_path.read_text())
        except Exception:
            meta_existing = {}

    # Determine feature set
    desired_features = meta_existing.get("feature_cols", ALL_FEATURES)
    # Stream a peek to ensure features exist and build scaler
    peek = next(_stream_rows(files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
    available = set(peek.columns.tolist())
    feature_cols = [c for c in desired_features if c in available]
    if len(feature_cols) < 4:
        raise ValueError(f"Too few features after engineering. Wanted={desired_features}, available={sorted(available)}")

    # Resolve training hyperparams from meta (fallback to CLI)
    window_size = int(meta_existing.get("window_size", cfg.window_size))
    hidden_size = int(meta_existing.get("hidden_size", cfg.hidden_size))
    num_layers  = int(meta_existing.get("num_layers",  cfg.num_layers))
    dropout     = float(meta_existing.get("dropout",   cfg.dropout))
    bidirectional = bool(meta_existing.get("bidirectional", cfg.bidirectional))

    # Scaler fit on sample
    from sklearn.preprocessing import StandardScaler
    sample = peek[feature_cols].astype(np.float32, copy=False).to_numpy()[:200_000]
    scaler = StandardScaler()
    scaler.fit(sample)

    def collate_batch(batch):
        xb, yb = zip(*batch)
        xb = torch.stack(list(xb), dim=0)  # [B,T,F]
        yb = torch.stack(list(yb), dim=0)
        B, T, F = xb.shape
        xflat = xb.reshape(B*T, F).numpy()
        xflat = scaler.transform(xflat).astype(np.float32, copy=False)
        xb = torch.from_numpy(xflat).view(B, T, F)
        return xb, yb

    train_files, val_files = _split_stream(files, cfg.val_frac if len(files) > 1 else 0.1)
    train_ds = StreamWindowDataset(train_files, feature_cols, cfg.price_col, window_size, chunksize=cfg.chunksize)
    val_ds   = StreamWindowDataset(val_files,   feature_cols, cfg.price_col, window_size, chunksize=cfg.chunksize)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.workers,
                              pin_memory=(device.type == "cuda"), collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=max(0, cfg.workers // 2),
                              pin_memory=(device.type == "cuda"), collate_fn=collate_batch)

    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler_obj = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        step = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                logits = model(xb)
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
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            step += 1

        model.eval()
        correct = 0
        total = 0
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

        print(f"Epoch {epoch}/{cfg.epochs} - train_loss={running/max(1, step):.4f} val_acc={val_acc:.4f}")

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.pt"
    last_path  = outdir / "model_last.pt"
    meta_path = outdir / "model_meta.json"
    scaler_path = outdir / "scaler.joblib"

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, last_path)
    torch.save(best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}, model_path)
    joblib.dump(scaler, scaler_path)

    # Write meta that exactly matches the checkpoint
    meta = dict(meta_existing)  # start from any existing settings
    meta.update({
        "model_type": "lstm_classifier",
        "framework": "pytorch",
        "feature_scaling": True,
        "scaler_type": "standard",
        "feature_cols": feature_cols,
        "label_def": "next_bar_up",
        "num_classes": 2,
        "price_col": cfg.price_col,
        "window_size": window_size,
        "input_size": len(feature_cols),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "buy_threshold": meta.get("buy_threshold", 0.60),
        "sell_threshold": meta.get("sell_threshold", 0.60),
        "tx_cost": meta.get("tx_cost", 0.0008),
        "model_state_path": "model.pt",
        "last_model_state_path": "model_last.pt",
        "scaler_path": "scaler.joblib",
        "notes": "Binary classification (1=buy, 0=no-trade). Streaming trainer with meta-locked features.",
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    (outdir / "training_summary.json").write_text(json.dumps({
        "val_acc_best": best_val,
        "feature_cols": feature_cols,
        "num_params": sum(p.numel() for p in model.parameters()),
        "config": vars(cfg) | {"meta_path": str(meta_path)},
        "config": vars(cfg) | {"meta_path": str(meta_path)},
    }, indent=2))

    print(f"Saved: {model_path}, {last_path}, scaler=True, meta={meta_path}")
>>>>>>> 8d962b1 (Updated launch_sagemaker_job and train_model)

def env_default(key: str, fallback: str) -> str:
    v = os.environ.get(key)
    return v if v else fallback

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memory-safe streaming trainer for LSTM classifier (meta-aware).")
    p.add_argument("--data-path", type=str, default=env_default("SM_CHANNEL_TRAIN", "eth_1m_data"))  # renamed
    p.add_argument("--output-dir", type=str, default=env_default("SM_MODEL_DIR", "./model"))
    p.add_argument("--meta-path", type=str, default="model/model_meta.json")  # default to inside output dir

    # Model / data (these are fallback defaults; meta can override)
    p.add_argument("--window-size", type=int, default=192)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", type=str2bool, default=True)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--price-col", type=str, default="close")
    # Triple barrier params
    p.add_argument("--tp-pct", type=float, default=0.01, help="Take-profit percent e.g., 0.01 for 1%")
    p.add_argument("--sl-pct", type=float, default=0.005, help="Stop-loss percent e.g., 0.005 for 0.5%")
    p.add_argument("--time-limit", type=int, default=30, help="Number of bars to look ahead")
    return p

def main():
    args = build_parser().parse_args()
    # If meta-path is inside output dir, ensure parent exists
    mp = Path(args.meta_path)
    if not mp.is_absolute():
        mp = Path(args.output_dir) / mp
    cfg = TrainConfig(
        data_path=args.data_path,     # <-- uses data_path now
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
    )
=======
    # If meta-path is inside output dir, ensure parent exists
    mp = Path(args.meta_path)
    if not mp.is_absolute():
        mp = Path(args.output_dir) / mp
    cfg = TrainConfig(
        data_path=args.data_path,     # <-- uses data_path now
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
        amp=args.amp,
        workers=args.workers,
        chunksize=args.chunksize,
    )
>>>>>>> 8d962b1 (Updated launch_sagemaker_job and train_model)
    train(cfg)

if __name__ == "__main__":
<<<<<<< HEAD
    main()

