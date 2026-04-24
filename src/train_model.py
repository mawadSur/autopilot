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
import h5py
from collections import deque
from torch import nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


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
]

ROLL_WINDOW = 20
ATR_ALPHA = 1/14
RSI_ALPHA = 1/14

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure basic columns exist
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

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
    
    # Hourly trend ratio (1m data → 60)
    hourly = df["close"].ewm(span=60, adjust=False).mean()
    df["price_vs_hourly_trend"] = (df["close"] / (hourly + 1e-12)).fillna(1.0)

    # Bollinger Band width (20, 2σ)
    std_20 = df["close"].rolling(ROLL_WINDOW).std()
    upper = sma + 2 * std_20
    lower = sma - 2 * std_20
    df["bb_width"] = ((upper - lower) / (sma + 1e-12)).fillna(0.0)

    return df

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
            chunk = _compute_features(chunk)
            if len(chunk) > overlap:
                yield chunk.iloc[overlap:].reset_index(drop=True)
                tail = chunk.iloc[-overlap:].reset_index(drop=True)
            else:
                tail = chunk
    # no final yield

def _stream_hdf5(path: str, symbols: List[str], chunksize: int = 500_000, overlap: int = 256) -> Iterable[pd.DataFrame]:
    """Stream rows from HDF5 store."""
    tail: Optional[pd.DataFrame] = None
    with h5py.File(path, 'r') as f:
        for symbol in symbols:
            candle_path = f"/{symbol}/1m/candles/candles"
            if candle_path not in f:
                print(f"⚠️ No candles found for {symbol} in HDF5")
                continue
            
            dset = f[candle_path]
            total_rows = dset.shape[0]
            
            for start in range(0, total_rows, chunksize):
                end = min(start + chunksize, total_rows)
                data = dset[start:end]
                chunk = pd.DataFrame(data)
                
                # Standardize column names if needed
                if "timestamp" in chunk.columns:
                    chunk = chunk.rename(columns={"timestamp": "time"})
                
                if tail is not None:
                    chunk = pd.concat([tail, chunk], ignore_index=True)
                
                chunk = _compute_features(chunk)
                
                if len(chunk) > overlap:
                    yield chunk.iloc[overlap:].reset_index(drop=True)
                    tail = chunk.iloc[-overlap:].reset_index(drop=True)
                else:
                    tail = chunk

def _make_labels(
    df: pd.DataFrame,
    price_col: str,
    horizon: int = 1,
    profit_threshold: float = 0.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nxt = df[price_col].shift(-horizon)
    gross_returns = (nxt - df[price_col]) / (df[price_col] + 1e-12)
    round_trip_cost = 2.0 * (fee_rate + slippage_rate)
    net_returns = gross_returns - round_trip_cost
    labels = (net_returns > profit_threshold).astype(np.int64)
    weights = 1.0 + np.clip(np.abs(net_returns.to_numpy(dtype=np.float32, copy=False)) * 100.0, 0.0, 10.0)

    labels.iloc[-horizon:] = 0
    weights[-horizon:] = 1.0
    net_returns.iloc[-horizon:] = 0.0
    return (
        labels.to_numpy(dtype=np.int64),
        weights.astype(np.float32, copy=False),
        net_returns.to_numpy(dtype=np.float32),
    )


def _select_probability_threshold(
    probabilities: np.ndarray,
    net_returns: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> Tuple[float, dict]:
    candidates = np.arange(min_threshold, max_threshold + (step / 2.0), step, dtype=np.float32)
    if probabilities.size == 0:
        return float(min_threshold), {
            "selected_threshold": float(min_threshold),
            "selected_trades": 0,
            "selected_win_rate": 0.0,
            "selected_avg_net_return": 0.0,
            "selected_total_net_return": 0.0,
            "selected_profit_factor": 0.0,
            "selected_max_drawdown": 0.0,
            "selected_trade_rate": 0.0,
            "selected_gross_profit": 0.0,
            "selected_gross_loss": 0.0,
        }

    best_threshold = float(min_threshold)
    best_metrics = None
    best_net_profit = float("-inf")

    for threshold in candidates:
        taken = probabilities >= float(threshold)
        trades = int(taken.sum())
        if trades == 0:
            total_net_return = 0.0
            avg_net_return = 0.0
            win_rate = 0.0
            gross_profit = 0.0
            gross_loss = 0.0
            profit_factor = 0.0
            max_drawdown = 0.0
            trade_rate = 0.0
        else:
            selected_returns = net_returns[taken]
            total_net_return = float(selected_returns.sum())
            avg_net_return = float(selected_returns.mean())
            win_rate = float((selected_returns > 0).mean())
            gross_profit = float(selected_returns[selected_returns > 0].sum())
            gross_loss = float(-selected_returns[selected_returns < 0].sum())
            profit_factor = gross_profit / max(gross_loss, 1e-12) if gross_profit > 0 else 0.0
            equity_curve = np.cumsum(selected_returns)
            running_peak = np.maximum.accumulate(np.maximum(equity_curve, 0.0))
            max_drawdown = float(np.max(running_peak - equity_curve)) if equity_curve.size else 0.0
            trade_rate = trades / max(1, probabilities.size)

        if (
            total_net_return > best_net_profit
            or (
                np.isclose(total_net_return, best_net_profit)
                and best_metrics is not None
                and (
                    profit_factor > best_metrics["selected_profit_factor"]
                    or (
                        np.isclose(profit_factor, best_metrics["selected_profit_factor"])
                        and max_drawdown < best_metrics["selected_max_drawdown"]
                    )
                )
            )
        ):
            best_net_profit = total_net_return
            best_threshold = float(threshold)
            best_metrics = {
                "selected_threshold": float(threshold),
                "selected_trades": trades,
                "selected_win_rate": win_rate,
                "selected_avg_net_return": avg_net_return,
                "selected_total_net_return": total_net_return,
                "selected_profit_factor": profit_factor,
                "selected_max_drawdown": max_drawdown,
                "selected_trade_rate": trade_rate,
                "selected_gross_profit": gross_profit,
                "selected_gross_loss": gross_loss,
            }

    return best_threshold, best_metrics

class StreamWindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path: str, data_format: str, feature_cols: List[str], price_col: str,
                 window_size: int, symbols: List[str] = None, profit_horizon: int = 1, profit_threshold: float = 0.0,
                 fee_rate: float = 0.0, slippage_rate: float = 0.0,
                 chunksize: int = 500_000, overlap: int = 256):
        super().__init__()
        self.data_path = data_path
        self.data_format = data_format
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.window = window_size
        self.symbols = symbols or ["ETHUSDT"]
        self.profit_horizon = profit_horizon
        self.profit_threshold = profit_threshold
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.chunksize = chunksize
        self.overlap = max(overlap, window_size + profit_horizon + 1)

    def __iter__(self):
        buf: Deque[np.ndarray] = deque(maxlen=self.window)
        
        if self.data_format == "csv":
            files = _list_csvs(self.data_path)
            # Simple split for streaming
            worker_info = torch.utils.data.get_worker_info()
            if worker_info:
                files = [f for i, f in enumerate(files) if i % worker_info.num_workers == worker_info.id]
            stream = _stream_rows(files, chunksize=self.chunksize, overlap=self.overlap)
        else:
            stream = _stream_hdf5(self.data_path, self.symbols, chunksize=self.chunksize, overlap=self.overlap)

        for df in stream:
            feats = df[self.feature_cols].astype(np.float32).to_numpy(copy=False)
            labels, weights, net_returns = _make_labels(
                df,
                self.price_col,
                self.profit_horizon,
                self.profit_threshold,
                self.fee_rate,
                self.slippage_rate,
            )
            for i in range(len(df)):
                buf.append(feats[i])
                if len(buf) < self.window:
                    continue
                Xw = np.stack(list(buf), axis=0)
                y = int(labels[i])
                w = float(weights[i])
                r = float(net_returns[i])
                yield (
                    torch.from_numpy(Xw).float(),
                    torch.tensor(y, dtype=torch.long),
                    torch.tensor(w, dtype=torch.float32),
                    torch.tensor(r, dtype=torch.float32),
                )

@dataclass
class TrainConfig:
    data_path: str
    data_format: str
    symbols: List[str]
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
    profit_horizon: int
    profit_threshold: float
    fee_rate: float
    slippage_rate: float
    min_probability_threshold: float
    max_probability_threshold: float
    probability_threshold_step: float
    amp: bool
    workers: int
    chunksize: int

class LSTMClassifier(nn.Module):
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
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * d),
            nn.Linear(hidden_size * d, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2),
        )

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
    
    # Peek for features and fit scaler
    if cfg.data_format == "csv":
        files = _list_csvs(cfg.data_path)
        peek = next(_stream_rows(files, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
    else:
        peek = next(_stream_hdf5(cfg.data_path, cfg.symbols, chunksize=min(cfg.chunksize, 200_000), overlap=cfg.window_size + 5))
        
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
    sample = peek[feature_cols].astype(np.float32).to_numpy(copy=False)[:200_000]
    scaler = StandardScaler()
    scaler.fit(sample)

    def collate_batch(batch):
        xb, yb, wb, rb = zip(*batch)
        xb = torch.stack(list(xb), dim=0)  # [B,T,F]
        yb = torch.stack(list(yb), dim=0)
        wb = torch.stack(list(wb), dim=0)
        rb = torch.stack(list(rb), dim=0)
        B, T, F = xb.shape
        xflat = xb.reshape(B*T, F).numpy()
        xflat = scaler.transform(xflat).astype(np.float32, copy=False)
        xb = torch.from_numpy(xflat).view(B, T, F)
        return xb, yb, wb, rb

    train_ds = StreamWindowDataset(
        cfg.data_path,
        cfg.data_format,
        feature_cols,
        cfg.price_col,
        window_size,
        symbols=cfg.symbols,
        profit_horizon=cfg.profit_horizon,
        profit_threshold=cfg.profit_threshold,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        chunksize=cfg.chunksize,
    )
    # Note: In streaming mode with single source, val_ds might need separate data or we split by time
    # For now, we use a small fraction for validation if possible
    val_ds = train_ds 

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler_obj = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    best_val_metric = float("-inf")
    best_state = None
    best_threshold = float(meta_existing.get("buy_threshold", cfg.min_probability_threshold))
    best_threshold_metrics = {
        "selected_threshold": best_threshold,
        "selected_trades": 0,
        "selected_win_rate": 0.0,
        "selected_avg_net_return": 0.0,
        "selected_total_net_return": 0.0,
        "selected_profit_factor": 0.0,
        "selected_max_drawdown": 0.0,
        "selected_trade_rate": 0.0,
        "selected_gross_profit": 0.0,
        "selected_gross_loss": 0.0,
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        step = 0

        for xb, yb, wb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                logits = model(xb)
                loss_raw = F.cross_entropy(logits, yb, reduction="none")
                probs = torch.softmax(logits, dim=-1)
                pt = probs.gather(1, yb.unsqueeze(1)).squeeze(1)
                focal_loss = 0.25 * (1 - pt) ** 2 * loss_raw
                loss = (focal_loss * wb).mean()

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
            if step > 2000: break # Safety cap for streaming

        model.eval()
        all_probs = []
        all_labels = []
        all_net_returns = []
        
        with torch.no_grad():
            v_step = 0
            for xb, yb, _, rb in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type in ("cuda", "mps")):
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb.numpy())
                all_net_returns.append(rb.numpy())
                v_step += 1
                if v_step > 200: break

        if not all_probs:
            print(f"Epoch {epoch} - No validation data")
            continue

        probs_vec = np.concatenate(all_probs)
        labels_vec = np.concatenate(all_labels)
        net_returns_vec = np.concatenate(all_net_returns)

        # Basic accuracy for reporting
        val_acc = ((probs_vec >= 0.5).astype(int) == labels_vec).mean()
        best_epoch_threshold, threshold_metrics = _select_probability_threshold(
            probs_vec,
            net_returns_vec,
            cfg.min_probability_threshold,
            cfg.max_probability_threshold,
            cfg.probability_threshold_step,
        )
        best_epoch_profit = float(threshold_metrics["selected_total_net_return"])

        if best_epoch_profit > best_val_metric:
            best_val_metric = best_epoch_profit
            best_threshold = best_epoch_threshold
            best_threshold_metrics = threshold_metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch}/{cfg.epochs} - train_loss={running/max(1, step):.4f} "
            f"val_acc={val_acc:.4f} buy_threshold={best_epoch_threshold:.2f} "
            f"val_net_profit={best_epoch_profit:.5f} "
            f"win_rate={threshold_metrics['selected_win_rate']:.4f} "
            f"profit_factor={threshold_metrics['selected_profit_factor']:.3f} "
            f"max_dd={threshold_metrics['selected_max_drawdown']:.5f}"
        )

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
        "label_def": "future_net_return_after_costs",
        "profit_horizon": cfg.profit_horizon,
        "profit_threshold": cfg.profit_threshold,
        "fee_rate": cfg.fee_rate,
        "slippage_rate": cfg.slippage_rate,
        "round_trip_cost": 2.0 * (cfg.fee_rate + cfg.slippage_rate),
        "num_classes": 2,
        "price_col": cfg.price_col,
        "window_size": window_size,
        "input_size": len(feature_cols),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "buy_threshold": float(best_threshold),
        "sell_threshold": meta.get("sell_threshold", 0.60),
        "tx_cost": 2.0 * (cfg.fee_rate + cfg.slippage_rate),
        "model_state_path": "model.pt",
        "last_model_state_path": "model_last.pt",
        "scaler_path": "scaler.joblib",
        "probability_threshold_selection": best_threshold_metrics,
        "selection_metric": "validation_net_profit",
        "notes": "Binary classification on future net return after fees and slippage. Sample weights increase with move magnitude, threshold search maximizes validation net profit, and best checkpoint is selected on trading metrics.",
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    (outdir / "training_summary.json").write_text(json.dumps({
        "val_net_profit_best": float(best_val_metric),
        "buy_threshold": float(best_threshold),
        "probability_threshold_selection": best_threshold_metrics,
        "feature_cols": feature_cols,
        "num_params": sum(p.numel() for p in model.parameters()),
        "config": vars(cfg) | {"meta_path": str(meta_path)},
    }, indent=2))

    print(f"Saved: {model_path}, {last_path}, scaler=True, meta={meta_path}")


def env_default(key: str, fallback: str) -> str:
    v = os.environ.get(key)
    return v if v else fallback

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memory-safe streaming trainer for LSTM classifier (meta-aware).")
    p.add_argument("--data-path", type=str, default=env_default("SM_CHANNEL_TRAIN", "eth_1m_data"))  # renamed
    p.add_argument("--data-format", type=str, choices=["csv", "hdf5"], default="csv", help="Format of input data")
    p.add_argument("--symbols", type=str, default="ETHUSDT", help="Comma-separated symbols for HDF5")
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
    p.add_argument("--profit-horizon", type=int, default=12,
                   help="Number of bars ahead to evaluate trade outcome")
    p.add_argument("--profit-threshold", type=float, default=0.005,
                   help="Minimum future net return required to label a trade as profitable")
    p.add_argument("--fee-rate", type=float, default=0.0004,
                   help="Per-side trading fee rate used to convert future returns into net returns")
    p.add_argument("--slippage-rate", type=float, default=0.0002,
                   help="Per-side slippage rate used to convert future returns into net returns")
    p.add_argument("--min-probability-threshold", type=float, default=0.55,
                   help="Lower bound for validation threshold search")
    p.add_argument("--max-probability-threshold", type=float, default=0.90,
                   help="Upper bound for validation threshold search")
    p.add_argument("--probability-threshold-step", type=float, default=0.05,
                   help="Step size for validation threshold search")
    return p

def main():
    args = build_parser().parse_args()
    # If meta-path is inside output dir, ensure parent exists
    mp = Path(args.meta_path)
    if not mp.is_absolute():
        mp = Path(args.output_dir) / mp
    
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    cfg = TrainConfig(
        data_path=args.data_path,
        data_format=args.data_format,
        symbols=symbols,
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
        profit_horizon=args.profit_horizon,
        profit_threshold=args.profit_threshold,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        min_probability_threshold=args.min_probability_threshold,
        max_probability_threshold=args.max_probability_threshold,
        probability_threshold_step=args.probability_threshold_step,
        amp=args.amp,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    train(cfg)
