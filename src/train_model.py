#!/usr/bin/env python3
"""
Train an LSTM price-movement classifier using ONLY the supported CLI flags.

Supported flags (all optional with profit-optimistic defaults):
--data (dir or csv; defaults to SM_CHANNEL_TRAIN or 'eth_1m_data')
--output-dir (defaults to SM_MODEL_DIR or './model')
--meta-path (defaults to 'model_meta.json')
--window-size
--hidden-size
--num-layers
--dropout
--bidirectional     (flag, no value -> True if set)
--epochs
--batch-size
--learning-rate
--weight-decay
--val-frac
--accumulate
--seed
--price-col
--disable-scaling   (flag, no value -> turn feature scaling OFF)

Everything else has been removed.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


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
    if torch.backends.mps.is_available():  # Apple MPS
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Data & Features
# ----------------------------
DEFAULT_FEATURES = [
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return",
    "sma_ratio_20",
    "ema_20",
    "rsi_14",
    "atr_14",
    "macd",
    "bb_width_20",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Candlestick geometry
    df["body"] = df["close"] - df["open"]
    df["range"] = (df["high"] - df["low"]).replace(0, 1e-12)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1))
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"])
    df["return"] = df["close"].pct_change().fillna(0.0)

    # SMA/EMA
    win = 20
    df["sma_20"] = df["close"].rolling(win).mean()
    df["sma_ratio_20"] = (df["close"] / (df["sma_20"] + 1e-12)).fillna(1.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI(14)
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # ATR(14)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(tr.mean())

    # MACD(12,26) minus signal(9)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = (macd - signal).fillna(0.0)

    # Bollinger Band width (20, 2σ)
    std_20 = df["close"].rolling(20).std()
    upper = df["sma_20"] + 2 * std_20
    lower = df["sma_20"] - 2 * std_20
    df["bb_width_20"] = ((upper - lower) / (df["sma_20"] + 1e-12)).fillna(0.0)

    return df


def make_labels(df: pd.DataFrame, price_col: str) -> pd.Series:
    """Binary label: next bar up (1) vs not-up (0)."""
    nxt = df[price_col].shift(-1)
    label = (nxt > df[price_col]).astype(np.int64)
    label.iloc[-1] = 0
    return label


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, window: int):
        self.X = features
        self.y = labels
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.window]
        y = self.y[idx + self.window]
        return torch.from_numpy(x).float(), torch.tensor(y).long()


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
    lr: float
    weight_decay: float
    val_frac: float
    accumulate: int
    seed: int
    price_col: str
    disable_scaling: bool


# ----------------------------
# Model
# ----------------------------
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
            nn.Linear(hidden_size * d, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return self.head(h)


# ----------------------------
# Training
# ----------------------------
def train_loop(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device()

    # Load data
    path = Path(cfg.data_path)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSVs found in {path}")
        df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    else:
        df = pd.read_csv(path)

    if cfg.price_col not in df.columns:
        raise ValueError(f"--price-col '{cfg.price_col}' not in data columns: {df.columns.tolist()}")

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    df = compute_features(df)

    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = make_labels(df, price_col=cfg.price_col).values.astype(np.int64)

    # Scaling
    scaler = None
    if not cfg.disable_scaling:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    # Dataset / split
    dataset = WindowDataset(X, y, window=cfg.window_size)
    n = len(dataset)
    if n < 1000:
        raise ValueError(f"Not enough samples after windowing: {n}")

    val_len = max(1, int(n * cfg.val_frac))
    train_len = n - val_len
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Model / Optim
    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train
    best_val = -1.0
    best_state = None
    model.train()
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (xb, yb) in enumerate(train_loader, 1):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            (loss / cfg.accumulate).backward()
            if i % cfg.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            global_step += 1

        # Validate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            val_acc = correct / max(1, total)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.train()

        print(f"Epoch {epoch}/{cfg.epochs} - train_loss={running/len(train_loader):.4f} val_acc={val_acc:.4f}")

    # Save artifacts
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.pt"
    last_path = outdir / "model_last.pt"
    scaler_path = outdir / "scaler.joblib"

    # save last state
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, last_path)

    # save best state as model.pt
    torch.save(best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)

    # Update meta
    meta_path = Path(cfg.meta_path)
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception:
        meta = {}

    meta.update({
        "model_type": "lstm_classifier",
        "framework": "pytorch",
        "feature_scaling": (not cfg.disable_scaling),
        "feature_cols": feature_cols,
        "price_col": cfg.price_col,
        "window_size": cfg.window_size,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "bidirectional": cfg.bidirectional,
        "buy_threshold": meta.get("buy_threshold", 0.52),
        "sell_threshold": meta.get("sell_threshold", 0.52),
        "tx_cost": meta.get("tx_cost", 0.0005),
        "model_state_path": "model.pt",
        "last_model_state_path": "model_last.pt",
        "scaler_path": "scaler.joblib",
        "notes": "Binary classification (1=buy, 0=no-trade). Optimistic defaults for profitability.",
    })

    meta_path.write_text(json.dumps(meta, indent=2))

    # Training summary
    (outdir / "training_summary.json").write_text(json.dumps({
        "val_acc_best": best_val,
        "feature_cols": feature_cols,
        "num_params": sum(p.numel() for p in model.parameters()),
        "config": vars(cfg),
    }, indent=2))

    print(f"Saved: {model_path}, {last_path}, scaler={not cfg.disable_scaling}, meta={meta_path}")


def env_default(key: str, fallback: str) -> str:
    v = os.environ.get(key)
    return v if v else fallback


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LSTM classifier (supported flags only).")
    p.add_argument("--data", type=str, default=env_default("SM_CHANNEL_TRAIN", "eth_1m_data"))
    p.add_argument("--output-dir", type=str, default=env_default("SM_MODEL_DIR", "./model"))
    p.add_argument("--meta-path", type=str, default="model_meta.json")

    # Model / data (optimistic defaults)
    p.add_argument("--window-size", type=int, default=192)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--disable-scaling", type=str2bool, default=False)
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)

    # Data specifics
    p.add_argument("--price-col", type=str, default="close")

    return p


def main():
    args = build_parser().parse_args()
    cfg = TrainConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        meta_path=args.meta_path,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        accumulate=args.accumulate,
        seed=args.seed,
        price_col=args.price_col,
        disable_scaling=args.disable_scaling,
    )
    train_loop(cfg)


if __name__ == "__main__":
    main()