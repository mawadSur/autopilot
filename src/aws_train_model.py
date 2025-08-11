from __future__ import annotations

import os
import argparse
import json
import random
import glob
from collections import deque
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Single-source feature pipeline
from utils import build_features, load_meta, DEFAULT_FEATURE_COLS

SM_TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


# ---------------- Model ----------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (hn, _) = self.lstm(x)
        last = hn[-1]  # (B, H)
        return self.head(last).squeeze(-1)  # logits


# -------------- Misc utils --------------
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Robust CSV normalization (works with headerless & Binance 12-col klines) ----------
def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _parse_ts(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() >= max(1, int(0.5 * len(series))):
        m = s.dropna().median()
        unit = "us" if m > 1e14 else ("ms" if m > 1e12 else "s")
        return pd.to_datetime(s.astype("int64"), unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

def _map_ohlcv_columns(df: pd.DataFrame, path: str) -> pd.DataFrame:
    norm_to_orig = {_norm(c): c for c in df.columns}
    def find(aliases):  # original name for first alias that exists
        for a in aliases:
            if a in norm_to_orig:
                return norm_to_orig[a]
        return None

    ts_col   = find(["timestamp","time","date","datetime","opentime","open_time","t","starttime"])
    open_col = find(["open","o","openprice"])
    high_col = find(["high","h","highprice"])
    low_col  = find(["low","l","lowprice"])
    close_col= find(["close","c","closeprice","adjclose"])
    vol_col  = find(["volume","vol","baseassetvolume","basevolume","takerbasevol","takerbasevolume","quoteassetvolume"])

    if not all([ts_col, open_col, high_col, low_col, close_col, vol_col]):
        raise KeyError(f"Missing OHLCV columns in {path}. Headers={list(df.columns)[:10]}")

    out = pd.DataFrame({
        "ts": _parse_ts(df[ts_col]),
        "open":   pd.to_numeric(df[open_col], errors="coerce"),
        "high":   pd.to_numeric(df[high_col], errors="coerce"),
        "low":    pd.to_numeric(df[low_col], errors="coerce"),
        "close":  pd.to_numeric(df[close_col], errors="coerce"),
        "volume": pd.to_numeric(df[vol_col], errors="coerce"),
    })
    out = out.dropna(subset=["ts","open","high","low","close","volume"]).set_index("ts").sort_index()
    return out[["open","high","low","close","volume"]]

def _read_csv_robust(path: str) -> pd.DataFrame:
    # Try normal headered CSV
    try:
        df = pd.read_csv(path)
        if all(str(c).replace(".", "", 1).isdigit() for c in df.columns):
            raise ValueError("numeric headers -> headerless")
        return _map_ohlcv_columns(df, path)
    except Exception:
        pass
    # Headerless fallbacks
    probe = pd.read_csv(path, header=None, nrows=1)
    n = probe.shape[1]
    if n == 6:
        names = ["timestamp","open","high","low","close","volume"]
    elif n >= 12:
        names = [
            "timestamp","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore",
        ] + [f"extra_{i}" for i in range(n - 12)]
    else:
        names = [f"col{i}" for i in range(n)]
    df2 = pd.read_csv(path, header=None, names=names)
    return _map_ohlcv_columns(df2, path)


# ---------- File iterator (bounded memory) ----------
def iter_month_frames(train_dir: str) -> Iterable[pd.DataFrame]:
    """Yield normalized OHLCV frames, one file (month) at a time."""
    paths = sorted(glob.glob(os.path.join(train_dir, "**/*.csv"), recursive=True))
    if not paths:
        raise RuntimeError(f"No CSV files found under {train_dir}")
    for p in paths:
        try:
            df = _read_csv_robust(p)
            yield df
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}")


# ---------- Sequence builder (streaming, with window carryover) ----------
def build_sequences_from_features(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    carryover_feat: np.ndarray | None,
    carryover_closes: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised sequences for ONE month of features, using carryovers from previous month
    so windows remain continuous across month boundaries.

    Returns: (X, y, next_carry_feat_tail, next_carry_close_tail)
    """
    feat_mat = feat_df[feature_cols].to_numpy(dtype=np.float32)
    closes = feat_df["close"].to_numpy(dtype=np.float32)

    if carryover_feat is not None:
        feat_mat = np.concatenate([carryover_feat, feat_mat], axis=0)
        closes = np.concatenate([carryover_closes, closes], axis=0)

    Xs, ys = [], []
    # Label definition: y_t = 1 if close_{t+1} > close_{t}, else 0
    # We emit (window ending at t) paired with label for t, so need t+1 to exist
    for end in range(window_size - 1, len(feat_mat) - 1):
        start = end - (window_size - 1)
        Xs.append(feat_mat[start : end + 1, :])
        ys.append(1.0 if closes[end + 1] > closes[end] else 0.0)

    X = np.stack(Xs, axis=0) if Xs else np.zeros((0, window_size, len(feature_cols)), np.float32)
    y = np.array(ys, dtype=np.float32)

    # Prepare tails for next month
    tail_feat = feat_mat[-(window_size - 1):] if len(feat_mat) >= window_size - 1 else feat_mat.copy()
    tail_close = closes[-(window_size - 1):] if len(closes) >= window_size - 1 else closes.copy()
    return X, y, tail_feat, tail_close


# -------------- Train (two-pass, streaming by month) --------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", type=bool, default=False)
    ap.add_argument("--window_size", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_months", type=int, default=1, help="Use the last N months for validation.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps.")
    args, _ = ap.parse_known_args()

    set_seeds(42)

    # ---- Metadata / feature schema ----
    meta_path = os.path.join(SM_MODEL_DIR, "model_meta.json")
    prev_meta = load_meta(meta_path) if os.path.exists(meta_path) else load_meta()
    feature_cols = prev_meta.get("feature_cols", DEFAULT_FEATURE_COLS)
    window_size = int(args.window_size)

    # ---- Enumerate files once to split train/val by month ----
    all_paths = sorted(glob.glob(os.path.join(SM_TRAIN_DIR, "**/*.csv"), recursive=True))
    if not all_paths:
        raise RuntimeError(f"No CSV files found under {SM_TRAIN_DIR}")
    split = max(1, int(args.val_months))
    train_paths = all_paths[:-split] if len(all_paths) > split else all_paths[:-1]
    val_paths = all_paths[-split:] if len(all_paths) > split else all_paths[-1:]

    print(f"[files] train months={len(train_paths)}  val months={len(val_paths)}")

    # ---- PASS 1: Fit scaler incrementally on TRAIN months only (row-wise features, not sequences) ----
    scaler = StandardScaler()
    rows_count = 0
    carry_raw = None
    for p in train_paths:
        df = _read_csv_robust(p)
        # Keep indicator continuity with a modest warmup tail of raw bars
        if carry_raw is not None:
            df = pd.concat([carry_raw, df]).sort_index()
        feat = build_features(df, compat_inf_to_zero=True)
        # Keep only the rows that belong to this file (avoid refitting on previous month)
        if carry_raw is not None:
            feat = feat.loc[df.index[len(carry_raw):]]
        arr = feat[feature_cols].to_numpy(dtype=np.float32)
        if len(arr):
            scaler.partial_fit(arr)
            rows_count += len(arr)
        # prepare small raw tail for next month (enough for indicators)
        carry_raw = df.tail(2000)
    print(f"[scaler] fitted on ~{rows_count:,} feature rows")

    # ---- Build model ----
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        bidirectional=bool(args.bidirectional),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.BCEWithLogitsLoss()

    def train_batches(X: np.ndarray, y: np.ndarray, batch_size: int, accumulate: int) -> float:
        if len(X) == 0:
            return 0.0
        model.train()
        total_loss, steps = 0.0, 0
        # shuffle per month chunk
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]; y = y[idx]
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y[i:i+batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            loss = loss_fn(logits, yb) / accumulate
            loss.backward()
            if (steps + 1) % accumulate == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * accumulate
            steps += 1
        # flush leftover grads
        if steps % accumulate != 0:
            opt.step(); opt.zero_grad(set_to_none=True)
        return total_loss / max(1, steps)

    def eval_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        if len(X) == 0:
            return 0.0
        model.eval()
        tot, steps = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
                yb = torch.tensor(y[i:i+batch_size], dtype=torch.float32, device=device)
                logits = model(xb)
                loss = loss_fn(logits, yb).item()
                tot += loss; steps += 1
        return tot / max(1, steps)

    # ---- PASS 2: Train epoch-by-epoch, month-by-month (bounded memory) ----
    best = float("inf")
    os.makedirs(SM_MODEL_DIR, exist_ok=True)

    for epoch in range(int(args.epochs)):
        # --- Train over train months ---
        carry_feat = None
        carry_close = None
        epoch_train_loss, month_ct = 0.0, 0
        for p in train_paths:
            df = _read_csv_robust(p)
            # maintain indicator continuity with small raw tail
            if month_ct > 0:
                df = pd.concat([carry_raw, df]).sort_index()
            feat = build_features(df, compat_inf_to_zero=True)
            # only new rows for this month
            if month_ct > 0:
                feat = feat.loc[df.index[len(carry_raw):]]

            # scale row-wise features then form sequences (with carryover windows)
            feat_scaled = feat.copy()
            feat_scaled[feature_cols] = scaler.transform(feat_scaled[feature_cols].to_numpy(np.float32))

            X, y, carry_feat, carry_close = build_sequences_from_features(
                feat_scaled, feature_cols, window_size, carry_feat, carry_close
            )

            # prepare next raw carry for indicators continuity
            carry_raw = df.tail(2000)

            if len(X):
                loss = train_batches(X, y, batch_size=int(args.batch_size), accumulate=int(args.accumulate))
                epoch_train_loss += loss
                month_ct += 1

        epoch_train_loss = epoch_train_loss / max(1, month_ct)

        # --- Evaluate on validation months without storing everything at once ---
        val_losses: List[float] = []
        carry_feat_v = None; carry_close_v = None; carry_raw_v = None
        for p in val_paths:
            df = _read_csv_robust(p)
            if carry_raw_v is not None:
                df = pd.concat([carry_raw_v, df]).sort_index()
            feat = build_features(df, compat_inf_to_zero=True)
            if carry_raw_v is not None:
                feat = feat.loc[df.index[len(carry_raw_v):]]
            feat_scaled = feat.copy()
            feat_scaled[feature_cols] = scaler.transform(feat_scaled[feature_cols].to_numpy(np.float32))
            Xv, yv, carry_feat_v, carry_close_v = build_sequences_from_features(
                feat_scaled, feature_cols, window_size, carry_feat_v, carry_close_v
            )
            carry_raw_v = df.tail(2000)
            if len(Xv):
                val_losses.append(eval_batches(Xv, yv, batch_size=int(args.batch_size)))

        v_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # Emit the metric the tuner looks for
        print(f"val_loss={v_loss:.6f}", flush=True)

        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), os.path.join(SM_MODEL_DIR, "model.pt"))

    # ---- Save scaler + meta ----
    dump(scaler, os.path.join(SM_MODEL_DIR, "scaler.joblib"))
    meta: Dict[str, Any] = {
        **prev_meta,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "input_size": len(feature_cols),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "bidirectional": bool(args.bidirectional),
        "scaler_type": "standard",
        "buy_threshold": float(prev_meta.get("buy_threshold", 0.5)),
        "sell_threshold": float(prev_meta.get("sell_threshold", 0.5)),
        "label_def": "next_bar_up",
    }
    with open(os.path.join(SM_MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
