# aws_train_model.py
from __future__ import annotations

import os
import argparse
import json
import random
import glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from joblib import dump

SM_TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


# ------------------------------
# Meta loader (honors env MODEL_META_PATH)
# ------------------------------
DEFAULT_FEATURE_COLS = [
    "open", "high", "low", "close", "body", "range", "upper_wick", "lower_wick",
    "return", "sma_ratio", "ema_20", "macd", "rsi_14", "vol_change", "atr",
    "price_vs_hourly_trend", "bb_width",
]

def load_model_meta() -> Dict[str, Any]:
    candidates = [
        os.getenv("MODEL_META_PATH"),
        os.path.join(os.getcwd(), "model_meta.json"),
        os.path.join(SM_MODEL_DIR, "model_meta.json"),
    ]
    for p in [c for c in candidates if c]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    j = json.load(f)
                if isinstance(j, dict):
                    if "features" in j and "feature_cols" not in j:
                        j["feature_cols"] = j["features"]
                    if "feature_cols" not in j:
                        j["feature_cols"] = DEFAULT_FEATURE_COLS
                    if "window_size" not in j:
                        j["window_size"] = 150
                    return j
            except Exception:
                pass
    return {"feature_cols": DEFAULT_FEATURE_COLS, "window_size": 150}


# ------------------------------
# Reproducibility
# ------------------------------
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# CSV schema normalization
# ------------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

BINANCE_12 = {0: "ts", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"}

def _map_ohlcv_columns(df: pd.DataFrame, path: str) -> pd.DataFrame:
    cols = list(df.columns)
    norm = [_norm(c) for c in cols]
    out = df.copy()

    # Typical headered CSV
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
        # Try Binance klines layout
        if len(cols) >= 6:
            rename = {c: BINANCE_12.get(i, f"c{i}") for i, c in enumerate(cols)}
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


# ------------------------------
# Feature engineering
# ------------------------------
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
    out = df.copy().sort_index()

    out["body"] = out["close"] - out["open"]
    out["range"] = out["high"] - out["low"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["return"] = out["close"].pct_change().fillna(0.0)

    sma20 = out["close"].rolling(20, min_periods=1).mean()
    out["sma_ratio"] = (out["close"] / (sma20 + 1e-12)).astype(np.float32)
    out["ema_20"] = _ema(out["close"], 20)

    ema12 = _ema(out["close"], 12)
    ema26 = _ema(out["close"], 26)
    out["macd"] = (ema12 - ema26)

    out["rsi_14"] = _rsi(out["close"], 14)

    out["vol_change"] = out["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    prev_close = out["close"].shift(1)
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14, min_periods=1).mean()

    hourly = out["close"].rolling(60, min_periods=1).mean()
    out["price_vs_hourly_trend"] = out["close"] / (hourly + 1e-12)

    std20 = out["close"].rolling(20, min_periods=1).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_width"] = (upper - lower) / (sma20 + 1e-12)

    if compat_inf_to_zero:
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


# ------------------------------
# Sequences (emit targets only from current month)
# ------------------------------
def build_sequences_from_features(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    carryover_feat: Optional[np.ndarray],
    carryover_closes: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_mat = feat_df[feature_cols].to_numpy(dtype=np.float32)
    closes = feat_df["close"].to_numpy(dtype=np.float32)

    carry_len = 0
    if carryover_feat is not None:
        carry_len = len(carryover_feat)
        feat_mat = np.concatenate([carryover_feat, feat_mat], axis=0)
        closes = np.concatenate([carryover_closes, closes], axis=0)

    start = max(window_size, carry_len)  # ensure targets land in "new" part
    X, y = [], []
    for i in range(start, len(feat_mat)):
        X.append(feat_mat[i - window_size : i])
        y.append(float(closes[i] > closes[i - 1]))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    tail_len = min(window_size, len(feat_mat))
    next_carry_feat = feat_mat[-tail_len:].copy()
    next_carry_close = closes[-tail_len:].copy()
    return X, y, next_carry_feat, next_carry_close


# ------------------------------
# Model
# ------------------------------
class LSTMClassifier(nn.Module):
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
        _, (hn, _) = self.lstm(x)
        last = hn[-1]
        return self.head(last).squeeze(-1)


# ------------------------------
# Training
# ------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", type=bool, default=False)
    ap.add_argument("--window_size", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_months", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--accumulate", type=int, default=1)
    args, _ = ap.parse_known_args()

    set_seeds(42)

    meta_in = load_model_meta()
    feature_cols = list(meta_in.get("feature_cols", DEFAULT_FEATURE_COLS))
    window_size = int(meta_in.get("window_size", args.window_size or 150))

    # Files & split
    all_paths = sorted(glob.glob(os.path.join(SM_TRAIN_DIR, "**/*.csv"), recursive=True))
    if not all_paths:
        raise RuntimeError(f"No CSV files found under {SM_TRAIN_DIR}")

    val_months = max(0, int(args.val_months))
    if val_months >= len(all_paths):
        val_months = max(0, len(all_paths) - 1)

    train_paths = all_paths[:-val_months] if val_months > 0 else all_paths
    val_paths = all_paths[-val_months:] if val_months > 0 else []

    if len(train_paths) == 0:
        raise RuntimeError(
            "No training files after splitting. Reduce --val_months or add more CSVs."
        )

    print(f"[files] train={len(train_paths)} val={len(val_paths)}")

    # -------- PASS 1: fit scaler on TRAIN months, skip empties --------
    scaler = StandardScaler()
    fitted_rows = 0
    carry_raw = None

    for p in train_paths:
        df = _read_csv_robust(p)
        if carry_raw is not None:
            df = pd.concat([carry_raw, df]).sort_index()

        feat_all = build_features(df, compat_inf_to_zero=True)
        offset = len(carry_raw) if carry_raw is not None else 0
        feat_new = feat_all.iloc[offset:]  # positional slice (key fix)

        if feat_new.empty:
            print(f"[warn] no new rows in {p} after stitching; skipping")
        else:
            # column presence check
            missing = [c for c in feature_cols if c not in feat_new.columns]
            if missing:
                raise RuntimeError(f"Missing features {missing} in file {p}")
            scaler.partial_fit(feat_new[feature_cols].to_numpy(np.float32))
            fitted_rows += len(feat_new)

        carry_raw = df.tail(2000)

    if not hasattr(scaler, "n_features_in_") or fitted_rows == 0:
        raise RuntimeError(
            "Scaler did not fit: no usable training rows. "
            "Ensure at least one non-empty TRAIN month and a valid --val_months."
        )
    print(f"[scaler] fitted on ~{fitted_rows:,} rows")

    # -------- Train / Validate --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        bidirectional=bool(args.bidirectional),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.BCEWithLogitsLoss()

    def train_batches(X: np.ndarray, y: np.ndarray, batch_size: int, accumulate: int) -> float:
        if len(X) == 0:
            return 0.0
        model.train()
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]

        total_loss, steps = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y[i:i+batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            loss = loss_fn(logits, yb) / max(1, accumulate)
            loss.backward()
            steps += 1
            if steps % accumulate == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * max(1, accumulate)
        if steps % max(1, accumulate) != 0:
            opt.step(); opt.zero_grad(set_to_none=True)
        return total_loss / max(1, steps)

    def eval_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        if len(X) == 0:
            return 0.0
        model.eval()
        total, steps = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
                yb = torch.tensor(y[i:i+batch_size], dtype=torch.float32, device=device)
                logits = model(xb)
                total += loss_fn(logits, yb).item()
                steps += 1
        return total / max(1, steps)

    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    best = float("inf")

    for epoch in range(int(args.epochs)):
        print(f"\n=== epoch {epoch+1}/{int(args.epochs)} ===")

        # ---- Train over months
        carry_feat = None
        carry_close = None
        month_losses: List[float] = []

        for p in train_paths:
            df = _read_csv_robust(p)
            feat_all = build_features(df, compat_inf_to_zero=True)

            # scale full month (continuity handled in sequence builder)
            miss = [c for c in feature_cols if c not in feat_all.columns]
            if miss:
                raise RuntimeError(f"Missing features {miss} in file {p}")

            feat_scaled = feat_all.copy()
            feat_scaled[feature_cols] = scaler.transform(
                feat_scaled[feature_cols].to_numpy(np.float32)
            )

            X, y, carry_feat, carry_close = build_sequences_from_features(
                feat_scaled, feature_cols, window_size, carry_feat, carry_close
            )
            if len(X) == 0:
                print(f"[warn] month {p} produced 0 training sequences; skipping")
                continue

            loss = train_batches(
                X, y, batch_size=int(args.batch_size), accumulate=int(args.accumulate)
            )
            month_losses.append(loss)

        if month_losses:
            print(f"[train] mean_loss={float(np.mean(month_losses)):.6f}")

        # ---- Validate
        val_losses: List[float] = []
        if val_paths:
            carry_feat_v = None
            carry_close_v = None
            for p in val_paths:
                df = _read_csv_robust(p)
                feat_all = build_features(df, compat_inf_to_zero=True)

                miss = [c for c in feature_cols if c not in feat_all.columns]
                if miss:
                    raise RuntimeError(f"Missing features {miss} in VAL file {p}")

                feat_scaled = feat_all.copy()
                feat_scaled[feature_cols] = scaler.transform(
                    feat_scaled[feature_cols].to_numpy(np.float32)
                )

                Xv, yv, carry_feat_v, carry_close_v = build_sequences_from_features(
                    feat_scaled, feature_cols, window_size, carry_feat_v, carry_close_v
                )
                if len(Xv) == 0:
                    print(f"[warn] month {p} produced 0 validation sequences; skipping")
                    continue

                val_losses.append(eval_batches(Xv, yv, batch_size=int(args.batch_size)))

        v_loss = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"val_loss={v_loss:.6f}", flush=True)  # metric for SageMaker/Hyperparam Tuning

        if val_losses and v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), os.path.join(SM_MODEL_DIR, "model.pt"))

    # ---- Save artifacts ----
    dump(scaler, os.path.join(SM_MODEL_DIR, "scaler.joblib"))
    meta_out: Dict[str, Any] = {
        **meta_in,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "input_size": len(feature_cols),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "bidirectional": bool(args.bidirectional),
        "scaler_type": "standard",
        "label_def": "next_bar_up",
    }
    with open(os.path.join(SM_MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)


if __name__ == "__main__":
    main()
