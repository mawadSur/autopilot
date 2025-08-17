#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified backtest script with pretty $ output.

Modes
-----
--mode simple
    Quick accuracy/precision-style report (no capital curve).

--mode portfolio
    Full equity simulation with configurable fees/slippage and shorting.

Assumptions
-----------
- Trained artifacts live in ./model (or pass --model-dir).
- Data lives in ./eth_1m_data (or pass --data-dir).
- If CSVs are headerless, we auto-assign OHLCV headers.

Examples
--------
python src/backtest.py --mode simple
python src/backtest.py --mode portfolio --capital 10000 --allow-shorts
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# optional scaler
try:
    import joblib
except Exception:
    joblib = None

# Our shared model helpers
try:
    # Requires models.py with LSTMClassifier + ModelMeta in PYTHONPATH
    from models import LSTMClassifier, ModelMeta  # type: ignore
except Exception as e:
    raise SystemExit(
        "Could not import LSTMClassifier/ModelMeta from models.py. "
        "Make sure models.py is on PYTHONPATH.\n"
        f"Underlying error: {e}"
    )

# ---------------------------
# CSV header repair (for headerless files)
# ---------------------------

DEFAULT_COLS_6 = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_COLS_7 = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
PRICE_CANDIDATES = ["close", "adj_close", "adj close", "close_price", "price", "last", "mid", "c"]

def _columns_look_headerless(cols: List[str]) -> bool:
    lowers = [str(c).strip().lower() for c in cols]
    if any(k in lowers for k in ["open", "high", "low", "close", "volume", "timestamp", "time", "c", "o", "h", "l", "v"]):
        return False
    numeric_like = 0
    for c in cols:
        s = str(c).strip().replace(".", "", 1).replace("-", "", 1)
        if s.isdigit():
            numeric_like += 1
    return numeric_like >= max(3, len(cols) // 2)

def _apply_default_headers(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[1]
    if n == 6:
        df.columns = DEFAULT_COLS_6
    elif n == 7:
        df.columns = DEFAULT_COLS_7
    else:
        df.columns = [f"col{i}" for i in range(n)]
    return df

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    if _columns_look_headerless(list(df.columns)):
        df = _apply_default_headers(df)
    return df

def read_csv_concat_sorted(data_dir: str) -> pd.DataFrame:
    """Read all CSVs (or a single file) and return one concatenated DataFrame with normalized headers."""
    p = Path(data_dir)
    files: List[str]
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
    else:
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Expected .csv file or directory, got: {data_dir}")
        files = [str(p)]
    parts = []
    for f in files:
        df = pd.read_csv(f)
        parts.append(_normalize_headers(df))
    out = pd.concat(parts, ignore_index=True)
    return out

def resolve_price_col(columns: List[str], preferred: Optional[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in columns}
    if preferred:
        if preferred in columns:
            return preferred
        if preferred.lower() in lower_map:
            return lower_map[preferred.lower()]
    for cand in PRICE_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]
    return None

# ---------------------------
# Model loading
# ---------------------------

def load_model_bundle(model_dir: str):
    """Load model_meta.json, weights, and scaler.joblib (optional)."""
    meta_path = Path(model_dir) / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta.json not found in {model_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = LSTMClassifier(
        input_size=int(meta["input_size"]),
        hidden_size=int(meta["hidden_size"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta.get("dropout", 0.0)),
        bidirectional=bool(meta.get("bidirectional", False)),
        num_classes=int(meta["num_classes"]),
    )
    weights_path = Path(model_dir) / meta.get("model_state_path", "model.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    scaler = None
    scaler_rel = meta.get("scaler_path")
    if scaler_rel:
        sp = Path(model_dir) / scaler_rel
        if sp.exists() and joblib is not None:
            scaler = joblib.load(sp)
    return model.eval(), scaler, meta

# ---------------------------
# Window building (same as training)
# ---------------------------

def build_windows(features: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = features.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=np.float32)
    stride0, stride1 = features.strides
    shape = (N - seq_len + 1, seq_len, F)
    strides = (stride0, stride0, stride1)
    return np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides).copy()

# ---------------------------
# Pretty formatting
# ---------------------------

def _fmt_money(x, currency="$"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1e12:
        return f"{currency}{x:.3e}"
    return f"{currency}{x:,.2f}"

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return str(x)

def print_portfolio_report(report: Dict, currency="$") -> None:
    m = (report or {}).get("metrics", {}) or {}
    p = (report or {}).get("portfolio", {}) or {}
    n = int(m.get("n", 0))
    start = p.get("start_capital", 0.0)
    end = p.get("end_equity", None)
    trades = int(p.get("trades", 0))
    mdd = p.get("max_drawdown", None)

    multiple = None
    ret_frac = None
    try:
        if start not in (None, 0) and end is not None and math.isfinite(float(start)) and math.isfinite(float(end)):
            multiple = float(end) / float(start)
            ret_frac = multiple - 1.0
    except Exception:
        pass

    print("\n=== PORTFOLIO MODE — SUMMARY ===")
    print(f"Bars processed : {n:,}")
    print(f"Trades         : {trades:,}")
    print(f"Start capital  : {_fmt_money(start, currency)}")
    print(f"End equity     : {_fmt_money(end, currency)}")
    if multiple is not None and math.isfinite(multiple):
        if multiple >= 1e6:
            print(f"Return         : ×{multiple:.3e}  (⚠️ extremely large; check return scaling)")
        else:
            print(f"Return         : {_fmt_pct(ret_frac)}  (×{multiple:.2f})")
    else:
        print("Return         : —")
    if mdd is not None:
        print(f"Max drawdown   : {_fmt_pct(mdd)}")
    print("")  # newline

# ---------------------------
# Backtest logic
# ---------------------------

def simulate_portfolio(close: np.ndarray,
                      signals: np.ndarray,
                      start_capital: float = 10_000.0,
                      allow_shorts: bool = False,
                      fee_bps: float = 1.0,
                      slippage_bps: float = 1.0) -> Tuple[Dict, pd.DataFrame]:
    """
    Vectorized equity simulation.
    - signals in {-1, 0, +1}. If allow_shorts=False, negatives are clipped to 0.
    - per-bar fractional return r_t = close[t+1]/close[t] - 1
    - cost (bps) applied when position changes.

    Safety guards to prevent absurd equity explosions:
      * Clip per-bar |r_t| to 20% to guard against bad ticks.
      * Bound signals to [-1, 1].
      * Ensure multiplicative gross return never drops below epsilon.
    """
    c = close.astype(float)
    r = np.zeros_like(c, dtype=float)
    r[:-1] = c[1:] / c[:-1] - 1.0
    # Clip outliers to ±20% per bar (defensive; adjust if your data warrants)
    np.clip(r, -0.2, 0.2, out=r)

    pos = signals.astype(float).copy()
    if not allow_shorts:
        pos = np.maximum(pos, 0.0)  # 0 or +1
    pos = np.clip(pos, -1.0, 1.0)

    # Trades: position changes
    pos_shift = np.roll(pos, 1)
    pos_shift[0] = 0.0
    changed = (pos != pos_shift).astype(float)
    trades = int(changed.sum())

    # Transaction costs when changing position
    cost_frac = (fee_bps + slippage_bps) / 1e4
    costs = changed * cost_frac

    # Equity curve
    equity = np.empty_like(r)
    equity[0] = start_capital
    for t in range(1, len(r)):
        gross = 1.0 + pos[t-1] * r[t-1]           # use prior position over [t-1, t)
        gross = max(gross, 1e-6)                  # never allow <= 0 multiplicative return
        equity[t] = equity[t-1] * gross * (1.0 - costs[t])  # pay cost when we *enter* new pos at t

    # Max drawdown (reported as positive fraction)
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    report = {
        "metrics": {"n": int(len(r))},
        "portfolio": {
            "start_capital": float(start_capital),
            "end_equity": float(equity[-1]),
            "return": float(equity[-1] / max(1e-12, start_capital) - 1.0),
            "max_drawdown": float(abs(max_dd)),
            "trades": int(trades),
        },
    }
    df_curve = pd.DataFrame({"close": c, "equity": equity, "pos": pos})
    return report, df_curve

# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified backtester")
    p.add_argument("--mode", choices=["simple", "portfolio"], default="portfolio")
    p.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or single CSV")
    p.add_argument("--model-dir", type=str, default="model", help="Where model_meta.json & model.pt live")
    p.add_argument("--seq-len", type=int, default=60, help="Sequence/window length used in training")
    p.add_argument("--label-col", type=str, default="label", help="Optional label col (not required)")
    p.add_argument("--price-col", type=str, default="close")

    # portfolio-only options
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--allow-shorts", action="store_true")
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)

    # decision threshold / mapping
    p.add_argument("--long-class", type=int, default=2, help="Which class means LONG")
    p.add_argument("--short-class", type=int, default=0, help="Which class means SHORT")

    return p

# ---------------------------
# Main
# ---------------------------

def main():
    args = build_argparser().parse_args()

    # Load data
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), args.price_col)
    if price_col is None:
        raise SystemExit(
            f"Could not find a price column. Tried '{args.price_col}' and aliases {PRICE_CANDIDATES}. "
            f"Available: {list(df.columns)}"
        )

    # Features for inference: numeric columns except obvious non-features
    drop_cols = {price_col, args.label_col, "timestamp", "time"}
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    if not feat_cols:
        raise SystemExit("No numeric feature columns found for inference.")

    X_flat = df[feat_cols].to_numpy(dtype=np.float32)
    X = build_windows(X_flat, args.seq_len)  # [N, T, F]
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce --seq-len.")

    # Align price series to window end
    close = df[price_col].to_numpy(dtype=float)[args.seq_len - 1:]

    # Load model & scaler
    model, scaler, meta = load_model_bundle(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Apply scaler if available
    if scaler is not None:
        n, t, f = X.shape
        X = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)

    # Predict
    with torch.no_grad():
        logits = []
        BS = 1024
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i+BS]).to(device)
            out = model(xb)  # [B, C]
            logits.append(out.cpu())
        logits = torch.cat(logits, dim=0)
        probs = F.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=-1)

    if args.mode == "simple":
        # quick counts if you have labels
        out = {"metrics": {"n": int(len(preds))}}
        if args.label_col in df.columns:
            y = df[args.label_col].to_numpy(dtype=int)[args.seq_len - 1:]
            acc = float((y == preds).mean())
            out["metrics"]["accuracy"] = acc
        print(json.dumps(out, indent=2))
        return

    # portfolio mode: map class -> position
    signals = np.zeros_like(preds, dtype=int)
    signals[preds == int(args.long_class)] = 1
    signals[preds == int(args.short_class)] = -1

    report, curve = simulate_portfolio(
        close=close,
        signals=signals,
        start_capital=float(args.capital),
        allow_shorts=bool(args.allow_shorts),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
    )

    # Pretty output ($ values)
    print_portfolio_report(report, currency="$")

    # If you still want the raw JSON, uncomment:
    # print("Raw JSON:\n" + json.dumps(report, indent=2))

if __name__ == "__main__":
    main()