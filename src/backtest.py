#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Backtesting Script (defaults to project-root data & model)

Defaults:
  --data      -> <project_root>/eth_1m_data
  --model-dir -> <project_root>          (expects model_meta.json, model.pt, scaler.joblib)

Usage:
  Simple/fast metrics:
    python src/backtest.py --mode simple

  Full portfolio simulation with starting capital:
    python src/backtest.py --mode portfolio --capital 10000 --allow-shorts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

# ----- project root & import helpers -----
# This file lives in <project_root>/src/backtest.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure we can import models.py from project root even when running `python src/backtest.py`
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import unified model utilities
from models import (  # type: ignore
    load_meta,
    build_model_from_meta,
    load_model_state,
    load_scaler,
    resolve_path,
)

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified backtesting script (simple vs portfolio modes).")

    # Modes
    p.add_argument("--mode", choices=["simple", "portfolio"], default="simple",
                   help="Backtesting mode: 'simple' for fast metrics, 'portfolio' for equity simulation.")

    # Data & columns
    p.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "eth_1m_data"),
        help="Path to CSV/Parquet file OR a directory containing multiple CSVs "
             f"(default: {PROJECT_ROOT / 'eth_1m_data'})"
    )
    p.add_argument("--time-col", type=str, default=None,
                   help="Optional time column name; if provided and reading a folder, result is sorted by this column.")
    p.add_argument("--price-col", type=str, default="close",
                   help="Price column used for P&L and portfolio sim.")
    p.add_argument("--label-col", type=str, default=None,
                   help="Optional ground-truth label column (0/1/2). Used for accuracy metrics.")
    p.add_argument("--feature-cols", type=str, nargs="*", default=None,
                   help="Explicit list of feature columns. If omitted, auto-select numeric columns excluding label/time/price.")
    p.add_argument("--sequence-column", type=str, default=None,
                   help="If your DataFrame already includes prebuilt sequences (list/np-array), give the column name here. "
                        "If set, we won't window raw features.")
    p.add_argument("--seq-len", type=int, default=60,
                   help="Sequence length when building windows from flat features.")
    p.add_argument("--drop-na", action="store_true",
                   help="Drop rows with NA after feature selection and sequence alignment.")

    # Inference controls
    p.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Directory containing model_meta.json and weights (default: project root)"
    )
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip model inference; use existing prediction columns.")
    p.add_argument("--pred-col", type=str, default=None,
                   help="Column with integer class predictions, used if --skip-inference.")
    p.add_argument("--prob-col", type=str, default=None,
                   help="Column with model confidence/probability for the predicted class, optional.")
    p.add_argument("--class-up", type=int, default=2,
                   help="Class index representing 'UP' signal.")
    p.add_argument("--class-down", type=int, default=0,
                   help="Class index representing 'DOWN' signal.")
    p.add_argument("--class-flat", type=int, default=1,
                   help="Class index representing 'FLAT' / no-trade.")

    # Decision thresholds
    p.add_argument("--min-prob-buy", type=float, default=0.0,
                   help="Minimum probability to accept a long (UP) signal.")
    p.add_argument("--min-prob-sell", type=float, default=0.0,
                   help="Minimum probability to accept a short (DOWN) signal.")
    p.add_argument("--allow-shorts", action="store_true",
                   help="Enable short trades for DOWN predictions.")

    # Simple-mode P&L params
    p.add_argument("--fee-bps", type=float, default=2.0,
                   help="Per-trade fee in basis points for simple P&L (0.01% = 1 bps).")
    p.add_argument("--slippage-bps", type=float, default=0.0,
                   help="Per-trade slippage in basis points for simple P&L.")

    # Portfolio mode params
    p.add_argument("--capital", type=float, default=10000.0,
                   help="Starting capital for portfolio simulation (used in --mode portfolio).")
    p.add_argument("--risk-per-trade", type=float, default=0.01,
                   help="Fraction of equity risked per trade (e.g., 0.01 = 1%).")
    p.add_argument("--stop-loss-bps", type=float, default=50.0,
                   help="Stop-loss distance in bps from entry (e.g., 50 = 0.5%).")
    p.add_argument("--take-profit-bps", type=float, default=100.0,
                   help="Take-profit distance in bps from entry (e.g., 100 = 1.0%).")
    p.add_argument("--portfolio-fee-bps", type=float, default=2.0,
                   help="Fee in bps applied on each entry and exit in portfolio mode.")
    p.add_argument("--portfolio-slippage-bps", type=float, default=0.0,
                   help="Slippage in bps applied on each entry and exit in portfolio mode.")
    p.add_argument("--max-hold", type=int, default=None,
                   help="Max bars to hold a position. If None, hold until SL/TP or signal flip.")

    # Output
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to write a CSV with predictions and backtest annotations.")
    return p

# -----------------------------
# Data Loading
# -----------------------------

def load_table(path: str, time_col: Optional[str]) -> pd.DataFrame:
    """
    - If `path` is a directory: read and concat all *.csv (sorted by filename). If time_col provided, sort by it at the end.
    - If `path` is a file: support CSV and Parquet.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        return df

    ext = p.suffix.lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        return df
    # default CSV
    df = pd.read_csv(p)
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    return df

# -----------------------------
# Feature Handling & Sequences
# -----------------------------

def select_features(df: pd.DataFrame, feature_cols: Optional[List[str]],
                    drop_cols: Iterable[str]) -> List[str]:
    if feature_cols:
        return feature_cols
    drop_cols = set([c for c in drop_cols if c is not None])
    cand = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cand if c not in drop_cols]

def build_windows_from_flat(features: np.ndarray, seq_len: int) -> np.ndarray:
    """
    features: [N, F] -> windows: [N - seq_len + 1, seq_len, F]
    """
    N, F = features.shape
    if N < seq_len:
        raise ValueError(f"Not enough rows ({N}) for seq_len={seq_len}")
    # Efficient rolling windows using stride tricks
    stride0, stride1 = features.strides
    shape = (N - seq_len + 1, seq_len, F)
    strides = (stride0, stride0, stride1)
    windows = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides).copy()
    return windows

# -----------------------------
# Inference
# -----------------------------

@dataclass
class InferenceArtifacts:
    model: torch.nn.Module
    device: str
    scaler: Any
    meta: Dict[str, Any]

def _get_paths(model_dir: str) -> Tuple[str, str, Optional[str]]:
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"model_meta.json not found in {model_dir}")

    meta = load_meta(meta_path)
    weights_path = resolve_path(model_dir, meta.model_state_path)
    scaler_path = resolve_path(model_dir, meta.scaler_path) if meta.feature_scaling else None

    if not os.path.exists(weights_path):
        # Common fallbacks
        for cand in ["model.pt", "best_model.pth", "weights.pt"]:
            alt = os.path.join(model_dir, cand)
            if os.path.exists(alt):
                weights_path = alt
                break

    if not os.path.exists(weights_path):
        raise FileNotFoundError("Model weights not found; check model_meta.json or export paths.")
    return meta_path, weights_path, scaler_path

def load_model_bundle(model_dir: str) -> InferenceArtifacts:
    meta_path, weights_path, scaler_path = _get_paths(model_dir)
    meta = load_meta(meta_path)

    model = build_model_from_meta(meta)
    load_model_state(model, weights_path, strict=False)
    model.eval()

    scaler = load_scaler(scaler_path) if meta.feature_scaling else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return InferenceArtifacts(model=model, device=device, scaler=scaler, meta=meta.to_dict())

def run_inference(art: InferenceArtifacts, X_btf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_btf: [B, T, F]
    Returns:
      probs: [B, C]
      preds: [B]
    """
    # Scale if applicable: reshape [B*T, F] for sklearn scalers.
    if art.scaler is not None:
        b, t, f = X_btf.shape
        flat = X_btf.reshape(b * t, f)
        flat = art.scaler.transform(flat)
        X_btf = flat.reshape(b, t, f)

    with torch.no_grad():
        x = torch.from_numpy(X_btf.astype(np.float32)).to(art.device)
        logits = art.model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return probs, preds

# -----------------------------
# Metrics
# -----------------------------

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    out = {}
    mask = ~np.isnan(y_true)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)
    if len(y_true) == 0:
        return {"n": 0}
    acc = (y_true == y_pred).mean()
    out["n"] = int(len(y_true))
    out["accuracy"] = float(acc)

    # macro precision/recall/f1
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    out["macro_precision"] = float(np.nanmean(precisions)) if precisions else np.nan
    out["macro_recall"] = float(np.nanmean(recalls)) if recalls else np.nan
    out["macro_f1"] = float(np.nanmean(f1s)) if f1s else np.nan
    return out

# -----------------------------
# Simple-mode P&L
# -----------------------------

def simple_pnl(prices: np.ndarray,
               preds: np.ndarray,
               probs: Optional[np.ndarray],
               class_up: int,
               class_down: int,
               min_prob_buy: float,
               min_prob_sell: float,
               allow_shorts: bool,
               fee_bps: float,
               slippage_bps: float) -> Dict[str, Any]:
    """
    Naive P&L: trade 1 unit per signal; enter/exit at next bar with fee+slippage applied each time.
    Long on UP; (optional) Short on DOWN.
    """
    n = len(prices)
    if n < 2:
        return {"n": n, "pnl_units": 0.0}

    fee = fee_bps / 1e4
    slip = slippage_bps / 1e4

    pnl = 0.0
    trades = 0
    for i in range(n - 1):
        p0 = prices[i]
        p1 = prices[i + 1]
        pred = preds[i]
        prob = None if probs is None else probs[i, pred]

        if pred == class_up and (prob is None or prob >= min_prob_buy):
            entry = p0 * (1 + slip)
            exit_ = p1 * (1 - slip)
            rtn = (exit_ / entry) - 1.0
            rtn -= fee
            rtn -= fee
            pnl += rtn
            trades += 1

        elif allow_shorts and pred == class_down and (prob is None or prob >= min_prob_sell):
            entry = p0 * (1 - slip)
            exit_ = p1 * (1 + slip)
            rtn = (entry / exit_) - 1.0
            rtn -= fee
            rtn -= fee
            pnl += rtn
            trades += 1

    return {"n": n, "trades": trades, "pnl_units": float(pnl)}

# -----------------------------
# Portfolio Simulation
# -----------------------------

@dataclass
class PortfolioState:
    equity: float
    position: int       # +1 long, -1 short, 0 flat
    entry_price: float
    bars_held: int

def bps(x: float) -> float:
    return x / 1e4

def simulate_portfolio(
    prices: np.ndarray,
    preds: np.ndarray,
    probs: Optional[np.ndarray],
    *,
    start_capital: float,
    class_up: int,
    class_down: int,
    min_prob_buy: float,
    min_prob_sell: float,
    allow_shorts: bool,
    fee_bps: float,
    slippage_bps: float,
    risk_per_trade: float,
    stop_loss_bps: float,
    take_profit_bps: float,
    max_hold: Optional[int],
) -> Dict[str, Any]:
    fee = bps(fee_bps)
    slip = bps(slippage_bps)
    sl = bps(stop_loss_bps)
    tp = bps(take_profit_bps)

    state = PortfolioState(equity=start_capital, position=0, entry_price=0.0, bars_held=0)
    shares = 0.0
    peak_equity = start_capital
    max_drawdown = 0.0
    n = len(prices)
    trades = 0

    def update_drawdown(eq: float):
        nonlocal peak_equity, max_drawdown
        peak_equity = max(peak_equity, eq)
        dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

    for i in range(n - 1):
        p0 = prices[i]
        p1 = prices[i + 1]
        pred = preds[i]
        prob = None if probs is None else probs[i, pred]

        # Mark-to-market equity at p0
        if state.position != 0:
            if state.position == 1:
                mtm = shares * p0
            else:
                mtm = shares * (2 * state.entry_price - p0)
            eq = mtm
        else:
            eq = state.equity
        update_drawdown(eq)

        # Exit conditions
        if state.position != 0:
            state.bars_held += 1

            if state.position == 1:
                sl_price = state.entry_price * (1 - sl)
                tp_price = state.entry_price * (1 + tp)
                hit_sl = p1 <= sl_price
                hit_tp = p1 >= tp_price
            else:
                sl_price = state.entry_price * (1 + sl)
                tp_price = state.entry_price * (1 - tp)
                hit_sl = p1 >= sl_price
                hit_tp = p1 <= tp_price

            time_exit = (max_hold is not None) and (state.bars_held >= max_hold)
            flip_exit = ((state.position == 1 and pred == class_down and (prob is None or prob >= min_prob_sell)) or
                         (state.position == -1 and pred == class_up and (prob is None or prob >= min_prob_buy)))

            if hit_sl or hit_tp or time_exit or flip_exit:
                if state.position == 1:
                    exit_price = p1 * (1 - slip)
                    proceeds = shares * exit_price
                else:
                    exit_price = p1 * (1 + slip)
                    proceeds = shares * (2 * state.entry_price - exit_price)
                proceeds *= (1 - fee)
                state.equity = proceeds
                update_drawdown(state.equity)
                state.position = 0
                shares = 0.0
                state.entry_price = 0.0
                state.bars_held = 0

        # Entry logic (only if flat)
        if state.position == 0:
            if pred == class_up and (prob is None or prob >= min_prob_buy):
                entry = p0 * (1 + slip)
                risk = sl * entry
                dollar_risk = state.equity * risk_per_trade
                shares = max(dollar_risk / risk, 0.0) if risk > 0 else 0.0
                if shares > 0:
                    cost = shares * entry
                    cost *= (1 + fee)
                    state.entry_price = entry
                    state.position = 1
                    state.bars_held = 0
                    trades += 1

            elif allow_shorts and pred == class_down and (prob is None or prob >= min_prob_sell):
                entry = p0 * (1 - slip)
                risk = sl * entry
                dollar_risk = state.equity * risk_per_trade
                shares = max(dollar_risk / risk, 0.0) if risk > 0 else 0.0
                if shares > 0:
                    cost = shares * entry
                    cost *= (1 + fee)
                    state.entry_price = entry
                    state.position = -1
                    state.bars_held = 0
                    trades += 1

    # Final mark-to-market
    last_price = prices[-1]
    if state.position != 0:
        if state.position == 1:
            equity_end = shares * last_price
        else:
            equity_end = shares * (2 * state.entry_price - last_price)
    else:
        equity_end = state.equity

    ret = (equity_end / start_capital) - 1.0
    return {
        "start_capital": float(start_capital),
        "end_equity": float(equity_end),
        "return": float(ret),
        "max_drawdown": float(max_drawdown),
        "trades": int(trades),
    }

# -----------------------------
# Main Flow
# -----------------------------

def main():
    args = build_arg_parser().parse_args()

    # Load data (dir of CSVs or single file); if time-col provided, ensure chronological order
    df = load_table(args.data, args.time_col)

    # Features
    drop_cols = [args.label_col, args.time_col, args.price_col]
    feat_cols = select_features(df, args.feature_cols, drop_cols)

    # Prepare sequences
    if args.sequence_column:
        # Expect column with arrays of shape [T, F]
        seqs = df[args.sequence_column].to_numpy()
        X = []
        for s in seqs:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Sequence must be 2D (T,F), got {arr.shape}")
            X.append(arr)
        X_btf = np.stack(X, axis=0)
        price_vec = df[args.price_col].to_numpy()
        y_true = df[args.label_col].to_numpy() if args.label_col and args.label_col in df else np.full(len(df), np.nan)
        valid_index = np.arange(len(df))
    else:
        # Flat -> window
        feat = df[feat_cols].to_numpy(dtype=np.float32)
        X_btf = build_windows_from_flat(feat, args.seq_len)
        tail = df.iloc[args.seq_len - 1:].copy()
        price_vec = tail[args.price_col].to_numpy()
        y_true = tail[args.label_col].to_numpy() if args.label_col and args.label_col in tail else np.full(len(tail), np.nan)
        valid_index = tail.index.to_numpy()

    if args.drop_na:
        ok = np.isfinite(price_vec)
        if args.label_col and len(y_true) == len(ok):
            ok = ok & np.isfinite(y_true)
        X_btf = X_btf[ok]
        price_vec = price_vec[ok]
        y_true = y_true[ok] if len(y_true) == len(ok) else y_true
        valid_index = valid_index[ok]

    # Predictions
    if args.skip_inference:
        if args.pred_col is None:
            raise ValueError("--skip-inference requires --pred-col")
        pred_series = df.loc[valid_index, args.pred_col]
        preds = pred_series.to_numpy().astype(int)
        probs = None
    else:
        art = load_model_bundle(args.model_dir)
        probs, preds = run_inference(art, X_btf)

    # SIMPLE MODE
    if args.mode == "simple":
        if args.label_col and np.isfinite(y_true).any():
            rep = classification_report(y_true, preds)
        else:
            rep = {"n": len(preds)}

        pnl = simple_pnl(
            prices=price_vec,
            preds=preds,
            probs=None if args.skip_inference else probs,
            class_up=args.class_up,
            class_down=args.class_down,
            min_prob_buy=args.min_prob_buy,
            min_prob_sell=args.min_prob_sell,
            allow_shorts=args.allow_shorts,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )

        print("\n=== SIMPLE MODE REPORT ===")
        print(json.dumps({"metrics": rep, "pnl": pnl}, indent=2))

        if args.out:
            out_df = pd.DataFrame({
                "index": valid_index,
                "price": price_vec,
                "pred": preds,
            }).set_index("index")
            if args.label_col and np.isfinite(y_true).any():
                out_df["label"] = y_true
            if not args.skip_inference:
                maxp = probs[np.arange(len(preds)), preds]
                out_df["prob"] = maxp
            out_df.to_csv(args.out)
            print(f"\nWrote: {args.out}")
        return

    # PORTFOLIO MODE
    elif args.mode == "portfolio":
        sim = simulate_portfolio(
            prices=price_vec,
            preds=preds,
            probs=None if args.skip_inference else probs,
            start_capital=args.capital,
            class_up=args.class_up,
            class_down=args.class_down,
            min_prob_buy=args.min_prob_buy,
            min_prob_sell=args.min_prob_sell,
            allow_shorts=args.allow_shorts,
            fee_bps=args.portfolio_fee_bps,
            slippage_bps=args.portfolio_slippage_bps,
            risk_per_trade=args.risk_per_trade,
            stop_loss_bps=args.stop_loss_bps,
            take_profit_bps=args.take_profit_bps,
            max_hold=args.max_hold,
        )

        if args.label_col and np.isfinite(y_true).any():
            rep = classification_report(y_true, preds)
        else:
            rep = {"n": len(preds)}

        print("\n=== PORTFOLIO MODE REPORT ===")
        print(json.dumps({"metrics": rep, "portfolio": sim}, indent=2))

        if args.out:
            out_df = pd.DataFrame({
                "index": valid_index,
                "price": price_vec,
                "pred": preds,
            }).set_index("index")
            if args.label_col and np.isfinite(y_true).any():
                out_df["label"] = y_true
            if not args.skip_inference:
                maxp = probs[np.arange(len(preds)), preds]
                out_df["prob"] = maxp
            out_df.to_csv(args.out)
            print(f"\nWrote: {args.out}")
        return


if __name__ == "__main__":
    main()