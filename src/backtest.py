#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py — Backtester with TP/SL, reading model_meta.json and
recomputing the same features used at training.

Usage
-----
python backtest.py --data-dir eth_1m_data --model-dir model
"""

from __future__ import annotations

import argparse
import json
import os
import math as _math
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    fmt_money, fmt_pct, compute_features, load_model_bundle, normalize_headers
)

# Features are now centralized in utils.compute_features

    

# =========================
# Pretty formatting
# =========================
def print_portfolio_report(report: Dict, currency: str = "$") -> None:
    m = (report or {}).get("metrics", {}) or {}
    p = (report or {}).get("portfolio", {}) or {}
    n = int(m.get("n", 0))
    start = p.get("start_capital", 0.0)
    end = p.get("end_equity", None)
    trades = int(p.get("trades", 0))
    wins = int(p.get("wins", 0))
    losses = int(p.get("losses", 0))
    mdd = p.get("max_drawdown", None)
    multiple = float(end) / float(start) if start not in (None, 0) and end is not None else None
    print("\n=== PORTFOLIO MODE — SUMMARY ===")
    print(f"Bars processed : {n:,}")
    print(f"Trades         : {trades:,}  (wins {wins}, losses {losses}, win rate {wins/max(1,trades):.2%})")
    print(f"Start capital  : {fmt_money(start, currency)}")
    print(f"End equity     : {fmt_money(end, currency)}")
    if multiple is not None and np.isfinite(multiple):
        print(f"Return         : {fmt_pct(multiple-1.0)}  (×{multiple:.2f})")
    else:
        print("Return         : —")
    if mdd is not None:
        print(f"Max drawdown   : {fmt_pct(mdd)}")
    print("")

# =========================
# Trade simulation
# =========================
def simulate_trades_with_tp_sl(opens, highs, lows, closes, classes, *, start_capital,
                               fee_pct=0.0008, tp_pct=0.005, sl_pct=0.0025,
                               atr: Optional[np.ndarray] = None,
                               atr_tp_mult: Optional[float] = None,
                               atr_sl_mult: Optional[float] = None,
                               cooldown: int = 0,
                               slippage_pct: float = 0.0) -> Tuple[Dict, pd.DataFrame]:
    """Simulate trades for a 3-class model: 0=short, 1=hold, 2=long.

    - Enters at next bar's open based on prior bar's signal.
    - One position at a time. Tracks TP/SL intrabar; opposing signals exit at open.
    - Equity is fully deployed on each trade; fees charged on entry and exit.
    """
    n = len(closes)
    cash = float(start_capital)
    equity_curve = np.empty(n, dtype=float)
    equity_curve[0] = cash

    pos = 0  # 0=flat, +1=long, -1=short
    entry_price = None
    tp_price = None
    sl_price = None
    cdn = 0  # cooldown counter

    trades = wins = losses = 0

    for i in range(1, n):
        o, hi, lo, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        sig_prev = int(classes[i - 1])  # signal decided at bar i-1, executed at i open

        if pos == 0:
            # Enter new position at open based on signal
            if cdn > 0:
                cdn -= 1
            elif sig_prev == 2:  # long
                cash *= (1.0 - fee_pct)
                entry_price = o * (1.0 + slippage_pct)
                if atr is not None and atr_tp_mult is not None and atr_sl_mult is not None:
                    a = float(max(1e-12, atr[i-1]))
                    tp_price = entry_price + atr_tp_mult * a
                    sl_price = entry_price - atr_sl_mult * a
                else:
                    tp_price = entry_price * (1.0 + tp_pct)
                    sl_price = entry_price * (1.0 - sl_pct)
                pos = +1
                trades += 1
            elif cdn == 0 and sig_prev == 0:  # short
                cash *= (1.0 - fee_pct)
                entry_price = o * (1.0 - slippage_pct)
                if atr is not None and atr_tp_mult is not None and atr_sl_mult is not None:
                    a = float(max(1e-12, atr[i-1]))
                    tp_price = entry_price - atr_tp_mult * a  # target below
                    sl_price = entry_price + atr_sl_mult * a  # stop above
                else:
                    tp_price = entry_price * (1.0 - tp_pct)  # target below
                    sl_price = entry_price * (1.0 + sl_pct)  # stop above
                pos = -1
                trades += 1
            equity_curve[i] = cash
            continue

        # Manage open position
        exit_price = None
        win = None

        if pos == +1:
            # First check intrabar SL, then TP (conservative)
            if lo <= sl_price <= hi:
                exit_price = sl_price
                win = False
            elif hi >= tp_price:
                exit_price = tp_price
                win = True
            # If no TP/SL, opposing signal exits at open
            elif sig_prev == 0:
                exit_price = o
                win = (exit_price >= entry_price)
        else:  # pos == -1 (short)
            # For short: SL if price rises to sl_price; TP if price drops to tp_price
            if hi >= sl_price:
                exit_price = sl_price
                win = False
            elif lo <= tp_price:
                exit_price = tp_price
                win = True
            elif sig_prev == 2:
                exit_price = o
                win = (exit_price <= entry_price)

        if exit_price is not None:
            if pos == +1:
                exit_exec = exit_price * (1.0 - slippage_pct)
                ret = (exit_exec / entry_price) - 1.0
            else:  # short
                exit_exec = exit_price * (1.0 + slippage_pct)
                ret = (entry_price / exit_exec) - 1.0
            cash *= (1.0 + ret)
            cash *= (1.0 - fee_pct)
            pos = 0
            entry_price = tp_price = sl_price = None
            wins += int(bool(win))
            losses += int(not bool(win))
            equity_curve[i] = cash
            cdn = max(cooldown, 0)
            continue

        # Still in trade → mark-to-market equity
        if pos == +1:
            mtm = (c / entry_price) - 1.0
        else:  # short
            mtm = (entry_price / c) - 1.0
        equity_curve[i] = cash * (1.0 + mtm)

    # Close any open position at last close
    if pos != 0 and entry_price is not None:
        c = float(closes[-1])
        if pos == +1:
            exit_exec = c * (1.0 - slippage_pct)
            ret = (exit_exec / entry_price) - 1.0
        else:
            exit_exec = c * (1.0 + slippage_pct)
            ret = (entry_price / exit_exec) - 1.0
        cash *= (1.0 + ret)
        cash *= (1.0 - fee_pct)

    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / peaks
    report = {
        "metrics": {"n": int(n)},
        "portfolio": {
            "start_capital": float(start_capital),
            "end_equity": float(equity_curve[-1]),
            "return": float(equity_curve[-1] / max(1e-12, start_capital) - 1.0),
            "max_drawdown": float(abs(np.min(dd)) if len(dd) else 0.0),
            "trades": int(trades), "wins": int(wins), "losses": int(losses),
        },
    }
    df_curve = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                             "equity": equity_curve, "class": classes})
    return report, df_curve


def apply_gating(probs: np.ndarray, *, thr_long: float, thr_short: float, margin: float, consensus: int) -> np.ndarray:
    """Map class probabilities [N,3] into discrete signals with thresholds and consensus.

    Returns array of classes encoded as 0=short, 1=hold, 2=long.
    Rules:
      - Long if p2 >= thr_long and p2 - max(p0,p1) >= margin
      - Short if p0 >= thr_short and p0 - max(p2,p1) >= margin
      - Else hold.
      - Require `consensus` consecutive identical non-hold signals.
    """
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must be [N,3]")
    N = probs.shape[0]
    raw = np.ones(N, dtype=np.int64)
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    long_mask = (p2 >= float(thr_long)) & ((p2 - np.maximum(p0, p1)) >= float(margin))
    short_mask = (p0 >= float(thr_short)) & ((p0 - np.maximum(p2, p1)) >= float(margin))
    raw[long_mask] = 2
    raw[short_mask] = 0
    if consensus <= 1:
        return raw
    out = np.ones(N, dtype=np.int64)
    run_sig = 1
    run_len = 0
    for i in range(N):
        sig = raw[i]
        if sig != 1 and sig == run_sig:
            run_len += 1
        else:
            run_sig = sig
            run_len = 1
        if sig != 1 and run_len >= consensus:
            out[i] = sig
        else:
            out[i] = 1
    return out

# =========================
# Robust prediction (auto-shrinks batch on CUDA OOM, CPU fallback)
# =========================
def predict_probs(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    probs = np.zeros(len(X), dtype=np.float32)
    i = 0
    bs = max(1, int(batch_size))
    while i < len(X):
        try:
            xb_np = X[i:i+bs]
            if len(xb_np) == 0:
                break
            xb = torch.from_numpy(xb_np).to(device, non_blocking=(device.type == "cuda"))
            with torch.no_grad():
                logits = model(xb)
                p = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs[i:i+len(p)] = p
            i += len(p)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda" and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM — reducing batch size to {bs}")
                continue
            # If still failing on CUDA with bs=1, fall back to CPU
            if "out of memory" in msg and device.type == "cuda":
                print("[WARN] CUDA OOM at batch size 1 — falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                torch.cuda.empty_cache()
                continue
            raise
    return probs

# =========================
# CLI
# =========================
def predict_classes(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device,
                    progress: bool = True) -> np.ndarray:
    """Robust class prediction with OOM handling. Returns np.int64 classes [N]."""
    classes = np.zeros(len(X), dtype=np.int64)
    i = 0
    bs = max(1, int(batch_size))
    N = len(X)
    next_mark = 0.1
    while i < len(X):
        try:
            xb_np = X[i:i+bs]
            if len(xb_np) == 0:
                break
            xb = torch.from_numpy(xb_np).to(device, non_blocking=(device.type == "cuda"))
            with torch.no_grad():
                logits = model(xb)
                pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            classes[i:i+len(pred)] = pred
            i += len(pred)
            if progress and N:
                frac = i / N
                if frac >= next_mark or i == N:
                    print(f"[predict] {i:,}/{N:,} ({frac:0.0%})")
                    next_mark += 0.1
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda" and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM — reducing batch size to {bs}")
                continue
            if "out of memory" in msg and device.type == "cuda":
                print("[WARN] CUDA OOM at batch size 1 — falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                torch.cuda.empty_cache()
                continue
            raise
    return classes


def predict_proba(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device,
                  progress: bool = True, temperature: float = 1.0) -> np.ndarray:
    """Return softmax probabilities [N,3] with robust batching and progress.

    If temperature != 1, logits are divided by T before softmax.
    """
    N = len(X)
    probs = np.zeros((N, 3), dtype=np.float32)
    i = 0
    bs = max(1, int(batch_size))
    next_mark = 0.1
    while i < N:
        try:
            xb_np = X[i:i+bs]
            if len(xb_np) == 0:
                break
            xb = torch.from_numpy(xb_np).to(device, non_blocking=(device.type == "cuda"))
            with torch.no_grad():
                logits = model(xb)
                if temperature and abs(float(temperature) - 1.0) > 1e-6:
                    logits = logits / float(temperature)
                p = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs[i:i+len(p)] = p
            i += len(p)
            if progress and N:
                frac = i / N
                if frac >= next_mark or i == N:
                    print(f"[predict] {i:,}/{N:,} ({frac:0.0%})")
                    next_mark += 0.1
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda" and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM — reducing batch size to {bs}")
                continue
            if "out of memory" in msg and device.type == "cuda":
                print("[WARN] CUDA OOM at batch size 1 — falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                torch.cuda.empty_cache()
                continue
            raise
    return probs


def predict_regression(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device,
                       progress: bool = True) -> np.ndarray:
    """Return scalar predictions [N] for regression models (num_outputs=1)."""
    N = len(X)
    preds = np.zeros(N, dtype=np.float32)
    i = 0
    bs = max(1, int(batch_size))
    next_mark = 0.1
    while i < N:
        try:
            xb_np = X[i:i+bs]
            if len(xb_np) == 0:
                break
            xb = torch.from_numpy(xb_np).to(device, non_blocking=(device.type == "cuda"))
            with torch.no_grad():
                out = model(xb)
                if out.ndim == 2 and out.shape[1] == 1:
                    p = out.squeeze(-1).detach().cpu().numpy()
                else:
                    p = out.squeeze(-1).detach().cpu().numpy()
            preds[i:i+len(p)] = p
            i += len(p)
            if progress and N:
                frac = i / N
                if frac >= next_mark or i == N:
                    print(f"[predict] {i:,}/{N:,} ({frac:0.0%})")
                    next_mark += 0.1
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda" and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM — reducing batch size to {bs}")
                continue
            if "out of memory" in msg and device.type == "cuda":
                print("[WARN] CUDA OOM at batch size 1 — falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                torch.cuda.empty_cache()
                continue
            raise
    return preds
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtester with TP/SL and probability threshold.")
    p.add_argument("--mode", choices=["simple", "portfolio"], default="portfolio")
    p.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or a single CSV")
    p.add_argument("--model-dir", type=str, default="model", help="Root where model_meta.json & model.pt live")
    p.add_argument("--threshold", type=float, default=None, help="Buy threshold (default from model_meta.json)")
    p.add_argument("--tp-pct", type=float, default=None, help="Take-profit as fraction (0.005 = 0.5%)")
    p.add_argument("--sl-pct", type=float, default=None, help="Stop-loss as fraction (0.0025 = 0.25%)")
    p.add_argument("--fee-pct", type=float, default=None, help="Per-side fee fraction (0.0008 = 0.08%)")
    p.add_argument("--capital", type=float, default=10_000.0, help="Starting capital for portfolio mode")
    p.add_argument("--batch-size", type=int, default=512, help="Prediction batch size (auto-shrinks if OOM)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device (default auto)")
    p.add_argument("--thr-long", type=float, default=0.75, help="Min probability for long signal")
    p.add_argument("--thr-short", type=float, default=0.75, help="Min probability for short signal")
    p.add_argument("--margin", type=float, default=0.25, help="Required margin vs next best class")
    p.add_argument("--consensus", type=int, default=2, help="Require N consecutive identical non-hold signals")
    p.add_argument("--cooldown", type=int, default=3, help="Bars to wait after an exit before re-entering")
    p.add_argument("--force-regress", action="store_true", help="Force regression mode (predict future price)")
    p.add_argument("--up-thr", type=float, default=0.03, help="Predicted return threshold for long (e.g., 0.03 = +3%)")
    p.add_argument("--down-thr", type=float, default=0.00, help="Absolute predicted return threshold for short (e.g., 0.00 = any negative)")
    p.add_argument("--horizon-mins", type=int, default=3, help="Prediction horizon in minutes for regression models")
    p.add_argument("--use-atr-stops", action="store_true", default=True, help="Use ATR multipliers for TP/SL instead of fixed percents")
    p.add_argument("--atr-tp-mult", type=float, default=1.8, help="ATR multiplier for take-profit")
    p.add_argument("--atr-sl-mult", type=float, default=1.0, help="ATR multiplier for stop-loss")
    p.add_argument("--last-csvs", type=int, default=None, help="If data-dir is a directory, only use the most recent N CSV files")
    p.add_argument("--days-back", type=int, default=None, help="Limit to the most recent N days (requires a timestamp column; else approximates by rows)")
    p.add_argument("--slippage-pct", type=float, default=0.0, help="Per-side slippage fraction applied to entry and exit prices")
    p.add_argument("--use-regime-filter", action="store_true", default=True, help="Only take longs if ema_50>=ema_200 and shorts if ema_50<=ema_200")
    p.add_argument("--min-atr-pct", type=float, default=0.001, help="Require atr/close >= this fraction or hold (e.g., 0.001 for 0.1%)")
    return p

# =========================
# Main
# =========================
def main():
    args = build_argparser().parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    if "feature_cols" not in meta:
        raise KeyError("Missing 'feature_cols' in model_meta.json; no fallback is allowed.")
    feature_cols = list(meta["feature_cols"])  # preserve order from meta
    meta_input = int(meta.get("input_size", len(feature_cols)))
    if meta_input != len(feature_cols):
        print(f"[WARN] meta.input_size ({meta_input}) != len(feature_cols) ({len(feature_cols)}). Using feature_cols order from meta.")
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)
    # Allow configurable class index for 'buy' via model_meta.json
    buy_class_index = (meta.get("class_map", {}) or {}).get("buy", 1)

    # Load raw data and compute the SAME features as training
    print(f"[load] Reading data from {args.data_dir} ...")
    data_path = Path(args.data_dir)
    if data_path.is_dir() and args.last_csvs:
        files = sorted(str(p) for p in data_path.glob("*.csv"))
        if not files:
            raise SystemExit(f"No CSV files found in directory: {args.data_dir}")
        sel = files[-int(args.last_csvs):]
        print(f"[load] Limiting to last {len(sel)} CSVs:")
        for s in sel:
            print(f"        {Path(s).name}")
        parts = [normalize_headers(pd.read_csv(s)) for s in sel]
        df = pd.concat(parts, ignore_index=True)
    else:
        df = read_csv_concat_sorted(args.data_dir)
    print(f"[load] Rows loaded: {len(df):,}")
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not locate a price column. Available: {list(df.columns)}")
    # Optional days-back restriction
    if args.days_back is not None and int(args.days_back) > 0:
        n_days = int(args.days_back)
        ts_col: Optional[str] = None
        for cand in ("timestamp", "time", "date"):
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            if ts.notna().any():
                cutoff = ts.max() - pd.Timedelta(days=n_days)
                df = df.loc[ts >= cutoff].reset_index(drop=True)
                print(f"[filter] Kept last {n_days} days by timestamp; rows now {len(df):,}")
        else:
            approx_rows = n_days * 1440  # assume 1-minute bars
            if len(df) > approx_rows:
                df = df.iloc[-approx_rows:].reset_index(drop=True)
                print(f"[filter] No timestamp column; kept last ~{n_days} days by rows; rows now {len(df):,}")

    print("[features] Engineering features ...")
    df = compute_features(df)
    # Add regime helper EMAs (not used by model unless included in meta)
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    print(f"[features] Done. Columns: {len(df.columns)}; Rows: {len(df):,}")

    # Build feature matrix in the SAME order as meta (do NOT drop 'close')
    drop_cols = {"timestamp", "time"}  # only drop non-features
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    missing = [c for c in feature_cols if c not in feat_cols]
    if missing:
        print(f"[WARN] Missing features in data (will drop): {missing}")
    if not feat_cols:
        raise SystemExit("No valid feature columns found in data for inference.")
    print(f"[INFO] Using {len(feat_cols)} features from meta: {feat_cols}")
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)

    # Windows
    print("[windows] Building windows ...")
    X = build_windows(X_flat, window_size)
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce window size.")
    print(f"[windows] Built {len(X):,} sequences of length {window_size} with {X.shape[-1]} features")

    # Align OHLC arrays to window ends
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes = df[price_col].to_numpy(dtype=float)[window_size - 1:]
    atr_arr = df["atr"].to_numpy(dtype=float)[window_size - 1:] if "atr" in df.columns else None
    ema50 = df["ema_50"].to_numpy(dtype=float)[window_size - 1:]
    ema200 = df["ema_200"].to_numpy(dtype=float)[window_size - 1:]

    # Scale using same scaler (fit on all meta features)
    if scaler is not None:
        n, t, f = X.shape
        print("[scale] Applying scaler to features ...")
        X = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)

    # Choose device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("[predict] Running model inference ...")
    task = str(meta.get("task", "")).lower() if isinstance(meta, dict) else ""
    is_regression = bool(args.force_regress or task == "regression" or int(meta.get("num_classes", 3)) == 1)
    if is_regression:
        preds = predict_regression(model, X, int(args.batch_size), device, progress=True)
        print("[predict] Done (regression).")
        ret = (preds / np.maximum(1e-12, closes)) - 1.0
        up_thr = float(args.up_thr)
        down_thr = float(args.down_thr)
        signals = np.ones(len(ret), dtype=np.int64)
        signals[ret >= up_thr] = 2
        signals[ret <= -max(0.0, down_thr)] = 0
    else:
        probs = predict_proba(
            model, X, int(args.batch_size), device,
            progress=True, temperature=float(meta.get("temperature", 1.0))
        )
        print("[predict] Done (classification).")
        signals = apply_gating(
            probs,
            thr_long=float(args.thr_long),
            thr_short=float(args.thr_short),
            margin=float(args.margin),
            consensus=int(args.consensus),
        )
    # Optional regime filter
    if args.use_regime_filter:
        long_ok = ema50 >= ema200
        short_ok = ema50 <= ema200
        before = signals.copy()
        signals[(signals == 2) & (~long_ok)] = 1
        signals[(signals == 0) & (~short_ok)] = 1
        changed = int((before != signals).sum())
        print(f"[filter] Regime filter applied; signals modified: {changed}")
    # Minimum ATR filter
    if args.min_atr_pct and atr_arr is not None:
        minp = float(args.min_atr_pct)
        mask = (atr_arr / np.maximum(1e-12, closes)) < minp
        chg = int(np.sum((signals != 1) & mask))
        signals[mask] = 1
        print(f"[filter] Min ATR filter applied; signals to HOLD: {chg}")

    if args.mode == "simple":
        unique, counts = np.unique(signals, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(json.dumps({
            "metrics": {"n": int(len(signals))},
            "class_dist": dist,
        }, indent=2))
        return

    # Portfolio simulation with TP/SL
    print("[simulate] Running portfolio simulation ...")
    use_atr = bool(args.use_atr_stops and atr_arr is not None)
    report, curve = simulate_trades_with_tp_sl(
        opens, highs, lows, closes, signals,
        start_capital=float(args.capital),
        fee_pct=fee_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        atr=(atr_arr if use_atr else None),
        atr_tp_mult=(float(args.atr_tp_mult) if use_atr else None),
        atr_sl_mult=(float(args.atr_sl_mult) if use_atr else None),
        cooldown=int(args.cooldown),
        slippage_pct=float(args.slippage_pct),
    )
    print("[simulate] Done.")
    print_portfolio_report(report, currency="$")

if __name__ == "__main__":
    main()




