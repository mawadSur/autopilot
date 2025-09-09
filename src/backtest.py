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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    fmt_money, fmt_pct, compute_features, load_model_bundle
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
def simulate_trades_with_tp_sl(opens, highs, lows, closes, probs, *, threshold, start_capital,
                               fee_pct=0.0008, tp_pct=0.005, sl_pct=0.0025) -> Tuple[Dict, pd.DataFrame]:
    n = len(closes); cash = float(start_capital); in_trade = False
    entry_price = tp_price = sl_price = None
    equity_curve = np.empty(n, dtype=float); equity_curve[0] = cash
    trades = wins = losses = 0
    for i in range(1, n):
        o, hi, lo, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        if not in_trade:
            if probs[i-1] >= threshold:
                cash *= (1.0 - fee_pct); entry_price = o
                tp_price = entry_price * (1.0 + tp_pct); sl_price = entry_price * (1.0 - sl_pct)
                in_trade = True; trades += 1
        else:
            exit_price = None; win = None
            if lo <= sl_price <= hi:   exit_price = sl_price; win = False
            elif hi >= tp_price:       exit_price = tp_price; win = True
            elif lo <= sl_price:       exit_price = sl_price; win = False
            if exit_price is not None:
                cash *= (1.0 + (exit_price / entry_price) - 1.0); cash *= (1.0 - fee_pct)
                in_trade = False; entry_price = tp_price = sl_price = None
                wins += int(win); losses += int(not win)
            else:
                mtm = (c / entry_price) - 1.0
                equity_curve[i] = cash * (1.0 + mtm); continue
        equity_curve[i] = cash
    if in_trade and entry_price is not None:
        cash *= (1.0 + (closes[-1] / entry_price) - 1.0); cash *= (1.0 - fee_pct)
    peaks = np.maximum.accumulate(equity_curve); dd = (equity_curve - peaks) / peaks
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
                             "equity": equity_curve, "prob": probs})
    return report, df_curve

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
    p.add_argument("--batch-size", type=int, default=512, help="Prediction batch size (auto-shrinks if OOM)")  # smaller default
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device (default auto)")
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
    feature_cols = list(meta["feature_cols"])  
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)
    # Allow configurable class index for 'buy' via model_meta.json
    buy_class_index = (meta.get("class_map", {}) or {}).get("buy", 1)

    # Load raw data and compute the SAME features as training
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not locate a price column. Available: {list(df.columns)}")
    df = compute_features(df)

    # Build feature matrix in the SAME order as meta (do NOT drop 'close')
    drop_cols = {"timestamp", "time"}  # only drop non-features
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    missing = [c for c in feature_cols if c not in feat_cols]
    if missing:
        print(f"[WARN] Missing features in data (will drop): {missing}")
    if not feat_cols:
        raise SystemExit("No valid feature columns found in data for inference.")
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)

    # Windows
    X = build_windows(X_flat, window_size)
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce window size.")

    # Align OHLC arrays to window ends
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes = df[price_col].to_numpy(dtype=float)[window_size - 1:]

    # Scale using same scaler (fit on all meta features)
    if scaler is not None:
        n, t, f = X.shape
        X = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)

    # Choose device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    probs = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        BS = int(args.batch_size)
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i + BS]).to(device)
            logits = model(xb)  # [B, C]
            p = F.softmax(logits, dim=-1)[:, buy_class_index].cpu().numpy()
            probs[i:i + BS] = p

    if args.mode == "simple":
        print(json.dumps({
            "metrics": {"n": int(len(probs))},
            "threshold": buy_threshold,
            "mean_prob": float(probs.mean()),
            "p90_prob": float(np.percentile(probs, 90)),
        }, indent=2))
        return

    # Portfolio simulation with TP/SL
    report, curve = simulate_trades_with_tp_sl(
        opens, highs, lows, closes, probs,
        threshold=buy_threshold,
        start_capital=float(args.capital),
        fee_pct=fee_pct, tp_pct=tp_pct, sl_pct=sl_pct,
    )
    print_portfolio_report(report, currency="$")

if __name__ == "__main__":
    main()
