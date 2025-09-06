#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_trade.py — forward-only paper trading with TP/SL and threshold

Defaults:
  • Data directory: ./eth_1m_data
  • Model artifacts: ./ (model_meta.json, model.pt, scaler.joblib)
  • Buy when P(class=1) >= threshold; exit via intra-bar TP/SL
  • Prints a trade blotter and a $-formatted summary

Examples
--------
python paper_trade.py --capital 10000
python paper_trade.py --threshold 0.65 --tp-pct 0.005 --sl-pct 0.0025 --fee-pct 0.0008
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    load_model_bundle, fmt_money
)

def main():
    ap = argparse.ArgumentParser(description="Paper (simulated) trading with TP/SL and threshold.")
    ap.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or single CSV")
    ap.add_argument("--model-dir", type=str, default=".", help="Where model_meta.json & model.pt live")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--tp-pct", type=float, default=None)
    ap.add_argument("--sl-pct", type=float, default=None)
    ap.add_argument("--fee-pct", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta["feature_cols"])  # strict: must exist
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)

    # Load data
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not find a price column. Available: {list(df.columns)}")

    # Features
    drop_cols = {price_col, "timestamp", "time"}
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    if not feat_cols:
        raise SystemExit("No valid feature columns found for inference.")
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)

    # Windows
    X = build_windows(X_flat, window_size)
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce window size.")

    # Align OHLC arrays to window ends
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows  = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes= df[price_col].to_numpy(dtype=float)[window_size - 1:]
    times = df["timestamp"].iloc[window_size - 1:] if "timestamp" in df.columns else pd.Series(range(len(closes)))

    # Scale if scaler exists
    if scaler is not None:
        n, t, f = X.shape
        X = scaler.transform(X.reshape(n*t, f)).reshape(n, t, f)

    # Predict probabilities (class 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    probs = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        BS = int(args.batch_size)
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i+BS]).to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs[i:i+BS] = p

    # Paper trading loop (same logic as backtester)
    cash = float(args.capital)
    in_trade = False
    entry_price = None
    tp_price = None
    sl_price = None
    trades = []
    wins = losses = 0

    for i in range(1, len(closes)):
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        t = times.iloc[i] if hasattr(times, "iloc") else times[i]

        if not in_trade:
            if probs[i-1] >= buy_threshold:
                entry_price = o
                tp_price = entry_price * (1.0 + tp_pct)
                sl_price = entry_price * (1.0 - sl_pct)
                cash *= (1.0 - fee_pct)  # entry fee
                in_trade = True
                trades.append({
                    "time": t, "side": "BUY", "price": entry_price, "prob": float(probs[i-1])
                })
        else:
            exit_px = None
            win = None
            # conservative order: SL before TP if both touched (path dependent)
            if l <= sl_price <= h:
                exit_px = sl_price
                win = False
            elif h >= tp_price:
                exit_px = tp_price
                win = True
            elif l <= sl_price:
                exit_px = sl_price
                win = False

            if exit_px is not None:
                gross_ret = (exit_px / entry_price) - 1.0
                cash *= (1.0 + gross_ret)
                cash *= (1.0 - fee_pct)  # exit fee
                in_trade = False
                trades.append({
                    "time": t, "side": "SELL", "price": exit_px, "result": "WIN" if win else "LOSS"
                })
                if win: wins += 1
                else: losses += 1

    # If still open at end, close at last close
    if in_trade and entry_price is not None:
        final_exit = float(closes[-1])
        gross_ret = (final_exit / entry_price) - 1.0
        cash *= (1.0 + gross_ret)
        cash *= (1.0 - fee_pct)
        trades.append({"time": times.iloc[-1] if hasattr(times,"iloc") else times[-1], "side": "SELL", "price": final_exit, "result": "CLOSE_END"})

    # Blotter + summary
    print("\n=== PAPER TRADE BLOTTER ===")
    for tr in trades:
        if tr["side"] == "BUY":
            print(f"{tr['time']}  BUY  @ {fmt_money(tr['price'])}  (prob={tr['prob']:.3f})")
        else:
            res = tr.get("result", "")
            print(f"{tr['time']}  SELL @ {fmt_money(tr['price'])}  {res}")

    start = float(args.capital)
    end = cash
    multiple = end / start if start else float("nan")
    ret = multiple - 1.0
    print("\n=== PAPER TRADE SUMMARY ===")
    print(f"Start capital  : {fmt_money(start)}")
    print(f"End equity     : {fmt_money(end)}")
    print(f"Return         : {ret*100:.2f}%  (×{multiple:.2f})")
    total_trades = sum(1 for tr in trades if tr["side"] == "SELL")
    if total_trades:
        print(f"Trades         : {total_trades}  (wins {wins}, losses {losses}, win rate {wins/max(1,total_trades):.2%})")
    print("")

if __name__ == "__main__":
    main()
