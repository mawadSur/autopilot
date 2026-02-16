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
    load_model_bundle, fmt_money, compute_features, FEATURE_COLUMNS
)

def main():
    ap = argparse.ArgumentParser(description="Paper (simulated) trading with TP/SL and threshold.")
    ap.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or single CSV")
    ap.add_argument("--model-dir", type=str, default="model", help="Where model_meta.json & model.pt live")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--threshold", type=float, default=None, help="Buy prob threshold (classification)")
    ap.add_argument("--up-thr", type=float, default=0.002,
                    help="Predicted return threshold for long entries in regression mode (+0.2%)")
    ap.add_argument("--tp-pct", type=float, default=None)
    ap.add_argument("--sl-pct", type=float, default=None)
    ap.add_argument("--fee-pct", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta["feature_cols"])  # strict: must exist
    if feature_cols != FEATURE_COLUMNS:
        raise ValueError(f"Feature list mismatch: meta has {feature_cols}, expected {FEATURE_COLUMNS}")
    window_size = int(meta.get("window_size", 150))
    # Prefer CLI override, otherwise meta, otherwise use a high-confidence fallback
    buy_threshold = float(meta.get("buy_threshold", 0.75)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = float(meta.get("tp_pct", 0.005)) if args.tp_pct is None else float(args.tp_pct)
    sl_pct = float(meta.get("sl_pct", 0.0025)) if args.sl_pct is None else float(args.sl_pct)

    # Load data
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not find a price column. Available: {list(df.columns)}")

    # Features
    print("[INFO] Computing features...")
    df = compute_features(df)
    print(f"[INFO] Features computed. Total columns: {len(df.columns)}")

    # Ensure all required feature columns from training are present
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Data is missing required features after computation: {missing_cols}")

    # Create feature matrix in the same order as training
    X_flat = df[feature_cols].to_numpy(dtype=np.float32)
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        assert list(scaler.feature_names_in_) == feature_cols, "Scaler feature order mismatch"
    print(f"[INFO] Feature matrix created with shape: {X_flat.shape}")

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

    # Predict signals (classification prob or regression return)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    num_classes = int(meta.get("num_classes", 3))
    task = str(meta.get("task", "classification")).lower()
    scores = np.zeros(len(X), dtype=np.float32)
    long_flags = np.zeros(len(X), dtype=bool)
    up_threshold = float(args.up_thr)
    with torch.no_grad():
        BS = int(args.batch_size)
        for start in range(0, len(X), BS):
            end = min(start + BS, len(X))
            xb = torch.from_numpy(X[start:end]).to(device)
            logits = model(xb)

            if num_classes == 1 or task == "regression":
                preds = logits.squeeze(-1).detach().cpu().numpy()
                close_slice = closes[start:end]
                returns = (preds / np.maximum(1e-12, close_slice)) - 1.0
                scores[start:end] = returns.astype(np.float32, copy=False)
                long_flags[start:end] = returns >= up_threshold
            else:
                probs_batch = F.softmax(logits, dim=-1).detach().cpu().numpy()
                long_idx = 2 if probs_batch.shape[1] >= 3 else 1
                long_probs = probs_batch[:, long_idx]
                scores[start:end] = long_probs.astype(np.float32, copy=False)
                long_flags[start:end] = long_probs >= buy_threshold

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
            if long_flags[i-1]:
                entry_price = o
                tp_price = entry_price * (1.0 + tp_pct)
                sl_price = entry_price * (1.0 - sl_pct)
                cash *= (1.0 - fee_pct)  # entry fee
                in_trade = True
                trades.append({
                    "time": t,
                    "side": "BUY",
                    "price": entry_price,
                    "signal": float(scores[i-1]),
                    "mode": task,
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
            sig = tr.get("signal")
            mode = tr.get("mode", task)
            if sig is None:
                sig_txt = "n/a"
            elif mode == "regression" or num_classes == 1:
                sig_txt = f"{sig*100:.2f}%"
            else:
                sig_txt = f"{sig:.3f}"
            print(f"{tr['time']}  BUY  @ {fmt_money(tr['price'])}  (signal={sig_txt})")
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


