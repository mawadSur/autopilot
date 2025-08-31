#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py — Unified backtester with TP/SL and pretty $ output

Defaults:
  • Data directory: ./eth_1m_data
  • Model artifacts: ./model  (expects model_meta.json, model.pt, scaler.joblib)
  • Binary classification: 1 = buy, 0 = no-trade
  • TP/SL checked intra-bar via high/low
  • Confidence threshold from model_meta.json (buy_threshold), CLI override possible

Examples
--------
python backtest.py --mode portfolio --capital 10000
python backtest.py --mode portfolio --capital 10000 --tp-pct 0.005 --sl-pct 0.0025 --fee-pct 0.0008 --threshold 0.65
python backtest.py --mode simple
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    load_model_bundle, fmt_money, fmt_pct, DEFAULT_FEATURE_COLS
)

# --- import your project model ---
try:
    from models import LSTMClassifier  # noqa: F401 (loaded inside utils.load_model_bundle)
except Exception as e:
    raise SystemExit(
        "Could not import LSTMClassifier from models.py. Ensure models.py is on PYTHONPATH.\n"
        f"Underlying error: {e}"
    )


# ---------------------------
# Pretty formatting
# ---------------------------
import math as _math  # noqa: E402


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
    ret_frac = None
    multiple = None
    if start not in (None, 0) and end is not None:
        multiple = float(end) / float(start)
        ret_frac = multiple - 1.0

    print("\n=== PORTFOLIO MODE — SUMMARY ===")
    print(f"Bars processed : {n:,}")
    print(f"Trades         : {trades:,}  (wins {wins}, losses {losses}, win rate {wins/max(1,trades):.2%})")
    print(f"Start capital  : {fmt_money(start, currency)}")
    print(f"End equity     : {fmt_money(end, currency)}")
    if multiple is not None and _math.isfinite(multiple):
        if multiple >= 1e6:
            print(f"Return         : ×{multiple:.3e}  (⚠️ extremely large; check return scaling)")
        else:
            print(f"Return         : {fmt_pct(ret_frac)}  (×{multiple:.2f})")
    else:
        print("Return         : —")
    if mdd is not None:
        print(f"Max drawdown   : {fmt_pct(mdd)}")
    print("")  # newline


# ---------------------------
# Trade simulation with TP/SL (intra-bar)
# ---------------------------

def simulate_trades_with_tp_sl(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    probs: np.ndarray, *,
    threshold: float,
    start_capital: float,
    fee_pct: float = 0.0008,
    tp_pct: float = 0.005,
    sl_pct: float = 0.0025,
) -> Tuple[Dict, pd.DataFrame]:
    """
    One-position-at-a-time long strategy:
      - Enter at next bar open when prob >= threshold and not already in a trade.
      - Set TP = entry * (1 + tp_pct), SL = entry * (1 - sl_pct).
      - While in trade, check **intra-bar**: if high >= TP => exit at TP; elif low <= SL => exit at SL; else hold.
      - Fees applied on both entry and exit.
    """
    n = len(closes)
    cash = float(start_capital)
    entry_price = None
    tp_price = None
    sl_price = None

    equity_curve = np.empty(n, dtype=float)
    equity_curve[0] = cash
    in_trade = False
    trades = 0
    wins = 0
    losses = 0

    for i in range(1, n):
        price_open = float(opens[i])
        price_high = float(highs[i])
        price_low = float(lows[i])
        price_close = float(closes[i])

        if not in_trade:
            if probs[i - 1] >= threshold:
                # enter at next bar open
                entry = price_open
                cash *= (1.0 - fee_pct)  # entry fee
                entry_price = entry
                tp_price = entry_price * (1.0 + tp_pct)
                sl_price = entry_price * (1.0 - sl_pct)
                in_trade = True
                trades += 1
        else:
            exit_price = None
            win = None
            # worst-case path if both touched within bar
            if price_low <= sl_price <= price_high:
                exit_price = sl_price
                win = False
            elif price_high >= tp_price:
                exit_price = tp_price
                win = True
            elif price_low <= sl_price:
                exit_price = sl_price
                win = False

            if exit_price is not None:
                gross_ret = (exit_price / entry_price) - 1.0
                cash *= (1.0 + gross_ret)
                cash *= (1.0 - fee_pct)  # exit fee
                in_trade = False
                entry_price = tp_price = sl_price = None
                if win:
                    wins += 1
                else:
                    losses += 1
            else:
                # still in trade: mark-to-market on close
                mtm = (price_close / entry_price) - 1.0
                equity_curve[i] = cash * (1.0 + mtm)
                continue

        equity_curve[i] = cash

    # close any open trade at final close with exit fee
    if in_trade and entry_price is not None:
        gross_ret = (closes[-1] / entry_price) - 1.0
        cash *= (1.0 + gross_ret)
        cash *= (1.0 - fee_pct)

    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / peaks
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    report = {
        "metrics": {"n": int(n)},
        "portfolio": {
            "start_capital": float(start_capital),
            "end_equity": float(equity_curve[-1]),
            "return": float(equity_curve[-1] / max(1e-12, start_capital) - 1.0),
            "max_drawdown": float(abs(max_dd)),
            "trades": int(trades),
            "wins": int(wins),
            "losses": int(losses),
        },
    }
    df_curve = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "equity": equity_curve, "prob": probs
    })
    return report, df_curve


# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtester with TP/SL and probability threshold.")
    p.add_argument("--mode", choices=["simple", "portfolio"], default="portfolio")
    p.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or a single CSV")
    # ⬇️ default to 'model' so artifacts in ./model/ are found without flags
    p.add_argument("--model-dir", type=str, default="model", help="Root where model_meta.json & model.pt live")

    # overrides / knobs
    p.add_argument("--threshold", type=float, default=None, help="Buy probability threshold (default from model_meta.json)")
    p.add_argument("--tp-pct", type=float, default=None, help="Take-profit as fraction (0.005 = 0.5%)")
    p.add_argument("--sl-pct", type=float, default=None, help="Stop-loss as fraction (0.0025 = 0.25%)")
    p.add_argument("--fee-pct", type=float, default=None, help="Per-side fee fraction (0.0008 = 0.08%)")
    p.add_argument("--capital", type=float, default=10_000.0, help="Starting capital for portfolio mode")

    # performance
    p.add_argument("--batch-size", type=int, default=2048, help="Prediction batch size")
    return p


# ---------------------------
# Main
# ---------------------------

def main():
    args = build_argparser().parse_args()

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    feature_cols = list(meta.get("feature_cols", DEFAULT_FEATURE_COLS))
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", 0.60)) if args.threshold is None else float(args.threshold)
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.005 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.0025 if args.sl_pct is None else float(args.sl_pct)

    # Load data
    df = read_csv_concat_sorted(args.data_dir)
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not locate a price column. Available: {list(df.columns)}")

    # Build features
    drop_cols = {price_col, "timestamp", "time"}
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    if len(feat_cols) != len(feature_cols):
        missing = [c for c in feature_cols if c not in feat_cols]
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

    # Scale if scaler exists
    if scaler is not None:
        n, t, f = X.shape
        X = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)

    # Predict probabilities (class 1 = buy)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    probs = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        BS = int(args.batch_size)
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i + BS]).to(device)
            logits = model(xb)  # [B, 2]
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
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
        fee_pct=fee_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )

    print_portfolio_report(report, currency="$")
    # If you want raw JSON too, uncomment:
    # print("Raw JSON:\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
