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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import (
    read_csv_concat_sorted,
    resolve_price_col,
    build_windows,
    load_model_bundle,
    compute_features,
    FEATURE_COLUMNS_PROFITABLE,
    align_feature_columns,
)
try:
    from simulator import simulate_trades_with_tp_sl, print_portfolio_report
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from simulator import simulate_trades_with_tp_sl, print_portfolio_report
from config import cfg
from logging_utils import setup_logging, logger

def main():
    setup_logging()
    args = cfg  # use shared settings

    # Load model + meta
    model, scaler, meta = load_model_bundle(args.model_dir)
    expected_size = int(meta.get("input_size") or len(meta.get("feature_cols") or []) or len(FEATURE_COLUMNS_PROFITABLE))
    try:
        feature_cols = align_feature_columns(meta.get("feature_cols"), expected_size=expected_size)
    except ValueError as exc:
        raise SystemExit(f"Model feature mismatch: {exc}")
    window_size = int(meta.get("window_size", 150))
    buy_threshold = float(meta.get("buy_threshold", cfg.thr_long))
    fee_pct = float(meta.get("tx_cost", cfg.fee_pct))
    tp_pct = float(meta.get("tp_pct", cfg.tp_pct))
    sl_pct = float(meta.get("sl_pct", cfg.sl_pct))

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
    up_threshold = float(cfg.up_thr)
    with torch.no_grad():
        BS = int(cfg.batch_size)
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

    # Build class signals: 2=long, 1=hold (no shorts here)
    classes = np.ones(len(long_flags), dtype=int)
    classes[long_flags] = 2

    report, curve = simulate_trades_with_tp_sl(
        opens,
        highs,
        lows,
        closes,
        classes,
        start_capital=cfg.capital,
        fee_pct=fee_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        slippage_pct=cfg.slippage_pct,
    )

    print_portfolio_report(report, currency="$")

if __name__ == "__main__":
    main()


