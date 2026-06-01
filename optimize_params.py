#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid sweep over gating + ATR params to maximize end equity
while keeping drawdown controlled. Uses backtest.py functions in-process.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401 - reserved for future enhancements
import torch

from utils import read_csv_concat_sorted, compute_features, build_windows, load_model_bundle
from backtest import apply_gating, simulate_trades_with_tp_sl

DATA_DIR = "eth_1m_data"
MODEL_DIR = "model_sanity"

SEARCH = {
    "thr_long": [0.70, 0.75, 0.80],
    "thr_short": [0.70, 0.75, 0.80],
    "margin": [0.15, 0.20, 0.25],
    "consensus": [1, 2, 3],
    "atr_tp": [1.8, 2.0, 2.2],
    "atr_sl": [0.9, 1.0, 1.1],
    "cooldown": [0, 1, 2],
}


def load_inference_inputs(data_dir: str, model_dir: str):
    model, scaler, meta = load_model_bundle(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    df = compute_features(read_csv_concat_sorted(data_dir))
    feature_cols = [c for c in meta["feature_cols"] if c in df.columns]

    window = int(meta.get("window_size", 192))
    feats = df[feature_cols].to_numpy(np.float32)
    if scaler is not None:
        feats = scaler.transform(feats)
    windows = build_windows(feats, window)

    usable = len(windows)
    opens = df["open"].to_numpy(float)[-usable:]
    highs = df["high"].to_numpy(float)[-usable:]
    lows = df["low"].to_numpy(float)[-usable:]
    price_col = meta.get("price_col", "close")
    closes = df[price_col].to_numpy(float)[-usable:]
    atr = df["atr"].to_numpy(float)[-usable:] if "atr" in df.columns else None

    return model, scaler, meta, windows, opens, highs, lows, closes, atr


def predict_probs(model, windows: np.ndarray, batch: int = 512) -> np.ndarray:
    import torch.nn.functional as F

    dev = next(model.parameters()).device
    n = len(windows)
    out = np.zeros((n, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch):
            end = start + batch
            xb = torch.from_numpy(windows[start:end]).to(dev)
            logits = model(xb)
            out[start:end] = F.softmax(logits, dim=-1).detach().cpu().numpy()
    return out


def score(end_equity: float, max_dd: float) -> float:
    return float(end_equity / (1.0 + 2.0 * max(0.0, max_dd)))


def main() -> None:
    model, scaler, meta, windows, opens, highs, lows, closes, atr = load_inference_inputs(DATA_DIR, MODEL_DIR)
    probs = predict_probs(model, windows)

    best = {"score": -1e18}
    keys = list(SEARCH.keys())
    for values in itertools.product(*[SEARCH[k] for k in keys]):
        cfg = dict(zip(keys, values))
        classes = apply_gating(
            probs,
            thr_long=cfg["thr_long"],
            thr_short=cfg["thr_short"],
            margin=cfg["margin"],
            consensus=int(cfg["consensus"]),
        )
        report, _ = simulate_trades_with_tp_sl(
            opens,
            highs,
            lows,
            closes,
            classes,
            start_capital=10_000,
            fee_pct=0.0008,
            atr=atr,
            atr_tp_mult=cfg["atr_tp"],
            atr_sl_mult=cfg["atr_sl"],
            cooldown=int(cfg["cooldown"]),
            slippage_pct=0.0002,
        )
        end_eq = report["portfolio"]["end_equity"]
        max_dd = report["portfolio"].get("max_drawdown", 0.0)
        composite = score(end_eq, max_dd)
        if composite > best["score"]:
            best = {"score": composite, "cfg": cfg, "report": report}
            print(json.dumps(best, indent=2))

    Path(MODEL_DIR, "best_live_config.json").write_text(json.dumps(best, indent=2))
    print("\n[FINAL BEST]\n" + json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
