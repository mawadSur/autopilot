#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py — Memory-safe streaming backtester.

Key upgrades vs prior version:
- Streams CSVs (directory or single file) in chunks with overlap.
- Computes features per chunk with continuity across files.
- Builds model windows incrementally (no giant X tensor).
- Runs inference in batches and simulates trades online.
- Supports both classification (3-class) and regression (1 output) models.

Usage
-----
python backtest.py --data-dir eth_1m_data --model-dir model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from logging_utils import setup_logging, logger

from utils import (
    fmt_money,
    fmt_pct,
    compute_features,
    load_model_bundle,
    normalize_headers,
    FEATURE_COLUMNS,
    DEFAULT_SEQ_LEN,
)
try:
    from simulator import (
        Bar,
        SimulationConfig,
        PortfolioSimulator,
        print_portfolio_report,
    )
    from streamer import KlineStreamer
except ModuleNotFoundError:
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from simulator import (
        Bar,
        SimulationConfig,
        PortfolioSimulator,
        print_portfolio_report,
    )
    from streamer import KlineStreamer
from config import cfg

# -------------------------
# CSV helper
# -------------------------
def _list_csvs(path: str) -> List[Path]:
    return KlineStreamer._list_csvs(Path(path))

# =========================
# Gating (online)
# =========================
def _raw_signal_from_probs(p: np.ndarray, thr_long: float, thr_short: float, margin: float) -> int:
    """
    p: [3] probabilities
    returns class: 0=short, 1=hold, 2=long
    """
    p0, p1, p2 = float(p[0]), float(p[1]), float(p[2])
    if (p2 >= thr_long) and (p2 - max(p0, p1) >= margin):
        return 2
    if (p0 >= thr_short) and (p0 - max(p2, p1) >= margin):
        return 0
    return 1

class ConsensusFilter:
    """Require N consecutive identical non-hold signals before outputting them."""
    def __init__(self, consensus: int):
        self.consensus = max(1, int(consensus))
        self.run_sig = 1
        self.run_len = 0

    def step(self, raw_sig: int) -> int:
        if self.consensus <= 1:
            return raw_sig
        if raw_sig != 1 and raw_sig == self.run_sig:
            self.run_len += 1
        else:
            self.run_sig = raw_sig
            self.run_len = 1
        if raw_sig != 1 and self.run_len >= self.consensus:
            return raw_sig
        return 1

# =========================
# Online portfolio simulator
# =========================
def _run_backtest_for_files(files: List[Path], model_dir: str, *, print_report: bool = True) -> Dict:
    model, scaler, meta = load_model_bundle(model_dir)
    feature_cols = list(meta.get("feature_cols", FEATURE_COLUMNS))
    if feature_cols != FEATURE_COLUMNS:
        raise ValueError(f"Feature list mismatch: meta has {feature_cols}, expected {FEATURE_COLUMNS}")

    window_size = int(cfg.window_size or meta.get("window_size", DEFAULT_SEQ_LEN))
    fee_pct = float(meta.get("tx_cost", cfg.fee_pct))
    tp_pct = float(meta.get("tp_pct", cfg.tp_pct))
    sl_pct = float(meta.get("sl_pct", cfg.sl_pct))

    task = str(meta.get("task", "")).lower()
    is_regression = bool(task == "regression" or int(meta.get("num_classes", 3)) == 1)
    temperature = float(meta.get("temperature", 1.0))

    # Device
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    overlap_rows = max(window_size + 5, 2000)

    print(f"[load] Files: {len(files)}")
    print(f"[load] Streaming chunksize={cfg.chunksize:,} overlap={overlap_rows:,}")
    print(f"[meta] window={window_size} features={len(feature_cols)} task={'regression' if is_regression else 'classification'}")

    sim_cfg = SimulationConfig(
        start_capital=float(cfg.capital),
        fee_pct=fee_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        cooldown=int(cfg.cooldown),
        slippage_pct=float(cfg.slippage_pct),
        use_atr_stops=bool(cfg.use_atr_stops),
        atr_tp_mult=float(cfg.atr_tp_mult),
        atr_sl_mult=float(cfg.atr_sl_mult),
        use_regime_filter=bool(cfg.use_regime_filter),
        min_atr_pct=float(cfg.min_atr_pct),
        allow_shorts=True,
    )
    sim = PortfolioSimulator(sim_cfg)

    consensus_filter = ConsensusFilter(int(cfg.consensus))

    X_batch: List[np.ndarray] = []
    bar_batch: List[Bar] = []

    total_pred = 0
    last_close_for_end = None

    streamer = KlineStreamer(
        cfg.data_dir,
        window_size=window_size,
        feature_cols=feature_cols,
        chunksize=int(cfg.chunksize),
        overlap_rows=int(overlap_rows),
    )

    def _flush_batch():
        nonlocal total_pred
        if not X_batch:
            return

        X = np.stack(X_batch, axis=0).astype(np.float32)

        if scaler is not None:
            if hasattr(scaler, "feature_names_in_"):
                assert list(scaler.feature_names_in_) == feature_cols
            B, T, F = X.shape
            X = scaler.transform(X.reshape(B * T, F)).reshape(B, T, F).astype(np.float32, copy=False)

        xb = torch.from_numpy(X).to(device)

        with torch.no_grad():
            out = model(xb)

        if is_regression:
            preds = out.squeeze(-1).detach().cpu().numpy()
            for k, bar in enumerate(bar_batch):
                ret = (float(preds[k]) / max(1e-12, float(bar.close))) - 1.0
                sig = 2 if ret >= float(cfg.up_thr) else (0 if ret <= -float(cfg.down_thr) else 1)
                sim.step(bar, signal=sig)
                total_pred += 1
        else:
            logits = out / temperature if abs(temperature - 1.0) > 1e-6 else out
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

            for k, bar in enumerate(bar_batch):
                raw_sig = _raw_signal_from_probs(probs[k], float(cfg.thr_long), float(cfg.thr_short), float(cfg.margin))
                sig = consensus_filter.step(raw_sig)
                sim.step(bar, signal=sig)
                total_pred += 1

        X_batch.clear()
        bar_batch.clear()

    print("[stream] Starting ...")
    for window, _label, meta in streamer:
        last_close_for_end = float(meta.get("close", np.nan))

        o = float(meta.get("open", np.nan))
        h = float(meta.get("high", np.nan))
        l = float(meta.get("low", np.nan))
        c = float(meta.get("close", np.nan))
        atr_prev = meta.get("atr")
        ema50 = float(window[-1, feature_cols.index("ema_50")]) if "ema_50" in feature_cols else np.nan
        ema200 = float(window[-1, feature_cols.index("ema_200")]) if "ema_200" in feature_cols else np.nan

        regime = None
        if cfg.use_regime_filter and np.isfinite(ema50) and np.isfinite(ema200):
            regime = 1 if ema50 > ema200 * 1.0005 else (-1 if ema50 < ema200 * 0.9995 else 0)

        X_batch.append(window)
        bar_batch.append(Bar(open=o, high=h, low=l, close=c, atr=atr_prev, ema_fast=ema50, ema_slow=ema200, regime=regime, timestamp=meta.get("timestamp")))

        if len(X_batch) >= int(cfg.batch_size):
            _flush_batch()

        if total_pred and total_pred % 200_000 == 0:
            print(f"[stream] predictions={total_pred:,} equity={fmt_money(sim.last_equity)} mdd={fmt_pct(sim.max_drawdown)}")

    _flush_batch()
    if last_close_for_end is not None:
        sim.finalize(last_close_for_end)

    report = sim.report()
    if print_report:
        print_portfolio_report(report, currency="$")

    return {
        "report": report,
        "meta": meta,
        "total_pred": int(total_pred),
        "last_close": last_close_for_end,
    }


    return {
        "report": report,
        "meta": meta,
        "total_pred": int(total_pred),
        "last_close": last_close_for_end,
    }


if __name__ == "__main__":
    setup_logging()
    files = _list_csvs(cfg.data_dir)
    _run_backtest_for_files(files, cfg.model_dir, print_report=True)
