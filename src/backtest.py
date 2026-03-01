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
    FEATURE_COLUMNS_PROFITABLE,
    DEFAULT_SEQ_LEN,
    update_strategy_registry,
)
from profitability import (
    compute_profitability_metrics,
    monte_carlo_permutation_test,
    kelly_fraction,
    vol_target_leverage,
    profitability_report,
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
class ProfitOptimizedBacktester:
    def __init__(self, model_dir: str, files: List[Path]) -> None:
        self.model_dir = model_dir
        self.files = files

    def _load_and_validate(self):
        model, scaler, meta = load_model_bundle(self.model_dir)
        feature_cols = list(meta.get("feature_names") or meta.get("feature_cols") or FEATURE_COLUMNS_PROFITABLE)
        if int(meta.get("feature_count") or 0) != len(feature_cols): raise ValueError("Model retrain required for profitability — delete old checkpoint in seq_dir and retrain.")
        return model, scaler, meta

    def _run_stream(self, model, scaler, meta, *, leverage: float = 1.0) -> Dict:
        feature_cols = list(meta.get("feature_names") or meta.get("feature_cols") or FEATURE_COLUMNS_PROFITABLE)

        window_size = int(cfg.window_size or meta.get("window_size", DEFAULT_SEQ_LEN))
        fee_pct = 0.00075
        tp_pct = float(meta.get("tp_pct", cfg.tp_pct))
        sl_pct = float(meta.get("sl_pct", cfg.sl_pct))

        task = str(meta.get("task", "")).lower()
        is_regression = bool(task == "regression" or int(meta.get("num_classes", 3)) == 1)
        temperature = float(meta.get("temperature", 1.0))

        device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device).eval()

        overlap_rows = max(window_size + 5, 2000)

        sim_cfg = SimulationConfig(
            start_capital=float(cfg.capital),
            fee_pct=fee_pct,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            cooldown=int(cfg.cooldown),
            slippage_pct=0.0,
            use_atr_stops=bool(cfg.use_atr_stops),
            atr_tp_mult=float(cfg.atr_tp_mult),
            atr_sl_mult=float(cfg.atr_sl_mult),
            use_regime_filter=bool(cfg.use_regime_filter),
            min_atr_pct=float(cfg.min_atr_pct),
            allow_shorts=True,
            leverage=float(leverage),
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
                try:
                    X = scaler.transform(X.reshape(B * T, F)).reshape(B, T, F).astype(np.float32, copy=False)
                except Exception:
                    pass

            xb = torch.from_numpy(X).to(device)
            with torch.no_grad():
                out = model(xb)

            if is_regression:
                preds = out.squeeze(-1).detach().cpu().numpy()
                for k, bar in enumerate(bar_batch):
                    ret = (float(preds[k]) / max(1e-12, float(bar.close))) - 1.0
                    sig = 2 if ret >= float(cfg.up_thr) else (0 if ret <= -float(cfg.down_thr) else 1)
                    atr = float(bar.atr) if bar.atr is not None and np.isfinite(bar.atr) else None
                    sim.config.slippage_pct = (0.5 * atr / max(1e-12, float(bar.close))) if atr is not None else 0.0
                    sim.step(bar, signal=sig)
                    total_pred += 1
            else:
                logits = out / temperature if abs(temperature - 1.0) > 1e-6 else out
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
                for k, bar in enumerate(bar_batch):
                    raw_sig = _raw_signal_from_probs(probs[k], float(cfg.thr_long), float(cfg.thr_short), float(cfg.margin))
                    sig = consensus_filter.step(raw_sig)
                    atr = float(bar.atr) if bar.atr is not None and np.isfinite(bar.atr) else None
                    sim.config.slippage_pct = (0.5 * atr / max(1e-12, float(bar.close))) if atr is not None else 0.0
                    sim.step(bar, signal=sig)
                    total_pred += 1

            X_batch.clear()
            bar_batch.clear()

        for window, _label, meta_row in streamer:
            last_close_for_end = float(meta_row.get("close", np.nan))
            o = float(meta_row.get("open", np.nan))
            h = float(meta_row.get("high", np.nan))
            l = float(meta_row.get("low", np.nan))
            c = float(meta_row.get("close", np.nan))
            atr_prev = meta_row.get("atr")
            ema50 = float(window[-1, feature_cols.index("ema_50")]) if "ema_50" in feature_cols else np.nan
            ema200 = float(window[-1, feature_cols.index("ema_200")]) if "ema_200" in feature_cols else np.nan
            regime = None
            if cfg.use_regime_filter and np.isfinite(ema50) and np.isfinite(ema200):
                regime = 1 if ema50 > ema200 * 1.0005 else (-1 if ema50 < ema200 * 0.9995 else 0)

            X_batch.append(window)
            bar_batch.append(Bar(open=o, high=h, low=l, close=c, atr=atr_prev, ema_fast=ema50, ema_slow=ema200, regime=regime, timestamp=meta_row.get("timestamp")))
            if len(X_batch) >= int(cfg.batch_size):
                _flush_batch()

        _flush_batch()
        if last_close_for_end is not None:
            sim.finalize(last_close_for_end)

        report = sim.report()
        equity_curve = sim.equity_curve()
        metrics = compute_profitability_metrics(report, equity_curve, sim.trade_log)
        return {
            "report": report,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trade_log": sim.trade_log,
        }

    def run(self) -> Dict:
        model, scaler, meta = self._load_and_validate()
        # Walk-forward windows (6 months in, 3 months out) driven by timestamps
        all_df = normalize_headers(pd.concat([pd.read_csv(p) for p in self.files], ignore_index=True))
        if "timestamp" not in all_df.columns:
            raise ValueError("timestamp column is required for walk-forward optimization")
        all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], errors="coerce", utc=True)
        all_df = all_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        months = all_df["timestamp"].dt.to_period("M").unique()
        if len(months) < 9:
            raise ValueError("Not enough data for 6+3 month walk-forward")

        results = []
        for i in range(0, len(months) - 8):
            in_months = months[i:i+6]
            out_months = months[i+6:i+9]

            in_df = all_df[all_df["timestamp"].dt.to_period("M").isin(in_months)]
            out_df = all_df[all_df["timestamp"].dt.to_period("M").isin(out_months)]

            in_files = [Path("__in_memory__.csv")]
            out_files = [Path("__in_memory__.csv")]

            in_df.to_csv(in_files[0], index=False)
            out_df.to_csv(out_files[0], index=False)

            tmp_cfg_data_dir = cfg.data_dir
            cfg.data_dir = str(in_files[0])
            in_run = self._run_stream(model, scaler, meta, leverage=1.0)
            cfg.data_dir = str(out_files[0])

            kelly = kelly_fraction(in_run["metrics"]["trade_returns"])
            lev_kelly = float(np.clip(kelly, 0.1, 3.0))
            lev_vol = vol_target_leverage(in_run["equity_curve"])
            leverage = lev_kelly if lev_kelly >= lev_vol else lev_vol

            out_run = self._run_stream(model, scaler, meta, leverage=leverage)
            cfg.data_dir = tmp_cfg_data_dir

            results.append(out_run)

        # Aggregate OOS metrics
        all_trade_returns = np.concatenate([r["metrics"]["trade_returns"] for r in results if r["metrics"]["trade_returns"].size])
        if all_trade_returns.size == 0:
            raise ValueError("No out-of-sample trades produced")

        final_metrics = compute_profitability_metrics(
            results[-1]["report"],
            results[-1]["equity_curve"],
            results[-1]["trade_log"],
        )
        mc = monte_carlo_permutation_test(all_trade_returns, runs=500)
        print(profitability_report(final_metrics, mc))
        ev = float(final_metrics.get("expectancy", 0.0))
        pf = float(final_metrics.get("profit_factor", 0.0))
        max_dd = float(final_metrics.get("max_drawdown_pct", 0.0))
        sharpe = float(final_metrics.get("sharpe_annualized", 0.0))
        trade_returns = final_metrics.get("trade_returns", np.zeros(0, dtype=float))
        win_rate = float(np.mean(trade_returns > 0.0) * 100.0) if trade_returns is not None and len(trade_returns) else 0.0
        print(
            f"EV per trade: {ev:.6f} | Profit Factor: {pf:.2f} | Max Drawdown: {max_dd:.2f}% | "
            f"Sharpe: {sharpe:.2f} | Win Rate: {win_rate:.2f}%"
        )
        if ev < 0.0005 or pf < 1.4:
            print("\033[91mRETRAIN WITH PROFIT MODE — current model is not profitable enough\033[0m")
        report = {
            "ev_per_trade": ev,
            "expectancy": ev,
            "profit_factor": pf,
            "max_drawdown_pct": max_dd,
            "sharpe": sharpe,
            "win_rate_pct": win_rate,
            "recommendation": "Retrain recommended" if ev < 0.0005 else "Hold model",
        }
        report_path = Path(self.model_dir) / "profit_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if final_metrics["profit_factor"] < 1.8 or final_metrics["max_drawdown_pct"] > 10.0:
            print("UNPROFITABLE - REJECTED")
        else:
            update_strategy_registry(Path(self.model_dir), final_metrics.get("recovery_factor", 0.0))

        return {"metrics": final_metrics, "monte_carlo": mc}

if __name__ == "__main__":
    setup_logging()
    files = _list_csvs(cfg.data_dir)
    model_root = Path(cfg.model_dir)
    meta_path = model_root / "model_meta.json"
    if meta_path.exists():
        ProfitOptimizedBacktester(str(model_root), files).run()
    else:
        seq_dirs = sorted([p for p in model_root.glob("seq_*") if (p / "model_meta.json").exists()])
        if not seq_dirs:
            raise FileNotFoundError(f"model_meta.json not found in {cfg.model_dir} or any seq_* subdir")
        for seq_dir in seq_dirs:
            print(f"\n[seq] Running backtest for {seq_dir.name} ...")
            ProfitOptimizedBacktester(str(seq_dir), files).run()
