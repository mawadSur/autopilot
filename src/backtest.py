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
from strategy_gate import StrategyGate

from utils import (
    fmt_money,
    fmt_pct,
    compute_features,
    load_model_bundle,
    load_inference_bundle,
    is_xgboost_model_dir,
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
        class_to_raw,
    )
    try:
        from src.streamer import KlineStreamer
    except ModuleNotFoundError:
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
        class_to_raw,
    )
    try:
        from src.streamer import KlineStreamer
    except ModuleNotFoundError:
        from streamer import KlineStreamer
from config import cfg

# -------------------------
# CSV helper
# -------------------------
def _list_csvs(path: str) -> List[Path]:
    return KlineStreamer._list_csvs(Path(path))


def _opt_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _side_has_usable_l2_depth(meta_row: Dict[str, object], side: str) -> bool:
    best_key = f"best_{side}"
    best = _opt_float(meta_row.get(best_key))
    if best is None or best <= 0.0:
        return False

    depth_keys = (
        (f"{side}_depth_5", f"vwap_{side}_5"),
        (f"{side}_depth_10", f"vwap_{side}_10"),
        (f"{side}_depth_20", f"vwap_{side}_20"),
    )
    for depth_key, vwap_key in depth_keys:
        depth = _opt_float(meta_row.get(depth_key))
        vwap = _opt_float(meta_row.get(vwap_key))
        if depth is not None and depth > 0.0 and vwap is not None and vwap > 0.0:
            return True
    return False


def _bar_has_usable_l2_depth(meta_row: Dict[str, object]) -> bool:
    return _side_has_usable_l2_depth(meta_row, "bid") and _side_has_usable_l2_depth(meta_row, "ask")


def _warn_on_sparse_l2_depth(total_bars: int, missing_bars: int, *, threshold: float = 0.10) -> None:
    total_bars = int(max(0, total_bars))
    missing_bars = int(max(0, missing_bars))
    if total_bars <= 0:
        return

    missing_ratio = float(missing_bars) / float(total_bars)
    if missing_ratio <= float(threshold):
        return

    logger.warning(
        f"[backtest] L2 depth data missing on {missing_ratio * 100.0:.2f}% "
        f"({missing_bars}/{total_bars}) of bars. "
        "Depth-aware execution will fall back to top-of-book or ATR slippage heuristics more often."
    )

# =========================
# Gating (online)
# =========================
def _raw_signal_from_probs(
    p: np.ndarray,
    thr_long: float,
    thr_short: float,
    margin: float,
    feature_row: Optional[Dict[str, float]] = None,
    window: Optional[np.ndarray] = None,
    strategy_gate: Optional[StrategyGate] = None,
) -> int:
    """
    p: [3] probabilities
    returns class: 0=short, 1=hold, 2=long
    """
    gate = strategy_gate or StrategyGate(
        thr_long=thr_long,
        thr_short=thr_short,
        margin=margin,
    )
    return gate.signal_from_probs(p, window=window, feature_row=feature_row)

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
# XGBoost inference adapter
# =========================
class XGBoostInferenceAdapter:
    """Wraps an XGBoost binary classifier to produce [N, 3] prob arrays.

    Columns: [1 - p_long, 0.0, p_long] so downstream StrategyGate (size>=3 path)
    treats col-0 as short-prob and col-2 as long-prob. The synthetic hold column
    is 0.0 which is fine — gating compares col-2 vs thr_long.
    """

    def __init__(self, model, scaler, feature_cols: List[str], meta: Dict) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.meta = meta

    def predict_probs(self, rows: np.ndarray) -> np.ndarray:
        """rows: [N, F] float32. Returns [N, 3] float64."""
        X = rows.astype(np.float64)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        p_long = self.model.predict_proba(X)[:, 1]
        N = p_long.shape[0]
        out = np.zeros((N, 3), dtype=np.float64)
        out[:, 2] = p_long
        out[:, 0] = 1.0 - p_long
        return out


# =========================
# Online portfolio simulator
# =========================
class ProfitOptimizedBacktester:
    def __init__(self, model_dir: str, files: List[Path]) -> None:
        self.model_dir = model_dir
        self.files = files
        self._kind: str = "legacy"

    def _load_and_validate(self):
        kind, model, scaler, meta = load_inference_bundle(self.model_dir)
        self._kind = kind
        if kind == "xgboost":
            feature_cols = list(meta.get("feature_cols") or [])
            if not feature_cols:
                raise ValueError("XGBoost meta.json missing feature_cols")
            return model, scaler, meta
        # legacy path: enforce feature_count sentinel
        feature_cols = list(meta.get("feature_names") or meta.get("feature_cols") or FEATURE_COLUMNS_PROFITABLE)
        if int(meta.get("feature_count") or 0) != len(feature_cols): raise ValueError("Model retrain required for profitability — delete old checkpoint in seq_dir and retrain.")
        return model, scaler, meta

    def _run_stream(self, model, scaler, meta, *, leverage: float = 1.0) -> Dict:
        is_xgb = self._kind == "xgboost"
        feature_cols = list(meta.get("feature_names") or meta.get("feature_cols") or FEATURE_COLUMNS_PROFITABLE)

        if is_xgb:
            xgb_adapter = XGBoostInferenceAdapter(model, scaler, feature_cols, meta)
            # Window size 2 is the minimum; XGBoost uses only the last row of each window.
            window_size = 2
        else:
            window_size = int(cfg.window_size or meta.get("window_size", DEFAULT_SEQ_LEN))

        # HONEST COSTS: use the *real* Coinbase Advanced retail fee schedule
        # (taker 60 bps / maker 40 bps) that the live adapter charges, instead of
        # the old fictional 7.5 bps. The previous ``fee_pct = 0.00075`` understated
        # round-trip cost by ~16x and left the maker path unwired, making every
        # backtest fictional. ``from_coinbase_fees()`` wires BOTH sides.
        # See trading/simulator.py COINBASE_*_FEE_PCT and
        # src/exchanges/adapters/coinbase_tradeable.py _DEFAULT_COINBASE_FEE_MODEL.
        tp_pct = float(meta.get("tp_pct", cfg.tp_pct))
        sl_pct = float(meta.get("sl_pct", cfg.sl_pct))

        if not is_xgb:
            task = str(meta.get("task", "")).lower()
            is_regression = bool(task == "regression" or int(meta.get("num_classes", 3)) == 1)
            temperature = float(meta.get("temperature", 1.0))
            device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
            model = model.to(device).eval()

        overlap_rows = max(window_size + 5, 2000)

        sim_cfg = SimulationConfig.from_coinbase_fees(
            start_capital=float(cfg.capital),
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            cooldown=int(cfg.cooldown),
            slippage_pct=0.0,
            use_market_depth=True,
            use_hard_gating=bool(getattr(cfg, "use_hard_gating", True)),
            post_only_entries=True,
            fallback_to_market_on_missing_book=True,
            fallback_to_market_on_post_only_miss=False,
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
        strategy_gate = StrategyGate(
            thr_long=float(cfg.thr_long),
            thr_short=float(cfg.thr_short),
            margin=float(cfg.margin),
            feature_cols=feature_cols if not is_xgb else [],
            use_hard_gating=bool(sim_cfg.use_hard_gating),
        )

        X_batch: List[np.ndarray] = []
        bar_batch: List[Bar] = []
        feature_meta_batch: List[Dict[str, float]] = []
        total_pred = 0
        last_close_for_end = None
        total_bars = 0
        missing_l2_bars = 0

        streamer = KlineStreamer(
            cfg.data_dir,
            window_size=window_size,
            feature_cols=feature_cols,
            chunksize=int(cfg.chunksize),
            overlap_rows=int(overlap_rows),
        )

        def _flush_batch_xgb() -> None:
            nonlocal total_pred
            if not X_batch:
                return
            rows = np.stack(X_batch, axis=0).astype(np.float64)
            probs = xgb_adapter.predict_probs(rows)  # [N, 3]
            for k, bar in enumerate(bar_batch):
                raw_sig = _raw_signal_from_probs(
                    probs[k],
                    float(cfg.thr_long),
                    float(cfg.thr_short),
                    float(cfg.margin),
                    feature_meta_batch[k],
                    window=None,
                    strategy_gate=strategy_gate,
                )
                sig = consensus_filter.step(raw_sig)
                atr = float(bar.atr) if bar.atr is not None and np.isfinite(bar.atr) else None
                sim.config.slippage_pct = (0.5 * atr / max(1e-12, float(bar.close))) if atr is not None else 0.0
                sim.step(bar, signal=class_to_raw(sig))
                total_pred += 1
            X_batch.clear()
            bar_batch.clear()
            feature_meta_batch.clear()

        def _flush_batch():
            nonlocal total_pred
            if not X_batch:
                return

            X = np.stack(X_batch, axis=0).astype(np.float32)
            if scaler is not None:
                if hasattr(scaler, "feature_names_in_"):
                    assert list(scaler.feature_names_in_) == feature_cols
                B, T, n_features = X.shape
                try:
                    X = scaler.transform(X.reshape(B * T, n_features)).reshape(B, T, n_features).astype(np.float32, copy=False)
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
                    sim.step(bar, signal=class_to_raw(sig))
                    total_pred += 1
            else:
                logits = out / temperature if abs(temperature - 1.0) > 1e-6 else out
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
                for k, bar in enumerate(bar_batch):
                    raw_sig = _raw_signal_from_probs(
                        probs[k],
                        float(cfg.thr_long),
                        float(cfg.thr_short),
                        float(cfg.margin),
                        feature_meta_batch[k],
                        window=X_batch[k],
                        strategy_gate=strategy_gate,
                    )
                    sig = consensus_filter.step(raw_sig)
                    atr = float(bar.atr) if bar.atr is not None and np.isfinite(bar.atr) else None
                    sim.config.slippage_pct = (0.5 * atr / max(1e-12, float(bar.close))) if atr is not None else 0.0
                    sim.step(bar, signal=class_to_raw(sig))
                    total_pred += 1

            X_batch.clear()
            bar_batch.clear()
            feature_meta_batch.clear()

        _flush_fn = _flush_batch_xgb if is_xgb else _flush_batch

        for window, _label, meta_row in streamer:
            total_bars += 1
            if sim_cfg.use_market_depth and not _bar_has_usable_l2_depth(meta_row):
                missing_l2_bars += 1
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

            # XGBoost: use only last row of window as 1D feature vector
            X_batch.append(window[-1] if is_xgb else window)
            feature_meta_batch.append(dict(meta_row))
            bar_batch.append(Bar(
                open=o,
                high=h,
                low=l,
                close=c,
                atr=atr_prev,
                ema_fast=ema50,
                ema_slow=ema200,
                regime=regime,
                timestamp=meta_row.get("timestamp"),
                best_bid=meta_row.get("best_bid"),
                best_ask=meta_row.get("best_ask"),
                bid_depth_5=meta_row.get("bid_depth_5"),
                ask_depth_5=meta_row.get("ask_depth_5"),
                bid_depth_10=meta_row.get("bid_depth_10"),
                ask_depth_10=meta_row.get("ask_depth_10"),
                bid_depth_20=meta_row.get("bid_depth_20"),
                ask_depth_20=meta_row.get("ask_depth_20"),
                vwap_bid_5=meta_row.get("vwap_bid_5"),
                vwap_ask_5=meta_row.get("vwap_ask_5"),
                vwap_bid_10=meta_row.get("vwap_bid_10"),
                vwap_ask_10=meta_row.get("vwap_ask_10"),
                vwap_bid_20=meta_row.get("vwap_bid_20"),
                vwap_ask_20=meta_row.get("vwap_ask_20"),
            ))
            if len(X_batch) >= int(cfg.batch_size):
                _flush_fn()

        _flush_fn()
        if last_close_for_end is not None:
            sim.finalize(last_close_for_end)
        _warn_on_sparse_l2_depth(total_bars, missing_l2_bars)

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

        # Hard reject gate — these thresholds are computed on the HONEST Coinbase
        # fee schedule (see _run_stream). A model only "passes" if it clears both
        # the profit-factor floor and the drawdown ceiling. We persist the verdict
        # into profit_report.json so the decision is auditable, not just printed.
        gate_min_profit_factor = 1.8
        gate_max_drawdown_pct = 10.0
        gate_rejected = bool(pf < gate_min_profit_factor or max_dd > gate_max_drawdown_pct)
        report = {
            "ev_per_trade": ev,
            "expectancy": ev,
            "profit_factor": pf,
            "max_drawdown_pct": max_dd,
            "sharpe": sharpe,
            "win_rate_pct": win_rate,
            "recommendation": "Retrain recommended" if ev < 0.0005 else "Hold model",
            "gate_min_profit_factor": gate_min_profit_factor,
            "gate_max_drawdown_pct": gate_max_drawdown_pct,
            "gate_passed": (not gate_rejected),
            "gate_verdict": "REJECTED" if gate_rejected else "ACCEPTED",
        }
        report_path = Path(self.model_dir) / "profit_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if gate_rejected:
            print("UNPROFITABLE - REJECTED")
        else:
            update_strategy_registry(Path(self.model_dir), final_metrics.get("recovery_factor", 0.0))

        return {"metrics": final_metrics, "monte_carlo": mc}

if __name__ == "__main__":
    setup_logging()
    files = _list_csvs(cfg.data_dir)
    model_root = Path(cfg.model_dir)
    # XGBoost bundle: model.joblib + meta.json
    if is_xgboost_model_dir(str(model_root)):
        logger.info("[backtest] Detected XGBoost bundle at %s", model_root)
        ProfitOptimizedBacktester(str(model_root), files).run()
    elif (model_root / "model_meta.json").exists():
        ProfitOptimizedBacktester(str(model_root), files).run()
    else:
        seq_dirs = sorted([p for p in model_root.glob("seq_*") if (p / "model_meta.json").exists()])
        if not seq_dirs:
            raise FileNotFoundError(
                f"No supported model found in {cfg.model_dir}. "
                "Expected model.joblib+meta.json (XGBoost) or model_meta.json (LegacyTransformer)."
            )
        for seq_dir in seq_dirs:
            print(f"\n[seq] Running backtest for {seq_dir.name} ...")
            ProfitOptimizedBacktester(str(seq_dir), files).run()
