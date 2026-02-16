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

from utils import (
    fmt_money,
    fmt_pct,
    compute_features,
    load_model_bundle,
    normalize_headers,
    FEATURE_COLUMNS,
    DEFAULT_SEQ_LEN,
)

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
# CSV streaming helpers
# =========================
def _list_csvs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return files
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file or directory, got: {path}")
    return [p]

def _iter_feature_chunks(
    files: List[Path],
    *,
    chunksize: int,
    overlap_rows: int,
) -> Iterable[pd.DataFrame]:
    """
    Yield feature-engineered chunks with continuity across file/chunk boundaries.

    We keep a tail of the last `overlap_rows` raw rows, prepend it to the next chunk,
    compute features on the combined frame, then yield only the "new" region (drop the
    overlap prefix) while saving a new tail from the end.
    """
    tail_raw: Optional[pd.DataFrame] = None

    for f in files:
        for chunk in pd.read_csv(f, chunksize=chunksize):
            chunk = normalize_headers(chunk)
            if tail_raw is not None:
                raw = pd.concat([tail_raw, chunk], ignore_index=True)
            else:
                raw = chunk

            # Feature engineering (uses rolling/ewm; needs overlap)
            feat = compute_features(raw)

            # Add regime helper EMAs (computed here so they are continuous too)
            feat["ema_50"] = feat["close"].ewm(span=50, adjust=False).mean()
            feat["ema_200"] = feat["close"].ewm(span=200, adjust=False).mean()

            if len(feat) > overlap_rows:
                # Yield only the "new" part (exclude overlap prefix)
                out = feat.iloc[overlap_rows:].reset_index(drop=True)
                # Keep tail *raw* rows for next iteration (use raw to recompute indicators cleanly)
                tail_raw = raw.iloc[-overlap_rows:].reset_index(drop=True)
                yield out
            else:
                # Not enough rows yet; just grow the tail
                tail_raw = raw.reset_index(drop=True)

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
class OnlinePortfolioSim:
    """
    Streaming version of simulate_trades_with_tp_sl().

    - Uses prev signal to decide entry/exit at current open.
    - TP/SL checked intra-bar; opposing signals exit at open.
    - Tracks max drawdown online.
    """
    def __init__(
        self,
        *,
        start_capital: float,
        fee_pct: float,
        tp_pct: float,
        sl_pct: float,
        cooldown: int,
        slippage_pct: float,
        use_atr_stops: bool,
        atr_tp_mult: float,
        atr_sl_mult: float,
    ):
        self.start_capital = float(start_capital)
        self.fee_pct = float(fee_pct)
        self.tp_pct = float(tp_pct)
        self.sl_pct = float(sl_pct)
        self.cooldown = max(0, int(cooldown))
        self.slippage_pct = float(slippage_pct)
        self.use_atr_stops = bool(use_atr_stops)
        self.atr_tp_mult = float(atr_tp_mult)
        self.atr_sl_mult = float(atr_sl_mult)

        self.cash = float(start_capital)
        self.pos = 0  # 0 flat, +1 long, -1 short
        self.entry_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.cdn = 0

        self.trades = 0
        self.wins = 0
        self.losses = 0

        self.peak_equity = float(start_capital)
        self.max_drawdown = 0.0

        self.n_bars = 0

    def _mark_equity(self, close_price: float) -> float:
        if self.pos == 0 or self.entry_price is None:
            equity = self.cash
        elif self.pos == +1:
            equity = self.cash * (1.0 + ((close_price / self.entry_price) - 1.0))
        else:
            equity = self.cash * (1.0 + ((self.entry_price / close_price) - 1.0))
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / max(1e-12, self.peak_equity)
        self.max_drawdown = max(self.max_drawdown, dd)
        return equity

    def step(self, *, o: float, h: float, l: float, c: float, sig_prev: int, atr_prev: Optional[float]) -> None:
        """
        Process one bar using previous bar's signal.
        """
        self.n_bars += 1

        # Entry logic (flat)
        if self.pos == 0:
            if self.cdn > 0:
                self.cdn -= 1
                self._mark_equity(c)
                return

            if sig_prev == 2:  # long
                self.cash *= (1.0 - self.fee_pct)
                entry = float(o) * (1.0 + self.slippage_pct)
                self.entry_price = entry
                if self.use_atr_stops and atr_prev is not None and np.isfinite(atr_prev):
                    a = float(max(1e-12, atr_prev))
                    self.tp_price = entry + self.atr_tp_mult * a
                    self.sl_price = entry - self.atr_sl_mult * a
                else:
                    self.tp_price = entry * (1.0 + self.tp_pct)
                    self.sl_price = entry * (1.0 - self.sl_pct)
                self.pos = +1
                self.trades += 1
                self._mark_equity(c)
                return

            if sig_prev == 0:  # short
                self.cash *= (1.0 - self.fee_pct)
                entry = float(o) * (1.0 - self.slippage_pct)
                self.entry_price = entry
                if self.use_atr_stops and atr_prev is not None and np.isfinite(atr_prev):
                    a = float(max(1e-12, atr_prev))
                    self.tp_price = entry - self.atr_tp_mult * a
                    self.sl_price = entry + self.atr_sl_mult * a
                else:
                    self.tp_price = entry * (1.0 - self.tp_pct)
                    self.sl_price = entry * (1.0 + self.sl_pct)
                self.pos = -1
                self.trades += 1
                self._mark_equity(c)
                return

            self._mark_equity(c)
            return

        # Manage open position
        assert self.entry_price is not None and self.tp_price is not None and self.sl_price is not None

        exit_price = None
        win = None

        if self.pos == +1:
            # SL first, then TP (conservative)
            if l <= self.sl_price <= h:
                exit_price = self.sl_price
                win = False
            elif h >= self.tp_price:
                exit_price = self.tp_price
                win = True
            elif sig_prev == 0:
                exit_price = float(o)
                win = exit_price >= self.entry_price
        else:
            # short: SL if rises; TP if drops
            if h >= self.sl_price:
                exit_price = self.sl_price
                win = False
            elif l <= self.tp_price:
                exit_price = self.tp_price
                win = True
            elif sig_prev == 2:
                exit_price = float(o)
                win = exit_price <= self.entry_price

        if exit_price is not None:
            if self.pos == +1:
                exit_exec = float(exit_price) * (1.0 - self.slippage_pct)
                ret = (exit_exec / self.entry_price) - 1.0
            else:
                exit_exec = float(exit_price) * (1.0 + self.slippage_pct)
                ret = (self.entry_price / exit_exec) - 1.0

            self.cash *= (1.0 + ret)
            self.cash *= (1.0 - self.fee_pct)

            self.pos = 0
            self.entry_price = self.tp_price = self.sl_price = None
            self.wins += int(bool(win))
            self.losses += int(not bool(win))
            self.cdn = self.cooldown

            self._mark_equity(c)
            return

        # Still in position
        self._mark_equity(c)

    def close_end(self, last_close: float) -> None:
        # If still open, close at last close
        if self.pos != 0 and self.entry_price is not None:
            c = float(last_close)
            if self.pos == +1:
                exit_exec = c * (1.0 - self.slippage_pct)
                ret = (exit_exec / self.entry_price) - 1.0
            else:
                exit_exec = c * (1.0 + self.slippage_pct)
                ret = (self.entry_price / exit_exec) - 1.0
            self.cash *= (1.0 + ret)
            self.cash *= (1.0 - self.fee_pct)
            self.pos = 0
            self.entry_price = self.tp_price = self.sl_price = None
            self._mark_equity(c)

    def report(self) -> Dict:
        return {
            "metrics": {"n": int(self.n_bars)},
            "portfolio": {
                "start_capital": float(self.start_capital),
                "end_equity": float(self.cash if self.pos == 0 else self.cash),  # cash updated through mark-to-market
                "return": float(self.cash / max(1e-12, self.start_capital) - 1.0),
                "max_drawdown": float(self.max_drawdown),
                "trades": int(self.trades),
                "wins": int(self.wins),
                "losses": int(self.losses),
            },
        }

# =========================
# CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Streaming backtester with TP/SL and probability threshold.")
    p.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or a single CSV")
    p.add_argument("--model-dir", type=str, default="model", help="Where model_meta.json & model.pt live")
    p.add_argument("--window-size", type=int, default=None,
                   help=f"Override sequence length (default from model_meta.json or {DEFAULT_SEQ_LEN}).")

    p.add_argument("--capital", type=float, default=10_000.0, help="Starting capital")
    p.add_argument("--batch-size", type=int, default=512, help="Inference batch size (windows)")
    p.add_argument("--chunksize", type=int, default=300_000, help="CSV read chunksize")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Classification gating controls
    p.add_argument("--thr-long", type=float, default=0.75)
    p.add_argument("--thr-short", type=float, default=0.75)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--consensus", type=int, default=2)

    # Portfolio controls
    p.add_argument("--cooldown", type=int, default=3)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--fee-pct", type=float, default=None)
    p.add_argument("--use-atr-stops", action="store_true", default=True)
    p.add_argument("--atr-tp-mult", type=float, default=1.8)
    p.add_argument("--atr-sl-mult", type=float, default=1.0)
    p.add_argument("--tp-pct", type=float, default=None, help="Legacy TP (used only if ATR stops are off/unavailable)")
    p.add_argument("--sl-pct", type=float, default=None, help="Legacy SL (used only if ATR stops are off/unavailable)")

    # Filters
    p.add_argument("--use-regime-filter", action="store_true", default=True)
    p.add_argument("--min-atr-pct", type=float, default=0.001)

    # Regression controls
    p.add_argument("--force-regress", action="store_true", help="Force regression mode")
    p.add_argument("--up-thr", type=float, default=0.002, help="Predicted return threshold for long (+0.2%)")
    p.add_argument("--down-thr", type=float, default=0.002, help="Predicted return threshold for short (-0.2%)")

    return p

# =========================
# Main
# =========================
def main():
    args = build_argparser().parse_args()

    # Load model bundle
    model, scaler, meta = load_model_bundle(args.model_dir)
    # Use meta feature list if present, else fall back to canonical FEATURE_COLUMNS
    feature_cols = list(meta.get("feature_cols", FEATURE_COLUMNS))
    # Assert canonical feature list consistency
    if feature_cols != FEATURE_COLUMNS:
        raise ValueError(f"Feature list mismatch: meta has {feature_cols}, expected {FEATURE_COLUMNS}")
    if args.window_size is not None:
        window_size = int(args.window_size)
    else:
        window_size = int(meta.get("window_size", DEFAULT_SEQ_LEN))
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = float(meta.get("tp_pct", 0.005)) if args.tp_pct is None else float(args.tp_pct)
    sl_pct = float(meta.get("sl_pct", 0.0025)) if args.sl_pct is None else float(args.sl_pct)

    task = str(meta.get("task", "")).lower()
    is_regression = bool(args.force_regress or task == "regression" or int(meta.get("num_classes", 3)) == 1)
    temperature = float(meta.get("temperature", 1.0))

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Streaming controls: overlap must cover indicator lookbacks + windowing alignment.
    # compute_features uses up to 1440 rolling for vwap proxy; keep some headroom.
    overlap_rows = max(window_size + 5, 2000)

    files = _list_csvs(args.data_dir)
    print(f"[load] Files: {len(files)}")
    print(f"[load] Streaming chunksize={args.chunksize:,} overlap_rows={overlap_rows:,}")
    print(f"[meta] window_size={window_size} features={len(feature_cols)} task={'regression' if is_regression else 'classification'}")

    # Online sim
    sim = OnlinePortfolioSim(
        start_capital=float(args.capital),
        fee_pct=fee_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        cooldown=int(args.cooldown),
        slippage_pct=float(args.slippage_pct),
        use_atr_stops=bool(args.use_atr_stops),
        atr_tp_mult=float(args.atr_tp_mult),
        atr_sl_mult=float(args.atr_sl_mult),
    )

    # Window buffer
    feat_buf: Deque[np.ndarray] = deque(maxlen=window_size)

    # For "signal executes on next bar open" logic:
    prev_signal: int = 1  # start HOLD
    have_prev_signal = False  # becomes true once we produce first prediction

    # For consensus gating (classification)
    consensus_filter = ConsensusFilter(int(args.consensus))

    # Batch buffers (windows + aligned bar data)
    X_batch: List[np.ndarray] = []
    bar_batch: List[Tuple[float, float, float, float, Optional[float], float, float]] = []
    # tuple: (open, high, low, close, atr_prev, ema50, ema200) aligned to window-end row

    total_pred = 0
    last_close_for_end = None

    def _flush_batch():
        nonlocal prev_signal, have_prev_signal, total_pred

        if not X_batch:
            return

        X = np.stack(X_batch, axis=0).astype(np.float32)  # [B,T,F]

        # Scale
        if scaler is not None:
            if hasattr(scaler, "feature_names_in_"):
                assert list(scaler.feature_names_in_) == feature_cols, "Scaler feature order mismatch"
            B, T, Fdim = X.shape
            X2 = scaler.transform(X.reshape(B * T, Fdim)).reshape(B, T, Fdim).astype(np.float32, copy=False)
        else:
            X2 = X

        xb = torch.from_numpy(X2).to(device, non_blocking=(device.type == "cuda"))

        with torch.no_grad():
            out = model(xb)

        if is_regression:
            preds = out.squeeze(-1).detach().cpu().numpy().astype(np.float32)
            for k in range(len(preds)):
                o, h, l, c, atr_prev, ema50, ema200 = bar_batch[k]
                # ret estimate matches your old logic: (pred / close) - 1
                ret = (float(preds[k]) / max(1e-12, float(c))) - 1.0
                sig = 1
                if ret >= float(args.up_thr):
                    sig = 2
                elif ret <= -max(0.0, float(args.down_thr)):
                    sig = 0

                # Regime filter
                if args.use_regime_filter:
                    if sig == 2 and not (ema50 >= ema200):
                        sig = 1
                    if sig == 0 and not (ema50 <= ema200):
                        sig = 1

                # Min ATR filter
                if args.min_atr_pct and atr_prev is not None and np.isfinite(atr_prev):
                    if (float(atr_prev) / max(1e-12, float(c))) < float(args.min_atr_pct):
                        sig = 1

                # Apply sim step using PREVIOUS signal on current bar
                if have_prev_signal:
                    sim.step(o=o, h=h, l=l, c=c, sig_prev=prev_signal, atr_prev=atr_prev)
                prev_signal = sig
                have_prev_signal = True
                total_pred += 1
        else:
            logits = out
            if temperature and abs(float(temperature) - 1.0) > 1e-6:
                logits = logits / float(temperature)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)  # [B,3]

            for k in range(len(probs)):
                o, h, l, c, atr_prev, ema50, ema200 = bar_batch[k]

                raw_sig = _raw_signal_from_probs(
                    probs[k],
                    thr_long=float(args.thr_long),
                    thr_short=float(args.thr_short),
                    margin=float(args.margin),
                )
                sig = consensus_filter.step(raw_sig)

                # Regime filter
                if args.use_regime_filter:
                    if sig == 2 and not (ema50 >= ema200):
                        sig = 1
                    if sig == 0 and not (ema50 <= ema200):
                        sig = 1

                # Min ATR filter
                if args.min_atr_pct and atr_prev is not None and np.isfinite(atr_prev):
                    if (float(atr_prev) / max(1e-12, float(c))) < float(args.min_atr_pct):
                        sig = 1

                if have_prev_signal:
                    sim.step(o=o, h=h, l=l, c=c, sig_prev=prev_signal, atr_prev=atr_prev)
                prev_signal = sig
                have_prev_signal = True
                total_pred += 1

        X_batch.clear()
        bar_batch.clear()

    print("[stream] Starting ...")
    for feat_chunk in _iter_feature_chunks(files, chunksize=int(args.chunksize), overlap_rows=int(overlap_rows)):
        # Ensure required features exist
        missing = [c for c in feature_cols if c not in feat_chunk.columns]
        if missing:
            raise SystemExit(f"Missing required features in engineered data: {missing}")

        # Prepare row-wise streaming
        # NOTE: We align bars to the SAME row as the window end.
        # Execution uses prev signal on NEXT bar open, handled inside sim.step() call order.
        for i in range(len(feat_chunk)):
            row = feat_chunk.iloc[i]

            # Update last close for end-close
            last_close_for_end = float(row["close"])

            # Append current feature row
            fvec = row[feature_cols].to_numpy(dtype=np.float32, copy=False)
            feat_buf.append(fvec)

            if len(feat_buf) < window_size:
                continue

            # Align current bar data (window ends at current row)
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            atr_prev = None
            if "atr" in feat_chunk.columns:
                atr_prev = float(row["atr"])
            ema50 = float(row.get("ema_50", np.nan))
            ema200 = float(row.get("ema_200", np.nan))

            X_batch.append(np.stack(list(feat_buf), axis=0))  # [T,F]
            bar_batch.append((o, h, l, c, atr_prev, ema50, ema200))

            if len(X_batch) >= int(args.batch_size):
                _flush_batch()

        # Progress-ish
        if total_pred and (total_pred % 200_000 == 0):
            print(f"[stream] predictions={total_pred:,} equity={fmt_money(sim.cash)} mdd={fmt_pct(sim.max_drawdown)}")

    # Flush last partial batch
    _flush_batch()

    # Close end
    if last_close_for_end is not None:
        sim.close_end(last_close_for_end)

    report = sim.report()
    print_portfolio_report(report, currency="$")

if __name__ == "__main__":
    main()
