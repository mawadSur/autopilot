#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_with_llm.py — Backtester extended with an optional Hugging Face LLM pipeline
that can be used to predict next-close price from the same features used by the
trained model.

This file is based on the user's backtest.py with the following additions:
- New CLI flags to enable an LLM-based prediction pipeline (--use-llm, --llm-model, etc.)
- Utility functions to build prompts from the feature windows and to parse LLM outputs
- Integration of the LLM predictions into the existing signal generation path

Limitations & notes:
- LLMs (text-generation models) are not optimized for precise numeric regression.
  Expect noisy outputs and tune the prompt carefully. Use the same features the
  training model used for best parity.
- Requires `transformers>=4.30` (or recent) and a model available on Hugging Face.
  Example model id: `tiiuae/grok-1.0-mini` or other Grok-family or Llama-family models.
- If running on GPU, ensure you have CUDA + PyTorch compatible with the model.

Usage example (portfolio mode with LLM predictions):

python backtest_with_llm.py --data-dir /path/to/csv --use-llm --llm-model "tiiuae/grok-1.0-mini" \
    --llm-batch-size 8 --device auto --capital 10000

"""

from __future__ import annotations

import argparse
import json
import os
import math as _math
import re
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Extended imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from utils import (
    read_csv_concat_sorted, resolve_price_col, build_windows,
    fmt_money, fmt_pct, compute_features, load_model_bundle, normalize_headers
)

FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'return', 'Range',
]
# --- (kept) existing functions from original backtest.py (simulate_trades_*, apply_gating, predict_*, etc.)
# For brevity in this code file, we import or reimplement only the functions needed below.
# The full implementations of simulate_trades_with_tp_sl, predict_proba/predict_regression and
# other helpers are expected to be present in the original file. If you prefer, merge this
# file back into your original `backtest.py` by copying the LLM-specific sections.

# For the purposes of distribution, we'll re-use the key functions present in the original
# backtest.py. Please ensure you merge these LLM additions into your canonical backtest.py
# if you want a single script.

# -----------------------
# LLM helper utilities
# -----------------------
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
def simulate_trades_with_tp_sl(opens, highs, lows, closes, classes, *, start_capital,
                               fee_pct=0.0008, tp_pct=0.005, sl_pct=0.005,
                               atr: Optional[np.ndarray] = None,
                               atr_tp_mult: Optional[float] = None,
                               atr_sl_mult: Optional[float] = None,
                               cooldown: int = 0,
                               slippage_pct: float = 0.0,
                               dynamic_sizing: bool = False,
                               max_risk_per_trade: float = 0.02,
                               leverage: float = 1.0) -> Tuple[Dict, pd.DataFrame]:
    """Simulate trades for a 3-class model: 0=short, 1=hold, 2=long.

    - Enters at next bar's open based on prior bar's signal.
    - One position at a time. Tracks TP/SL intrabar; opposing signals exit at open.
    - Equity is fully deployed on each trade; fees charged on entry and exit.
    """
    n = len(closes)
    cash = float(start_capital)
    equity_curve = np.empty(n, dtype=float)
    equity_curve[0] = cash

    pos = 0  # 0=flat, +1=long, -1=short
    entry_price = None
    tp_price = None
    sl_price = None
    cdn = 0  # cooldown counter

    trades = wins = losses = 0

    for i in range(1, n):
        o, hi, lo, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        sig_prev = int(classes[i - 1])  # signal decided at bar i-1, executed at i open

        if pos == 0:
            # Enter new position at open based on signal
            if cdn > 0:
                cdn -= 1
            elif sig_prev == 2:  # long
                entry_price = o * (1.0 + slippage_pct)
                # Determine TP and SL prices first
                if atr is not None and atr_tp_mult is not None and atr_sl_mult is not None:
                    a = float(max(1e-12, atr[i-1]))
                    tp_price = entry_price + atr_tp_mult * a
                    sl_price = entry_price - atr_sl_mult * a
                else:
                    tp_price = entry_price * (1.0 + tp_pct)
                    sl_price = entry_price * (1.0 - sl_pct)
                # Determine position size by risk percent or full equity
                if dynamic_sizing:
                    dist = entry_price - sl_price
                    risk_amount = cash * max_risk_per_trade
                    position_size = risk_amount / max(dist, 1e-9)
                    fee_amount = fee_pct * position_size * leverage
                    cash -= fee_amount
                else:
                    position_size = cash * leverage
                    cash -= fee_pct * position_size
                pos = +1
                trades += 1
            elif cdn == 0 and sig_prev == 0:  # short
                entry_price = o * (1.0 - slippage_pct)
                # Determine TP and SL prices first
                if atr is not None and atr_tp_mult is not None and atr_sl_mult is not None:
                    a = float(max(1e-12, atr[i-1]))
                    tp_price = entry_price - atr_tp_mult * a  # target below
                    sl_price = entry_price + atr_sl_mult * a  # stop above
                else:
                    tp_price = entry_price * (1.0 - tp_pct)
                    sl_price = entry_price * (1.0 + sl_pct)
                # Determine position size by risk percent or full equity
                if dynamic_sizing:
                    dist = sl_price - entry_price
                    risk_amount = cash * max_risk_per_trade
                    position_size = risk_amount / max(dist, 1e-9)
                    fee_amount = fee_pct * position_size * leverage
                    cash -= fee_amount
                else:
                    position_size = cash * leverage
                    cash -= fee_pct * position_size
                pos = -1
                trades += 1
            equity_curve[i] = cash
            continue

        # Manage open position
        exit_price = None
        win = None

        if pos == +1:
            # First check intrabar SL, then TP (conservative)
            if lo <= sl_price <= hi:
                exit_price = sl_price
                win = False
            elif hi >= tp_price:
                exit_price = tp_price
                win = True
            # If no TP/SL, opposing signal exits at open
            elif sig_prev == 0:
                exit_price = o
                win = (exit_price >= entry_price)
        else:  # pos == -1 (short)
            # For short: SL if price rises to sl_price; TP if price drops to tp_price
            if hi >= sl_price:
                exit_price = sl_price
                win = False
            elif lo <= tp_price:
                exit_price = tp_price
                win = True
            elif sig_prev == 2:
                exit_price = o
                win = (exit_price <= entry_price)

        if exit_price is not None:
            # Calculate P&L on sized position
            if pos == +1:
                exit_exec = exit_price * (1.0 - slippage_pct)
                pnl = (exit_exec / entry_price - 1.0) * position_size
            else:
                exit_exec = exit_price * (1.0 + slippage_pct)
                pnl = (entry_price / exit_exec - 1.0) * position_size
            # deduct exit fee on leveraged position
            cash += leverage * pnl - fee_pct * position_size * leverage
            pos = 0
            entry_price = tp_price = sl_price = None
            wins += int(bool(win))
            losses += int(not bool(win))
            equity_curve[i] = cash
            cdn = max(cooldown, 0)
            continue

        # Still in trade → mark-to-market equity
        if pos == +1:
            mtm = (c / entry_price) - 1.0
        else:  # short
            mtm = (entry_price / c) - 1.0
        equity_curve[i] = cash * (1.0 + mtm)

    # Close any open position at last close
    if pos != 0 and entry_price is not None:
        c = float(closes[-1])
        # Close final position similarly with leverage
        if pos == +1:
            exit_exec = c * (1.0 - slippage_pct)
            pnl = (exit_exec / entry_price - 1.0) * position_size
        else:
            exit_exec = c * (1.0 + slippage_pct)
            pnl = (entry_price / exit_exec - 1.0) * position_size
        cash += leverage * pnl - fee_pct * position_size * leverage

    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / peaks
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
                             "equity": equity_curve, "class": classes})
    return report, df_curve


def simulate_trades_with_tp_sl_more_aggressive(opens, highs, lows, closes, classes, *, start_capital,
                               fee_pct=0.0008, tp_pct=0.005, sl_pct=0.005,
                               atr: Optional[np.ndarray] = None,
                               atr_tp_mult: Optional[float] = None,
                               atr_sl_mult: Optional[float] = None,
                               cooldown: int = 0,
                               slippage_pct: float = 0.0,
                               dynamic_sizing: bool = False,
                               max_risk_per_trade: float = 0.02,
                               leverage: float = 1.0,
                               # Aggressive trailing stop and breakeven logic
                               trail_stop_long: float = 0.0002, # 0.02% trailing stop for long
                               trail_stop_short: float = 0.0002, # 0.02% trailing stop for short
                               breakeven_trigger_long: float = 1.0005,
                               breakeven_trigger_short: float = 0.9995) -> Tuple[Dict, pd.DataFrame]:
    
    """Simulate trades for a 3-class model: 0=short, 1=hold, 2=long.

    - Enters at next bar's open based on prior bar's signal.
    - One position at a time. Tracks TP/SL intrabar; opposing signals exit at open.
    - Equity is fully deployed on each trade; fees charged on entry and exit.
    """
    n = len(closes)
    cash = float(start_capital)
    equity_curve = np.empty(n, dtype=float)
    equity_curve[0] = cash

    pos = 0  # 0=flat, +1=long, -1=short
    entry_price = None
    tp_price = None
    sl_price = None
    cdn = 0  # cooldown counter

    trades = wins = losses = 0

    for i in range(1, n):
        o, hi, lo, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        sig_prev = int(classes[i - 1])

        if pos == 0:
            if cdn > 0:
                cdn -= 1
            elif sig_prev == 2:
                entry_price = o * (1.0 + slippage_pct)
                if atr is not None and atr_tp_mult is not None and atr_sl_mult is not None:
                    a = float(max(1e-12, atr[i-1]))
                    tp_price = entry_price + atr_tp_mult * a
                    sl_price = entry_price - atr_sl_mult * a
                else:
                    tp_price = entry_price * (1.0 + tp_pct)
                    sl_price = entry_price * (1.0 - sl_pct)
                if dynamic_sizing:
                    dist = entry_price - sl_price
                    risk_amount = cash * max_risk_per_trade
                    position_size = risk_amount / max(dist, 1e-9)
                    fee_amount = fee_pct * position_size * leverage
                    cash -= fee_amount
                else:
                    position_size = cash * leverage
                    cash -= fee_pct * position_size
                pos = +1
                trades += 1
                trail_price = entry_price
            elif cdn == 0 and sig_prev == 0:
                entry_price = o * (1.0 - slippage_pct)
                if atr is not None and atr_tp_mult is not None and atr_sl_mult is not None:
                    a = float(max(1e-12, atr[i-1]))
                    tp_price = entry_price - atr_tp_mult * a
                    sl_price = entry_price + atr_sl_mult * a
                else:
                    tp_price = entry_price * (1.0 - tp_pct)
                    sl_price = entry_price * (1.0 + sl_pct)
                if dynamic_sizing:
                    dist = sl_price - entry_price
                    risk_amount = cash * max_risk_per_trade
                    position_size = risk_amount / max(dist, 1e-9)
                    fee_amount = fee_pct * position_size * leverage
                    cash -= fee_amount
                else:
                    position_size = cash * leverage
                    cash -= fee_pct * position_size
                pos = -1
                trades += 1
                trail_price = entry_price
            equity_curve[i] = cash
            continue

        exit_price = None
        win = None

        if pos == +1:
            # Aggressive trailing stop for long
            if hi > trail_price:
                trail_price = hi
            # Exit if price drops more than 0.02% from high since entry
            if c < trail_price * (1 - trail_stop_long):
                exit_price = c
                win = (exit_price > entry_price)
            # Breakeven: exit if price falls back to entry after being up at least 0.05%
            elif trail_price > entry_price * breakeven_trigger_long and c <= entry_price:
                exit_price = c
                win = (exit_price > entry_price)
            elif lo <= sl_price <= hi:
                exit_price = sl_price
                win = False
            elif hi >= tp_price:
                exit_price = tp_price
                win = True
            elif sig_prev == 0:
                exit_price = o
                win = (exit_price >= entry_price)
        else:
            # Aggressive trailing stop for short
            if lo < trail_price:
                trail_price = lo
            # Exit if price rises more than 0.02% from low since entry
            if c > trail_price * (1 + trail_stop_short):
                exit_price = c
                win = (exit_price < entry_price)
            # Breakeven: exit if price rises back to entry after being down at least 0.05%
            elif trail_price < entry_price * breakeven_trigger_short and c >= entry_price:
                exit_price = c
                win = (exit_price < entry_price)
            elif hi >= sl_price:
                exit_price = sl_price
                win = False
            elif lo <= tp_price:
                exit_price = tp_price
                win = True
            elif sig_prev == 2:
                exit_price = o
                win = (exit_price <= entry_price)

        if exit_price is not None:
            if pos == +1:
                exit_exec = exit_price * (1.0 - slippage_pct)
                pnl = (exit_exec / entry_price - 1.0) * position_size
            else:
                exit_exec = exit_price * (1.0 + slippage_pct)
                pnl = (entry_price / exit_exec - 1.0) * position_size
            cash += leverage * pnl - fee_pct * position_size * leverage
            pos = 0
            entry_price = tp_price = sl_price = None
            wins += int(bool(win))
            losses += int(not bool(win))
            equity_curve[i] = cash
            cdn = max(cooldown, 0)
            continue

        # Still in trade → mark-to-market equity
        if pos == +1:
            mtm = (c / entry_price) - 1.0
        else:
            mtm = (entry_price / c) - 1.0
        equity_curve[i] = cash * (1.0 + mtm)

    # Close any open position at last close
    if pos != 0 and entry_price is not None:
        c = float(closes[-1])
        # Close final position similarly with leverage
        if pos == +1:
            exit_exec = c * (1.0 - slippage_pct)
            pnl = (exit_exec / entry_price - 1.0) * position_size
        else:
            exit_exec = c * (1.0 + slippage_pct)
            pnl = (entry_price / exit_exec - 1.0) * position_size
        cash += leverage * pnl - fee_pct * position_size * leverage

    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / peaks
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
                             "equity": equity_curve, "class": classes})
    return report, df_curve


def predict_regression(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device,
                       progress: bool = True) -> np.ndarray:
    """Return scalar predictions [N] for regression models (num_outputs=1)."""
    N = len(X)
    preds = np.zeros(N, dtype=np.float32)
    i = 0
    bs = max(1, int(batch_size))
    next_mark = 0.1
    while i < N:
        try:
            xb_np = X[i:i+bs]
            if len(xb_np) == 0:
                break
            xb = torch.from_numpy(xb_np).to(device, non_blocking=(device.type == "cuda"))
            with torch.no_grad():
                out = model(xb)
                if out.ndim == 2 and out.shape[1] == 1:
                    p = out.squeeze(-1).detach().cpu().numpy()
                else:
                    p = out.squeeze(-1).detach().cpu().numpy()
            preds[i:i+len(p)] = p
            i += len(p)
            if progress and N:
                frac = i / N
                if frac >= next_mark or i == N:
                    print(f"[predict] {i:,}/{N:,} ({frac:0.0%})")
                    next_mark += 0.1
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda" and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM — reducing batch size to {bs}")
                continue
            if "out of memory" in msg and device.type == "cuda":
                print("[WARN] CUDA OOM at batch size 1 — falling back to CPU")
                device = torch.device("cpu")
                model = model.to(device)
                torch.cuda.empty_cache()
                continue
            raise
    return preds


def build_prompt_from_window(feature_names: List[str], window: np.ndarray, prompt_template: Optional[str] = None) -> str:
    """Build a compact prompt from a single window of features.

    - feature_names: ordered list of feature column names used in `window`.
    - window: np.ndarray of shape [window_len, n_features] — we will summarize by using
      only the last row (most recent values) to predict the next close. If you want the
      LLM to see more history, you can modify this to include averages, mins, recent trend, etc.
    """
    last = window[-1]
    pairs = [f"{name}: {float(val):.8f}" for name, val in zip(feature_names, last)]
    features_text = "\\n".join(pairs)
    if prompt_template is None:
        prompt = (
            "You are given the latest feature values for an ETH 1-minute bar.\n"
            "Using these features, predict the next bar's closing price (a single numeric value).\n"
            "Output ONLY the predicted close as a plain number (no extra words).\n\n"
            "Features:\n"
            f"{features_text}\n\n"
            "Answer:"
        )
    else:
        prompt = prompt_template.replace("{features}", features_text)
    return prompt


NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")


def parse_number_from_text(text: str) -> Optional[float]:
    """Attempt to extract the first sensible floating number from model output."""
    if not isinstance(text, str):
        return None
    m = NUMBER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def llm_predict_batch_textgen(llm_pipeline, prompts: List[str], *, max_new_tokens: int = 16, device: Optional[int] = None) -> List[Optional[float]]:
    """Use a HF text-generation pipeline to predict numeric closes from prompts.

    Returns a list of floats or None when parsing fails.
    """
    outputs: List[Optional[float]] = [None] * len(prompts)
    if llm_pipeline is None:
        raise RuntimeError("LLM pipeline is not initialized")

    # Process in batches based on pipeline semantics
    for i in range(0, len(prompts), 1):
        prompt = prompts[i]
        try:
            raw = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            # pipeline returns list of dicts; get generated_text
            if isinstance(raw, list) and raw:
                gen = raw[0].get("generated_text") or raw[0].get("text") or ""
            elif isinstance(raw, dict):
                gen = raw.get("generated_text") or raw.get("text") or ""
            else:
                gen = str(raw)
            # remove the prompt prefix if present
            if gen.startswith(prompt):
                gen = gen[len(prompt):]
            val = parse_number_from_text(gen)
            outputs[i] = val
        except Exception as e:
            print(f"[llm] generation failed for index {i}: {e}")
            outputs[i] = None
    return outputs


# -----------------------
# CLI additions
# -----------------------
from huggingface_hub import login

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtester with TP/SL and optional LLM predictions.")
    # (keep relevant args from original backtest.py)
    p.add_argument("--mode", choices=["simple", "portfolio"], default="portfolio")
    p.add_argument("--data-dir", type=str, default="eth_1m_data", help="Dir or a single CSV")
    p.add_argument("--model-dir", type=str, default="/opt/ml/processing/input/model/", help="Root where model_meta.json & model.pt live")
    p.add_argument("--capital", type=float, default=10_000.0, help="Starting capital for portfolio mode")
    p.add_argument("--batch-size", type=int, default=512, help="Prediction batch size (auto-shrinks if OOM)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device (default auto)")
    p.add_argument("--last-csvs", type=int, default=1, help="If data-dir is a directory, only use the most recent N CSV files")

    # gating and portfolio args
    p.add_argument("--thr-long", type=float, default=0.55)
    p.add_argument("--thr-short", type=float, default=0.55)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--consensus", type=int, default=2)
    p.add_argument("--tp-pct", type=float, default=0.002)
    p.add_argument("--sl-pct", type=float, default=0.005)
    p.add_argument("--fee-pct", type=float, default=0.0008, help="Per-side fee fraction (0.0 for no fees)")
    p.add_argument("--atr-tp-mult", type=float, default=5.0, help="ATR multiplier for take-profit (larger wins)")
    p.add_argument("--atr-sl-mult", type=float, default=1.0, help="ATR multiplier for stop-loss (tight risk)")
    p.add_argument("--cooldown", type=int, default=0, help="Bars to wait after an exit before re-entering (faster)")
    p.add_argument("--use-atr-stops", action="store_true", default=False, help="Use ATR multipliers for TP/SL (disabled by default)")
    p.add_argument("--slippage-pct", type=float, default=0.0002, help="Per-side slippage fraction for realism (0.02%)")
    p.add_argument("--dynamic-sizing", action="store_true", default=True, help="Enable position sizing by risk percent")
    p.add_argument("--max-risk-per-trade", type=float, default=0.01, help="Max percent of equity to risk per trade (e.g. 0.01)")
    p.add_argument("--leverage", type=float, default=1.0, help="Leverage factor (e.g. 2.0 for 2x)")
    p.add_argument("--currency", type=str, default="$", help="Currency symbol for reporting (e.g. $ or €)")
    p.add_argument("--switch-less-aggressive", action="store_true", default=True, help="Switch to less aggressive TP/SL logic")

    # LLM-specific flags
    p.add_argument("--use-llm", action="store_true", help="Use a Hugging Face LLM pipeline to predict next close instead of the saved torch model")
    p.add_argument("--llm-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Hugging Face model id for text generation (e.g. grok or llama families)")
    p.add_argument("--llm-device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for the LLM pipeline")
    p.add_argument("--llm-batch-size", type=int, default=32, help="Number of prompts to send to the LLM in a loop (pipeline may not support large batch sizes for older transformers)")
    p.add_argument("--llm-max-new-tokens", type=int, default=8, help="Tokens to generate from the LLM (keep small — we only need a short numeric answer)")
    p.add_argument("--llm-prompt-template", type=str, default=None, help="Optional prompt template. Use {features} to inject features.")

    return p


# -----------------------
# Main logic
# -----------------------

def main():
    
    # add your huggingface token here. 
    login(token="your_huggingface_token_here")
    args = build_argparser().parse_args()
    model, scaler, meta = None, None, {}
   
    feature_cols = FEATURES  # preserve order from meta
    #fixed window size for LLM
    window_size = 60
    fee_pct = float(meta.get("tx_cost", 0.0008)) if args.fee_pct is None else float(args.fee_pct)
    tp_pct = 0.0025 if args.tp_pct is None else float(args.tp_pct)
    sl_pct = 0.005 if args.sl_pct is None else float(args.sl_pct)
   
    # Load data and compute features (same as original)
    args.data_dir = "/opt/ml/processing/input/data"
    print(f"[load] Reading data from {args.data_dir} ...")
    data_path = Path(args.data_dir)
    if data_path.is_dir() and args.last_csvs:
        files = sorted(str(p) for p in data_path.glob("*.csv"))
        if not files:
            raise SystemExit(f"No CSV files found in directory: {args.data_dir}")
        sel = files[-int(args.last_csvs):]
        print(f"[load] Limiting to last {len(sel)} CSVs:")
        for s in sel:
            print(f"        {Path(s).name}")
        parts = [normalize_headers(pd.read_csv(s)) for s in sel]
        df = pd.concat(parts, ignore_index=True)
    else:
        df = read_csv_concat_sorted(args.data_dir)
    print(f"[load] Rows loaded: {len(df):,}")
    print("meta price col: ", meta.get("price_col", "close"))
    price_col = resolve_price_col(df.columns.tolist(), meta.get("price_col", "close"))
    if price_col is None:
        raise SystemExit(f"Could not locate a price column. Available: {list(df.columns)}")

    print("[features] Engineering features ...")
    df = compute_features(df)
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # Build feature columns used by meta
    drop_cols = {"timestamp", "time"}
    feat_cols = [c for c in feature_cols if c in df.columns and c not in drop_cols]
    missing = [c for c in feature_cols if c not in feat_cols]
    if missing:
        print(f"[WARN] Missing features in data (will drop): {missing}")
    if not feat_cols:
        raise SystemExit("No valid feature columns found in data for inference.")
    print(f"[INFO] Using {len(feat_cols)} features from meta: {feat_cols}")
 
    # # Scale in-place if scaler present
    # if scaler is not None: 
   #     print("[scale] Applying RobustScaler in-place (vectorized) ...")
    #     df[feat_cols] = df[feat_cols].astype("float32")
    #     centers = scaler.center_.astype("float32")
    #     scales = scaler.scale_.astype("float32")
    #     eps = 1e-8
    #     scales = np.where(scales == 0, eps, scales)
    #     X_flat = df[feat_cols].to_numpy(dtype=np.float32)
    #     X_flat -= centers
    #     X_flat /= scales
    # else:
    #     X_flat = df[feat_cols].to_numpy(dtype=np.float32)
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)
    # Windows
    print("[windows] Building windows ...")
    X = build_windows(X_flat, window_size)
    if len(X) == 0:
        raise SystemExit("Not enough rows to build any sequences. Increase data or reduce window size.")
    print(f"[windows] Built {len(X):,} sequences of length {window_size} with {X.shape[-1]} features")

    # Align OHLC arrays to window ends
    opens = df["open"].to_numpy(dtype=float)[window_size - 1:]
    highs = df["high"].to_numpy(dtype=float)[window_size - 1:]
    lows = df["low"].to_numpy(dtype=float)[window_size - 1:]
    closes = df[price_col].to_numpy(dtype=float)[window_size - 1:]

    # If user requested LLM predictions, spin up the pipeline and produce preds
    use_llm = True
    llm_preds = None
    if use_llm:
        if pipeline is None:
            raise SystemExit("transformers not available; please `pip install transformers` to use --use-llm")
        # Decide device
        if args.llm_device == "cpu":
            hf_device = -1
        elif args.llm_device == "cuda":
            hf_device = 0
        else:
            # auto: use cuda if available
            hf_device = 0 if torch.cuda.is_available() else -1

        print(f"[llm] Loading text-generation pipeline for {args.llm_model} (device={hf_device}) ...")
        # Some models require tokenizer + model config; pipeline will attempt to load them
        try:
            llm_pipe = pipeline("text-generation", model=args.llm_model, device=hf_device)
        except Exception as e:
            print(f"[llm] pipeline() failed to load model '{args.llm_model}': {e}")
            print("[llm] Trying to load model with AutoTokenizer/AutoModelForCausalLM then pipeline...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.llm_model,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,  # or fp16 if supported
                    trust_remote_code=True,
                )
                llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            except Exception as e2:
                raise SystemExit(f"Failed to create LLM pipeline for {args.llm_model}: {e2}")

        # Build prompts (we only use the last row of each window by default)
        prompts = [build_prompt_from_window(feat_cols, w, prompt_template=args.llm_prompt_template) for w in X]

        print(f"[llm] Generating predictions for {len(prompts):,} prompts (this may be slow)." )
        preds_list: List[Optional[float]] = []
        batch_size = max(1, int(args.llm_batch_size))
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            out_vals = llm_predict_batch_textgen(llm_pipe, batch, max_new_tokens=int(args.llm_max_new_tokens))
            preds_list.extend(out_vals)
            print(f"[llm] Processed {min(i+batch_size, len(prompts))}/{len(prompts)}")

        # Convert list into numeric array; if None -> fill with NaN
        llm_preds = np.array([float(x) if x is not None else np.nan for x in preds_list], dtype=np.float64)
        print(f"[llm] Done. Successful predictions: {int(np.isfinite(llm_preds).sum())}/{len(llm_preds)}")
    
    # If using LLM predictions, create signals from predicted close
    if use_llm and llm_preds is not None:
        # Convert predicted close to return relative to current close
        ret = (llm_preds - closes) / np.maximum(1e-12, closes)
        up_thr = float(0.002)  # default 0.2% threshold; you can expose as CLI if desired
        down_thr = float(0.002)
        signals = np.ones(len(ret), dtype=np.int64)
        signals[np.isnan(ret)] = 1
        signals[ret >= up_thr] = 2
        signals[ret <= -down_thr] = 0
    else:
        # Fall back to original torch model inference path (classification/regression)
        print("[predict] Running torch model inference (original behavior) ...")
        # We expect the original predict_proba / predict_regression implementations to exist
        task = str(meta.get("task", "")).lower() if isinstance(meta, dict) else ""
        is_regression = bool(task == "regression" or int(meta.get("num_classes", 3)) == 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if is_regression:
            ret = predict_regression(model, X, int(args.batch_size), device, progress=True)
            up_thr = float(0.002)
            down_thr = float(0.002)
            signals = np.ones(len(ret), dtype=np.int64)
            signals[ret >= up_thr] = 2
            signals[ret <= -down_thr] = 0
        # else:
        #     probs = predict_proba(model, X, int(args.batch_size), device, progress=True, temperature=float(meta.get("temperature", 1.0)))
        #     signals = apply_gating(probs, thr_long=float(args.thr_long), thr_short=float(args.thr_short), margin=float(args.margin), consensus=int(args.consensus))
    
    # Portfolio simulation with TP/SL
    print("[simulate] Running portfolio simulation ...")
    if args.switch_less_aggressive:
        report, curve = simulate_trades_with_tp_sl(
            opens, highs, lows, closes, signals,
            start_capital=float(args.capital),
            fee_pct=fee_pct,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            cooldown=int(args.cooldown),
        slippage_pct=float(args.slippage_pct),
        dynamic_sizing=bool(args.dynamic_sizing),
        max_risk_per_trade=float(args.max_risk_per_trade),
        leverage=float(args.leverage),
        )
    else:
        report, curve = simulate_trades_with_tp_sl_more_aggressive(
            opens, highs, lows, closes, signals,
            start_capital=float(args.capital),
            fee_pct=fee_pct,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            cooldown=int(args.cooldown),
        slippage_pct=float(args.slippage_pct),
        dynamic_sizing=bool(args.dynamic_sizing),
        max_risk_per_trade=float(args.max_risk_per_trade),
        leverage=float(args.leverage),
        )

    print("[simulate] Done.")
    print_portfolio_report(report, currency="$")


if __name__ == "__main__":
    main()