#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_trader.py — live loop with paper or real orders.
Mirrors backtester semantics: gating (thr/margin/consensus), regime filter,
ATR TP/SL, cooldown, and optional shorts. Volatility-scaled position sizing.

Usage
-----
Paper by default:
  python live_trader.py --symbol ETH/USDT --interval 60

Real (spot) if allowed + env set:
  BINANCE_API_KEY=... BINANCE_API_SECRET=... python live_trader.py --real true

Reads:
- model/model_meta.json   (feature order, window size, num_classes)
- model/best_live_config.json (auto-selected thresholds from optimize_params.py)

Requires:
- utils.py (compute_features, build_windows, load_model_bundle, fmt_money, etc.)
- ccxt (for real mode; optional for paper)
"""

from __future__ import annotations

import collections
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd
from logging_utils import setup_logging, logger

try:
    import ccxt  # optional for real trading
except Exception:  # pragma: no cover - ccxt optional
    ccxt = None

from utils import compute_features, load_model_bundle, fmt_money, FEATURE_COLUMNS
from config import cfg
try:
    from simulator import SimulationConfig, PortfolioSimulator, Bar
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from simulator import SimulationConfig, PortfolioSimulator, Bar

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / "model"
DATA_DIR = PROJECT / "eth_1m_data"


# ----------------------------
# Helpers
# ----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.rolling(n, min_periods=1).mean()


def load_best_live_config(model_dir: Path) -> Dict[str, float]:
    path = model_dir / "best_live_config.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text())
            return obj.get("live_config", {})
        except Exception:
            return {}
    return {}



# ----------------------------
# Signal logic (gating + regime)
# ----------------------------


@dataclass
class GatingCfg:
    thr_long: float = 0.70
    thr_short: float = 0.70
    margin: float = 0.15       # gap between top-2 class probabilities
    consensus: int = 2
    cooldown: int = 2


@dataclass
class RiskCfg:
    atr_tp_mult: float = 1.8
    atr_sl_mult: float = 1.1
    risk_pct_per_trade: float = 0.004  # 0.4% equity risked
    max_leverage: float = 1.0
    allow_shorts: bool = False


class ConsensusBuffer:
    def __init__(self, n: int):
        self.n = max(1, int(n))
        self.buf: Deque[int] = collections.deque(maxlen=self.n)

    def push(self, direction: int) -> None:
        self.buf.append(direction)

    def ok(self, expect: int) -> bool:
        if not self.buf:
            return False
        if self.n == 1:
            return self.buf[-1] == expect
        if len(self.buf) < self.n:
            return False
        return all(v == expect for v in self.buf)


def regime_filter(df: pd.DataFrame) -> int:
    """Return +1 (up), -1 (down), or 0 (neutral) using EMA50/EMA200 cross."""
    if len(df) < 200:
        return 0
    ema50 = ema(df["close"], 50)
    ema200 = ema(df["close"], 200)
    fast = float(ema50.iloc[-1])
    slow = float(ema200.iloc[-1])
    if fast > slow * 1.0005:
        return +1
    if fast < slow * 0.9995:
        return -1
    return 0


def decide_direction(probs: np.ndarray, gating: GatingCfg) -> int:
    """Return -1 (short), 0 (flat), or +1 (long) from class probabilities."""
    probs = np.asarray(probs).flatten()
    if probs.size >= 3:
        p_short, p_hold, p_long = probs[:3]
        max_prob = max(p_long, p_short)
        top_two = np.sort(probs[:3])[-2:]
        second_max = float(top_two[-2]) if len(top_two) == 2 else 0.0
        gap = float(max_prob - second_max)
        if p_long >= p_short and max_prob >= gating.thr_long and gap >= gating.margin:
            return +1
        if p_short > p_long and max_prob >= gating.thr_short and gap >= gating.margin:
            return -1
        return 0
    if probs.size == 2:
        # Treat final element as long probability, first as short/hold fallback.
        p_short = float(probs[0])
        p_long = float(probs[1])
        if p_long >= gating.thr_long and (p_long - p_short) >= gating.margin:
            return +1
        if p_short >= gating.thr_short and (p_short - p_long) >= gating.margin:
            return -1
        return 0
    if probs.size == 1:
        p_long = float(probs[0])
        return +1 if p_long >= gating.thr_long else 0
    raise ValueError("Unsupported probability vector shape")


# ----------------------------
# Exchange (optional)
# ----------------------------


def make_exchange(testnet: bool) -> Optional["ccxt.binance"]:
    if ccxt is None:
        return None
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "spot"},
    })
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange


# ----------------------------
# Main loop
# ----------------------------


def run_loop() -> None:
    import torch

    setup_logging()
    args = cfg
    setattr(args, "_last_close", None)
    setattr(args, "interval", getattr(cfg, "interval_sec", 5.0))
    if cfg.capital is not None:
        setattr(args, "starting_cash", cfg.capital)

    model_dir = Path(cfg.model_dir).expanduser().resolve()
    model, scaler, meta = load_model_bundle(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    num_classes = int(meta.get("num_classes", 1))
    window = int(meta.get("window_size", 1))
    feature_cols = list(meta.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("model_meta.json missing feature_cols; cannot stream.")
    if feature_cols != FEATURE_COLUMNS:
        raise ValueError(f"Feature list mismatch: meta has {feature_cols}, expected {FEATURE_COLUMNS}")

    # Load optimizer-derived live config overrides
    best_cfg = load_best_live_config(model_dir)
    gating = GatingCfg(
        thr_long=float(best_cfg.get("thr_long", cfg.thr_long)),
        thr_short=float(best_cfg.get("thr_short", cfg.thr_short)),
        margin=float(best_cfg.get("margin", cfg.margin)),
        consensus=int(best_cfg.get("consensus", cfg.consensus)),
        cooldown=int(best_cfg.get("cooldown", cfg.cooldown)),
    )
    risk = RiskCfg(
        atr_tp_mult=float(best_cfg.get("atr_tp", args.atr_tp_mult)),
        atr_sl_mult=float(best_cfg.get("atr_sl", args.atr_sl_mult)),
        risk_pct_per_trade=cfg.risk_pct_per_trade,
        max_leverage=cfg.max_leverage,
        allow_shorts=cfg.allow_shorts,
    )

    consensus = ConsensusBuffer(gating.consensus)
    sim_cfg = SimulationConfig(
        start_capital=float(cfg.starting_cash),
        fee_pct=float(cfg.fee_pct),
        slippage_pct=float(cfg.slippage_pct),
        cooldown=int(gating.cooldown),
        use_atr_stops=True,
        atr_tp_mult=risk.atr_tp_mult,
        atr_sl_mult=risk.atr_sl_mult,
        dynamic_sizing=True,
        max_risk_per_trade=risk.risk_pct_per_trade,
        leverage=risk.max_leverage if hasattr(risk, "max_leverage") else 1.0,
        allow_shorts=risk.allow_shorts,
        use_regime_filter=True,
        record_trades=True,
    )
    sim = PortfolioSimulator(sim_cfg)
    prev_log_len = 0

    exchange = make_exchange(testnet=args.testnet)

    history = load_initial_data(args)
    if not history.empty:
        try:
            args._last_close = float(history["close"].iloc[-1])
        except Exception:
            args._last_close = None

    print("[INFO] Starting live loop — initial equity:", fmt_money(sim.last_equity))
    while True:
        kline = fetch_kline(args, exchange)
        if kline is not None:
            history = pd.concat([history, pd.DataFrame([kline])], ignore_index=True)
            history = history.tail(max(5000, window + 400)).reset_index(drop=True)

        if len(history) < window:
            time.sleep(max(0.5, args.interval))
            continue

        try:
            feats = compute_features(history.copy())
        except Exception as exc:
            print(f"[WARN] compute_features failed: {exc}")
            time.sleep(max(0.5, args.interval))
            continue

        missing = [c for c in feature_cols if c not in feats.columns]
        if missing:
            print(f"[WARN] Missing feature columns {missing}; waiting for more data")
            time.sleep(max(0.5, args.interval))
            continue

        regime = regime_filter(feats)
        feats["atr14"] = atr(feats, 14)
        last_atr = float(feats["atr14"].iloc[-1])
        last_close = float(feats["close"].iloc[-1])

        window_frame = feats[feature_cols].tail(window).astype(np.float32)
        if len(window_frame) < window:
            time.sleep(max(0.5, args.interval))
            continue

        arr = window_frame.to_numpy()
        if scaler is not None:
            if hasattr(scaler, "feature_names_in_"):
                assert list(scaler.feature_names_in_) == feature_cols, "Scaler feature order mismatch"
            arr = scaler.transform(arr)
        arr = arr.reshape(1, window, -1)

        tensor = torch_tensor(arr, device)
        with torch.no_grad():
            logits = model(tensor)
            if num_classes <= 1:
                probs_tensor = torch.sigmoid(logits)
            else:
                if logits.shape[-1] == 1:
                    probs_tensor = torch.sigmoid(logits)
                else:
                    probs_tensor = torch.softmax(logits, dim=-1)
        probs = probs_tensor.squeeze(0).detach().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([float(probs)])

        direction = decide_direction(probs, gating)
        consensus.push(direction)

        signal = 0
        if direction == +1 and consensus.ok(+1) and regime >= 0:
            signal = +1
        elif direction == -1 and consensus.ok(-1) and regime <= 0 and risk.allow_shorts:
            signal = -1

        # Build bar for simulator (executes previous pending signal)
        last_row = feats.iloc[-1]
        bar = Bar(
            open=float(last_row["open"]),
            high=float(last_row["high"]),
            low=float(last_row["low"]),
            close=last_close,
            atr=last_atr,
            regime=regime,
            timestamp=last_row.get("timestamp"),
        )
        sim.step(bar, signal=signal)

        # Emit new trades as they happen
        if sim.trade_log and len(sim.trade_log) > prev_log_len:
            new_logs = sim.trade_log[prev_log_len:]
            prev_log_len = len(sim.trade_log)
            for tr in new_logs:
                side = tr.get("side", "").upper()
                action = tr.get("action", "").upper()
                price = tr.get("price")
                pnl = tr.get("pnl")
                ret = tr.get("ret")
                eq = tr.get("equity", sim.last_equity)
                reason = tr.get("reason", "")
                if action == "ENTER":
                    print(f"[ENTER {side}] price={price:.2f} tp={tr.get('tp'):.2f} sl={tr.get('sl'):.2f} eq={fmt_money(eq)}")
                elif action == "EXIT":
                    print(f"[EXIT  {side}] price={price:.2f} pnl={pnl:.2f} ret={ret:.4f} reason={reason} eq={fmt_money(eq)}")

        time.sleep(max(0.5, args.interval))


# ----------------------------
# Minimal torch helpers
# ----------------------------

def torch_tensor(array: np.ndarray, device) -> "torch.Tensor":
    import torch

    tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    return tensor.to(device)


# ----------------------------
# Data fetch
# ----------------------------

def load_initial_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv_path:
        path = Path(args.csv_path).expanduser()
    else:
        path = DATA_DIR / "seed.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            cols = [c for c in ("date", "timestamp", "open", "high", "low", "close", "volume") if c in df.columns]
            df = df[cols]
            return df.tail(5000).reset_index(drop=True)
        except Exception as exc:
            print(f"[WARN] Failed to load seed CSV: {exc}")
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def fetch_kline(args: argparse.Namespace, exchange) -> Optional[Dict[str, float]]:
    """Return latest bar dictionary or None if data not yet ready."""
    if args.real and exchange is not None:
        try:
            ohlcv = exchange.fetch_ohlcv(args.symbol, timeframe="1m", limit=2)
            if not ohlcv:
                return None
            # Use last closed candle for features/signals
            ts, o, h, l, c, v = ohlcv[-2]
            args._last_close = float(c)
            return {
                "date": ts,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        except Exception as exc:
            print(f"[WARN] fetch_ohlcv failed: {exc}")
            return None

    import random

    base = args._last_close if args._last_close is not None else 3000.0
    drift = random.uniform(-0.0005, 0.0005)
    high_spread = random.uniform(0.0, 0.0008)
    low_spread = random.uniform(0.0, 0.0008)
    close = base * (1.0 + drift)
    high = max(close, base) * (1.0 + high_spread)
    low = min(close, base) * (1.0 - low_spread)
    args._last_close = float(close)
    return {
        "date": int(time.time() * 1000),
        "open": float(base),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": random.uniform(50, 150),
    }


# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    run_loop()
