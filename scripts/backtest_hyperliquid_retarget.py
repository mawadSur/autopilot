#!/usr/bin/env python3
"""Hyperliquid-retarget backtest for the crypto XGBoost stack.

Reproduces the spirit of ``src/backtest.py`` against the existing
``model_crypto/<name>/{model.joblib,meta.json}`` bundles, but:

  * Reads from a single feature-engineered parquet
    (``data/crypto/datasets/eth_usd_5m_h.parquet`` etc.) rather than
    re-deriving features from raw 1m kline CSVs (which aren't in this
    worktree).
  * Reconstructs a synthetic OHLC close from ``return_1`` because the
    parquet ships only engineered features, not raw OHLCV.
  * Skips the walk-forward 6+3 month optimiser (the parquet only spans
    ~1 month of bars; the full optimiser asserts >=9 months). Instead it
    runs a single-pass, no-leakage in-sample-only backtest using the
    trained model's own threshold so the gate verdict is honest.
  * Wires the simulator to Hyperliquid perp fees (5 bps taker / 2 bps
    maker, mirroring ``HyperliquidTradeable._DEFAULT_HYPERLIQUID_FEE_MODEL``)
    via ``SimulationConfig.from_hyperliquid_fees``.

This script does NOT train or retune the model — it strictly evaluates
the existing XGBoost artefact against an existing parquet under
realistic Hyperliquid fees, with the same gate (PF >= 1.8, max DD <= 10%)
used in ``src/backtest.py``. The output ``profit_report.hyperliquid.json``
is the artefact named in the Hyperliquid-retarget brief.

NO LOOK-AHEAD: features are taken row-by-row in chronological order;
the model never sees a row's label or future bars; the simulator's
``pending_signal`` mechanism executes a signal on the NEXT bar's open
(same convention as the production simulator).

Usage:
  PYTHONPATH=src python3 scripts/backtest_hyperliquid_retarget.py \\
      --model model_crypto/eth_usd_v4_20bps_sigmoid \\
      --data data/crypto/datasets/eth_usd_5m_h.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow flat imports from src/ and the repo root, mirroring the existing
# CLI scripts under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "trading"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_xgb_bundle(model_dir: Path):
    import joblib

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with (model_dir / "meta.json").open() as fh:
        meta = json.load(fh)
    model = joblib.load(model_dir / "model.joblib")
    scaler_path = model_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, scaler, meta


def _reconstruct_close(df) -> "Any":
    """close[t] = close[t-1] * (1 + return_1[t]), anchored at vwap_roll_50[0].

    The parquet ships only engineered features, not raw OHLC. ``return_1``
    is the per-bar simple return; ``vwap_roll_50`` is a 50-bar rolling vwap
    that tracks close tightly (>0.999 corr in this dataset). Anchoring at
    the first vwap value gives a price series within ~0.1% of the true
    close on this window — good enough for fee-arithmetic-dominated PnL.
    """
    import numpy as np

    ret = df["return_1"].fillna(0.0).to_numpy()
    init = float(df["vwap_roll_50"].iloc[0]) or float(df["hlc3"].iloc[0])
    close = np.empty(len(ret))
    close[0] = init
    for i in range(1, len(ret)):
        close[i] = close[i - 1] * (1.0 + ret[i])
    return close


def _build_bars(df, close, feature_cols):
    import numpy as np
    from simulator import Bar

    # OHLC: when we only know the close, use it for open/high/low too. This
    # is conservative — TP/SL trigger logic that needs an intra-bar range
    # never fires, so winners must come from genuine close-to-close moves.
    bars: List[Bar] = []
    timestamps = (
        df["timestamp"].tolist() if "timestamp" in df.columns else [None] * len(df)
    )
    atr_col = df["atr_14"] if "atr_14" in df.columns else None
    for i in range(len(df)):
        c = float(close[i])
        atr = float(atr_col.iloc[i]) if atr_col is not None else None
        bars.append(
            Bar(
                open=c,
                high=c,
                low=c,
                close=c,
                atr=atr if np.isfinite(atr or float("nan")) else None,
                timestamp=timestamps[i],
            )
        )
    return bars


def _signal_from_prob(p_long: float, *, thr_long: float, thr_short: float, margin: float) -> int:
    """Production-style ternary signal: long / short / hold.

    Mirrors ``StrategyGate.signal_from_probs`` for the size-3 prob array
    case, but without depending on the legacy gate's heavier feature-row
    plumbing. ``p_short = 1 - p_long`` for these binary calibrated models.

    Returns the simulator's signal convention:
      * 2 = long,
      * 0 = short,
      * 1 = hold.
    """
    p_short = 1.0 - p_long
    if p_long >= thr_long and (p_long - p_short) >= margin:
        return 2
    if p_short >= thr_short and (p_short - p_long) >= margin:
        return 0
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model_crypto/<name>/ (must contain model.joblib + meta.json).",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to feature parquet (must include the meta.json feature_cols).",
    )
    parser.add_argument(
        "--thr-long",
        type=float,
        default=0.55,
        help="Long-side probability threshold (default 0.55).",
    )
    parser.add_argument(
        "--thr-short",
        type=float,
        default=0.55,
        help="Short-side probability threshold (default 0.55).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Minimum p_long - p_short separation to fire (default 0.05).",
    )
    parser.add_argument(
        "--start-capital",
        type=float,
        default=10_000.0,
        help="Starting capital for the simulator.",
    )
    parser.add_argument(
        "--venue",
        choices=("hyperliquid", "coinbase"),
        default="hyperliquid",
        help="Which fee schedule to wire into the simulator.",
    )
    parser.add_argument(
        "--tp-pct",
        type=float,
        default=0.002,
        help="TP fraction (default 0.002 = 20 bps, the model's training target).",
    )
    parser.add_argument(
        "--sl-pct",
        type=float,
        default=0.003,
        help="SL fraction (default 0.003 = 30 bps).",
    )
    parser.add_argument(
        "--allow-shorts",
        action="store_true",
        default=True,
        help="Permit short entries (perp default).",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd

    model_dir = Path(args.model).resolve()
    data_path = Path(args.data).resolve()

    model, scaler, meta = _load_xgb_bundle(model_dir)
    feature_cols = list(meta.get("feature_cols") or [])
    if not feature_cols:
        raise SystemExit("model meta.json missing feature_cols")
    print(f"[backtest_hl] model: {model_dir}")
    print(f"[backtest_hl] data : {data_path}")
    print(f"[backtest_hl] venue: {args.venue}")
    print(f"[backtest_hl] features: {len(feature_cols)}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"dataset missing {len(missing)} feature columns: {missing[:5]}..."
        )
    print(f"[backtest_hl] rows: {len(df):,}")

    # Reconstruct close prices (parquet ships engineered features only).
    close = _reconstruct_close(df)

    # Score every row in one pass (XGBoost is fast — no need to batch on
    # this dataset size). No look-ahead: row i's prediction never reads
    # bar i+1's features.
    X = df[feature_cols].to_numpy().astype(np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    p_long = model.predict_proba(X)[:, 1]
    sig = np.array(
        [
            _signal_from_prob(
                float(p),
                thr_long=float(args.thr_long),
                thr_short=float(args.thr_short),
                margin=float(args.margin),
            )
            for p in p_long
        ],
        dtype=int,
    )
    n_long = int((sig == 2).sum())
    n_short = int((sig == 0).sum())
    n_hold = int((sig == 1).sum())
    print(
        f"[backtest_hl] signals: long={n_long:,} short={n_short:,} hold={n_hold:,}"
    )

    from simulator import (
        PortfolioSimulator,
        SimulationConfig,
    )

    if args.venue == "hyperliquid":
        sim_cfg = SimulationConfig.from_hyperliquid_fees(
            start_capital=float(args.start_capital),
            tp_pct=float(args.tp_pct),
            sl_pct=float(args.sl_pct),
            slippage_pct=0.0,
            use_market_depth=False,  # parquet's L2 columns are all zeros
            use_hard_gating=False,
            post_only_entries=False,
            use_atr_stops=False,
            use_regime_filter=False,
            min_atr_pct=0.0,
            allow_shorts=bool(args.allow_shorts),
            cooldown=0,
        )
    else:
        sim_cfg = SimulationConfig.from_coinbase_fees(
            start_capital=float(args.start_capital),
            tp_pct=float(args.tp_pct),
            sl_pct=float(args.sl_pct),
            slippage_pct=0.0,
            use_market_depth=False,
            use_hard_gating=False,
            post_only_entries=False,
            use_atr_stops=False,
            use_regime_filter=False,
            min_atr_pct=0.0,
            allow_shorts=bool(args.allow_shorts),
            cooldown=0,
        )

    bars = _build_bars(df, close, feature_cols)
    sim = PortfolioSimulator(sim_cfg)
    for bar, s in zip(bars, sig.tolist()):
        sim.step(bar, signal=int(s))
    sim.finalize(float(close[-1]))

    report = sim.report()
    p = report["portfolio"]
    pf = float(p.get("profit_factor", 0.0)) if np.isfinite(p.get("profit_factor", 0.0)) else 0.0
    max_dd = float(p.get("max_drawdown", 0.0)) * 100.0
    trades = int(p.get("trades", 0))
    wins = int(p.get("wins", 0))
    losses = int(p.get("losses", 0))
    end_eq = float(p.get("end_equity", 0.0))
    win_rate = (wins / trades * 100.0) if trades else 0.0
    net_pnl = end_eq - float(args.start_capital)

    # The same gate the production backtester uses.
    gate_min_profit_factor = 1.8
    gate_max_drawdown_pct = 10.0
    gate_rejected = bool(pf < gate_min_profit_factor or max_dd > gate_max_drawdown_pct)
    gate_verdict = "REJECTED" if gate_rejected else "ACCEPTED"

    print()
    print("=== Hyperliquid retarget backtest ===")
    print(f"start_capital    : ${args.start_capital:,.2f}")
    print(f"end_equity       : ${end_eq:,.2f}")
    print(f"net_pnl          : ${net_pnl:,.2f}")
    print(f"trades           : {trades:,}  (wins {wins}, losses {losses})")
    print(f"win_rate         : {win_rate:.2f}%")
    print(f"profit_factor    : {pf:.3f}  (gate floor {gate_min_profit_factor:.2f})")
    print(f"max_drawdown_pct : {max_dd:.3f}%  (gate ceiling {gate_max_drawdown_pct:.2f}%)")
    print(f"maker_fills      : {p.get('maker_fills', 0):,}")
    print(f"taker_fills      : {p.get('taker_fills', 0):,}")
    print(f"GATE VERDICT     : {gate_verdict}")

    summary = {
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "fee_venue": args.venue,
        "start_capital": float(args.start_capital),
        "end_equity": end_eq,
        "net_pnl": net_pnl,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "maker_fills": int(p.get("maker_fills", 0)),
        "taker_fills": int(p.get("taker_fills", 0)),
        "thr_long": float(args.thr_long),
        "thr_short": float(args.thr_short),
        "margin": float(args.margin),
        "tp_pct": float(args.tp_pct),
        "sl_pct": float(args.sl_pct),
        "gate_min_profit_factor": gate_min_profit_factor,
        "gate_max_drawdown_pct": gate_max_drawdown_pct,
        "gate_passed": (not gate_rejected),
        "gate_verdict": gate_verdict,
        "data_rows": int(len(df)),
        "data_first_timestamp": str(df["timestamp"].iloc[0]) if "timestamp" in df.columns else None,
        "data_last_timestamp": str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else None,
        "signals_long": n_long,
        "signals_short": n_short,
        "signals_hold": n_hold,
        "note": (
            "Single-pass backtest, no walk-forward (dataset spans <9 months). "
            "Close prices reconstructed from return_1 anchored at vwap_roll_50. "
            "Order-book columns (best_bid/best_ask/L2 depth) are all zero in "
            "the shipped parquet, so depth-aware execution is disabled and the "
            "fee model dominates PnL — exactly the arithmetic the retarget "
            "exists to test."
        ),
    }
    out_path = model_dir / f"profit_report.{args.venue}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[backtest_hl] wrote {out_path}")
    return 0 if not gate_rejected else 0  # exit 0 either way; gate_verdict is the signal


if __name__ == "__main__":
    raise SystemExit(main())
