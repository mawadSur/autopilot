#!/usr/bin/env python3
"""Post-fee branch-ablation audit for the crypto XGBoost models.

Goal (per request): *better net execution* — audit every decision branch for
whether it actually improves **post-fee expectancy**, and keep only the ones
that pay for themselves once honest Coinbase fees are charged.

Method
------
1. Load a ``model_crypto/<sym>_*`` bundle (``model.joblib`` + ``meta.json``).
2. Load its featurized dataset (``meta.dataset_path``) for the 135 features +
   ``timestamp`` + ``label``, and JOIN the *real* OHLC price path from the raw
   per-day files under ``data/crypto/<SYM>/1m/*.csv`` on timestamp. (The
   featurized dataset carries book columns by name but they are identically 0 —
   the L2 book was never backfilled — so execution is modelled from the OHLC
   path + ATR-slippage, and the microstructure branches are flagged heuristic.)
3. Take the **out-of-sample TEST slice only** (last 15%, matching the trainer's
   time-based split) — no leakage, the model never saw it and no threshold was
   fit on it.
4. Score once with the frozen model, then re-run the *execution simulator*
   (``trading.simulator``) under a BASELINE config and under each branch flipped
   one at a time. Report the delta in **post-fee** expectancy / profit-factor /
   net-return / drawdown / trade count.

Honest-costs note
-----------------
Post-fee per-trade returns are reconstructed from the equity sequence. Under the
simulator's all-in single-position sizing, cash-when-flat == equity, so
``equity_i / equity_{i-1} - 1`` is the exact net-of-both-fees return of trade
``i``. The simulator now also nets fees into ``trade_log['ret']``/``['pnl']``
directly (defect fixed), so this reconstruction and ``compute_profitability_metrics``
agree — the equity-derived version is kept as an independent cross-check.

Usage
-----
    ./.venv/bin/python src/crypto_training/branch_audit.py \
        --model-dir model_crypto/eth_usd_v4_20bps_sigmoid [--split test] [--out <path>]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent          # .../src
_ROOT = _SRC.parent                 # repo root
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib  # noqa: E402

from trading.simulator import Bar, PortfolioSimulator, SimulationConfig  # noqa: E402
from strategy_gate import StrategyGate  # noqa: E402
from regime_router import RegimeRouter  # noqa: E402

# Class-style signal constants (StrategyGate / ConsensusFilter convention).
CLS_SHORT, CLS_HOLD, CLS_LONG = 0, 1, 2
# Raw signal convention fed to the simulator (unambiguous: +1 long / 0 flat).
RAW_LONG, RAW_FLAT = 1, 0

VAL_FRAC = 0.15
TEST_FRAC = 0.15
# Neutral zone (in per-trade return terms) inside which a branch's effect on
# post-fee expectancy is treated as "no real difference".
EPS_EXPECTANCY = 2e-5  # 0.2 bps per trade


# ---------------------------------------------------------------------------
# Consensus filter (local copy of backtest.ConsensusFilter — class-style).
# ---------------------------------------------------------------------------
class ConsensusFilter:
    """Require N consecutive identical non-hold signals before emitting them."""

    def __init__(self, consensus: int) -> None:
        self.consensus = max(1, int(consensus))
        self.run_sig = CLS_HOLD
        self.run_len = 0

    def step(self, raw_sig: int) -> int:
        if self.consensus <= 1:
            return raw_sig
        if raw_sig != CLS_HOLD and raw_sig == self.run_sig:
            self.run_len += 1
        else:
            self.run_sig = raw_sig
            self.run_len = 1
        if raw_sig != CLS_HOLD and self.run_len >= self.consensus:
            return raw_sig
        return CLS_HOLD


# ---------------------------------------------------------------------------
# Branch configuration — every ablatable decision branch as one flag.
# ---------------------------------------------------------------------------
@dataclass
class BranchConfig:
    """Branch toggles. The BASELINE is a *minimal trading* config: take every
    model long signal (thr only), execution branches ON, optional FILTERS OFF.
    This guarantees the baseline actually trades so each branch's marginal
    post-fee effect is measurable — the legacy-transformer defaults (consensus=2,
    margin, hard-gate) silence these sparse binary models to zero trades."""

    thr_long: float = 0.5
    # --- Optional FILTERS: default OFF, audited as "add" (does ON help EV?) ---
    use_hard_gate: bool = False        # confluence hard gate (liq-sweep/AVWAP/GP)
    use_consensus: bool = False
    consensus: int = 2
    use_regime_filter: bool = False    # EMA-fast vs EMA-slow sim filter
    use_min_atr: bool = False
    min_atr_pct: float = 0.001
    use_regime_routing: bool = False   # NEW: per-regime threshold/enable
    # --- Execution/exit branches: default ON, audited as "remove" ------------
    use_post_only: bool = True         # maker entries (heuristic: no book data)
    use_market_depth: bool = True      # book-walk fills  (heuristic: no book data)
    use_dynamic_slippage: bool = True  # per-bar 0.5*ATR taker slippage
    use_atr_stops: bool = True
    use_cooldown: bool = True
    cooldown: int = 3
    # fixed-stop fallbacks (used when use_atr_stops is off)
    tp_pct: float = 0.005
    sl_pct: float = 0.0025
    atr_tp_mult: float = 1.8
    atr_sl_mult: float = 1.0
    leverage: float = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_SYM_DIR = {"eth": "ETH-USD", "btc": "BTC-USD", "sol": "SOL-USD"}


def _read_table(path: str) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _ohlc_dir_for(dataset_path: str, override: Optional[str]) -> str:
    if override:
        return override
    name = Path(dataset_path).name.lower()
    for pref, d in _SYM_DIR.items():
        if name.startswith(pref):
            return str(Path("data/crypto") / d / "1m")
    raise ValueError(f"Cannot infer OHLC dir from {dataset_path}; pass --ohlc-dir")


def _load_ohlc(ohlc_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(ohlc_dir) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No OHLC day files in {ohlc_dir}")
    frames = [pd.read_csv(f) for f in files]
    ohlc = pd.concat(frames, ignore_index=True)
    need = {"timestamp", "open", "high", "low", "close"}
    missing = need - set(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC files missing columns {missing}")
    ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"], utc=True, errors="coerce")
    ohlc = ohlc.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
    return ohlc[["timestamp", "open", "high", "low", "close"]]


def load_joined_test_slice(
    model_dir: str, *, split: str = "test", ohlc_dir: Optional[str] = None
) -> Tuple[object, Dict, pd.DataFrame, List[str]]:
    """Load model + meta and the OOS feature/OHLC-joined slice for ``split``."""
    mdir = Path(model_dir)
    meta = json.loads((mdir / "meta.json").read_text())
    model = joblib.load(mdir / "model.joblib")
    feature_cols = list(meta["feature_cols"])

    feats = _read_table(meta["dataset_path"])
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce")
    feats = feats.dropna(subset=["timestamp"]).sort_values("timestamp")

    ohlc = _load_ohlc(_ohlc_dir_for(meta["dataset_path"], ohlc_dir))
    df = feats.merge(ohlc, on="timestamp", how="inner").sort_values("timestamp")
    df = df.reset_index(drop=True)

    # Time-based split identical to train_xgboost._time_based_split.
    n = len(df)
    n_test = int(n * TEST_FRAC)
    n_val = int(n * VAL_FRAC)
    n_train = n - n_val - n_test
    if split == "train":
        sl = df.iloc[:n_train]
    elif split == "val":
        sl = df.iloc[n_train:n_train + n_val]
    else:  # test (default) — true OOS
        sl = df.iloc[n_train + n_val:]
    return model, meta, sl.reset_index(drop=True), feature_cols


# ---------------------------------------------------------------------------
# Simulation of one branch config
# ---------------------------------------------------------------------------
def _predict_p_long(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.asarray(proba)[:, 1].astype(np.float64)


def _build_bars(sl: pd.DataFrame) -> List[Bar]:
    o = sl["open"].to_numpy(float)
    h = sl["high"].to_numpy(float)
    lo = sl["low"].to_numpy(float)
    c = sl["close"].to_numpy(float)
    atr = sl["atr_14"].to_numpy(float) if "atr_14" in sl.columns else np.full(len(sl), np.nan)
    ema_fast = sl["ema_21"].to_numpy(float) if "ema_21" in sl.columns else np.full(len(sl), np.nan)
    ema_slow = sl["ema_50"].to_numpy(float) if "ema_50" in sl.columns else np.full(len(sl), np.nan)
    ts = sl["timestamp"].astype(str).to_list()
    bars: List[Bar] = []
    for i in range(len(sl)):
        a = float(atr[i]) if np.isfinite(atr[i]) else None
        ef = float(ema_fast[i]) if np.isfinite(ema_fast[i]) else None
        es = float(ema_slow[i]) if np.isfinite(ema_slow[i]) else None
        bars.append(Bar(open=o[i], high=h[i], low=lo[i], close=c[i], atr=a,
                        ema_fast=ef, ema_slow=es, timestamp=ts[i]))
    return bars


def run_branch(
    bc: BranchConfig,
    *,
    p_long: np.ndarray,
    bars: List[Bar],
    feature_rows: List[dict],
    feature_cols: List[str],
    start_capital: float = 10_000.0,
    return_sim: bool = False,
):
    """Run the execution simulator under one branch config; return post-fee metrics."""
    sim_cfg = SimulationConfig.from_coinbase_fees(
        start_capital=float(start_capital),
        tp_pct=float(bc.tp_pct),
        sl_pct=float(bc.sl_pct),
        cooldown=int(bc.cooldown) if bc.use_cooldown else 0,
        slippage_pct=0.0,
        use_market_depth=bool(bc.use_market_depth),
        use_hard_gating=bool(bc.use_hard_gate),
        post_only_entries=bool(bc.use_post_only),
        fallback_to_market_on_missing_book=True,
        fallback_to_market_on_post_only_miss=False,
        use_atr_stops=bool(bc.use_atr_stops),
        atr_tp_mult=float(bc.atr_tp_mult),
        atr_sl_mult=float(bc.atr_sl_mult),
        use_regime_filter=bool(bc.use_regime_filter),
        min_atr_pct=float(bc.min_atr_pct) if bc.use_min_atr else 0.0,
        allow_shorts=False,  # models are binary long-only
        leverage=float(bc.leverage),
    )
    sim = PortfolioSimulator(sim_cfg)
    gate = StrategyGate(
        thr_long=bc.thr_long, thr_short=1.0, margin=0.0,
        feature_cols=feature_cols, use_hard_gating=bool(bc.use_hard_gate),
    )
    consensus = ConsensusFilter(bc.consensus if bc.use_consensus else 1)
    router = RegimeRouter() if bc.use_regime_routing else None

    for k, bar in enumerate(bars):
        p = float(p_long[k])
        row = feature_rows[k]
        thr = bc.thr_long
        enabled = True
        if router is not None:
            enabled, thr, _reg = router.route(row)
        cls_sig = CLS_LONG if (enabled and p >= thr) else CLS_HOLD
        # Confluence hard gate (long-only): may demote LONG -> HOLD.
        if bc.use_hard_gate and cls_sig == CLS_LONG:
            cls_sig = gate.apply_hard_gate(cls_sig, feature_row=row)
        cls_sig = consensus.step(cls_sig)
        raw_sig = RAW_LONG if cls_sig == CLS_LONG else RAW_FLAT
        if bc.use_dynamic_slippage and bar.atr is not None and bar.close:
            sim.config.slippage_pct = 0.5 * float(bar.atr) / max(1e-12, float(bar.close))
        else:
            sim.config.slippage_pct = 0.0
        sim.step(bar, signal=raw_sig)
    if bars:
        sim.finalize(bars[-1].close, bars[-1].timestamp)

    metrics = _post_fee_metrics(sim, start_capital)
    if return_sim:
        return metrics, sim
    return metrics


def _post_fee_metrics(sim: PortfolioSimulator, start_capital: float) -> Dict:
    """Reconstruct HONEST post-fee per-trade returns from the equity sequence.

    All-in single-position sizing => cash-when-flat == equity, so the equity
    recorded on each ``exit`` row already nets entry fee + exit fee + gross P/L.
    """
    exits = [t for t in sim.trade_log if t.get("action") == "exit"]
    equities = [float(t["equity"]) for t in exits]
    net_rets: List[float] = []
    prev = float(start_capital)
    for eq in equities:
        net_rets.append(eq / max(1e-12, prev) - 1.0)
        prev = eq
    arr = np.array(net_rets, dtype=float)
    n = int(arr.size)
    if n:
        pnls = np.diff(np.concatenate([[start_capital], equities]))
        gain = float(pnls[pnls > 0].sum())
        loss = float(-pnls[pnls < 0].sum())
        pf = (gain / loss) if loss > 0 else (float("inf") if gain > 0 else 0.0)
        expectancy = float(arr.mean())
        win_rate = float((arr > 0).mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        sqn = float(expectancy / std * np.sqrt(n)) if std > 0 else 0.0
    else:
        pf = expectancy = win_rate = sqn = 0.0
    rep = sim.report()["portfolio"]
    return {
        "n_trades": n,
        "post_fee_expectancy": expectancy,     # mean net-of-fee per-trade return
        "post_fee_profit_factor": pf,
        "post_fee_win_rate": win_rate,
        "sqn": sqn,
        "net_return_pct": float(sim.last_equity / max(1e-12, start_capital) - 1.0) * 100.0,
        "max_drawdown_pct": float(rep.get("max_drawdown", 0.0)) * 100.0,
        "maker_fills": int(rep.get("maker_fills", 0)),
        "taker_fills": int(rep.get("taker_fills", 0)),
        "missed_entries": int(rep.get("missed_entries", 0)),
        "exposure": float(rep.get("exposure", 0.0)),
    }


def _regime_breakdown(sim, start_capital: float, ts2regime: Dict[str, str]) -> Dict:
    """Group baseline trades by their ENTRY regime and report post-fee EV each.

    This is the real test of regime routing: if one regime's trades are
    materially less-negative (or positive) post-fee, routing to it pays; if all
    regimes are uniformly underwater, routing cannot rescue the strategy.
    """
    entry_ts: Optional[str] = None
    prev_eq = float(start_capital)
    buckets: Dict[str, List[float]] = {}
    for t in sim.trade_log:
        if t.get("action") == "enter":
            entry_ts = str(t.get("timestamp"))
        elif t.get("action") == "exit":
            eq = float(t["equity"])
            net_ret = eq / max(1e-12, prev_eq) - 1.0
            prev_eq = eq
            reg = ts2regime.get(entry_ts, "unknown")
            buckets.setdefault(reg, []).append(net_ret)
    out = {}
    for reg, rets in buckets.items():
        a = np.array(rets, dtype=float)
        out[reg] = {
            "n_trades": int(a.size),
            "post_fee_expectancy": float(a.mean()) if a.size else 0.0,
            "post_fee_win_rate": float((a > 0).mean()) if a.size else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# The audit: baseline + one-branch-flipped ablations + threshold sweep + combo
# ---------------------------------------------------------------------------
# Each "remove" ablation flips a beneficial-by-assumption branch OFF; a branch
# "PAYS" (KEEP) if removing it lowers post-fee expectancy. Additive branches
# (regime routing) flip ON; they earn KEEP only if they raise expectancy.
_ABLATIONS = [
    # Optional filters — baseline OFF, flip ON; KEEP only if EV rises.
    ("hard_gate", dict(use_hard_gate=True), "add"),
    ("consensus", dict(use_consensus=True, consensus=2), "add"),
    ("regime_filter", dict(use_regime_filter=True), "add"),
    ("min_atr_filter", dict(use_min_atr=True), "add"),
    ("regime_routing", dict(use_regime_routing=True), "add"),
    # Execution/exit STRATEGY choices — baseline ON, flip OFF; KEEP if removing hurts.
    ("post_only_maker", dict(use_post_only=False), "remove"),
    ("atr_stops", dict(use_atr_stops=False), "remove"),
    ("cooldown", dict(use_cooldown=False), "remove"),
    # COST-REALISM models — always ON. Flipping OFF just fabricates optimistic
    # fills, so these are report-only (never auto-cut): the delta shows how much
    # the cost model subtracts, not a strategy lever.
    ("dynamic_slippage", dict(use_dynamic_slippage=False), "cost"),
    ("market_depth", dict(use_market_depth=False), "cost"),
]


def _verdict(kind: str, base_exp: float, ablate_exp: float, ablate_trades: int) -> str:
    delta = ablate_exp - base_exp  # effect of the flip
    if kind == "cost":
        return "COST (always on)"
    if kind == "remove":
        # Removing the branch changed expectancy by `delta`.
        if delta > EPS_EXPECTANCY:
            return "CUT"          # removing it improved post-fee EV
        if delta < -EPS_EXPECTANCY:
            return "KEEP"         # removing it hurt -> branch pays
        return "CUT (neutral)"    # no EV difference -> drop the complexity/cost
    else:  # add
        if ablate_trades < 10:
            return "SKIP (too few trades)"
        if delta > EPS_EXPECTANCY:
            return "KEEP (adopt)"
        return "SKIP"


def audit_model(
    model_dir: str, *, split: str = "test", ohlc_dir: Optional[str] = None,
    thr_long: Optional[float] = None, start_capital: float = 10_000.0,
) -> Dict:
    model, meta, sl, feature_cols = load_joined_test_slice(
        model_dir, split=split, ohlc_dir=ohlc_dir)
    X = sl[feature_cols].to_numpy(np.float64)
    p_long = _predict_p_long(model, X)
    bars = _build_bars(sl)
    feature_rows = sl.to_dict("records")

    default_thr = thr_long
    if default_thr is None:
        default_thr = float(meta.get("optimal_threshold") or 0.5)
    baseline = BranchConfig(thr_long=float(default_thr))

    base_metrics, base_sim = run_branch(
        baseline, p_long=p_long, bars=bars, feature_rows=feature_rows,
        feature_cols=feature_cols, start_capital=start_capital, return_sim=True)
    base_exp = base_metrics["post_fee_expectancy"]

    # Per-regime post-fee EV of baseline trades (the real regime-routing test).
    router0 = RegimeRouter()
    ts2regime = {str(r["timestamp"]): router0.classify(r) for r in feature_rows}
    regime_ev = _regime_breakdown(base_sim, start_capital, ts2regime)

    branches: List[Dict] = []
    for name, override, kind in _ABLATIONS:
        bc = replace(baseline, **override)
        m = run_branch(bc, p_long=p_long, bars=bars, feature_rows=feature_rows,
                       feature_cols=feature_cols, start_capital=start_capital)
        branches.append({
            "branch": name, "kind": kind,
            "delta_expectancy": m["post_fee_expectancy"] - base_exp,
            "verdict": _verdict(kind, base_exp, m["post_fee_expectancy"], m["n_trades"]),
            "metrics": m,
        })

    # Threshold sweep (report the post-fee-EV-maximizing long threshold).
    thr_grid = [round(t, 2) for t in np.linspace(0.30, 0.80, 11)]
    sweep = []
    for t in thr_grid:
        m = run_branch(replace(baseline, thr_long=float(t)), p_long=p_long, bars=bars,
                       feature_rows=feature_rows, feature_cols=feature_cols,
                       start_capital=start_capital)
        sweep.append({"thr_long": t, "post_fee_expectancy": m["post_fee_expectancy"],
                      "post_fee_profit_factor": m["post_fee_profit_factor"],
                      "n_trades": m["n_trades"], "net_return_pct": m["net_return_pct"]})
    tradeable = [s for s in sweep if s["n_trades"] >= 10] or sweep
    best_thr = max(tradeable, key=lambda s: s["post_fee_expectancy"])["thr_long"]

    # Combined config: apply every CUT + adopt routing if KEEP, at best thr.
    combo = replace(baseline, thr_long=float(best_thr))
    for b in branches:
        name, kind, verdict = b["branch"], b["kind"], b["verdict"]
        # "add" branch that KEEPs -> adopt it; "remove" branch that CUTs -> drop it.
        # Both are expressed by applying the branch's ablation override.
        if (kind == "add" and verdict.startswith("KEEP")) or (
            kind == "remove" and verdict.startswith("CUT")
        ):
            combo = replace(combo, **dict(_ABLATIONS_MAP[name]))
    combo_metrics = run_branch(combo, p_long=p_long, bars=bars, feature_rows=feature_rows,
                               feature_cols=feature_cols, start_capital=start_capital)

    return {
        "model_dir": str(model_dir),
        "dataset_path": meta.get("dataset_path"),
        "split": split,
        "n_bars": int(len(sl)),
        "time_span": [str(sl["timestamp"].iloc[0]), str(sl["timestamp"].iloc[-1])] if len(sl) else [],
        "fee_schedule": {"taker_bps": 60.0, "maker_bps": 40.0, "roundtrip_taker_bps": 120.0},
        "label_hurdle_bps": _infer_hurdle_bps(meta),
        "baseline_config": vars(baseline),
        "baseline_metrics": base_metrics,
        "regime_ev_breakdown": regime_ev,
        "branches": branches,
        "threshold_sweep": sweep,
        "best_thr_long": best_thr,
        "recommended_config": vars(combo),
        "recommended_metrics": combo_metrics,
        "caveats": [
            "Book columns (best_bid/ask, depth, spread) are identically 0 in the "
            "datasets -> post_only_maker / market_depth results are HEURISTIC "
            "(ATR-slippage/taker fallback), not real book fills. Rebuild datasets "
            "with L2 book data to validate those two branches for real.",
            "Post-fee per-trade returns are reconstructed from the equity curve "
            "(cross-check); the simulator now also nets fees into trade_log ret/pnl.",
            "Intrabar TP/SL uses real high/low from the OHLC join.",
        ],
    }


_ABLATIONS_MAP = {name: override for name, override, _ in _ABLATIONS}


def _infer_hurdle_bps(meta: Dict) -> Optional[float]:
    p = str(meta.get("dataset_path", "")).lower()
    if "20bps" in p:
        return 20.0
    if "10bps" in p:
        return 10.0
    return None


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def _fmt(res: Dict) -> str:
    b = res["baseline_metrics"]
    lines = []
    lines.append("=" * 92)
    lines.append(f"POST-FEE BRANCH AUDIT — {res['model_dir']}  (split={res['split']}, "
                 f"{res['n_bars']:,} bars)")
    if res.get("time_span"):
        lines.append(f"  OOS span: {res['time_span'][0]} -> {res['time_span'][1]}")
    lines.append(f"  Fees: taker 60bps / maker 40bps (roundtrip taker ~120bps) | "
                 f"label hurdle: {res.get('label_hurdle_bps')} bps")
    lines.append("-" * 92)
    lines.append(f"BASELINE  thr={res['baseline_config']['thr_long']:.2f}  "
                 f"trades={b['n_trades']}  EV/trade={b['post_fee_expectancy']*1e4:+.2f}bps  "
                 f"PF={b['post_fee_profit_factor']:.2f}  win={b['post_fee_win_rate']*100:.1f}%  "
                 f"net={b['net_return_pct']:+.1f}%  maxDD={b['max_drawdown_pct']:.1f}%")
    lines.append("-" * 92)
    lines.append(f"{'BRANCH':<20}{'ΔEV/trade':>12}{'trades':>8}{'EV/trade':>11}"
                 f"{'PF':>7}{'net%':>8}   VERDICT")
    for br in res["branches"]:
        m = br["metrics"]
        lines.append(
            f"{br['branch']:<20}{br['delta_expectancy']*1e4:>+11.2f}b{m['n_trades']:>8}"
            f"{m['post_fee_expectancy']*1e4:>+10.2f}b{m['post_fee_profit_factor']:>7.2f}"
            f"{m['net_return_pct']:>+8.1f}   {br['verdict']}")
    lines.append("-" * 92)
    reg = res.get("regime_ev_breakdown", {})
    if reg:
        lines.append("PER-REGIME post-fee EV (baseline trades by entry regime):")
        for name, rm in sorted(reg.items()):
            lines.append(f"    {name:<12} trades={rm['n_trades']:<4} "
                         f"EV/trade={rm['post_fee_expectancy']*1e4:+.2f}bps  "
                         f"win={rm['post_fee_win_rate']*100:.0f}%")
    lines.append("-" * 92)
    sw = res["threshold_sweep"]
    lines.append("THRESHOLD SWEEP (EV/trade bps @ thr): " +
                 "  ".join(f"{s['thr_long']:.2f}:{s['post_fee_expectancy']*1e4:+.1f}" for s in sw))
    lines.append(f"  -> best thr_long = {res['best_thr_long']:.2f}")
    r = res["recommended_metrics"]
    lines.append("-" * 92)
    lines.append(f"RECOMMENDED (best thr + all CUTs + routing-if-kept): "
                 f"trades={r['n_trades']}  EV/trade={r['post_fee_expectancy']*1e4:+.2f}bps  "
                 f"PF={r['post_fee_profit_factor']:.2f}  net={r['net_return_pct']:+.1f}%  "
                 f"maxDD={r['max_drawdown_pct']:.1f}%")
    lines.append("=" * 92)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ohlc-dir", default=None)
    ap.add_argument("--thr-long", type=float, default=None)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--out", default=None, help="JSON output path (default: <model-dir>/branch_audit.json)")
    args = ap.parse_args()

    res = audit_model(args.model_dir, split=args.split, ohlc_dir=args.ohlc_dir,
                      thr_long=args.thr_long, start_capital=args.capital)
    print(_fmt(res))
    out = args.out or str(Path(args.model_dir) / "branch_audit.json")
    Path(out).write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[branch_audit] wrote {out}")


if __name__ == "__main__":
    main()
