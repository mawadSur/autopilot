"""Probe: how much does each confluence filter lift precision of the
calibrated XGBoost crypto entry model, and at what cost in trade
frequency?

For each (symbol, model) combo:
  1. Load the test slice (last 15% of ``data/crypto/datasets/{sym}_usd_1m.csv``).
  2. Load the trained model + meta from ``model_crypto/{sym}_usd_v*/``.
  3. Run ``predict_proba`` on the test slice.
  4. Apply the operator-chosen threshold (ETH=0.30, BTC=0.30, SOL=0.50)
     to get "model says trigger" mask. The label column is the binary
     "forward 5-bar return > 10 bps" used in training.
  5. For each filter stack, compute:
       * triggers   = #rows where model fires AND filter accepts
       * precision  = mean(label[triggers])
       * Δ precision vs baseline
       * frequency kept = triggers / baseline_triggers
  6. Print a per-symbol markdown table.
  7. Print a recommendation block + final caveats.

Run::

    ./.venv/bin/python probe_confluence_filters.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# libomp dance before numpy/xgboost.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Allow ``import confluence_filters`` from src/ without PYTHONPATH gymnastics.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
import numpy as np
import pandas as pd

from confluence_filters import (
    vectorised_atr_not_extreme,
    vectorised_spread_ok,
    vectorised_trend_align,
    vectorised_volume_above_ma,
    vectorised_volume_above_ma_proxy,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS: List[Tuple[str, str, str, float]] = [
    # (symbol_label, dataset_csv, model_dir, model_threshold)
    ("ETH/USD", "data/crypto/datasets/eth_usd_1m.csv", "model_crypto/eth_usd_v2", 0.30),
    ("BTC/USD", "data/crypto/datasets/btc_usd_1m.csv", "model_crypto/btc_usd_v1", 0.30),
    ("SOL/USD", "data/crypto/datasets/sol_usd_1m.csv", "model_crypto/sol_usd_v1", 0.50),
]

TEST_FRAC = 0.15
TRAIN_FRAC = 0.70  # used only for ATR percentile cap calibration
ATR_PERCENTILE = 80


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@dataclass
class StackResult:
    name: str
    triggers: int
    precision: float
    delta_precision: float  # in percentage POINTS (not %)
    frequency_kept: float  # fraction of baseline triggers


def _load_meta(model_dir: Path) -> dict:
    return json.loads((model_dir / "meta.json").read_text())


def _load_test_slice(csv_path: Path, test_frac: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    test = df.iloc[n - n_test :].reset_index(drop=True)
    return test


def _load_train_slice_for_atr(csv_path: Path, train_frac: float) -> pd.DataFrame:
    """Just the columns we need to compute the ATR cap (saves RAM)."""
    df = pd.read_csv(csv_path, usecols=["timestamp", "atrp_14"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    return df.iloc[:n_train]


def _baseline(probs: np.ndarray, labels: np.ndarray, thr: float) -> Tuple[np.ndarray, int, float]:
    """Return (trigger_mask, n_triggers, precision)."""
    mask = probs >= thr
    n = int(mask.sum())
    prec = float(labels[mask].mean()) if n > 0 else float("nan")
    return mask, n, prec


def _eval_stack(
    name: str,
    trigger_mask: np.ndarray,
    labels: np.ndarray,
    baseline_n: int,
    baseline_prec: float,
) -> StackResult:
    n = int(trigger_mask.sum())
    prec = float(labels[trigger_mask].mean()) if n > 0 else float("nan")
    delta = (prec - baseline_prec) * 100.0 if n > 0 else float("nan")
    freq = (n / baseline_n) if baseline_n > 0 else 0.0
    return StackResult(name=name, triggers=n, precision=prec, delta_precision=delta, frequency_kept=freq)


def _fmt_row(r: StackResult) -> str:
    def _pct(x: float) -> str:
        return "n/a" if not np.isfinite(x) else f"{x * 100:5.1f}%"

    def _signed(x: float) -> str:
        if not np.isfinite(x):
            return "  n/a"
        return f"{x:+5.1f}"

    return f"| {r.name:<37} | {r.triggers:>8} | {_pct(r.precision):>9} | {_signed(r.delta_precision):>11} | {_pct(r.frequency_kept):>14} |"


# ---------------------------------------------------------------------------
# Main per-symbol loop
# ---------------------------------------------------------------------------


def probe_symbol(symbol: str, csv_rel: str, model_rel: str, thr: float) -> Tuple[str, List[StackResult]]:
    csv_path = _REPO_ROOT / csv_rel
    model_dir = _REPO_ROOT / model_rel

    print(f"\n--- {symbol} -------------------------------------------------------")
    print(f"  dataset: {csv_rel}")
    print(f"  model:   {model_rel}")
    print(f"  thr:     {thr}")

    meta = _load_meta(model_dir)
    feature_cols: List[str] = list(meta["feature_cols"])

    # 1) Load test slice + compute features array
    test_df = _load_test_slice(csv_path, TEST_FRAC)
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise RuntimeError(f"{symbol}: test slice missing model features: {missing[:5]}")
    labels = test_df["label"].astype(int).to_numpy()

    X = test_df[feature_cols].to_numpy(dtype=np.float32)
    # NaN guard (mirrors what the predictor adapter does at inference time).
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Predict
    model = joblib.load(model_dir / "model.joblib")
    probs = model.predict_proba(X)[:, 1]
    pos_rate = float(labels.mean())
    print(
        f"  test rows: {len(test_df)}, label positive rate: {pos_rate * 100:.2f}%, "
        f"probs range: [{probs.min():.3f}, {probs.max():.3f}], mean: {probs.mean():.3f}"
    )

    # 3) Baseline
    base_mask, base_n, base_prec = _baseline(probs, labels, thr)
    if base_n == 0:
        print(f"  WARNING: zero triggers at thr={thr}. Lower the threshold or retrain.")
        return symbol, []

    # 4) Compute per-row gate masks (only meaningful at trigger rows, but
    #    we compute over the whole test slice and AND with base_mask).
    train_atr = _load_train_slice_for_atr(csv_path, TRAIN_FRAC)["atrp_14"].dropna()
    atr_cap = float(np.percentile(train_atr.to_numpy(), ATR_PERCENTILE))
    print(f"  atrp_14 P{ATR_PERCENTILE} (from train slice): {atr_cap:.6f}")

    g_vol_lit = vectorised_volume_above_ma(test_df, multiplier=1.5)
    g_vol_proxy = vectorised_volume_above_ma_proxy(test_df, multiplier=1.5)
    g_atr = vectorised_atr_not_extreme(test_df, cap=atr_cap)
    g_trend = vectorised_trend_align(test_df)
    g_spread = vectorised_spread_ok(test_df, max_spread_bps=2.0)

    # Sanity-report the degenerate filters so the user can see WHY a stack
    # collapses to zero (e.g. spec-literal volume gate -> 0 because
    # volume_quote is identically 0 in this dataset).
    print(
        f"  filter pass rates (over full test slice): "
        f"vol_lit={g_vol_lit.mean()*100:5.2f}%, "
        f"vol_proxy={g_vol_proxy.mean()*100:5.2f}%, "
        f"atr<=P{ATR_PERCENTILE}={g_atr.mean()*100:5.2f}%, "
        f"trend={g_trend.mean()*100:5.2f}%, "
        f"spread<=2bps={g_spread.mean()*100:5.2f}%"
    )

    # 5) Build stacks. We deliberately report BOTH the spec-literal vol
    #    gate and the proxy variant -- the spec-literal one will be 0
    #    triggers on the current data and that's worth surfacing.
    stacks: List[Tuple[str, np.ndarray]] = [
        ("model only (baseline)", base_mask),
        ("+ volume_above_ma(1.5) [SPEC LITERAL]", base_mask & g_vol_lit),
        ("+ volume_above_ma_proxy(1.5)", base_mask & g_vol_proxy),
        (f"+ atr_not_extreme(P{ATR_PERCENTILE})", base_mask & g_atr),
        ("+ trend_align", base_mask & g_trend),
        ("+ spread_ok(2bps)", base_mask & g_spread),
        # Pairwise combos that look promising on intuition (high signal +
        # cheap frequency cost).
        ("+ vol_proxy + trend", base_mask & g_vol_proxy & g_trend),
        ("+ vol_proxy + atr", base_mask & g_vol_proxy & g_atr),
        ("+ trend + atr", base_mask & g_trend & g_atr),
        # ALL gates (spec-literal volume so 0 triggers expected) + ALL with proxy
        ("+ ALL gates [SPEC LITERAL]", base_mask & g_vol_lit & g_atr & g_trend & g_spread),
        ("+ ALL gates (proxy volume)", base_mask & g_vol_proxy & g_atr & g_trend & g_spread),
    ]

    results: List[StackResult] = []
    for name, mask in stacks:
        results.append(_eval_stack(name, mask, labels, base_n, base_prec))

    return symbol, results


def render_table(symbol: str, thr: float, results: List[StackResult]) -> str:
    if not results:
        return f"\n### {symbol} @ thr={thr}\n\n_no triggers; skipping_\n"
    base = results[0]
    header = (
        f"\n### {symbol} @ thr={thr} "
        f"(baseline {base.triggers} trig, {base.precision * 100:.1f}% precision)\n\n"
        "| Filter stack                          | Triggers | Precision | Δ precision | Frequency kept |\n"
        "|---------------------------------------|---------:|----------:|------------:|---------------:|\n"
    )
    body = "\n".join(_fmt_row(r) for r in results)
    return header + body + "\n"


def make_recommendation(
    per_symbol: Dict[str, Tuple[float, List[StackResult]]], precision_target: float = 0.60
) -> str:
    out: List[str] = []
    out.append("\n## Recommendation\n")
    out.append(f"Target precision: {precision_target * 100:.0f}%\n")
    out.append("")
    out.append(f"Best per-symbol filter stack hitting >= {precision_target * 100:.0f}% precision")
    out.append("(prioritise highest precision; break ties by more triggers):\n")

    hopeless: List[str] = []
    for symbol, (thr, results) in per_symbol.items():
        if not results:
            hopeless.append(f"{symbol} (no triggers at thr={thr})")
            continue
        # Exclude the baseline row itself from "best filter" picks unless
        # nothing else helps.
        candidates = [r for r in results if r.triggers > 0 and r.precision >= precision_target]
        if candidates:
            best = max(candidates, key=lambda r: (r.precision, r.triggers))
            # Triggers/day estimate: test slice is the last 15% (~19k rows
            # of 1m bars => ~13.4 days). best.triggers / 13.4
            n_days = (len(_load_test_slice(_REPO_ROOT / "data/crypto/datasets/eth_usd_1m.csv", TEST_FRAC)) / 1440.0)
            trig_per_day = best.triggers / max(n_days, 1e-9)
            out.append(
                f"  * {symbol}: {best.name}  ->  {best.precision * 100:.1f}% precision, "
                f"{best.triggers} triggers in test (~{trig_per_day:.1f}/day)"
            )
        else:
            best_anyway = max(results[1:], key=lambda r: r.precision if r.triggers > 0 else -1)
            out.append(
                f"  * {symbol}: NO STACK HITS {precision_target * 100:.0f}%. "
                f"Best achievable: {best_anyway.name} -> {best_anyway.precision * 100:.1f}% "
                f"@ {best_anyway.triggers} triggers (baseline {results[0].precision * 100:.1f}%)"
            )
            hopeless.append(symbol)

    if hopeless:
        out.append("")
        out.append(f"Symbols where {precision_target * 100:.0f}% is NOT reachable with these gates alone:")
        for h in hopeless:
            out.append(f"  * {h}")

    return "\n".join(out) + "\n"


def main() -> int:
    per_symbol: Dict[str, Tuple[float, List[StackResult]]] = {}
    tables: List[str] = []

    for symbol, csv_rel, model_rel, thr in SYMBOLS:
        sym, results = probe_symbol(symbol, csv_rel, model_rel, thr)
        per_symbol[sym] = (thr, results)
        tables.append(render_table(sym, thr, results))

    print("\n\n=================================================================")
    print("                       PROBE REPORT")
    print("=================================================================")
    for t in tables:
        print(t)

    rec = make_recommendation(per_symbol, precision_target=0.60)
    print(rec)

    print("\n## Caveats (read this)\n")
    print(
        "  * `volume_quote` and `spread_pct` are identically 0 in the current\n"
        "    1m datasets -- the OHLCV backfill didn't carry quote-currency volume\n"
        "    or L1 book columns. The SPEC LITERAL volume gate therefore always\n"
        "    rejects (0 triggers); the spread gate is a trivial pass-through.\n"
        "    The `_proxy` variant uses `expm1(vol_log)` (base volume) and is\n"
        "    what reflects the genuine 'volume above MA' intent on this dataset.\n"
        "  * The ATR cap is calibrated from the 70% TRAIN slice (P{p}), so it's\n"
        "    out-of-sample for the test slice. If you re-run after a retrain,\n"
        "    re-compute the cap.\n"
        "  * Test slice is ~15% of 129k 1m bars per symbol -> ~13.4 days. The\n"
        "    triggers/day estimate is back-of-envelope.\n"
        "  * Label is the same one the model trained on (binary, forward 5-bar\n"
        "    return > 10 bps), so we're measuring SAME-LABEL precision lift, not\n"
        "    real PnL. Gross-of-fees, gross-of-slippage."
        .format(p=ATR_PERCENTILE)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
