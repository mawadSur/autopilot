"""Live-vs-training feature distribution diagnostic.

Pulls the exact feature vector the live XGBoost predictor would compute right
now (same buffer + ``compute_features`` path the supervisor uses), then
compares it to the training parquet distribution.

Settles: is the model correctly bearish on today's regime, or is some feature
silently broken at inference time?

Usage:
    ./.venv/bin/python scripts/diagnose_live_features.py \\
        --symbol ETH/USD \\
        --model-dir model_crypto/eth_usd_voln_v1 \\
        --parquet data/crypto/datasets/eth_usd_voln.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from exchanges.coinbase import CoinbaseExchange, Ticker
from predictor import XGBoostPredictor


CONSTANT_TOL = 1e-12


def _train_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> pd.DataFrame:
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return df.iloc[:n_train].copy()


def _fmt(v: float, width: int = 12) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'nan':>{width}}"
    if abs(v) >= 1e6 or (0 < abs(v) < 1e-3):
        return f"{v:>{width}.4e}"
    return f"{v:>{width}.4f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETH/USD")
    p.add_argument("--model-dir", default="model_crypto/eth_usd_voln_v1")
    p.add_argument("--parquet", default="data/crypto/datasets/eth_usd_voln.parquet")
    p.add_argument("--top", type=int, default=20, help="Top-N drifted features to print")
    p.add_argument("--leave-one-out", type=int, default=10,
                   help="Top-N features to test for leave-one-out sensitivity")
    args = p.parse_args()

    model_dir = (REPO / args.model_dir).resolve()
    parquet_path = (REPO / args.parquet).resolve()

    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
          f"diagnostic: {args.symbol} model={model_dir.name}")
    print()

    # ---------------- training distribution ----------------
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    train = _train_split(df)
    print(f"training parquet: rows={len(df):,} train={len(train):,} cols={len(df.columns)}")

    # ---------------- live predictor ----------------
    exch = CoinbaseExchange()
    predictor = XGBoostPredictor(
        model_dir=str(model_dir),
        exchange=exch,
        thr_long=None,  # use meta.optimal_threshold
    )
    feature_cols = predictor.feature_cols

    # Force a buffer refresh + one inference. ticker is unused by the predictor
    # (it pulls candles itself from the exchange) but we need a valid Ticker.
    ticker = exch.get_ticker(args.symbol)
    print(f"live ticker: {args.symbol} bid={ticker.bid} ask={ticker.ask} last={ticker.last}")
    result = predictor.predict_full(args.symbol, ticker)
    if result.feature_buffer is None:
        print("FATAL: predictor returned None feature_buffer "
              "(probably buffer still warming up — re-run after ~5min)")
        return 1
    live_p_long = result.model_probs.get("long") if result.model_probs else None
    print(f"live P(long)={live_p_long:.4f} thr={predictor.thr_long:.3f} "
          f"side={result.side} conf={result.confidence:.3f}")
    print()

    # ---------------- per-feature comparison ----------------
    rows = []
    for col in feature_cols:
        if col not in train.columns:
            rows.append({"feature": col, "status": "MISSING_FROM_TRAIN"})
            continue
        train_col = train[col].astype(float)
        mu = float(train_col.mean())
        sd = float(train_col.std())
        live_v = result.feature_buffer.get(col)
        # None means the predictor flagged NaN/inf at inference time
        if live_v is None or not np.isfinite(live_v):
            rows.append({
                "feature": col, "live": np.nan, "train_mean": mu,
                "train_std": sd, "z": np.nan, "status": "LIVE_NAN",
            })
            continue
        # Constant training feature (the 50 zombie bucket-2 cols)
        if sd < CONSTANT_TOL:
            status = "CONSTANT" if abs(live_v - mu) < CONSTANT_TOL else "CONSTANT_BUT_LIVE_DIFFERS"
            rows.append({
                "feature": col, "live": live_v, "train_mean": mu,
                "train_std": sd, "z": np.nan, "status": status,
            })
            continue
        z = (live_v - mu) / sd
        # P5/P95 to know whether live is within training support
        p5 = float(train_col.quantile(0.05))
        p95 = float(train_col.quantile(0.95))
        in_support = p5 <= live_v <= p95
        rows.append({
            "feature": col, "live": live_v, "train_mean": mu,
            "train_std": sd, "z": z, "p5": p5, "p95": p95,
            "in_support": in_support,
            "status": "OK" if abs(z) <= 2 else ("OUTSIDE_2SIGMA" if abs(z) <= 3 else "OUTSIDE_3SIGMA"),
        })

    summary = pd.DataFrame(rows)

    n_total = len(summary)
    n_constant = int((summary["status"] == "CONSTANT").sum())
    n_constant_diff = int((summary["status"] == "CONSTANT_BUT_LIVE_DIFFERS").sum())
    n_live_nan = int((summary["status"] == "LIVE_NAN").sum())
    n_missing = int((summary["status"] == "MISSING_FROM_TRAIN").sum())
    n_outside_2 = int((summary["status"] == "OUTSIDE_2SIGMA").sum())
    n_outside_3 = int((summary["status"] == "OUTSIDE_3SIGMA").sum())
    n_ok = int((summary["status"] == "OK").sum())

    print("=" * 80)
    print("FEATURE STATUS SUMMARY")
    print("=" * 80)
    print(f"  total features:                   {n_total}")
    print(f"  OK (live within 2σ of train):     {n_ok}")
    print(f"  outside 2σ (<= 3σ):               {n_outside_2}")
    print(f"  outside 3σ (potential drift):     {n_outside_3}")
    print(f"  CONSTANT in train (dead inputs):  {n_constant}")
    print(f"  CONSTANT but live differs (BUG):  {n_constant_diff}")
    print(f"  LIVE NaN/inf:                     {n_live_nan}")
    print(f"  MISSING from train parquet:       {n_missing}")
    print()

    # Constant-but-live-differs is the smoking gun for a feature engineering bug.
    if n_constant_diff > 0:
        print(">>> FLAG: training-constant features that are non-zero at inference:")
        bug = summary[summary["status"] == "CONSTANT_BUT_LIVE_DIFFERS"]
        print(bug[["feature", "live", "train_mean"]].to_string(index=False))
        print()

    if n_live_nan > 0:
        print(">>> FLAG: features with NaN/inf at live inference:")
        nan_rows = summary[summary["status"] == "LIVE_NAN"]
        print(nan_rows[["feature", "train_mean", "train_std"]].to_string(index=False))
        print()

    # Top drifted (real features only)
    real = summary[summary["status"].isin(["OK", "OUTSIDE_2SIGMA", "OUTSIDE_3SIGMA"])].copy()
    real["abs_z"] = real["z"].abs()
    top = real.sort_values("abs_z", ascending=False).head(args.top)
    print(f"TOP {args.top} FEATURES BY |z| (live vs training mean)")
    print("=" * 80)
    print(f"{'feature':<32} {'live':>12} {'train_mean':>12} {'train_std':>12} {'z':>8} {'in_p5_p95':>10}")
    for _, r in top.iterrows():
        in_sup = "yes" if r.get("in_support") else "no"
        print(f"{r['feature']:<32} {_fmt(r['live'])} {_fmt(r['train_mean'])} "
              f"{_fmt(r['train_std'])} {r['z']:>8.2f} {in_sup:>10}")
    print()

    # ---------------- leave-one-out sensitivity ----------------
    # For the top-K drifted features, replace one at a time with the training
    # median and see how P(long) moves. Tells us which feature(s) are pulling
    # the prediction down.
    print(f"LEAVE-ONE-OUT SENSITIVITY (top {args.leave_one_out} drifted)")
    print("=" * 80)
    print("Replace one feature at a time with train_median; observe P(long).")
    print(f"Baseline P(long) = {live_p_long:.4f}")
    print()

    # Build a single-row DataFrame from feature_buffer in feature_cols order
    row_vals = []
    for col in feature_cols:
        v = result.feature_buffer.get(col)
        if v is None or not np.isfinite(v):
            v = 0.0
        row_vals.append(float(v))
    base_row = np.array(row_vals, dtype="float32").reshape(1, -1)

    # Verify our baseline reproduces what predict_full returned.
    reproduced = float(predictor.model.predict_proba(base_row)[0, 1])
    if abs(reproduced - live_p_long) > 1e-4:
        print(f"WARNING: reproduced P(long)={reproduced:.4f} != "
              f"predict_full P(long)={live_p_long:.4f} (diff={reproduced - live_p_long:+.4f})")
        print("  This means leave-one-out is being computed on a slightly different "
              "vector than what produced the logged probability. Likely a NaN-fill "
              "boundary case. Results below are still directionally informative.")
        print()

    train_medians = {col: float(train[col].astype(float).median())
                     for col in feature_cols if col in train.columns}

    impacts = []
    for _, r in top.iterrows():
        col = r["feature"]
        if col not in train_medians:
            continue
        i = feature_cols.index(col)
        mod = base_row.copy()
        mod[0, i] = train_medians[col]
        p_mod = float(predictor.model.predict_proba(mod)[0, 1])
        delta = p_mod - reproduced
        impacts.append({
            "feature": col, "live": r["live"], "median": train_medians[col],
            "z": r["z"], "P_long_if_median": p_mod, "delta": delta,
        })

    impacts_df = pd.DataFrame(impacts).sort_values("delta", key=lambda s: s.abs(), ascending=False)
    print(f"{'feature':<32} {'live':>12} {'median':>12} {'z':>8} "
          f"{'P(long)_repl':>12} {'delta':>8}")
    for _, r in impacts_df.head(args.leave_one_out).iterrows():
        print(f"{r['feature']:<32} {_fmt(r['live'])} {_fmt(r['median'])} "
              f"{r['z']:>8.2f} {r['P_long_if_median']:>12.4f} {r['delta']:>+8.4f}")
    print()

    # Maximum upside: replace ALL top-K drifted features with their training
    # medians at once. If P(long) jumps over thr_long, the drifted features
    # are collectively responsible for the silence.
    swap_all = base_row.copy()
    for _, r in top.iterrows():
        col = r["feature"]
        if col in train_medians:
            i = feature_cols.index(col)
            swap_all[0, i] = train_medians[col]
    p_swap_all = float(predictor.model.predict_proba(swap_all)[0, 1])
    print(f"P(long) if ALL top-{args.top} drifted features set to train_median: "
          f"{p_swap_all:.4f} (Δ={p_swap_all - reproduced:+.4f})")
    print(f"  thr_long={predictor.thr_long:.3f} -> "
          f"{'WOULD TRIGGER' if p_swap_all >= predictor.thr_long else 'still neutral'}")
    print()

    # ---------------- verdict ----------------
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    if n_constant_diff > 0:
        print("BUG: training-constant features are non-zero at inference. "
              "This is a feature-engineering parity gap and is the most likely "
              "cause of the live silence.")
    elif n_live_nan > 0:
        print(f"BUG: {n_live_nan} feature(s) NaN/inf at inference. "
              "Check compute_features warmup + ensure_optional_microstructure_columns.")
    elif n_outside_3 > 5:
        print(f"DRIFT: {n_outside_3} features outside 3σ of training distribution. "
              "Today's market regime is genuinely far from the training period. "
              "Model is correctly idle — not a bug, just no edge in this regime.")
    elif p_swap_all >= predictor.thr_long:
        print("DRIFT: replacing the top drifted features with training medians "
              "would push P(long) over threshold. Today's regime is the cause "
              "of the silence; the model itself is healthy.")
    else:
        print("NO BUG, NO DRIFT: features are mostly in distribution and "
              "replacing them with training medians still does not push P(long) "
              "over threshold. The model genuinely has no long signal here. "
              "Conclusion: the 60.7% test-window winrate is regime-specific and "
              "we are not currently in such a regime. Plan A (real microstructure) "
              "is the only path forward.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
