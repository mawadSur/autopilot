"""Fit + evaluate a hi-win regime gate for ETH v2 blend09_alt.

Methodology:

1. Reproduce test-split predictions at thr=0.57.
2. Split the test window in half by time: ``fit`` (first half) and
   ``eval`` (second half).
3. Grid-search a 2-feature gate ``bb_width_20 >= theta_bb AND
   range_pct >= theta_range`` over the ``fit`` slice to maximise
   ``win_rate * sqrt(retention)`` (balances precision and volume).
4. Score the chosen gate on the ``eval`` slice without further tuning.
5. As a robustness check, run leave-one-day-out CV across the 14 test days.

Output answers: does the gate generalise out-of-fit, or is it just an
overfit to the same window the model already saw?
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
for p in (SRC, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import joblib
import numpy as np
import pandas as pd

from crypto_training.train_xgboost import _time_based_split  # noqa: E402

DATASET = REPO / "data/crypto/datasets/eth_usd_voln.parquet"
MODEL_DIR = REPO / "model_crypto/eth_usd_voln_v2_blend09_alt"
THR = 0.57

# Features the validation diagnostic flagged with Cohen's d >= 0.5.
GATE_FEATURES = ("bb_width_20", "range_pct")
# Minimum fraction of fires we insist on keeping. Below this, the gate
# may technically maximise objective but we'd be down to noise volume.
MIN_RETENTION = 0.30


def _score(fired: pd.DataFrame, gate_pass: np.ndarray) -> dict:
    n_total = len(fired)
    kept = fired.loc[gate_pass]
    n = len(kept)
    if n == 0:
        return {"n": 0, "wr": None, "retention": 0.0, "score": -1.0}
    wr = float(kept["label"].mean())
    retention = n / n_total
    score = wr * (retention**0.5) if wr is not None else -1.0
    return {"n": int(n), "wr": wr, "retention": float(retention), "score": float(score)}


def _grid_search(fit: pd.DataFrame) -> tuple[dict, list]:
    """Search a (theta_bb, theta_range) grid on the fit slice."""
    bb = fit["bb_width_20"].astype(float)
    rg = fit["range_pct"].astype(float)
    # Use quantile grid so the search adapts to the distribution.
    bb_grid = [float(bb.quantile(q)) for q in np.arange(0.0, 0.81, 0.05)]
    rg_grid = [float(rg.quantile(q)) for q in np.arange(0.0, 0.81, 0.05)]

    rows = []
    best = None
    for tb, tr in itertools.product(bb_grid, rg_grid):
        pass_mask = (bb.to_numpy() >= tb) & (rg.to_numpy() >= tr)
        stats = _score(fit, pass_mask)
        if stats["retention"] < MIN_RETENTION:
            continue
        row = {"theta_bb_width": tb, "theta_range_pct": tr, **stats}
        rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
    return best, rows


def main() -> int:
    df = pd.read_parquet(DATASET)
    train_df, val_df, test_df = _time_based_split(df, val_frac=0.15, test_frac=0.15)
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    feature_cols = meta["feature_cols"]
    model = joblib.load(MODEL_DIR / "model.joblib")

    X_test = test_df[feature_cols].to_numpy(np.float32)
    y_test = test_df["label"].astype(int).to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    fired = test_df.loc[proba >= THR].copy()
    fired["proba"] = proba[proba >= THR]
    fired["label"] = y_test[proba >= THR]
    fired["timestamp"] = pd.to_datetime(fired["timestamp"], utc=True)
    fired["date"] = fired["timestamp"].dt.date
    fired = fired.sort_values("timestamp").reset_index(drop=True)
    print(f"fires: n={len(fired)} overall_wr={float(fired['label'].mean()):.4f}")

    # ---------------- time-based fit/eval split ----------------
    midpoint = len(fired) // 2
    fit = fired.iloc[:midpoint].copy()
    evl = fired.iloc[midpoint:].copy()
    print(
        f"fit slice: n={len(fit)} wr={float(fit['label'].mean()):.4f} "
        f"{fit['timestamp'].iloc[0]} -> {fit['timestamp'].iloc[-1]}"
    )
    print(
        f"eval slice: n={len(evl)} wr={float(evl['label'].mean()):.4f} "
        f"{evl['timestamp'].iloc[0]} -> {evl['timestamp'].iloc[-1]}"
    )

    best, all_rows = _grid_search(fit)
    if best is None:
        print("\nno gate found that retains >= 30% of fires -- abort")
        return 1
    print(
        f"\nbest gate on fit: bb_width>={best['theta_bb_width']:.5f} "
        f"AND range_pct>={best['theta_range_pct']:.5f}"
    )
    print(
        f"  fit:  n={best['n']} wr={best['wr']:.4f} retention={best['retention']:.3f}"
    )

    # Evaluate the chosen gate on the held-out eval slice
    bb_e = evl["bb_width_20"].astype(float).to_numpy()
    rg_e = evl["range_pct"].astype(float).to_numpy()
    pass_eval = (bb_e >= best["theta_bb_width"]) & (rg_e >= best["theta_range_pct"])
    eval_stats = _score(evl, pass_eval)
    print(f"  eval: n={eval_stats['n']} wr={eval_stats['wr']:.4f} retention={eval_stats['retention']:.3f}")
    eval_baseline = float(evl["label"].mean())
    print(f"  eval baseline (no gate): wr={eval_baseline:.4f}")
    eval_lift = (eval_stats["wr"] - eval_baseline) if eval_stats["wr"] is not None else None
    print(f"  eval lift: {eval_lift:+.4f}" if eval_lift is not None else "  eval lift: n/a")

    # ---------------- leave-one-day-out CV ----------------
    print("\nleave-one-day-out CV (gate fit on other 13 days, eval on held-out day):")
    days = sorted(fired["date"].unique())
    loo_rows = []
    for held in days:
        train_slice = fired.loc[fired["date"] != held].copy()
        eval_slice = fired.loc[fired["date"] == held].copy()
        local_best, _ = _grid_search(train_slice)
        if local_best is None:
            continue
        bb_e = eval_slice["bb_width_20"].astype(float).to_numpy()
        rg_e = eval_slice["range_pct"].astype(float).to_numpy()
        pass_mask = (
            (bb_e >= local_best["theta_bb_width"])
            & (rg_e >= local_best["theta_range_pct"])
        )
        kept = eval_slice.loc[pass_mask]
        baseline_wr = float(eval_slice["label"].mean())
        kept_wr = float(kept["label"].mean()) if len(kept) else None
        loo_rows.append(
            {
                "held_day": str(held),
                "n_total": int(len(eval_slice)),
                "n_kept": int(len(kept)),
                "kept_wr": kept_wr,
                "baseline_wr": baseline_wr,
                "lift": (kept_wr - baseline_wr) if kept_wr is not None else None,
                "theta_bb_width": local_best["theta_bb_width"],
                "theta_range_pct": local_best["theta_range_pct"],
            }
        )

    loo_df = pd.DataFrame(loo_rows)
    print(loo_df.to_string(index=False))
    aggr = {
        "n_total_sum": int(loo_df["n_total"].sum()),
        "n_kept_sum": int(loo_df["n_kept"].sum()),
        "kept_wr_weighted": float(
            (loo_df["kept_wr"] * loo_df["n_kept"]).sum() / max(int(loo_df["n_kept"].sum()), 1)
        ) if loo_df["n_kept"].sum() > 0 else None,
        "baseline_wr_weighted": float(
            (loo_df["baseline_wr"] * loo_df["n_total"]).sum() / int(loo_df["n_total"].sum())
        ),
    }
    print(f"\nLOO aggregate: {aggr}")

    out = REPO / "model_sanity/hiwin_gate_fit.json"
    out.write_text(
        json.dumps(
            {
                "thr": THR,
                "n_fires": int(len(fired)),
                "fit_eval_split": {
                    "best_gate": best,
                    "eval_stats": eval_stats,
                    "eval_baseline_wr": eval_baseline,
                    "eval_lift": eval_lift,
                },
                "loo_cv": loo_rows,
                "loo_aggregate": aggr,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
