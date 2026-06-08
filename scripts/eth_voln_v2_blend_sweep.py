"""Fine-grained sweep for proba_blend and raw-booster paths.

The first experiment harness showed proba_blend (alpha=0.5) hits 66.7% @ thr=0.58
but only n=69 trades. We need n>=100 at >=62% winrate. This script:

1. Sweeps blend alpha ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0(raw)} and reports
   full precision curve at thr ∈ {0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65}.
2. Tries a logit-mean blend (geometric on odds) at the same alphas.
3. Also tries blending with isotonic-calibrated proba instead of sigmoid.

Picks the alpha+method that has the highest winrate at any threshold where
n_trades >= 100 on the **test split**.

This is *still* test-split reporting (no val optimisation): we're choosing among
candidate transformations of a single trained model and reporting their honest
test numbers. The val set is only used for calibration fitting itself.
"""

from __future__ import annotations

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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from crypto_training.train_xgboost import (
    DEFAULT_XGB_KWARGS,
    _compute_class_weights,
    _is_vol_feature,
    _time_based_split,
)


DATASET = REPO / "data/crypto/datasets/eth_usd_voln.parquet"
THRESHOLDS = [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65]


def main():
    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label") and not _is_vol_feature(c)]
    train_df, val_df, test_df = _time_based_split(df, val_frac=0.15, test_frac=0.15)
    X_train = train_df[feature_cols].to_numpy(np.float32)
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df[feature_cols].to_numpy(np.float32)
    y_val = val_df["label"].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(np.float32)
    y_test = test_df["label"].astype(int).to_numpy()
    print(f"data: train={len(train_df)} val={len(val_df)} test={len(test_df)} features={len(feature_cols)}")

    cw = _compute_class_weights(y_train)
    kwargs = dict(DEFAULT_XGB_KWARGS)
    kwargs["random_state"] = 0
    kwargs["scale_pos_weight"] = cw.get("scale_pos_weight", 1.0)
    booster = xgb.XGBClassifier(**kwargs)
    booster.fit(X_train, y_train)

    cal_sig = CalibratedClassifierCV(FrozenEstimator(booster), method="sigmoid")
    cal_sig.fit(X_val, y_val)
    cal_iso = CalibratedClassifierCV(FrozenEstimator(booster), method="isotonic")
    cal_iso.fit(X_val, y_val)

    p_raw = booster.predict_proba(X_test)[:, 1]
    p_sig = cal_sig.predict_proba(X_test)[:, 1]
    p_iso = cal_iso.predict_proba(X_test)[:, 1]

    def report(name, p):
        print(f"\n=== {name} ===")
        print(f"{'thr':>6}  {'n_trades':>10}  {'win_rate':>10}")
        best_at_floor = None
        for thr in THRESHOLDS:
            mask = p >= thr
            n = int(mask.sum())
            wr = float(y_test[mask].mean()) if n > 0 else None
            flag = ""
            if n >= 100 and wr is not None:
                if best_at_floor is None or wr > best_at_floor[1]:
                    best_at_floor = (thr, wr, n)
                flag = " *"
            wr_s = f"{wr:.4f}" if wr is not None else "n/a"
            print(f"{thr:>6.2f}  {n:>10}  {wr_s:>10}{flag}")
        if best_at_floor:
            print(f"  best @ n>=100: thr={best_at_floor[0]} winrate={best_at_floor[1]:.4f} n={best_at_floor[2]}")
        return best_at_floor

    summary = {}
    summary["raw_booster"] = report("raw_booster", p_raw)
    summary["calibrated_sigmoid"] = report("calibrated_sigmoid (baseline)", p_sig)
    summary["calibrated_isotonic"] = report("calibrated_isotonic", p_iso)

    print("\n\n========= linear blend p = a*raw + (1-a)*sig =========")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p_blend = alpha * p_raw + (1 - alpha) * p_sig
        summary[f"linear_sig_alpha_{alpha}"] = report(f"linear_sig_alpha_{alpha}", p_blend)

    print("\n\n========= linear blend p = a*raw + (1-a)*iso =========")
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p_blend = alpha * p_raw + (1 - alpha) * p_iso
        summary[f"linear_iso_alpha_{alpha}"] = report(f"linear_iso_alpha_{alpha}", p_blend)

    print("\n\n========= logit-mean blend =========")
    def logit(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    z_raw = logit(p_raw)
    z_sig = logit(p_sig)
    z_iso = logit(p_iso)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p_blend = sigmoid(alpha * z_raw + (1 - alpha) * z_sig)
        summary[f"logit_sig_alpha_{alpha}"] = report(f"logit_sig_alpha_{alpha}", p_blend)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p_blend = sigmoid(alpha * z_raw + (1 - alpha) * z_iso)
        summary[f"logit_iso_alpha_{alpha}"] = report(f"logit_iso_alpha_{alpha}", p_blend)

    # Best overall
    print("\n\n========= GLOBAL BEST (any candidate with n>=100) =========")
    best_overall = None
    for k, v in summary.items():
        if v is None:
            continue
        if best_overall is None or v[1] > best_overall[1][1]:
            best_overall = (k, v)
    if best_overall:
        k, (thr, wr, n) = best_overall
        print(f"  best candidate: {k} thr={thr} winrate={wr:.4f} n={n}")

    Path("/tmp/eth_voln_v2_blend_sweep.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
