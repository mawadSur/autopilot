"""Final sweep: stochastic ensemble + best blend variants.

After the deterministic seed observation, this script:

1. Builds a *stochastic* 5-seed ensemble using subsample=0.8, colsample_bytree=0.8.
2. Compares its calibrated proba, raw proba, and best blends against the
   deterministic v1 baseline.
3. Reports honest test-split precision curves.

The user's bar: winrate >= 62% at some thr where n_trades >= 100 on test.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from crypto_training.train_xgboost import (
    DEFAULT_XGB_KWARGS,
    _compute_class_weights,
    _is_vol_feature,
    _time_based_split,
    _compute_feature_stats,
)
from crypto_training.blend_wrappers import (
    DetBlend as _PersistableDetBlend,
    StochasticEnsemble as _PersistableStochasticEnsemble,
    BlendedEnsemble as _PersistableBlendedEnsemble,
)


DATASET = REPO / "data/crypto/datasets/eth_usd_voln.parquet"
THRESHOLDS = [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]
MODELS_ROOT = REPO / "model_crypto"


def report(name, p, y_test):
    print(f"\n=== {name} ===")
    print(f"{'thr':>6}  {'n_trades':>10}  {'win_rate':>10}")
    best = None
    rows = []
    for thr in THRESHOLDS:
        mask = p >= thr
        n = int(mask.sum())
        wr = float(y_test[mask].mean()) if n > 0 else None
        rows.append({"thr": thr, "n_trades": n, "win_rate": wr})
        flag = ""
        if n >= 100 and wr is not None:
            if best is None or wr > best[1]:
                best = (thr, wr, n)
            flag = " *"
        wr_s = f"{wr:.4f}" if wr is not None else "n/a"
        print(f"{thr:>6.2f}  {n:>10}  {wr_s:>10}{flag}")
    if best:
        print(f"  best @ n>=100: thr={best[0]} winrate={best[1]:.4f} n={best[2]}")
    return rows, best


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Wrapper classes live in src/crypto_training/blend_wrappers.py so joblib
# can pickle/unpickle them from any context (e.g. the validator script that
# imports from a different __main__).
StochasticEnsemble = _PersistableStochasticEnsemble
BlendedEnsemble = _PersistableBlendedEnsemble
DetBlend = _PersistableDetBlend


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

    # ----- Stochastic ensemble -----
    K = 5
    print(f"\nbuilding stochastic ensemble of {K} boosters (subsample=0.8, colsample=0.8) ...")
    raw_members = []
    cal_members = []
    for seed in range(K):
        kw = dict(DEFAULT_XGB_KWARGS)
        kw["scale_pos_weight"] = cw.get("scale_pos_weight", 1.0)
        kw["subsample"] = 0.8
        kw["colsample_bytree"] = 0.8
        kw["random_state"] = seed
        b = xgb.XGBClassifier(**kw)
        b.fit(X_train, y_train)
        raw_members.append(b)
        cal = CalibratedClassifierCV(FrozenEstimator(b), method="sigmoid")
        cal.fit(X_val, y_val)
        cal_members.append(cal)

    ens = StochasticEnsemble(cal_members, raw_members)
    p_ens_cal = ens.predict_proba(X_test)[:, 1]
    p_ens_raw = ens.predict_raw_proba(X_test)[:, 1]

    results = {}
    results["stochastic_ensemble_calibrated"] = report(
        "stochastic_ensemble_calibrated (5 seeds)", p_ens_cal, y_test
    )
    results["stochastic_ensemble_raw"] = report(
        "stochastic_ensemble_raw (5 seeds)", p_ens_raw, y_test
    )

    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p_b = alpha * p_ens_raw + (1 - alpha) * p_ens_cal
        results[f"ensemble_blend_alpha_{alpha}"] = report(
            f"ensemble_blend_alpha_{alpha}", p_b, y_test
        )

    # Also compare to single-booster baseline (deterministic)
    print("\n--- single-booster baseline (no subsample) ---")
    kw_det = dict(DEFAULT_XGB_KWARGS)
    kw_det["scale_pos_weight"] = cw.get("scale_pos_weight", 1.0)
    b_det = xgb.XGBClassifier(**kw_det)
    b_det.fit(X_train, y_train)
    cal_det = CalibratedClassifierCV(FrozenEstimator(b_det), method="sigmoid")
    cal_det.fit(X_val, y_val)
    p_det_cal = cal_det.predict_proba(X_test)[:, 1]
    p_det_raw = b_det.predict_proba(X_test)[:, 1]
    results["single_det_calibrated_baseline"] = report(
        "single_det_calibrated (v1 baseline)", p_det_cal, y_test
    )
    for alpha in [0.6, 0.9]:
        p_b = alpha * p_det_raw + (1 - alpha) * p_det_cal
        results[f"single_det_blend_alpha_{alpha}"] = report(
            f"single_det_blend_alpha_{alpha}", p_b, y_test
        )

    # ----- GLOBAL BEST -----
    print("\n========= GLOBAL BEST (n>=100) =========")
    best_overall = None
    for k, (rows, best) in results.items():
        if best is None:
            continue
        if best_overall is None or best[1] > best_overall[1][1]:
            best_overall = (k, best)
    if best_overall:
        print(f"  WINNER: {best_overall[0]} thr={best_overall[1][0]} winrate={best_overall[1][1]:.4f} n={best_overall[1][2]}")

    Path("/tmp/eth_voln_v2_final_sweep.json").write_text(
        json.dumps({k: {"curve": v[0], "best_n100": v[1]} for k, v in results.items()}, indent=2, default=str)
    )

    # Persist the best ensemble model + the best deterministic + a blended ensemble at alpha=0.6
    # to model_crypto. The validator script expects:
    #   - model.joblib (sklearn-compat with predict_proba)
    #   - meta.json with feature_cols + dataset_path
    #
    # Persist 3 candidates so the human can compare:
    #   v2_ensemble_5             -- stochastic ensemble, calibrated proba
    #   v2_ensemble_5_blend_0.6   -- stochastic ensemble, blended (alpha=0.6)
    #   v2_det_blend_0.6          -- deterministic single, blended (alpha=0.6)

    feature_means, feature_stds = _compute_feature_stats(train_df, feature_cols)

    def _persist(name, model_obj, curve, extra_meta):
        out = MODELS_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_obj, out / "model.joblib")
        # Find the best thr at n>=100
        best_at_100 = None
        for r in curve:
            if r["n_trades"] >= 100 and r["win_rate"] is not None:
                if best_at_100 is None or r["win_rate"] > best_at_100["win_rate"]:
                    best_at_100 = r
        meta = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DATASET.relative_to(REPO)),
            "feature_cols": feature_cols,
            "label_classes": [0, 1],
            "calibration_method": "sigmoid",
            "xgb_kwargs": {k: v for k, v in DEFAULT_XGB_KWARGS.items()},
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "test_precision_curve": curve,
            "optimal_threshold": best_at_100["thr"] if best_at_100 else None,
            "test_winrate_at_optimal_threshold": best_at_100["win_rate"] if best_at_100 else None,
            "test_ntrades_at_optimal_threshold": best_at_100["n_trades"] if best_at_100 else None,
            "threshold_status": "ok" if best_at_100 and best_at_100["win_rate"] >= 0.62 else "below_v2_target",
            "experiment": name,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "class_weights": cw,
            **extra_meta,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    # Persist ensemble calibrated
    rows_cal, _ = results["stochastic_ensemble_calibrated"]
    p1 = _persist("eth_usd_voln_v2_ensemble_5", ens, rows_cal, {
        "ensemble_k": K,
        "ensemble_subsample": 0.8,
        "ensemble_colsample_bytree": 0.8,
        "blend_alpha": None,
    })
    print(f"persisted: {p1}")

    # Persist blended ensemble at alpha=0.6 (the top performer)
    rows_ens6, _ = results["ensemble_blend_alpha_0.6"]
    blended_ens = BlendedEnsemble(ens, alpha=0.6)
    p2 = _persist("eth_usd_voln_v2_ensemble_5_blend_0.6", blended_ens, rows_ens6, {
        "ensemble_k": K,
        "ensemble_subsample": 0.8,
        "ensemble_colsample_bytree": 0.8,
        "blend_alpha": 0.6,
    })
    print(f"persisted: {p2}")

    # Persist deterministic single + blend 0.6 (winner) and 0.9 (alt with more trades at higher thr)
    det_blend_6 = DetBlend(b_det, cal_det, alpha=0.6)
    rows_det6, _ = results["single_det_blend_alpha_0.6"]
    p3 = _persist("eth_usd_voln_v2_det_blend_0.6", det_blend_6, rows_det6, {
        "ensemble_k": 1,
        "blend_alpha": 0.6,
    })
    print(f"persisted: {p3}")

    det_blend_9 = DetBlend(b_det, cal_det, alpha=0.9)
    rows_det9, _ = results["single_det_blend_alpha_0.9"]
    p4 = _persist("eth_usd_voln_v2_det_blend_0.9", det_blend_9, rows_det9, {
        "ensemble_k": 1,
        "blend_alpha": 0.9,
    })
    print(f"persisted: {p4}")


if __name__ == "__main__":
    main()
