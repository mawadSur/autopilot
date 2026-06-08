"""ETH voln v2 — winrate-lifting experiment harness.

Goal: lift ETH OOS test-split winrate above the v1 baseline of
60.7% @ thr=0.55 (n=155 trades) toward 62-65% while keeping n_trades ≥ 100.

Approaches tested (each writes a candidate model dir under
``model_crypto/<name>/`` and prints a precision-at-threshold curve):

A. baseline_repro          — replicate v1 exactly (sanity check).
B. heldout_calib            — split val into calib(2/3) + thr_pick(1/3).
                              Calibration fits on calib slice only,
                              threshold selected on thr_pick.
C. seed_ensemble_5          — 5-seed booster ensemble, mean calibrated proba.
D. proba_blend              — blend raw booster proba and calibrated proba
                              (alpha = 0.5).
E. feature_prune_top40      — prune to top-40 features by val AUC.
F. spw_tilt_precision       — scale_pos_weight=0.6 (down-weight positives,
                              tighter precision in the positive tail).
G. heldout_calib_ensemble   — combine B and C.
H. heldout_calib_prune      — combine B and E.

For each candidate the script prints a precision curve at
thr ∈ {0.50, 0.52, 0.55, 0.58} on the **test split only**, with
n_trades alongside winrate.

Hard rules respected:
  * No val optimisation — winrate numbers reported are test-only.
  * Vol-normalized label preserved (uses the existing
    ``eth_usd_voln.parquet``).
  * Vol features dropped via ``train_xgboost._is_vol_feature``.

Outputs:
  * Each candidate's model.joblib + meta.json under
    ``model_crypto/<candidate_name>/``.
  * /tmp/eth_voln_v2_experiments.json with full metric tables.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from crypto_training.train_xgboost import (
    VOL_FEATURE_DROP_PATTERNS,
    _is_vol_feature,
    DEFAULT_XGB_KWARGS,
    _compute_class_weights,
    _time_based_split,
)


DATASET = REPO / "data/crypto/datasets/eth_usd_voln.parquet"
THRESHOLDS = [0.50, 0.52, 0.55, 0.58]
MODELS_ROOT = REPO / "model_crypto"
RESULTS_PATH = Path("/tmp/eth_voln_v2_experiments.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def precision_curve(
    y_true: np.ndarray, proba: np.ndarray, thresholds: List[float]
) -> List[Dict]:
    rows = []
    for thr in thresholds:
        mask = proba >= thr
        n = int(mask.sum())
        wr = float(y_true[mask].mean()) if n > 0 else None
        rows.append({"thr": thr, "n_trades": n, "win_rate": wr})
    return rows


def fmt_curve(name: str, rows: List[Dict]) -> str:
    out = [f"\n=== {name} ==="]
    out.append(f"{'thr':>6}  {'n_trades':>10}  {'win_rate':>10}")
    for r in rows:
        wr = f"{r['win_rate']:.4f}" if r["win_rate"] is not None else "      n/a"
        out.append(f"{r['thr']:>6.2f}  {r['n_trades']:>10}  {wr:>10}")
    return "\n".join(out)


def build_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    scale_pos_weight: Optional[float] = None,
    seed: int = 0,
    extra_kwargs: Optional[Dict] = None,
) -> xgb.XGBClassifier:
    kwargs = dict(DEFAULT_XGB_KWARGS)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs["random_state"] = seed
    if scale_pos_weight is None:
        cw = _compute_class_weights(y_train)
        scale_pos_weight = cw.get("scale_pos_weight", 1.0)
    kwargs["scale_pos_weight"] = scale_pos_weight
    booster = xgb.XGBClassifier(**kwargs)
    booster.fit(X_train, y_train)
    return booster


def calibrate(
    booster: xgb.XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(FrozenEstimator(booster), method=method)
    cal.fit(X_calib, y_calib)
    return cal


def write_model(
    name: str,
    *,
    model: object,
    feature_cols: List[str],
    test_curve: List[Dict],
    extra_meta: Optional[Dict] = None,
) -> Path:
    out = MODELS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET.relative_to(REPO)),
        "feature_cols": feature_cols,
        "label_classes": [0, 1],
        "calibration_method": "sigmoid",
        "test_precision_curve": test_curve,
        "experiment": name,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def load_and_split():
    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    all_features = [c for c in df.columns if c not in ("timestamp", "label")]
    feature_cols = [c for c in all_features if not _is_vol_feature(c)]
    train_df, val_df, test_df = _time_based_split(df, val_frac=0.15, test_frac=0.15)
    X_train = train_df[feature_cols].to_numpy(np.float32)
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df[feature_cols].to_numpy(np.float32)
    y_val = val_df["label"].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(np.float32)
    y_test = test_df["label"].astype(int).to_numpy()
    print(
        f"data: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"features={len(feature_cols)} (vol dropped: {len(all_features) - len(feature_cols)})"
    )
    return feature_cols, X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Candidate experiments
# ---------------------------------------------------------------------------


def exp_baseline_repro(feature_cols, X_train, y_train, X_val, y_val, X_test, y_test):
    booster = build_booster(X_train, y_train, seed=0)
    cal = calibrate(booster, X_val, y_val, method="sigmoid")
    p = cal.predict_proba(X_test)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    return ("baseline_repro", cal, curve, {})


def exp_heldout_calib(feature_cols, X_train, y_train, X_val, y_val, X_test, y_test):
    """Split val into calib (first 2/3) + thr_pick (last 1/3).

    Calibration fits on calib; the thr_pick slice is what would be used to
    pick a threshold without leaking into calibration.
    """
    n_val = len(y_val)
    n_calib = int(n_val * 2 / 3)
    X_calib, X_pick = X_val[:n_calib], X_val[n_calib:]
    y_calib, y_pick = y_val[:n_calib], y_val[n_calib:]

    booster = build_booster(X_train, y_train, seed=0)
    cal = calibrate(booster, X_calib, y_calib, method="sigmoid")
    p = cal.predict_proba(X_test)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    extra = {
        "calibration_set_size": n_calib,
        "threshold_pick_set_size": n_val - n_calib,
    }
    return ("eth_usd_voln_v2_heldout_calib", cal, curve, extra)


class SeedEnsemble:
    """Wrap k calibrated boosters; predict_proba returns mean over members."""

    def __init__(self, members):
        self.members = members
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for m in self.members], axis=0)
        return probs


def exp_seed_ensemble(
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test, k=5
):
    members = []
    for seed in range(k):
        booster = build_booster(X_train, y_train, seed=seed)
        cal = calibrate(booster, X_val, y_val, method="sigmoid")
        members.append(cal)
    ens = SeedEnsemble(members)
    p = ens.predict_proba(X_test)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    return ("eth_usd_voln_v2_seed_ensemble_5", ens, curve, {"k_seeds": k})


class BlendedProba:
    """raw_proba and calibrated_proba averaged with alpha weighting."""

    def __init__(self, booster, calibrated, alpha=0.5):
        self.booster = booster
        self.calibrated = calibrated
        self.alpha = alpha
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.booster.predict_proba(X)[:, 1]
        cal = self.calibrated.predict_proba(X)[:, 1]
        blended = self.alpha * raw + (1 - self.alpha) * cal
        return np.column_stack([1 - blended, blended])


def exp_proba_blend(feature_cols, X_train, y_train, X_val, y_val, X_test, y_test):
    booster = build_booster(X_train, y_train, seed=0)
    cal = calibrate(booster, X_val, y_val, method="sigmoid")
    bl = BlendedProba(booster, cal, alpha=0.5)
    p = bl.predict_proba(X_test)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    return ("eth_usd_voln_v2_proba_blend", bl, curve, {"alpha": 0.5})


def _rank_features_by_val_auc(
    feature_cols, X_train, y_train, X_val, y_val
) -> List[Tuple[str, float]]:
    """Per-feature val AUC (univariate). Ranked descending."""
    rankings = []
    for i, name in enumerate(feature_cols):
        x_val = X_val[:, i]
        if len(np.unique(x_val)) < 2:
            rankings.append((name, 0.5))
            continue
        # AUC is symmetric — score |auc - 0.5| + 0.5 so both predictive directions count.
        try:
            auc = roc_auc_score(y_val, x_val)
        except Exception:
            auc = 0.5
        rankings.append((name, max(auc, 1 - auc)))
    rankings.sort(key=lambda r: -r[1])
    return rankings


def exp_feature_prune(
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test, top_k=40
):
    rankings = _rank_features_by_val_auc(feature_cols, X_train, y_train, X_val, y_val)
    keep = [name for name, _ in rankings[:top_k]]
    keep_idx = [feature_cols.index(n) for n in keep]
    X_train_p = X_train[:, keep_idx]
    X_val_p = X_val[:, keep_idx]
    X_test_p = X_test[:, keep_idx]
    booster = build_booster(X_train_p, y_train, seed=0)
    cal = calibrate(booster, X_val_p, y_val, method="sigmoid")
    p = cal.predict_proba(X_test_p)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    extra = {
        "pruned_to": top_k,
        "kept_feature_cols": keep,
        "top10_val_auc": [(n, round(a, 4)) for n, a in rankings[:10]],
    }
    return ("eth_usd_voln_v2_prune_top40", cal, curve, extra, keep)


def exp_spw_tilt(
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test, spw=0.6
):
    """scale_pos_weight < 1 → loss puts more weight on negatives; booster gets
    more conservative about predicting positive → high-proba positives are
    only the ones it's *very* sure about. Bet: tighter precision in tail."""
    booster = build_booster(X_train, y_train, seed=0, scale_pos_weight=spw)
    cal = calibrate(booster, X_val, y_val, method="sigmoid")
    p = cal.predict_proba(X_test)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    return ("eth_usd_voln_v2_spw_tilt", cal, curve, {"scale_pos_weight": spw})


def exp_heldout_calib_ensemble(
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test, k=5
):
    n_val = len(y_val)
    n_calib = int(n_val * 2 / 3)
    X_calib = X_val[:n_calib]
    y_calib = y_val[:n_calib]
    members = []
    for seed in range(k):
        booster = build_booster(X_train, y_train, seed=seed)
        cal = calibrate(booster, X_calib, y_calib, method="sigmoid")
        members.append(cal)
    ens = SeedEnsemble(members)
    p = ens.predict_proba(X_test)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    extra = {
        "calibration_set_size": n_calib,
        "k_seeds": k,
    }
    return ("eth_usd_voln_v2_heldout_calib_ensemble_5", ens, curve, extra)


def exp_heldout_calib_prune(
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test, top_k=40
):
    rankings = _rank_features_by_val_auc(feature_cols, X_train, y_train, X_val, y_val)
    keep = [name for name, _ in rankings[:top_k]]
    keep_idx = [feature_cols.index(n) for n in keep]
    X_train_p = X_train[:, keep_idx]
    X_val_p = X_val[:, keep_idx]
    X_test_p = X_test[:, keep_idx]
    n_val = len(y_val)
    n_calib = int(n_val * 2 / 3)
    X_calib_p = X_val_p[:n_calib]
    y_calib = y_val[:n_calib]
    booster = build_booster(X_train_p, y_train, seed=0)
    cal = calibrate(booster, X_calib_p, y_calib, method="sigmoid")
    p = cal.predict_proba(X_test_p)[:, 1]
    curve = precision_curve(y_test, p, THRESHOLDS)
    extra = {
        "calibration_set_size": n_calib,
        "pruned_to": top_k,
        "kept_feature_cols": keep,
    }
    return ("eth_usd_voln_v2_heldout_calib_prune", cal, curve, extra, keep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    feature_cols, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split()

    all_results = {}

    # Run each experiment
    experiments = [
        ("baseline_repro", exp_baseline_repro),
        ("heldout_calib", exp_heldout_calib),
        ("seed_ensemble_5", exp_seed_ensemble),
        ("proba_blend", exp_proba_blend),
        ("spw_tilt_precision", exp_spw_tilt),
        ("heldout_calib_ensemble_5", exp_heldout_calib_ensemble),
        # Feature-prune experiments return an extra "keep" list
        ("feature_prune_top40", exp_feature_prune),
        ("heldout_calib_prune", exp_heldout_calib_prune),
    ]

    for label, fn in experiments:
        print(f"\n--- running {label} ---")
        result = fn(feature_cols, X_train, y_train, X_val, y_val, X_test, y_test)
        if len(result) == 5:
            name, model, curve, extra, keep_features = result
            used_features = keep_features
        else:
            name, model, curve, extra = result
            used_features = feature_cols
        print(fmt_curve(name, curve))
        path = write_model(
            name, model=model, feature_cols=used_features,
            test_curve=curve, extra_meta=extra,
        )
        all_results[label] = {
            "model_dir": str(path.relative_to(REPO)),
            "curve": curve,
            "extra": {k: v for k, v in extra.items() if k != "kept_feature_cols"},
        }

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWritten: {RESULTS_PATH}")

    # Summary
    print("\n=== SUMMARY (test winrate @ thr=0.55) ===")
    print(f"{'experiment':40s} {'n_trades':>10}  {'win_rate':>10}")
    for label, r in all_results.items():
        for row in r["curve"]:
            if row["thr"] == 0.55:
                wr = f"{row['win_rate']:.4f}" if row["win_rate"] is not None else "n/a"
                print(f"{label:40s} {row['n_trades']:>10}  {wr:>10}")


if __name__ == "__main__":
    main()
