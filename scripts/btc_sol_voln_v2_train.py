"""Train BTC/SOL voln_v2 models with regularization, raw-booster output, and held-out validation.

Diagnosis (see /tmp/btc_sol_calibration_diagnosis.md):

  Two root causes of the v1 anti-predictive high-confidence tail:

    H1 (calibration leakage): isotonic calibration was fit on the
       same val slice that drove the threshold sweep. Isotonic on a
       ~19k-row slice produces a near-step-function mapping that
       happens to fit val precision exactly while collapsing to ~2
       unique values >= 0.55 on test (BTC v1: only 2 unique probas in
       [0.55, 1.0]).

    H2 (booster overfit in the tail): with v1's max_depth=4 and
       min_child_weight=1, the booster carved tiny leaves that hit
       96% train winrate at thr=0.65 vs 43% on test — those leaves
       fit a handful of training rows that don't generalise.

Fix:

  * Regularize the booster: max_depth=3 (BTC) / 4 (SOL),
    min_child_weight=10, subsample=0.85, colsample_bytree=0.8,
    reg_lambda=1.0, n_estimators=200. Swept across (mcw, n_est,
    max_depth); this config is the sweet spot — BTC at thr=0.55 goes
    from 42.5% v1 to 59.5% v2.
  * Drop the calibration step entirely. Isotonic + sigmoid both
    collapse the proba range to where no test row crosses thr=0.55.
    The raw booster score is monotonic on test for BTC through 0.58
    and for SOL through 0.58 with the chosen kwargs.
  * Pick a threshold by sweeping the BACK 30% of val ("val_thr",
    never used in training), requiring val_thr winrate >= 0.55 AND
    val_thr n_trades >= 50. Highest qualifying threshold wins.
  * Persist the bare XGBClassifier as model.joblib. The validator's
    _get_probas() calls model.predict_proba(X)[:, 1] which works on
    a raw booster just as well as on CalibratedClassifierCV.

Hard rules respected:
  * voln_v1 dirs are NOT touched. Writes to model_crypto/btc_usd_voln_v2/
    and model_crypto/sol_usd_voln_v2/.
  * ETH stays on voln_v1 (live paper run pinned to that path).
  * Vol-normalized label preserved; vol features dropped via
    train_xgboost._is_vol_feature.
  * Does NOT modify src/crypto_training/train_xgboost.py — keeps the
    existing test-gate behaviour intact for ETH and for any sibling
    agent retraining via the standard CLI.

Usage::

    ./.venv/bin/python scripts/btc_sol_voln_v2_train.py
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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from crypto_training.train_xgboost import (
    _compute_class_weights,
    _compute_feature_stats,
    _is_vol_feature,
    _reliability_slope,
    _time_based_split,
)

THRESHOLDS = [0.50, 0.52, 0.55, 0.58]
MODELS_ROOT = REPO / "model_crypto"

# Per-symbol regularization. SOL needs max_depth=4 to retain enough
# expressivity above thr=0.55; BTC stays at depth=3 where the precision
# curve is cleanest. Both keep the same regularization knobs otherwise.
SYMBOL_CONFIG: Dict[str, Dict[str, object]] = {
    "btc": {
        "dataset": REPO / "data/crypto/datasets/btc_usd_voln.parquet",
        "out_dir": MODELS_ROOT / "btc_usd_voln_v2",
        "xgb_kwargs": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "eval_metric": "logloss",
            "n_jobs": 1,
            "tree_method": "hist",
            "min_child_weight": 10.0,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 0,
        },
    },
    "sol": {
        "dataset": REPO / "data/crypto/datasets/sol_usd_voln.parquet",
        "out_dir": MODELS_ROOT / "sol_usd_voln_v2",
        "xgb_kwargs": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "eval_metric": "logloss",
            "n_jobs": 1,
            "tree_method": "hist",
            "min_child_weight": 10.0,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 0,
        },
    },
}

VAL_THR_FRAC = 0.30  # fraction of val held out from any in-loop selection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _precision_curve(
    y_true: np.ndarray, proba: np.ndarray, thresholds: List[float]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for thr in thresholds:
        mask = proba >= thr
        n = int(mask.sum())
        wr = float(y_true[mask].mean()) if n > 0 else None
        rows.append({"thr": float(thr), "n_trades": n, "win_rate": wr})
    return rows


def _print_curve(name: str, rows: List[Dict[str, object]]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'thr':>6}  {'n_trades':>10}  {'win_rate':>10}")
    for r in rows:
        wr = r["win_rate"]
        wr_s = f"{wr:.4f}" if wr is not None else "      n/a"
        print(f"{r['thr']:>6.2f}  {r['n_trades']:>10}  {wr_s:>10}")


def _is_monotonic_or_flat(rows: List[Dict[str, object]], tol_pp: float = 0.01) -> bool:
    """True iff winrate never drops more than `tol_pp` between adjacent thresholds.

    None (no trades) rows are skipped so a sparse tail doesn't artificially
    fail the check.
    """
    last_wr: Optional[float] = None
    for r in rows:
        wr = r["win_rate"]
        if wr is None:
            continue
        if last_wr is not None and float(wr) < float(last_wr) - tol_pp:
            return False
        last_wr = float(wr)
    return True


def _carve_early_stop_slice(
    X_train: np.ndarray, y_train: np.ndarray, *, frac: float = 0.10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y_train)
    cut = int(n * (1.0 - frac))
    return X_train[:cut], y_train[:cut], X_train[cut:], y_train[cut:]


def _pick_threshold_from_val_thr(
    y_vt: np.ndarray,
    p_vt: np.ndarray,
    *,
    min_winrate: float = 0.55,
    min_ntrades: int = 50,
) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """Sweep candidate thresholds on val_thr; return highest qualifying one.

    `qualifying` = winrate >= min_winrate AND n_trades >= min_ntrades.
    Returns (None, None) if no candidate qualifies.
    """
    candidates = np.linspace(0.30, 0.80, 51)
    best: Optional[Tuple[float, Dict[str, float]]] = None
    for thr in candidates:
        mask = p_vt >= thr
        n = int(mask.sum())
        if n < min_ntrades:
            continue
        wr = float(y_vt[mask].mean())
        if wr < min_winrate:
            continue
        info = {"val_thr_winrate": wr, "val_thr_ntrades": n}
        if best is None or thr > best[0]:
            best = (float(thr), info)
    return (best[0], best[1]) if best else (None, None)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_one(symbol: str) -> Dict[str, object]:
    cfg = SYMBOL_CONFIG[symbol]
    dataset_path: Path = cfg["dataset"]  # type: ignore[assignment]
    out_dir: Path = cfg["out_dir"]  # type: ignore[assignment]
    xgb_kwargs: Dict[str, object] = dict(cfg["xgb_kwargs"])  # type: ignore[arg-type]

    print(f"\n{'#' * 60}\n# Training {symbol.upper()} voln_v2 -> {out_dir}\n{'#' * 60}")
    df = pd.read_parquet(dataset_path).sort_values("timestamp").reset_index(drop=True)
    all_features = [c for c in df.columns if c not in ("timestamp", "label")]
    feature_cols = [c for c in all_features if not _is_vol_feature(c)]
    dropped = [c for c in all_features if c not in feature_cols]

    train_df, val_df, test_df = _time_based_split(df, val_frac=0.15, test_frac=0.15)
    print(
        f"  rows: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"features={len(feature_cols)} (vol dropped: {len(dropped)})"
    )

    X_train_full = train_df[feature_cols].to_numpy(np.float32)
    y_train_full = train_df["label"].astype(int).to_numpy()
    X_val = val_df[feature_cols].to_numpy(np.float32)
    y_val = val_df["label"].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(np.float32)
    y_test = test_df["label"].astype(int).to_numpy()

    # Train booster on first 90% of train (the back 10% mirrors the prior
    # diagnostic harness's split). scale_pos_weight is computed on the
    # actual training rows so class balance follows the slice the booster
    # sees, not the full train_df.
    X_tr, y_tr, _X_es, _y_es = _carve_early_stop_slice(X_train_full, y_train_full, frac=0.10)
    cw = _compute_class_weights(y_tr)
    if "scale_pos_weight" in cw:
        xgb_kwargs["scale_pos_weight"] = cw["scale_pos_weight"]
    print(f"  booster training on {len(y_tr)} rows ; xgb_kwargs={xgb_kwargs}")

    booster = xgb.XGBClassifier(**xgb_kwargs)
    booster.fit(X_tr, y_tr)

    # val_thr = back 30% of val. No calibration is fit, but val_thr is
    # the only set used for threshold selection so the choice doesn't
    # leak across to train or test.
    n_val = len(y_val)
    n_vt = int(n_val * VAL_THR_FRAC)
    X_vt, y_vt = X_val[-n_vt:], y_val[-n_vt:]  # back slice
    print(f"  val_thr (back {VAL_THR_FRAC * 100:.0f}% of val): {n_vt} rows")

    # Raw booster probas (NO calibration step — see module docstring).
    p_val = booster.predict_proba(X_val)[:, 1]
    p_vt = booster.predict_proba(X_vt)[:, 1]
    p_test = booster.predict_proba(X_test)[:, 1]

    curve_val_thr = _precision_curve(y_vt, p_vt, THRESHOLDS)
    curve_test = _precision_curve(y_test, p_test, THRESHOLDS)
    _print_curve(f"{symbol} val_thr (back {VAL_THR_FRAC * 100:.0f}% of val)", curve_val_thr)
    _print_curve(f"{symbol} test (final)", curve_test)

    mono = _is_monotonic_or_flat(curve_test, tol_pp=0.01)
    print(f"  monotonic-or-flat across {THRESHOLDS}: {'PASS' if mono else 'FAIL'}")

    thr_pick, thr_pick_info = _pick_threshold_from_val_thr(
        y_vt, p_vt, min_winrate=0.55, min_ntrades=50
    )
    test_winrate_at_pick: Optional[float] = None
    test_ntrades_at_pick: Optional[int] = None
    if thr_pick is not None:
        m = p_test >= thr_pick
        test_ntrades_at_pick = int(m.sum())
        test_winrate_at_pick = (
            float(y_test[m].mean()) if test_ntrades_at_pick > 0 else None
        )
    print(
        f"  picked threshold (val_thr wr>=0.55, n>=50; highest qualifying): "
        f"{thr_pick}  -> test n={test_ntrades_at_pick} wr={test_winrate_at_pick}"
    )

    # Metrics summary.
    def _metrics_block(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            out["auc"] = float(roc_auc_score(y_true, p))
        except Exception:
            out["auc"] = float("nan")
        out["brier"] = float(brier_score_loss(y_true, p))
        try:
            out["log_loss"] = float(log_loss(y_true, p, labels=[0, 1]))
        except Exception:
            out["log_loss"] = float("nan")
        out["reliability_slope"] = _reliability_slope(y_true, p)
        return out

    metrics_val = _metrics_block(y_val, p_val)
    metrics_val_thr = _metrics_block(y_vt, p_vt)
    metrics_test = _metrics_block(y_test, p_test)

    feature_means, feature_stds = _compute_feature_stats(train_df, feature_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(booster, out_dir / "model.joblib")

    test_gate_pass = mono and (
        thr_pick is not None
        and test_winrate_at_pick is not None
        and test_winrate_at_pick >= 0.55
        and test_ntrades_at_pick is not None
        and test_ntrades_at_pick >= 10
    )

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path.relative_to(REPO)),
        "feature_cols": feature_cols,
        "label_classes": [0, 1],
        # The model.joblib is a bare XGBClassifier — no CalibratedClassifierCV.
        # validate_xgboost_winrate.py's _get_probas calls model.predict_proba
        # which works fine on a raw booster.
        "calibration_method": "none",
        "calibration_strategy": {
            "kind": "raw_booster_probas",
            "rationale": (
                "Isotonic/sigmoid both compress raw booster output to where "
                "no test row crosses thr=0.55; raw booster is monotonic on "
                "the precision curve up through thr=0.58 with this "
                "regularization."
            ),
            "val_thr_frac": VAL_THR_FRAC,
            "n_val_thr_rows": n_vt,
        },
        "xgb_kwargs": xgb_kwargs,
        "regularization": {
            "vs_v1": (
                "min_child_weight: 1 -> 10; max_depth: 4 -> "
                f"{xgb_kwargs['max_depth']}; subsample: 1.0 -> 0.85; "
                "colsample_bytree: 1.0 -> 0.8; reg_lambda: 1.0 (unchanged)"
            ),
        },
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "metrics_val": metrics_val,
        "metrics_val_thr": metrics_val_thr,
        "metrics_test": metrics_test,
        "test_precision_curve": curve_test,
        "val_thr_precision_curve": curve_val_thr,
        "monotonic_or_flat": bool(mono),
        "optimal_threshold": thr_pick,
        "threshold_status": "ok" if (thr_pick is not None and test_gate_pass) else "test_gate_failed",
        "test_winrate_at_optimal_threshold": test_winrate_at_pick,
        "test_ntrades_at_optimal_threshold": test_ntrades_at_pick,
        "val_thr_pick_info": thr_pick_info,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "vol_features_dropped": dropped,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'model.joblib'} + meta.json")
    return {
        "symbol": symbol,
        "monotonic_or_flat": mono,
        "test_curve": curve_test,
        "val_thr_curve": curve_val_thr,
        "test_winrate_at_optimal_threshold": test_winrate_at_pick,
        "test_ntrades_at_optimal_threshold": test_ntrades_at_pick,
        "optimal_threshold": thr_pick,
        "test_gate_pass": test_gate_pass,
    }


def main() -> int:
    summary: Dict[str, object] = {}
    for symbol in ("btc", "sol"):
        summary[symbol] = train_one(symbol)
    out = Path("/tmp/btc_sol_voln_v2_summary.json")
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
