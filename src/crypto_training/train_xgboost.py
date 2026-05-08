"""Train an XGBoost calibration model on a crypto OHLCV dataset.

Loads a parquet/csv produced by ``build_dataset.py``, does a time-based
train/val/test split (sorted by ``timestamp``), fits an
``XGBClassifier`` on train, then wraps the booster in
``CalibratedClassifierCV(method="isotonic", cv="prefit")`` fit on the
validation slice. Persists the calibrated model + a ``.meta.json``
sidecar that the supervisor's predictor adapter can use later.

CLI::

    ./.venv/bin/python src/crypto_training/train_xgboost.py \\
        --dataset data/crypto/datasets/eth_usd.parquet \\
        --out model_crypto/eth_usd_v1/

Quality gates printed at the end (informational, not enforced):
  * brier  -- calibration sharpness; lower is better
  * log_loss
  * accuracy
  * AUC
  * reliability slope (1.0 = perfectly calibrated)
"""

from __future__ import annotations

import os

# Same libomp dance as the prediction-market trainer.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# sys.path shim so the CLI runs without PYTHONPATH=src.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


DEFAULT_XGB_KWARGS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    # Single-thread XGBoost so sklearn's CalibratedClassifierCV doesn't
    # reinitialise libomp on macOS.
    "n_jobs": 1,
    "tree_method": "hist",
}


@dataclass
class TrainSummary:
    """Returned by ``train``. Useful for tests + the CLI report."""

    rows_train: int
    rows_val: int
    rows_test: int
    feature_count: int
    metrics_test: Dict[str, float]
    metrics_val: Dict[str, float]
    output_dir: Path
    model_filename: str = "model.joblib"
    meta_filename: str = "meta.json"


def _load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset (parquet or csv) and validate basic schema."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Dataset {path} missing required 'timestamp' column")
    if "label" not in df.columns:
        raise ValueError(f"Dataset {path} missing required 'label' column")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _time_based_split(
    df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sequentially split sorted-by-time data into train/val/test."""
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Not enough rows ({n}) for val_frac={val_frac} + test_frac={test_frac}"
        )
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def walk_forward_cv(
    df: pd.DataFrame,
    *,
    window_size: int = 1000,
    step_size: int = 200,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Anchored rolling walk-forward folds.

    Each fold's train slice is ``df[0:val_start]`` and val slice is
    ``df[val_start:val_end]`` where ``val_start`` advances by
    ``step_size`` between folds. ``window_size`` sets the minimum
    training history before the first fold's val begins, and
    ``step_size`` bounds the per-fold validation length.

    **Invariants** (covered by tests):

      * ``min(val.timestamp) > max(train.timestamp)`` for every fold
        (no leakage; train ends strictly before val begins),
      * val slices across folds don't overlap.

    Returns ``[]`` if the dataset has fewer than ``window_size + 1``
    rows -- caller should fall back to single-split ``train()``.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError(
            f"walk_forward_cv: window_size and step_size must be positive "
            f"(got window_size={window_size}, step_size={step_size})"
        )
    n = len(df)
    if n < window_size + 1:
        return []

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    val_start = window_size
    while val_start < n:
        val_end = min(val_start + step_size, n)
        train_slice = df.iloc[:val_start].copy()
        val_slice = df.iloc[val_start:val_end].copy()
        if len(val_slice) == 0:
            break
        folds.append((train_slice, val_slice))
        val_start += step_size
    return folds


@dataclass
class WalkForwardSummary:
    """Aggregated metrics across all walk-forward folds.

    ``per_fold`` is the per-fold metrics dict (one entry per fold);
    ``mean_metrics`` is a flat ``{metric_name: mean_value}`` dict over
    folds.
    """

    n_folds: int
    per_fold: List[Dict[str, Any]]
    mean_metrics: Dict[str, float]
    feature_count: int
    rows_total: int


def _reliability_slope(y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10) -> float:
    """Slope of (mean_pred, observed_freq) regression. 1.0 = perfectly calibrated.

    Useful complement to brier: a low brier with very flat predictions can
    look good in aggregate but be useless for thresholding. Slope catches
    that.
    """
    if len(y_true) == 0 or len(np.unique(y_prob)) < 2:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, bins - 1)
    means_pred = []
    freqs_obs = []
    for b in range(bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        means_pred.append(float(y_prob[mask].mean()))
        freqs_obs.append(float(y_true[mask].mean()))
    if len(means_pred) < 2:
        return float("nan")
    x = np.asarray(means_pred, dtype=np.float64)
    y = np.asarray(freqs_obs, dtype=np.float64)
    # Slope of best-fit line through origin would be biased; use full OLS.
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, *, label_classes: List[int]
) -> Dict[str, float]:
    """Compute calibration + classification metrics."""
    out: Dict[str, float] = {}
    if len(label_classes) == 2:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["log_loss"] = float(log_loss(y_true, y_prob, labels=label_classes))
    y_pred = (y_prob >= 0.5).astype(int) if len(label_classes) == 2 else np.argmax(
        y_prob.reshape(-1, len(label_classes)), axis=1
    )
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    if len(label_classes) == 2 and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["auc"] = float("nan")
    if len(label_classes) == 2:
        out["reliability_slope"] = _reliability_slope(y_true, y_prob)
    return out


def train(
    *,
    dataset_path: Path,
    output_dir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic",
    xgb_kwargs: Optional[Dict[str, Any]] = None,
) -> TrainSummary:
    """Train + calibrate + persist."""
    df = _load_dataset(dataset_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    label_classes = sorted({int(v) for v in df["label"].unique()})
    if len(label_classes) < 2:
        raise ValueError(
            f"Dataset has only one label class ({label_classes}); cannot train."
        )

    train_df, val_df, test_df = _time_based_split(
        df, val_frac=val_frac, test_frac=test_frac
    )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["label"].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].astype(int).to_numpy()

    LOGGER.info(
        "fitting XGBClassifier on %d rows (val=%d, test=%d, features=%d, classes=%s)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(feature_cols),
        label_classes,
    )
    booster_kwargs = dict(DEFAULT_XGB_KWARGS)
    if xgb_kwargs:
        booster_kwargs.update(xgb_kwargs)
    booster = xgb.XGBClassifier(**booster_kwargs)
    booster.fit(X_train, y_train)

    LOGGER.info("calibrating with method=%s on val set", calibration_method)
    # sklearn 1.6 deprecated ``cv="prefit"`` -- wrap the booster in
    # FrozenEstimator instead so calibration just fits the calibration
    # curve on the val set without retraining the underlying booster.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(booster), method=calibration_method
    )
    calibrated.fit(X_val, y_val)

    # Eval on val + test.
    if len(label_classes) == 2:
        prob_val = calibrated.predict_proba(X_val)[:, 1]
        prob_test = calibrated.predict_proba(X_test)[:, 1]
    else:
        # Multi-class: pass full probability vector to log_loss; argmax for
        # accuracy. AUC + brier + reliability are skipped for multi-class.
        prob_val = calibrated.predict_proba(X_val)
        prob_test = calibrated.predict_proba(X_test)

    metrics_val = _evaluate(y_val, prob_val, label_classes=label_classes)
    metrics_test = _evaluate(y_test, prob_test, label_classes=label_classes)

    # Persist.
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    meta_path = output_dir / "meta.json"
    joblib.dump(calibrated, model_path)

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "feature_cols": feature_cols,
        "label_classes": label_classes,
        "calibration_method": calibration_method,
        "xgb_kwargs": booster_kwargs,
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = TrainSummary(
        rows_train=len(train_df),
        rows_val=len(val_df),
        rows_test=len(test_df),
        feature_count=len(feature_cols),
        metrics_test=metrics_test,
        metrics_val=metrics_val,
        output_dir=output_dir,
    )
    LOGGER.info(
        "saved model to %s\nval=%s\ntest=%s",
        output_dir,
        metrics_val,
        metrics_test,
    )
    return summary


def train_with_walk_forward_cv(
    df: pd.DataFrame,
    *,
    window_size: int = 1000,
    step_size: int = 200,
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic",
    xgb_kwargs: Optional[Dict[str, Any]] = None,
) -> WalkForwardSummary:
    """Walk-forward CV: fit + calibrate on each fold, aggregate metrics.

    A single train/val/test split can hide regime-dependent overfitting
    (model looks great on the val slice but only because val happens to
    be a calm period). Walk-forward folds let the operator see metric
    variance across rolling time slices.

    If ``walk_forward_cv`` returns no folds (dataset too small), raises
    so the caller knows to fall back to ``train()``.
    """
    if "timestamp" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "walk-forward CV requires 'timestamp' and 'label' columns"
        )
    df = df.sort_values("timestamp").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    label_classes = sorted({int(v) for v in df["label"].unique()})
    if len(label_classes) < 2:
        raise ValueError(
            f"Walk-forward CV requires at least 2 label classes; got {label_classes}"
        )

    folds = walk_forward_cv(df, window_size=window_size, step_size=step_size)
    if not folds:
        raise ValueError(
            f"walk_forward_cv produced no folds for n={len(df)} "
            f"window_size={window_size}; fall back to single-split train()"
        )

    booster_kwargs = dict(DEFAULT_XGB_KWARGS)
    if xgb_kwargs:
        booster_kwargs.update(xgb_kwargs)

    per_fold: List[Dict[str, Any]] = []
    for i, (train_df, val_df) in enumerate(folds):
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].astype(int).to_numpy()
        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = val_df["label"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            # Degenerate fold (one class only). Skip rather than crash --
            # the booster's predict_proba is meaningless there.
            LOGGER.warning(
                "walk-forward fold %d has single-class slice; skipping", i
            )
            continue
        booster = xgb.XGBClassifier(**booster_kwargs)
        booster.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(booster), method=calibration_method
        )
        calibrated.fit(X_val, y_val)
        if len(label_classes) == 2:
            prob_val = calibrated.predict_proba(X_val)[:, 1]
        else:
            prob_val = calibrated.predict_proba(X_val)
        metrics: Dict[str, Any] = _evaluate(
            y_val, prob_val, label_classes=label_classes
        )
        metrics["fold_index"] = i
        metrics["rows_train"] = int(len(train_df))
        metrics["rows_val"] = int(len(val_df))
        # Recorded as strings so they survive JSON round-trips.
        metrics["train_first_ts"] = str(train_df["timestamp"].iloc[0])
        metrics["train_last_ts"] = str(train_df["timestamp"].iloc[-1])
        metrics["val_first_ts"] = str(val_df["timestamp"].iloc[0])
        metrics["val_last_ts"] = str(val_df["timestamp"].iloc[-1])
        per_fold.append(metrics)

    if not per_fold:
        raise ValueError(
            "All walk-forward folds were single-class; cannot aggregate"
        )

    # Aggregate numeric metrics by mean across folds.
    skip_for_mean = {"fold_index", "rows_train", "rows_val"}
    numeric_keys: List[str] = []
    for k, v in per_fold[0].items():
        if (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and k not in skip_for_mean
        ):
            numeric_keys.append(k)
    mean_metrics: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [
            float(fold[k])
            for fold in per_fold
            if k in fold and np.isfinite(fold[k])
        ]
        mean_metrics[k] = float(np.mean(vals)) if vals else float("nan")

    return WalkForwardSummary(
        n_folds=len(per_fold),
        per_fold=per_fold,
        mean_metrics=mean_metrics,
        feature_count=len(feature_cols),
        rows_total=len(df),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_xgboost",
        description="Train + calibrate an XGBoost model on a crypto dataset.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to parquet/csv produced by build_dataset.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write model.joblib + meta.json",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of dataset for validation/calibration (default 0.15)",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of dataset for test (default 0.15)",
    )
    p.add_argument(
        "--calibration",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method (default isotonic)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    summary = train(
        dataset_path=args.dataset,
        output_dir=args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        calibration_method=args.calibration,
    )
    print(json.dumps(asdict(summary), default=str, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
