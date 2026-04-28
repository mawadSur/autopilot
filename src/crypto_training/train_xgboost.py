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
