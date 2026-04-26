"""Train the XGBoost calibration baseline.

Loads a dataset assembled by :mod:`calibration_agent.build_dataset`, performs a
time-based train/val/test split (sorted by ``captured_at_utc``), fits an
``xgboost.XGBClassifier`` against :data:`ALL_FEATURE_COLUMNS`, wraps it in
``CalibratedClassifierCV(cv="prefit")`` fit on the validation set, and persists
the calibrated model via ``joblib.dump`` alongside a ``.meta.json`` sidecar.
"""

from __future__ import annotations

import os

# xgboost ships its own libomp; sklearn / numpy may already have loaded a
# different OpenMP runtime in the same process. Set both env vars BEFORE any
# numerical import so libomp is initialised exactly once. Required on macOS
# under unittest discover; harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union

# Mirror the sys.path shim used by build_dataset.py so this CLI runs
# without the caller setting PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
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

from calibration_agent.ml_service import ALL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
CalibrationMethod = Literal["isotonic", "sigmoid"]

DEFAULT_XGB_KWARGS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    # Single-threaded by default. Avoids an OMP/Apple-libc init collision
    # when sklearn's CalibratedClassifierCV later wraps the booster on macOS.
    # Override via xgb_kwargs={"n_jobs": -1} for production training.
    "n_jobs": 1,
}

MIN_TOTAL_ROWS = 10


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load a dataset from Parquet (default) or CSV based on file suffix."""

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    return pd.read_parquet(dataset_path)


def _validate_dataset(df: pd.DataFrame) -> None:
    """Sanity-check the dataset before splitting."""

    if len(df) < MIN_TOTAL_ROWS:
        raise ValueError(
            f"Dataset must contain at least {MIN_TOTAL_ROWS} rows; got {len(df)}."
        )
    missing_features = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataset missing required feature columns: {missing_features}"
        )
    if "market_outcome" not in df.columns:
        raise ValueError("Dataset missing required 'market_outcome' column.")
    if "captured_at_utc" not in df.columns:
        raise ValueError("Dataset missing required 'captured_at_utc' column.")


def _time_based_split(
    df: pd.DataFrame, val_fraction: float, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by ``captured_at_utc`` ascending, then carve out train/val/test.

    Oldest rows train, then val, then most-recent rows are held out for test.
    """

    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be positive.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction must be < 1.0 to leave room for train."
        )

    sorted_df = df.sort_values("captured_at_utc", kind="mergesort").reset_index(
        drop=True
    )
    n = len(sorted_df)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Time-based split produced empty train set (n={n}, val={n_val}, "
            f"test={n_test}). Lower val_fraction/test_fraction."
        )
    train_df = sorted_df.iloc[:n_train].copy()
    val_df = sorted_df.iloc[n_train : n_train + n_val].copy()
    test_df = sorted_df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _compute_metrics(y_true: pd.Series, y_proba: Any) -> Dict[str, float]:
    """Compute Brier, log-loss, accuracy, and AUC against binary labels."""

    y_true_arr = y_true.astype(int).to_numpy()
    metrics: Dict[str, float] = {
        "brier": float(brier_score_loss(y_true_arr, y_proba)),
        "log_loss": float(log_loss(y_true_arr, y_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true_arr, (y_proba >= 0.5).astype(int))),
    }
    # AUC is undefined when the test set has only one class.
    if len(set(y_true_arr.tolist())) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_proba))
    else:
        metrics["auc"] = float("nan")
    return metrics


def train(
    dataset_path: PathLike,
    *,
    output_path: PathLike,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    calibration_method: CalibrationMethod = "isotonic",
    random_state: int = 42,
    **xgb_kwargs: Any,
) -> Dict[str, Any]:
    """Train an XGBoost + calibrated baseline and persist it via joblib."""

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    df = _load_dataset(dataset_path)
    _validate_dataset(df)

    train_df, val_df, test_df = _time_based_split(df, val_fraction, test_fraction)

    feature_cols = list(ALL_FEATURE_COLUMNS)
    x_train = train_df[feature_cols].astype(float)
    y_train = train_df["market_outcome"].astype(int)
    x_val = val_df[feature_cols].astype(float)
    y_val = val_df["market_outcome"].astype(int)
    x_test = test_df[feature_cols].astype(float)
    y_test = test_df["market_outcome"].astype(int)

    merged_xgb_kwargs: Dict[str, Any] = {**DEFAULT_XGB_KWARGS, **xgb_kwargs}
    merged_xgb_kwargs.setdefault("random_state", random_state)

    base_model = xgb.XGBClassifier(**merged_xgb_kwargs)
    base_model.fit(x_train, y_train)

    # Calibration requires both classes in y_val for isotonic to be meaningful;
    # if only one class is present, fall back to sigmoid which degrades more
    # gracefully and surface a warning so operators notice.
    effective_method: CalibrationMethod = calibration_method
    if len(set(y_val.tolist())) < 2 and calibration_method == "isotonic":
        logger.warning(
            "Validation set is single-class; falling back to sigmoid calibration."
        )
        effective_method = "sigmoid"

    # sklearn 1.6 deprecated ``cv="prefit"`` in favour of wrapping the trained
    # estimator in :class:`FrozenEstimator` first, then passing it to
    # ``CalibratedClassifierCV`` (which now treats a frozen estimator as a
    # no-fit base learner and only fits the calibrator on the val split).
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base_model), method=effective_method
    )
    calibrated.fit(x_val, y_val)

    test_proba = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_proba)

    baseline_proba = test_df["implied_prob"].astype(float).to_numpy()
    baseline_metrics = _compute_metrics(y_test, baseline_proba)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    captured = pd.to_datetime(df["captured_at_utc"], utc=True, errors="coerce")
    training_range = {
        "min_captured_at_utc": (
            captured.min().isoformat() if pd.notna(captured.min()) else None
        ),
        "max_captured_at_utc": (
            captured.max().isoformat() if pd.notna(captured.max()) else None
        ),
    }

    metadata: Dict[str, Any] = {
        "model_path": str(output_path),
        "meta_path": str(meta_path),
        "dataset_path": str(dataset_path),
        "training_range": training_range,
        "n_samples": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "total": int(len(df)),
        },
        "calibration_method": effective_method,
        "requested_calibration_method": calibration_method,
        "xgb_kwargs": merged_xgb_kwargs,
        "random_state": random_state,
        "feature_columns": feature_cols,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return metadata


def _parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the XGBoost calibration baseline against a dataset assembled "
            "by calibration_agent.build_dataset."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the Parquet (.parquet) or CSV (.csv) training dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the joblib-serialised calibrated model.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--calibration",
        choices=("isotonic", "sigmoid"),
        default="isotonic",
        help="Calibration method passed to CalibratedClassifierCV.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    args = _parse_args(argv)
    summary = train(
        args.dataset_path,
        output_path=args.output,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        calibration_method=args.calibration,
        random_state=args.random_state,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
