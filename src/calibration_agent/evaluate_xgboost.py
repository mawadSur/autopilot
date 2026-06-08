"""Evaluate a saved XGBoost calibration baseline against quality thresholds.

Loads a joblib-serialised model produced by :mod:`calibration_agent.train_xgboost`
and a labelled dataset, computes Brier / log-loss / accuracy / AUC / reliability
slope for both the model and the implied-prob baseline, then renders a pass/fail
verdict against fixed quality gates.
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
from typing import Any, Dict, Union

# Mirror the sys.path shim used by build_dataset.py so this CLI runs
# without the caller setting PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from calibration_agent.ml_service import ALL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Quality gate thresholds (echoed by the CLI/report).
BRIER_MAX = 0.20
LOGLOSS_MAX = 0.55
SLOPE_MIN = 0.8
SLOPE_MAX = 1.2
N_RELIABILITY_BINS = 10


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load a dataset from Parquet (default) or CSV based on file suffix."""

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    return pd.read_parquet(dataset_path)


def _reliability_slope(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Bin into deciles by predicted prob, fit a line through (mean_pred, mean_actual).

    Slope=1 indicates perfect calibration. Returns ``nan`` if fewer than two
    distinct non-empty bins are produced (insufficient signal to fit a line).
    """

    if len(y_proba) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, N_RELIABILITY_BINS + 1)
    # ``np.digitize`` with right=False maps probs into bins 1..N; clamp the
    # last edge so probabilities of exactly 1.0 fall into the final bin.
    bin_idx = np.clip(
        np.digitize(y_proba, bin_edges[1:-1], right=False), 0, N_RELIABILITY_BINS - 1
    )
    mean_preds: list[float] = []
    mean_actuals: list[float] = []
    for b in range(N_RELIABILITY_BINS):
        mask = bin_idx == b
        if not mask.any():
            continue
        mean_preds.append(float(y_proba[mask].mean()))
        mean_actuals.append(float(y_true[mask].mean()))
    if len(mean_preds) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(mean_preds, mean_actuals, deg=1)
    return float(slope)


def _compute_full_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute Brier, log-loss, accuracy, AUC, and reliability slope."""

    y_true_arr = y_true.astype(int).to_numpy()
    metrics: Dict[str, float] = {
        "brier": float(brier_score_loss(y_true_arr, y_proba)),
        "log_loss": float(log_loss(y_true_arr, y_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true_arr, (y_proba >= 0.5).astype(int))),
    }
    if len(set(y_true_arr.tolist())) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_proba))
    else:
        metrics["auc"] = float("nan")
    metrics["reliability_slope"] = _reliability_slope(y_true_arr, np.asarray(y_proba))
    return metrics


def _evaluate_criteria(
    model_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
) -> Dict[str, bool]:
    """Apply pass/fail thresholds to the computed metrics."""

    slope = model_metrics.get("reliability_slope", float("nan"))
    slope_in_band = (
        not np.isnan(slope) and SLOPE_MIN <= slope <= SLOPE_MAX
    )
    return {
        "brier_lt_0_20": bool(model_metrics["brier"] < BRIER_MAX),
        "logloss_lt_0_55": bool(model_metrics["log_loss"] < LOGLOSS_MAX),
        "slope_in_band": bool(slope_in_band),
        "beats_baseline_brier": bool(
            model_metrics["brier"] < baseline_metrics["brier"]
        ),
    }


def evaluate(model_path: PathLike, dataset_path: PathLike) -> Dict[str, Any]:
    """Evaluate a saved model against the dataset's labels and the implied-prob baseline."""

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = _load_dataset(dataset_path)
    missing = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")
    if "market_outcome" not in df.columns or "implied_prob" not in df.columns:
        raise ValueError(
            "Dataset must contain 'market_outcome' and 'implied_prob' columns."
        )

    model = joblib.load(model_path)

    x = df[list(ALL_FEATURE_COLUMNS)].astype(float)
    y = df["market_outcome"].astype(int)
    proba_full = model.predict_proba(x)
    # Binary classifiers expose two columns; positive class is index 1.
    y_proba = (
        np.asarray(proba_full)[:, 1]
        if np.ndim(proba_full) == 2 and np.shape(proba_full)[1] >= 2
        else np.asarray(proba_full).ravel()
    )

    baseline_proba = df["implied_prob"].astype(float).to_numpy()

    model_metrics = _compute_full_metrics(y, y_proba)
    baseline_metrics = _compute_full_metrics(y, baseline_proba)
    criteria = _evaluate_criteria(model_metrics, baseline_metrics)

    return {
        "pass": all(criteria.values()),
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "criteria": criteria,
        "thresholds": {
            "brier_max": BRIER_MAX,
            "logloss_max": LOGLOSS_MAX,
            "slope_min": SLOPE_MIN,
            "slope_max": SLOPE_MAX,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
    }


def _format_verdict(result: Dict[str, Any]) -> str:
    verdict = "PASS" if result["pass"] else "FAIL"
    lines = [f"Verdict: {verdict}"]
    lines.append("Model metrics:")
    for k, v in result["model_metrics"].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("Baseline (implied_prob) metrics:")
    for k, v in result["baseline_metrics"].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("Pass criteria:")
    for k, v in result["criteria"].items():
        flag = "ok" if v else "FAIL"
        lines.append(f"  [{flag}] {k}")
    return "\n".join(lines)


def _parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved XGBoost calibration baseline against the "
            "implied-prob baseline and pass/fail thresholds."
        )
    )
    parser.add_argument("model_path", type=Path, help="Path to a joblib-saved model.")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to a labelled Parquet/CSV dataset for evaluation.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path; when given, the verdict dict is written as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    args = _parse_args(argv)
    result = evaluate(args.model_path, args.dataset_path)
    print(_format_verdict(result))
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
