"""One-shot script: re-run threshold sweep on already-trained eth_usd_v1.

Loads model_crypto/eth_usd_v1/meta.json + model.joblib, re-derives the
same val/test split the trainer used (temporal, val_frac=0.15,
test_frac=0.15), runs _sweep_thresholds_for_sharpe on the val set, and
writes threshold_metrics + optimal_threshold back into meta.json.

Does NOT retrain the model. Does NOT touch src/crypto_training/.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Allow importing from src/ flat (same pattern as train_xgboost.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_SRC_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib
import numpy as np
import pandas as pd

# Import the sweep + split helpers from the trainer without calling train().
from crypto_training.train_xgboost import (
    _sweep_thresholds_for_sharpe,
    _time_based_split,
)

MODEL_DIR = _REPO_ROOT / "model_crypto" / "eth_usd_v1"
META_PATH = MODEL_DIR / "meta.json"
MODEL_PATH = MODEL_DIR / "model.joblib"

# Match the trainer defaults exactly.
VAL_FRAC = 0.15
TEST_FRAC = 0.15
FEE_BPS = 200.0


def _load_meta() -> dict:
    with META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _load_dataset(meta: dict) -> pd.DataFrame:
    dataset_path = _REPO_ROOT / meta["dataset_path"]
    if not dataset_path.exists():
        sys.exit(
            f"ERROR: dataset not found at {dataset_path}\n"
            "Cannot derive val split without the original parquet. Aborting."
        )
    df = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _atomic_write_meta(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _print_summary_table(threshold_metrics: dict) -> None:
    print(f"\n{'thr':>6}  {'win_rate':>10}  {'n_trades':>10}  {'sharpe':>10}")
    print("-" * 44)
    for thr_str in sorted(threshold_metrics):
        thr = float(thr_str)
        if thr < 0.44 or thr > 0.76:
            continue
        m = threshold_metrics[thr_str]
        print(
            f"{thr:>6.2f}  {m['win_rate']:>10.4f}  {int(m['n_trades']):>10}  {m['sharpe']:>10.4f}"
        )


def main() -> None:
    meta = _load_meta()

    if meta.get("optimal_threshold") is not None:
        print(f"NOTE: meta.json already has optimal_threshold={meta['optimal_threshold']}; re-sweeping anyway.")

    df = _load_dataset(meta)
    feature_cols = meta["feature_cols"]

    # Validate all feature columns are present.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {len(missing)} feature columns missing from parquet: {missing[:10]}")

    _, val_df, _ = _time_based_split(df, val_frac=VAL_FRAC, test_frac=TEST_FRAC)

    # Sanity-check row counts match meta.
    if len(val_df) != meta.get("rows_val", len(val_df)):
        print(
            f"WARNING: val split has {len(val_df)} rows but meta.rows_val={meta.get('rows_val')}. "
            "Proceeding — dataset may have grown since training."
        )

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["label"].astype(int).to_numpy()

    model = joblib.load(MODEL_PATH)
    label_classes = meta.get("label_classes", [0, 1])
    if len(label_classes) != 2:
        sys.exit("ERROR: multi-class model; threshold sweep is binary-only. Aborting.")

    proba_val = model.predict_proba(X_val)[:, 1]

    optimal_threshold, threshold_metrics, threshold_status = _sweep_thresholds_for_sharpe(
        y_val, proba_val, fee_bps=FEE_BPS
    )

    print(f"Sweep done: optimal_threshold={optimal_threshold:.4f}  status={threshold_status}")
    _print_summary_table(threshold_metrics)

    # Patch meta.json in-place, preserving all other fields.
    meta["optimal_threshold"] = optimal_threshold
    meta["threshold_metrics"] = threshold_metrics
    meta["threshold_status"] = threshold_status
    _atomic_write_meta(META_PATH, meta)
    print(f"\nPatched {META_PATH}")

    # Verification read-back.
    patched = _load_meta()
    assert isinstance(patched["optimal_threshold"], float), "optimal_threshold not a float after patch"
    assert isinstance(patched["threshold_metrics"], dict), "threshold_metrics not a dict after patch"
    first_key = next(iter(patched["threshold_metrics"]))
    assert "sim_win_rate" in patched["threshold_metrics"][first_key] or "win_rate" in patched["threshold_metrics"][first_key], \
        "threshold_metrics entry missing win_rate"
    print("Verification: optimal_threshold and threshold_metrics confirmed in patched meta.json")


if __name__ == "__main__":
    main()
