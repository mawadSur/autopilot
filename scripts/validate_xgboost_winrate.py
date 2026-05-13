"""Parquet-native win-rate validation for XGBoost crypto models.

NOTE on label semantics: ``label = 1`` means "forward return over the
horizon exceeded the dataset's threshold_bps, NET of fees." In other
words, the label is *fee-aware*: a correctly labeled 1 already cleared
round-trip trade costs. So ``win_rate`` here == precision at threshold X
== the fraction of triggered trades where the forward return was
profitable enough to cover fees.  That is exactly the user-defined
"win-rate" target (65%).

Complements ``src/backtest.py`` (which reads 1m OHLC CSVs and can
simulate TP/SL) with a fast parquet-native path that works for all
three crypto models (BTC, ETH, SOL) using the same data the model was
trained on.  Does NOT touch any model files or training pipeline code.

Usage::

    ./.venv/bin/python scripts/validate_xgboost_winrate.py \\
        --model-dir model_crypto/btc_usd_v1 \\
        --split test \\
        --thresholds 0.50,0.55,0.60,0.65,0.70
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_SRC_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_meta(model_dir: Path) -> dict:
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        sys.exit(f"ERROR: meta.json not found at {meta_path}")
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_parquet(meta: dict) -> pd.DataFrame:
    raw_path = meta.get("dataset_path", "")
    dataset_path = _REPO_ROOT / raw_path
    if not dataset_path.exists():
        sys.exit(
            f"ERROR: dataset not found at {dataset_path}\n"
            f"(meta.dataset_path = {raw_path!r})"
        )
    df = pd.read_parquet(dataset_path)
    if "timestamp" not in df.columns:
        sys.exit(f"ERROR: parquet at {dataset_path} missing 'timestamp' column")
    if "label" not in df.columns:
        sys.exit(f"ERROR: parquet at {dataset_path} missing 'label' column")
    return df.sort_values("timestamp").reset_index(drop=True)


def _time_based_split(
    df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Matches train_xgboost._time_based_split exactly."""
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train: n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def _select_split(
    df: pd.DataFrame, split: str, meta: dict
) -> pd.DataFrame:
    train_df, val_df, test_df = _time_based_split(df)
    split_map = {"train": train_df, "val": val_df, "test": test_df, "all": df}
    if split not in split_map:
        sys.exit(f"ERROR: --split must be one of train|val|test|all; got {split!r}")

    result = split_map[split]
    expected_key = f"rows_{split}" if split != "all" else None
    if expected_key and expected_key in meta:
        expected_rows = meta[expected_key]
        if len(result) != expected_rows:
            print(
                f"WARNING: {split} split has {len(result)} rows but "
                f"meta.{expected_key}={expected_rows}. "
                "Dataset may have changed since training."
            )
    return result


# ---------------------------------------------------------------------------
# Feature + model helpers
# ---------------------------------------------------------------------------


def _validate_features(df: pd.DataFrame, feature_cols: List[str]) -> None:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} feature column(s) missing from parquet: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )


def _load_model_and_scaler(model_dir: Path) -> Tuple[object, Optional[object]]:
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        sys.exit(f"ERROR: model.joblib not found at {model_path}")
    model = joblib.load(model_path)

    scaler_path = model_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, scaler


def _get_probas(
    model: object,
    df_split: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[object],
) -> np.ndarray:
    X = df_split[feature_cols].to_numpy(dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Win-rate computation
# ---------------------------------------------------------------------------


def _compute_win_rate_table(
    y_true: np.ndarray,
    p_long: np.ndarray,
    thresholds: List[float],
    min_n: int = 10,
) -> List[Dict]:
    rows = []
    for thr in sorted(thresholds):
        mask = p_long >= thr
        n_trades = int(mask.sum())
        if n_trades == 0:
            win_rate: Optional[float] = None
            meets_65: Optional[bool] = None
        else:
            win_rate = float(y_true[mask].mean())
            meets_65 = (win_rate >= 0.65) and (n_trades >= min_n)
        rows.append(
            {
                "threshold": thr,
                "n_trades": n_trades,
                "win_rate": win_rate,
                "meets_65_target": meets_65,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_table(rows: List[Dict], model_name: str, split: str) -> None:
    print(f"\nModel: {model_name}  |  Split: {split}")
    header = f"{'thr':>6}  {'n_trades':>10}  {'win_rate':>10}  {'meets_65?':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        wr = f"{r['win_rate']:.4f}" if r["win_rate"] is not None else "      n/a"
        m65 = str(r["meets_65_target"]) if r["meets_65_target"] is not None else "    n/a"
        print(f"{r['threshold']:>6.2f}  {r['n_trades']:>10}  {wr:>10}  {m65:>10}")


def _write_json(
    rows: List[Dict], model_name: str, split: str, out_dir: Path
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"xgb_validation_{model_name}_{split}_{ts}.json"
    out_path = out_dir / filename
    payload = {"model": model_name, "split": split, "generated_utc": ts, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _default_thresholds() -> List[float]:
    return [round(x, 2) for x in np.arange(0.30, 0.81, 0.05).tolist()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_xgboost_winrate",
        description="Parquet-native win-rate validation for XGBoost crypto models.",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model dir (e.g. model_crypto/btc_usd_v1)",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to evaluate (default: test)",
    )
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated floats, e.g. 0.50,0.55,0.60 (default: 0.30-0.80 step 0.05)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for JSON output. Defaults to runs/ if it exists + is "
            "writable, else /tmp."
        ),
    )
    return p.parse_args(argv)


def _resolve_out_dir(cli_out: Optional[Path]) -> Path:
    if cli_out is not None:
        cli_out.mkdir(parents=True, exist_ok=True)
        return cli_out
    runs_dir = _REPO_ROOT / "runs"
    if runs_dir.exists() and os.access(runs_dir, os.W_OK):
        return runs_dir
    return Path("/tmp")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    model_dir = (
        args.model_dir
        if args.model_dir.is_absolute()
        else _REPO_ROOT / args.model_dir
    )
    model_name = model_dir.name

    thresholds = (
        [float(t.strip()) for t in args.thresholds.split(",")]
        if args.thresholds
        else _default_thresholds()
    )

    meta = _load_meta(model_dir)
    label_classes = meta.get("label_classes", [0, 1])
    if len(label_classes) != 2:
        sys.exit("ERROR: multi-class model; win-rate validation is binary-only.")

    feature_cols: List[str] = meta["feature_cols"]
    df_full = _load_parquet(meta)
    _validate_features(df_full, feature_cols)

    df_split = _select_split(df_full, args.split, meta)

    model, scaler = _load_model_and_scaler(model_dir)
    p_long = _get_probas(model, df_split, feature_cols, scaler)
    y_true = df_split["label"].astype(int).to_numpy()

    rows = _compute_win_rate_table(y_true, p_long, thresholds)
    _print_table(rows, model_name, args.split)

    out_dir = _resolve_out_dir(args.out_dir)
    out_path = _write_json(rows, model_name, args.split, out_dir)
    print(f"\nJSON written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
