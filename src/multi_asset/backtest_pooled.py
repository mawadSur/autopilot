"""SHADOW backtest for a trained pooled LSTM — read-only, no orders.

Replays the held-out tail of the pooled dataset through a trained model and
reports, per-asset and pooled, the precision (win-rate) and coverage at the
model's blessed threshold. This is the honest "would this signal have been
right?" check on out-of-sample data — it never simulates fills or PnL beyond
the vol-normalized hit-rate the label encodes.

CLI::

    ./.venv/bin/python src/multi_asset/backtest_pooled.py \\
        --dataset data/pooled/pooled_1d.parquet \\
        --model model_multi/pooled_1d_v1/ \\
        --out runs/pooled_backtest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch

from multi_asset.model import PooledLSTMClassifier
from multi_asset.predictor import PooledLSTMPredictor
from multi_asset.sequences import build_sequences
from multi_asset.train_pooled_lstm import (
    _predict_proba,
    load_pooled,
    per_asset_winrate,
    time_split,
)


def backtest(
    *,
    dataset_path: Path,
    model_dir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    force_cpu: bool = True,
) -> dict:
    pred = PooledLSTMPredictor(str(model_dir), device="cpu" if force_cpu else None)
    df = load_pooled(Path(dataset_path))

    _, _, test_df = time_split(df, val_frac=val_frac, test_frac=test_frac)
    test_seq = build_sequences(
        test_df, feature_cols=pred.feature_cols, asset_to_idx=pred.asset_vocab,
        window=pred.window, gap_multiplier=float(pred.meta.get("gap_multiplier", 5.0)),
    )
    if len(test_seq) == 0:
        return {"error": "no test sequences", "n_test_seq": 0}

    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    proba = _predict_proba(pred.model, test_seq, pred.scaler, device)
    thr = pred.threshold if pred.threshold is not None else 0.5
    idx_to_asset = {v: k for k, v in pred.asset_vocab.items()}

    mask = proba >= thr
    n_sig = int(mask.sum())
    pooled_wr = float(test_seq.y[mask].mean()) if n_sig else 0.0
    report = {
        "model_dir": str(model_dir),
        "dataset": str(dataset_path),
        "threshold": thr,
        "threshold_status": pred.meta.get("threshold_status"),
        "base_rate": float(test_seq.y.mean()),
        "n_test_seq": len(test_seq),
        "pooled": {"win_rate": pooled_wr, "n_signals": n_sig, "coverage": n_sig / len(test_seq)},
        "per_asset": per_asset_winrate(test_seq.asset_idx, test_seq.y, proba, thr, idx_to_asset),
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SHADOW backtest a pooled LSTM (read-only).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--out", default=None, help="optional path to write the JSON report")
    args = p.parse_args(argv)

    report = backtest(
        dataset_path=Path(args.dataset), model_dir=Path(args.model),
        val_frac=args.val_frac, test_frac=args.test_frac,
    )
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
