"""Rebuild the per-symbol regime memory stores against the current predictor.

The regime store's embedding dimension is `len(feature_cols) * 4`. If the
predictor's feature_cols change between training and the regime backfill
(e.g. the voln-normalized v2 rework dropped 15 features from 135 to 120),
the on-disk store's dim no longer matches the live encoder and every
predict raises `query embedding dim mismatch`. The predictor falls back to
the static threshold gracefully but the regime flywheel (Kelly sizing,
regime label, OutcomeAdjuster) goes dark.

This script reads each model's `meta.json` for the canonical feature_cols,
filters the source voln parquet down to those exact columns (in the same
order), and rebuilds the store via the existing `regime_memory.backfill.
build_store` entry point.

Run from repo root:

    PYTHONPATH=src ./.venv/bin/python scripts/rebuild_regime_stores.py

Wall-clock: ~2 minutes for all three symbols on a local machine. The
output paths overwrite the existing stores; the supervisor's
REGIME_STORE_PATH_<SAFE_SYMBOL> env vars don't change.

If you change the symbol set or rename a model directory, edit `TARGETS`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from regime_memory.backfill import build_store

REPO = Path(__file__).resolve().parent.parent

# (symbol_tag, model_dir_name, parquet_name, store_filename)
TARGETS = [
    ("ETH-USD", "eth_usd_voln_v2_blend09_alt", "eth_usd_voln.parquet", "eth_usd.npz"),
    ("BTC-USD", "btc_usd_voln_v2",             "btc_usd_voln.parquet", "btc_usd.npz"),
    ("SOL-USD", "sol_usd_voln_v2",             "sol_usd_voln.parquet", "sol_usd.npz"),
]


def main() -> None:
    for symbol_tag, model_dir, parquet_name, store_name in TARGETS:
        t0 = time.time()
        print(f"\n=== {symbol_tag} ({model_dir}) ===")

        meta_path = REPO / "model_crypto" / model_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta["feature_cols"]
        print(f"  model feature_cols: {len(feature_cols)}")

        parquet_path = REPO / "data" / "crypto" / "datasets" / parquet_name
        df = pd.read_parquet(parquet_path)
        print(f"  parquet shape: {df.shape}")

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise SystemExit(
                f"  ERROR: {len(missing)} feature_cols missing from parquet: "
                f"{missing[:5]}"
            )

        # Drop everything except the model's feature columns. Backfill picks
        # encoder columns from the dataframe's columns; restricting here is
        # what makes the store dim match the live predictor's encoder.
        df_filtered = df[feature_cols].copy()
        print(
            f"  filtered shape: {df_filtered.shape}, "
            f"expected_dim = {len(feature_cols) * 4}"
        )

        # window_bars must equal label_horizon_bars (the function expects
        # future_rets and future_signal arrays of the same length). The
        # 2026-05-11 backfill used the defaults (60 + 60) per its memory note.
        store = build_store(
            df_filtered,
            window_bars=60,
            label_horizon_bars=60,
            symbol=symbol_tag,
        )
        print(f"  built store: size = {len(store)}, dim = {store.dim}")

        out_path = REPO / "data" / "regime_stores" / store_name
        store.save(str(out_path))
        elapsed = time.time() - t0
        print(f"  saved -> {out_path} ({elapsed:.1f}s)")

    print("\nAll three regime stores rebuilt.")


if __name__ == "__main__":
    main()
