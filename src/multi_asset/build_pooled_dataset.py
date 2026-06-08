"""Assemble a POOLED, scale-invariant, leakage-safe training table.

One row per (timestamp, asset). Columns::

    timestamp, asset_id, asset_class, <SCALE_INVARIANT_FEATURES...>, label

Why this shape defends the pooled model from scale leakage
----------------------------------------------------------
* **Features**: only :data:`SCALE_INVARIANT_FEATURES` survive — returns, ratios,
  z-scores, normalized volatility, cyclical time. Every one is dimensionless or
  price-relative, so BTC at $60k and a $40 stock produce comparable values. The
  raw price/size/EMA/MACD columns (which would let the net read the price tag
  off the magnitude) are excluded, as are the all-zero L2/microstructure columns
  that candle-only feeds never populate.
* **Label**: ``label_kind="vol_normalized"`` — "did the forward return clear
  ``k * ATR%``?" — so "profitable" means the same number of volatility units on
  every asset, not "moved a lot of dollars."
* **asset_id**: identity is carried explicitly (consumed by the model's learned
  embedding), not smuggled through feature scale.

CLI::

    ./.venv/bin/python src/multi_asset/build_pooled_dataset.py \\
        --granularity 1d --horizon 1 --out data/pooled/pooled_1d.parquet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from utils import FEATURE_COLUMNS_PROFITABLE, compute_features
from crypto_training.build_dataset import label_forward_return_binary
from multi_asset.sources import OHLCV_COLUMNS
from multi_asset.universe import DEFAULT_UNIVERSE, Instrument, Universe

# Curated dimensionless / price-relative subset of FEATURE_COLUMNS_PROFITABLE.
# Verified scale-free against utils.compute_features definitions:
#   ema_spread_* = (ema_a - ema_b)/close ; *_to_prev_close = price ratio - 1 ;
#   dist_to_golden_pocket = dist/ATR ; close_over_* = price ratio ; adx in 0..100.
SCALE_INVARIANT_FEATURES: List[str] = [
    # returns
    "return_1", "return_5", "return_15", "log_ret", "zret_20", "zret_60",
    # candle geometry (ratios in ~[0,1])
    "range_pct", "body_to_range_ratio", "body_to_range", "abs_body_to_range",
    "upper_wick_ratio", "lower_wick_ratio", "close_pos_in_range",
    # trend structure (price-relative)
    "ema_spread_9_21", "ema_spread_21_50",
    "close_over_ema_9", "close_over_ema_21", "close_over_ema_50", "close_over_vwap_50",
    "price_z_60", "price_z_240", "price_pos_donchian20",
    # normalized volatility
    "atrp_14", "ret_std_30", "bb_width_20", "bb_pctb_20", "vol_z_20",
    "rv_5", "rv_15", "rv_60", "rv_240", "vol_of_vol_60",
    # gaps / micro-momentum (ratios)
    "open_to_prev_close", "close_to_prev_close",
    # structural events (binary) + normalized distance
    "liq_sweep_high", "liq_sweep_low", "in_golden_pocket", "dist_to_golden_pocket",
    # cyclical time
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    # multi-timeframe (all normalized)
    "tf5_log_ret_1", "tf5_rv_20", "tf5_atrp_14", "tf5_ema_spread",
    "tf15_log_ret_1", "tf15_rv_20", "tf15_atrp_14", "tf15_ema_spread",
    "tf60_log_ret_1", "tf60_rv_20", "tf60_atrp_14", "tf60_ema_spread",
    # trend strength (bounded 0..100, scale-free across assets)
    "adx",
]
# Defensive: never accidentally ship a feature that isn't actually produced.
SCALE_INVARIANT_FEATURES = [c for c in SCALE_INVARIANT_FEATURES if c in FEATURE_COLUMNS_PROFITABLE]


@dataclass
class PooledSummary:
    granularity: str
    horizon_bars: int
    vol_normalize_k: float
    warmup_bars: int
    n_assets: int
    asset_ids: List[str]
    feature_count: int
    rows_per_asset: Dict[str, int]
    rows_total: int
    label_positive_rate: float
    dropped_constant_features: List[str]
    output_path: Optional[str] = None


def build_for_instrument(
    ohlcv: pd.DataFrame,
    *,
    asset_id: str,
    asset_class: str,
    horizon_bars: int,
    vol_normalize_k: float,
    warmup_bars: int,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Per-asset slice of the pooled table: features + vol-normalized label.

    ``ohlcv`` must carry the canonical :data:`sources.OHLCV_COLUMNS`. Row order is
    preserved through ``compute_features`` (which 0-fills, never drops rows), so
    the captured timestamp re-aligns by position.
    """
    clean = ohlcv[OHLCV_COLUMNS].copy().reset_index(drop=True)
    ts = clean["timestamp"].astype(str)

    feats = compute_features(clean)  # -> FEATURE_COLUMNS_PROFITABLE, same length/order
    feats = feats.reset_index(drop=True)

    label = label_forward_return_binary(
        feats, horizon_bars=horizon_bars, threshold_bps=0.0,
        label_kind="vol_normalized", vol_normalize_k=vol_normalize_k,
    )

    out = feats[feature_cols].copy()
    out.insert(0, "timestamp", ts.to_numpy())
    out.insert(1, "asset_id", asset_id)
    out.insert(2, "asset_class", asset_class)
    out["label"] = label.to_numpy()

    # Drop per-asset warmup (indicators 0-filled before they stabilize) and the
    # trailing rows whose forward label is NaN. Both are per-asset so no asset
    # contaminates another's warmup/horizon.
    if warmup_bars > 0:
        out = out.iloc[warmup_bars:]
    out = out.dropna(subset=["label"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out


def assemble_pooled(
    frames: Dict[str, pd.DataFrame],
    *,
    instruments: Dict[str, Instrument],
    horizon_bars: int = 1,
    vol_normalize_k: float = 0.5,
    warmup_bars: int = 60,
    feature_cols: Optional[List[str]] = None,
) -> "tuple[pd.DataFrame, PooledSummary]":
    """Build the pooled table from ``{asset_id: ohlcv_df}``. Pure (no I/O)."""
    feature_cols = list(feature_cols or SCALE_INVARIANT_FEATURES)
    per_asset: List[pd.DataFrame] = []
    rows_per_asset: Dict[str, int] = {}
    for asset_id, ohlcv in frames.items():
        inst = instruments[asset_id]
        block = build_for_instrument(
            ohlcv, asset_id=asset_id, asset_class=inst.asset_class,
            horizon_bars=horizon_bars, vol_normalize_k=vol_normalize_k,
            warmup_bars=warmup_bars, feature_cols=feature_cols,
        )
        if len(block):
            per_asset.append(block)
            rows_per_asset[asset_id] = len(block)

    if not per_asset:
        raise ValueError("No rows produced for any asset — check inputs / warmup / horizon.")

    pooled = pd.concat(per_asset, ignore_index=True)
    # Global wall-clock order (then asset for determinism). The trainer splits on
    # this time axis, so look-ahead is impossible across the train/val/test cuts.
    pooled = pooled.assign(_k=pd.to_datetime(pooled["timestamp"], utc=True))
    pooled = pooled.sort_values(["_k", "asset_id"]).drop(columns="_k").reset_index(drop=True)

    # Drop features that are constant across the WHOLE pool (dead inputs).
    dropped_constant: List[str] = []
    for col in feature_cols:
        if pooled[col].nunique(dropna=False) <= 1:
            dropped_constant.append(col)
    if dropped_constant:
        pooled = pooled.drop(columns=dropped_constant)

    kept = [c for c in feature_cols if c not in dropped_constant]
    summary = PooledSummary(
        granularity="",  # filled by the CLI
        horizon_bars=horizon_bars,
        vol_normalize_k=vol_normalize_k,
        warmup_bars=warmup_bars,
        n_assets=len(rows_per_asset),
        asset_ids=sorted(rows_per_asset),
        feature_count=len(kept),
        rows_per_asset=rows_per_asset,
        rows_total=len(pooled),
        label_positive_rate=float(pooled["label"].mean()),
        dropped_constant_features=dropped_constant,
    )
    return pooled, summary


def _load_ohlcv_csv(inst: Instrument, data_root: Path, granularity: str) -> Optional[pd.DataFrame]:
    """Concat every CSV under the instrument's data dir; None if absent."""
    d = inst.data_dir(data_root, granularity)
    if not d.exists():
        return None
    files = sorted(d.glob("*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.assign(_k=pd.to_datetime(df["timestamp"], utc=True)).sort_values("_k").drop(columns="_k")
    return df.reset_index(drop=True)


def _write(pooled: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pooled.to_parquet(out_path, index=False)
        return out_path
    except Exception:  # pyarrow not installed -> CSV fallback (same convention as build_dataset)
        csv_path = out_path.with_suffix(".csv")
        pooled.to_csv(csv_path, index=False)
        return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the pooled multi-asset LSTM dataset.")
    p.add_argument("--universe", default=None, help="universe JSON (default: built-in)")
    p.add_argument("--granularity", default=None)
    p.add_argument("--horizon", type=int, default=1, help="forward bars for the label")
    p.add_argument("--vol-k", type=float, default=0.5, help="vol-normalized label multiplier (k*ATR%%)")
    p.add_argument("--warmup", type=int, default=60, help="per-asset warmup bars to drop")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="data/pooled/pooled.parquet")
    args = p.parse_args(argv)

    universe = Universe.from_json(Path(args.universe)) if args.universe else DEFAULT_UNIVERSE
    granularity = args.granularity or universe.granularity
    data_root = Path(args.data_root)

    frames: Dict[str, pd.DataFrame] = {}
    instruments: Dict[str, Instrument] = {}
    for inst in universe.instruments:
        df = _load_ohlcv_csv(inst, data_root, granularity)
        if df is None or df.empty:
            print(f"  {inst.asset_id}: no backfilled data (run backfill.py) — skipped")
            continue
        frames[inst.asset_id] = df
        instruments[inst.asset_id] = inst

    if not frames:
        print("No data found for any instrument. Run src/multi_asset/backfill.py first.")
        return 1

    pooled, summary = assemble_pooled(
        frames, instruments=instruments, horizon_bars=args.horizon,
        vol_normalize_k=args.vol_k, warmup_bars=args.warmup,
    )
    summary.granularity = granularity
    written = _write(pooled, Path(args.out))
    summary.output_path = str(written)

    import json
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
