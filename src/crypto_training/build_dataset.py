"""Assemble a training dataset from backfilled OHLCV CSVs.

Pipeline:
  1. Load every day-CSV under
     ``<data_root>/<symbol>/<granularity>/*.csv`` and concat oldest-first.
  2. Compute the 36 derived features via :func:`utils.compute_features` --
     same feature contract as the legacy transformer in ``model_sanity/``.
  3. Apply a forward-return label. Default: binary "did close move > +bps
     in the next ``horizon_bars`` bars after fees?" (a "long is profitable"
     gate suitable for an HFT long-only model). Optional ``--label-mode
     three_class`` produces ``[-1, 0, 1]`` with a symmetric +/- threshold.
  4. Drop the warmup window (rows with any NaN feature) and the trailing
     ``horizon_bars`` (no forward target).
  5. Persist to parquet (or CSV fallback if pyarrow isn't installed).

CLI::

    ./.venv/bin/python src/crypto_training/build_dataset.py \\
        --symbol ETH/USD --data-root data/crypto/ \\
        --out data/crypto/datasets/eth_usd_1m_5m_h.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

# sys.path shim so this CLI runs without PYTHONPATH=src.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


_GRANULARITY_LABEL_TO_DIR = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
}


@dataclass
class DatasetSummary:
    """What ``build_dataset`` produced."""

    symbol: str
    granularity: str
    rows_in: int
    rows_out: int
    feature_count: int
    label_mode: Literal["binary", "three_class"]
    horizon_bars: int
    threshold_bps: float
    label_distribution: dict
    label_kind: Literal["fixed_bps", "vol_normalized"] = "fixed_bps"
    vol_normalize_k: float = 0.5
    output_path: Optional[Path] = None


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "-")


def load_ohlcv(
    *, data_root: Path, symbol: str, granularity_label: str = "1m"
) -> pd.DataFrame:
    """Concatenate every day CSV under ``<root>/<sym>/<granularity>/`` oldest-first."""
    sym_dir = Path(data_root).expanduser().resolve() / _safe_symbol(symbol) / (
        _GRANULARITY_LABEL_TO_DIR[granularity_label]
    )
    if not sym_dir.exists():
        raise FileNotFoundError(
            f"No backfill directory at {sym_dir}. Run backfill_ohlcv first."
        )
    files = sorted(sym_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {sym_dir}")
    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Sanity: drop dupes by timestamp, sort.
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def label_forward_return_binary(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    threshold_bps: float,
    label_kind: Literal["fixed_bps", "vol_normalized"] = "fixed_bps",
    vol_normalize_k: float = 0.5,
) -> pd.Series:
    """1 if forward return over ``horizon_bars`` exceeds the label threshold.

    The forward return is computed in basis points:
    ``(close[t+horizon] - close[t]) / close[t] * 10000``.

    Two label schemes are supported via ``label_kind``:

    ``fixed_bps`` (default, backwards-compatible):
        ``label = forward_return_bps > threshold_bps``
        Simpler but volatility-confounded: in high-vol periods, more bars
        trivially cross any fixed threshold regardless of direction. The model
        then learns "high vol -> label=1" instead of "direction -> label=1".

    ``vol_normalized``:
        ``label = forward_return_bps > vol_normalize_k * atrp_14_bps``
        where ``atrp_14_bps = atr_14 / close * 10000``.
        This normalizes the threshold by local volatility so label=1 means
        "forward move exceeded k standard-deviation moves", not just "moved a
        lot in absolute terms." The model must learn directional alpha to
        predict this, not just detect high-volatility regimes.

        Rows where ``atrp_14`` is NaN (typically the first 14 bars of warmup)
        produce NaN labels and are dropped downstream by ``build_dataset``.

        Requires ``atrp_14`` and ``close`` columns in ``df`` (both are present
        in any frame returned by ``compute_features``).
    """
    closes = df["close"].astype(float).to_numpy()
    forward = np.full_like(closes, np.nan, dtype=np.float64)
    if len(closes) > horizon_bars:
        forward[:-horizon_bars] = (
            (closes[horizon_bars:] - closes[:-horizon_bars])
            / np.where(closes[:-horizon_bars] == 0, 1e-12, closes[:-horizon_bars])
            * 10000.0
        )

    if label_kind == "fixed_bps":
        label_arr = np.where(
            np.isnan(forward), np.nan, (forward > threshold_bps).astype(float)
        )
    elif label_kind == "vol_normalized":
        if "atrp_14" not in df.columns:
            raise ValueError(
                "label_kind='vol_normalized' requires an 'atrp_14' column. "
                "Run compute_features() before calling label_forward_return_binary()."
            )
        # atrp_14 is already in percent (ATR / close * 100); convert to bps.
        atrp_bps = df["atrp_14"].astype(float).to_numpy() * 100.0
        dynamic_thr = vol_normalize_k * atrp_bps
        label_arr = np.where(
            np.isnan(forward) | np.isnan(atrp_bps),
            np.nan,
            (forward > dynamic_thr).astype(float),
        )
    else:
        raise ValueError(
            f"Unknown label_kind {label_kind!r}; expected 'fixed_bps' or 'vol_normalized'."
        )

    # Use a nullable float series so we can express NaN for the trailing
    # rows that have no forward target. The trainer downstream casts to int.
    return pd.Series(label_arr, index=df.index, name="label")


def label_forward_return_three_class(
    df: pd.DataFrame, *, horizon_bars: int, threshold_bps: float
) -> pd.Series:
    """Three-class label.

    Returns:
      *  1 if forward return > +threshold_bps
      * -1 if forward return < -threshold_bps
      *  0 otherwise (chop / no edge)
    NaN for the trailing rows that have no forward target.
    """
    closes = df["close"].astype(float).to_numpy()
    forward = np.full_like(closes, np.nan, dtype=np.float64)
    if len(closes) > horizon_bars:
        forward[:-horizon_bars] = (
            (closes[horizon_bars:] - closes[:-horizon_bars])
            / np.where(closes[:-horizon_bars] == 0, 1e-12, closes[:-horizon_bars])
            * 10000.0
        )
    label = np.where(
        np.isnan(forward),
        np.nan,
        np.where(forward > threshold_bps, 1, np.where(forward < -threshold_bps, -1, 0)),
    )
    return pd.Series(label, index=df.index, name="label")


def build_dataset(
    *,
    data_root: Path,
    symbol: str,
    granularity_label: str = "1m",
    horizon_bars: int = 5,
    threshold_bps: float = 10.0,
    label_mode: Literal["binary", "three_class"] = "binary",
    label_kind: Literal["fixed_bps", "vol_normalized"] = "fixed_bps",
    vol_normalize_k: float = 0.5,
    feature_cols: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, DatasetSummary]:
    """Return the train-ready DataFrame + a summary dict.

    If ``feature_cols`` is None, every numeric column produced by
    ``compute_features`` is kept (useful for trainer-side feature
    selection). Pass an explicit list (eg the legacy 36-feature set) to
    lock the schema.
    """
    from utils import compute_features  # lazy import (heavy: pulls pandas)

    raw = load_ohlcv(
        data_root=data_root, symbol=symbol, granularity_label=granularity_label
    )
    rows_in = len(raw)
    LOGGER.info("loaded %d raw OHLCV rows for %s/%s", rows_in, symbol, granularity_label)

    # Preserve timestamp -- compute_features drops it during feature
    # engineering but we need it for time-based train/val/test splits.
    timestamp_series = raw["timestamp"].copy()
    feats = compute_features(raw.copy())
    feats["timestamp"] = timestamp_series.values

    # Label.
    if label_mode == "binary":
        feats["label"] = label_forward_return_binary(
            feats,
            horizon_bars=horizon_bars,
            threshold_bps=threshold_bps,
            label_kind=label_kind,
            vol_normalize_k=vol_normalize_k,
        )
    elif label_mode == "three_class":
        feats["label"] = label_forward_return_three_class(
            feats, horizon_bars=horizon_bars, threshold_bps=threshold_bps
        )
    else:
        raise ValueError(f"Unknown label_mode {label_mode!r}")

    # Choose feature columns: caller-supplied or every numeric col except
    # raw OHLCV / timestamp / label.
    if feature_cols is None:
        exclude = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
        }
        feature_cols = [
            c
            for c in feats.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(feats[c])
        ]

    # Keep only what we need; drop rows missing label or any feature.
    keep_cols = ["timestamp", *feature_cols, "label"]
    out = feats[keep_cols].dropna(subset=["label"]).copy()
    out = out.dropna(subset=feature_cols)
    out = out.reset_index(drop=True)

    # Label distribution.
    label_dist = (
        out["label"]
        .astype("int64", errors="ignore")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    label_dist = {int(k): int(v) for k, v in label_dist.items()}

    summary = DatasetSummary(
        symbol=symbol,
        granularity=granularity_label,
        rows_in=rows_in,
        rows_out=len(out),
        feature_count=len(feature_cols),
        label_mode=label_mode,
        horizon_bars=horizon_bars,
        threshold_bps=threshold_bps,
        label_distribution=label_dist,
        label_kind=label_kind,
        vol_normalize_k=vol_normalize_k,
    )

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Prefer parquet (compact + fast), fall back to CSV if pyarrow missing.
        if output_path.suffix == ".parquet":
            try:
                out.to_parquet(output_path, index=False)
            except Exception as exc:  # noqa: BLE001 -- fallback path
                LOGGER.warning(
                    "parquet write failed (%s); falling back to CSV at %s",
                    exc,
                    output_path.with_suffix(".csv"),
                )
                output_path = output_path.with_suffix(".csv")
                out.to_csv(output_path, index=False)
        else:
            out.to_csv(output_path, index=False)
        summary.output_path = output_path
        LOGGER.info(
            "wrote %d rows / %d features to %s (label dist %s)",
            len(out),
            len(feature_cols),
            output_path,
            label_dist,
        )

    return out, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_dataset",
        description="Assemble a training dataset from backfilled Coinbase OHLCV.",
    )
    p.add_argument("--symbol", required=True, help="e.g. ETH/USD")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/crypto"),
        help="Where backfill_ohlcv wrote its CSVs (default ./data/crypto)",
    )
    p.add_argument(
        "--granularity",
        default="1m",
        choices=sorted(_GRANULARITY_LABEL_TO_DIR.keys()),
        help="Bar size to load (default 1m)",
    )
    p.add_argument(
        "--horizon-bars",
        type=int,
        default=5,
        help="Forward bars used for the label (default 5)",
    )
    p.add_argument(
        "--threshold-bps",
        type=float,
        default=10.0,
        help="Basis-point threshold for label (default 10 bps)",
    )
    p.add_argument(
        "--label-mode",
        choices=["binary", "three_class"],
        default="binary",
        help="Label scheme (default binary)",
    )
    p.add_argument(
        "--label-kind",
        choices=["fixed_bps", "vol_normalized"],
        default="fixed_bps",
        help=(
            "Label threshold kind for binary mode (default fixed_bps). "
            "'vol_normalized' uses forward_return_bps > k * atrp_14_bps so "
            "the threshold scales with local volatility, removing the "
            "volatility-confound that causes models to learn 'high vol -> label=1'."
        ),
    )
    p.add_argument(
        "--vol-normalize-k",
        type=float,
        default=0.5,
        help=(
            "Multiplier for the vol-normalized label threshold: "
            "label=1 when forward_return_bps > k * atrp_14_bps (default 0.5). "
            "Only used when --label-kind=vol_normalized."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the dataset (.parquet or .csv)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    _, summary = build_dataset(
        data_root=args.data_root,
        symbol=args.symbol,
        granularity_label=args.granularity,
        horizon_bars=args.horizon_bars,
        threshold_bps=args.threshold_bps,
        label_mode=args.label_mode,
        label_kind=args.label_kind,
        vol_normalize_k=args.vol_normalize_k,
        output_path=args.out,
    )
    LOGGER.info(
        "dataset summary: %d rows in -> %d rows out, %d features, "
        "label dist %s, horizon=%d bars, threshold=%.1f bps, mode=%s",
        summary.rows_in,
        summary.rows_out,
        summary.feature_count,
        summary.label_distribution,
        summary.horizon_bars,
        summary.threshold_bps,
        summary.label_mode,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
