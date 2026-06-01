"""Backfill CLI: ingest a historical OHLCV/feature parquet into a regime store.

Usage::

    ./.venv/bin/python -m regime_memory.backfill \
        --dataset data/crypto/datasets/eth_usd_v2.parquet \
        --output regime_store_eth.npz \
        --window-bars 60 \
        --label-horizon-bars 60

For each rolling ``window_bars`` slice of the dataset we:

1. Compute a 144-dim embedding via :class:`RegimeEncoder`.
2. Look ``label_horizon_bars`` into the future to compute synthetic
   metadata that captures "what would have worked here":

   * ``optimal_threshold`` — best classification threshold (over a coarse
     sweep) for the next-N bars, scoring on a simple sign-of-return
     classifier. This is intentionally cheap; the canonical optimal
     threshold belongs in a separate per-symbol calibration sweep, not
     in the backfill loop.
   * ``realized_sharpe`` — annualized Sharpe of the next-N bar returns.
     The backfill assumes 1m bars (525,600 minutes per year); on 5m bars
     consumers should pass ``--bars-per-year`` accordingly.
   * ``regime_label`` — coarse numeric label encoding the next-N return
     distribution: ``2.0`` (trending up), ``0.0`` (trending down), or
     ``1.0`` (choppy / sideways). Numeric so the
     similarity-weighted-average inside :class:`RegimeLookup` produces a
     usable scalar (a string label couldn't be averaged).

Synthetic-label heuristic
-------------------------
The post-hoc regime labels are deliberately simple — this is the v0
heuristic, intended to be replaced by a learned regime detector later:

* Compute ``net_return`` (sum of next-N log returns) and ``vol`` (std).
* If ``|net_return| < vol`` → choppy (label 1.0).
* Else if ``net_return > 0`` → trending up (label 2.0).
* Else → trending down (label 0.0).

The optimal-threshold sweep runs ``np.linspace(0.40, 0.70, 7)``; for each
candidate ``thr`` we treat "predict up if some-feature > thr" as the
classifier. The "some-feature" we use is ``ema_spread_9_21`` when present
(it's a near-zero-centered momentum proxy in the project's feature set);
otherwise we fall back to the *first* numeric column in the frame. Returns
the threshold that maximizes the directional-accuracy score on the next-N
bars. Coarse and not load-bearing — it's a starting point, not a final
answer.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from regime_memory.encoder import RegimeEncoder, RegimeWindow
from regime_memory.store import NaiveRegimeStore, RegimeStore, make_regime_store

logger = logging.getLogger(__name__)


# Default bars-per-year for 1m bars (525,600 minutes / year). Override with
# ``--bars-per-year`` when ingesting 5m, 15m, etc.
_BARS_PER_YEAR_1M = 525_600

# Threshold sweep for the synthetic optimal-threshold field. Coarse on
# purpose — the real optimization belongs in a separate sweep, not the
# backfill hot loop. 7 candidates × N windows is fast enough on a parquet
# with 100k+ rows.
_THRESHOLD_GRID: tuple[float, ...] = tuple(np.linspace(0.40, 0.70, 7).tolist())


def _detect_return_column(df: pd.DataFrame) -> str:
    """Pick the column we'll treat as the per-bar return for forward stats.

    Preference order matches the project's feature naming:
    ``log_ret`` → ``return_1`` → ``close_to_prev_close``. Falls back to a
    raw close-pct-change if none of those exist (which means callers passed
    a less-rich frame — fine for tests).
    """

    for candidate in ("log_ret", "return_1", "close_to_prev_close"):
        if candidate in df.columns:
            return candidate
    if "close" in df.columns:
        df["_synthetic_log_ret"] = np.log(df["close"]).diff().fillna(0.0)
        return "_synthetic_log_ret"
    raise ValueError(
        "no recognizable return column found in dataset; expected one of "
        "log_ret, return_1, close_to_prev_close, close"
    )


def _detect_signal_column(df: pd.DataFrame) -> str:
    """Pick the feature we'll sweep thresholds against.

    ``ema_spread_9_21`` is the project's go-to momentum proxy. If absent we
    fall back to the first numeric, non-timestamp column — good enough for
    tests with synthetic data.
    """

    if "ema_spread_9_21" in df.columns:
        return "ema_spread_9_21"
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude any obvious label / timestamp columns the trainer wrote.
    for skip in ("label", "timestamp", "minute_of_day", "day_of_week"):
        if skip in numeric:
            numeric.remove(skip)
    if not numeric:
        raise ValueError("no numeric column available to use as signal")
    return numeric[0]


def _regime_label_from_returns(rets: np.ndarray) -> float:
    """Numeric label: 2.0 (up), 1.0 (chop), 0.0 (down). See module docstring."""

    if rets.size == 0:
        return 1.0
    net = float(np.nansum(rets))
    vol = float(np.nanstd(rets))
    # A constant non-zero stream (vol == 0 but net != 0) is the strongest
    # possible trend signal — we route it to up/down rather than to chop.
    # The "choppy" branch only fires when vol > 0 and the within-window
    # noise dominates the net drift.
    if vol > 0.0 and abs(net) < vol:
        return 1.0
    if net == 0.0:
        return 1.0
    return 2.0 if net > 0 else 0.0


def _realized_sharpe(rets: np.ndarray, bars_per_year: int) -> float:
    """Annualized Sharpe of a (possibly short) return series.

    Returns ``0.0`` for degenerate inputs (all-zero std, empty array). We
    don't subtract a risk-free rate — the backfill is a relative-regime
    fingerprint, not a portfolio metric.
    """

    if rets.size < 2:
        return 0.0
    mean = float(np.nanmean(rets))
    std = float(np.nanstd(rets, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return mean / std * math.sqrt(bars_per_year)


def _optimal_threshold(
    signal: np.ndarray,
    rets: np.ndarray,
) -> float:
    """Return the threshold from :data:`_THRESHOLD_GRID` with highest dir-accuracy.

    Classifier: ``signal > thr`` → predict up, else predict down. Score:
    fraction of bars where the predicted direction matches the realized
    sign of the return. Ties resolved by the lowest threshold (numpy
    ``argmax`` semantics).
    """

    if signal.size == 0 or rets.size == 0:
        return float(_THRESHOLD_GRID[len(_THRESHOLD_GRID) // 2])
    realized_up = (rets > 0).astype(np.int8)
    best_score = -1.0
    best_thr = float(_THRESHOLD_GRID[0])
    for thr in _THRESHOLD_GRID:
        pred = (signal > thr).astype(np.int8)
        # Element-wise match rate over the horizon.
        match = float(np.mean(pred == realized_up))
        if match > best_score:
            best_score = match
            best_thr = float(thr)
    return best_thr


def _iter_windows(
    df: pd.DataFrame,
    *,
    window_bars: int,
    label_horizon_bars: int,
    max_windows: Optional[int],
):
    """Yield ``(window_df, future_rets, future_signal)`` for each rolling slice.

    Stride is ``window_bars`` (non-overlapping) so we don't blow up the
    store with N nearly-identical neighbors. Callers who want denser
    coverage can pass ``--stride`` once we expose it (not in v0).
    """

    return_col = _detect_return_column(df)
    signal_col = _detect_signal_column(df)
    n = len(df)
    yielded = 0
    start = 0
    while start + window_bars + label_horizon_bars <= n:
        window = df.iloc[start : start + window_bars]
        future = df.iloc[start + window_bars : start + window_bars + label_horizon_bars]
        future_rets = future[return_col].to_numpy(dtype=np.float64, copy=False)
        future_signal = window[signal_col].to_numpy(dtype=np.float64, copy=False)
        yield window, future_rets, future_signal
        yielded += 1
        if max_windows is not None and yielded >= max_windows:
            return
        start += window_bars


def _window_end_utc(window: pd.DataFrame) -> str:
    """Pick a stable window-end timestamp string for the RegimeWindow id.

    Falls back to a synthesized 'window-N' string if no timestamp column
    exists (lets the test suite use entirely synthetic frames).
    """

    if "timestamp" in window.columns:
        return str(window["timestamp"].iloc[-1])
    last_idx = window.index[-1]
    return f"row-{int(last_idx)}"


def build_store(
    dataset: pd.DataFrame,
    *,
    symbol: str,
    window_bars: int = 60,
    label_horizon_bars: int = 60,
    max_windows: Optional[int] = None,
    bars_per_year: int = _BARS_PER_YEAR_1M,
    encoder: Optional[RegimeEncoder] = None,
    prefer_faiss: bool = True,
):
    """Build a regime store from an in-memory dataset frame.

    Splits the heavy work out of :func:`main` so the test suite can drive
    it without a CLI roundtrip. Returns the populated store instance.
    """

    if window_bars < 1:
        raise ValueError(f"window_bars must be >= 1, got {window_bars!r}")
    if label_horizon_bars < 1:
        raise ValueError(
            f"label_horizon_bars must be >= 1, got {label_horizon_bars!r}"
        )

    # Drop rows where ALL feature columns would be NaN; these come from the
    # head of feature parquets where rolling indicators haven't warmed up.
    if "label" in dataset.columns:
        feature_cols = [c for c in dataset.columns if c not in ("label", "timestamp")]
    else:
        feature_cols = [c for c in dataset.columns if c != "timestamp"]

    if encoder is None:
        encoder = RegimeEncoder(feature_cols=feature_cols)

    dim = encoder.expected_dim()
    store = make_regime_store(dim=dim, prefer_faiss=prefer_faiss)

    windows_built = 0
    for window_df, future_rets, future_signal in _iter_windows(
        dataset,
        window_bars=window_bars,
        label_horizon_bars=label_horizon_bars,
        max_windows=max_windows,
    ):
        embedding = encoder.encode_features(window_df, window_size=window_bars)
        metadata: Dict[str, float] = {
            "optimal_threshold": _optimal_threshold(future_signal, future_rets),
            "realized_sharpe": _realized_sharpe(future_rets, bars_per_year),
            "regime_label": _regime_label_from_returns(future_rets),
            # ``kelly_size_pct`` placeholder — derive from realized_sharpe so
            # consumers get a non-zero default. Capped at 0.25 to avoid a
            # high-Sharpe regime suggesting reckless sizing during backfill.
            # The real risk manager re-applies its own cap downstream.
            "kelly_size_pct": float(
                min(
                    0.25,
                    max(
                        0.0,
                        _realized_sharpe(future_rets, bars_per_year) / 10.0,
                    ),
                )
            ),
        }
        store.add(
            RegimeWindow(
                symbol=symbol,
                window_end_utc=_window_end_utc(window_df),
                bars=window_bars,
                embedding=embedding,
                metadata=metadata,
            )
        )
        windows_built += 1

    logger.info("backfill: built %d windows for %s", windows_built, symbol)
    return store


# --- CLI --------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill a regime memory store from a historical OHLCV/feature parquet.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the historical feature parquet (e.g. data/crypto/datasets/eth_usd_v2.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the persisted store (.npz).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETH-USD",
        help="Symbol tag stored on each RegimeWindow. Default ETH-USD.",
    )
    parser.add_argument(
        "--window-bars",
        type=int,
        default=60,
        help="Bars per regime window. Default 60 (1h on 1m bars).",
    )
    parser.add_argument(
        "--label-horizon-bars",
        type=int,
        default=60,
        help="Bars of forward-return data used to compute synthetic metadata.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on windows ingested (useful for tests / smoke runs).",
    )
    parser.add_argument(
        "--bars-per-year",
        type=int,
        default=_BARS_PER_YEAR_1M,
        help="Annualization factor for realized Sharpe. Default 525600 (1m bars).",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Force the numpy-backed NaiveRegimeStore even if FAISS is installed.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.dataset)
    store = build_store(
        df,
        symbol=args.symbol,
        window_bars=args.window_bars,
        label_horizon_bars=args.label_horizon_bars,
        max_windows=args.max_windows,
        bars_per_year=args.bars_per_year,
        prefer_faiss=not args.no_faiss,
    )
    store.save(args.output)
    print(
        f"wrote {len(store)} regime windows to {args.output} "
        f"(symbol={args.symbol}, dim={store.dim})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_store", "main"]
