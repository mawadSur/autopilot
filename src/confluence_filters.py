"""Standalone confluence entry filters (hard gates) layered on top of the
calibrated XGBoost ``P(long-profitable)`` from the crypto stack.

These filters do NOT modify the model or the supervisor -- they're a
prototype intended to be measured offline (see ``probe_confluence_filters.py``).
The idea: the model already weighs volume, ATR, trend, spread *softly* via
its 135 features. But there's evidence that turning them into a HARD gate
(don't trade when feature X is on the wrong side of a threshold) can lift
precision at the cost of trade frequency.

Each gate is a pure function: ``(row) -> bool`` where ``row`` is the FULL
135-feature row (pandas Series, dict, or anything subscriptable).
``True`` means "this bar passes the gate / it's OK to trade".
``False`` means "skip this signal".

Why pure functions and not a class:
  * easy to unit-test in isolation
  * trivial to compose via ``gate_all``
  * no hidden state / no rolling buffer needed -- the upstream feature
    pipeline already did all the rolling work

A note on data realities (see probe report for details):
  * ``volume_quote`` is identically 0 in the current 1m datasets because
    the OHLCV backfill didn't carry the quote-currency volume column.
    The raw base volume is recoverable as ``expm1(vol_log)`` -- that's
    what the model's ``vol_ma_20`` rolling stat is built on. The
    spec-literal volume gate will therefore always reject; the
    ``_proxy`` variant below uses ``expm1(vol_log)`` instead and is
    what the probe actually exercises.
  * ``spread_pct`` is identically 0 in the current datasets (no L1 book
    data in the backfill). The spread gate is therefore a trivial
    pass-through right now; it'll only have teeth once the dataset is
    rebuilt with order-book data.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Per-symbol ATR percentile caps (P80 of atrp_14, pre-computed from the
# 70% TRAIN slice -- not val/test -- so the cap is "out-of-sample" for the
# probe's test slice).
# Methodology: see probe_confluence_filters._precompute_atr_caps().
# Numbers are eye-balled from running build/probe; if a symbol is missing
# we fall back to a global default.
# ---------------------------------------------------------------------------
ATR_PERCENTILE_CAPS: dict[str, dict[int, float]] = {
    # Filled in lazily by the probe before use; the gate falls back to the
    # ``cap`` keyword if no symbol-specific override exists.
    "ETH/USD": {},
    "BTC/USD": {},
    "SOL/USD": {},
}


def _get(row: Any, key: str, default: float = float("nan")) -> float:
    """Subscript a pandas Series or dict-like row, returning a float."""
    try:
        # pandas Series supports both .get and [] but .get returns the
        # default cleanly if the column is missing.
        if hasattr(row, "get"):
            val = row.get(key, default)
        else:
            val = row[key]
    except (KeyError, IndexError):
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _is_finite_positive(x: float) -> bool:
    """True if x is a finite positive number (excludes 0, NaN, inf)."""
    return math.isfinite(x) and x > 0.0


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def gate_volume_above_ma(row: Any, multiplier: float = 1.5) -> bool:
    """Require current-bar quote volume to be at least ``multiplier`` x the
    20-bar rolling average. Filters out low-volume noise minutes.

    Spec-literal: compares ``volume_quote >= multiplier * vol_ma_20``.
    On the current datasets ``volume_quote`` is identically 0 (the
    OHLCV backfill didn't carry quote-currency volume), so this gate
    always returns False on real data. See ``gate_volume_above_ma_proxy``
    for the equivalent gate that uses the base-volume signal
    (``expm1(vol_log)``) actually present in the dataset.
    """
    vol_q = _get(row, "volume_quote")
    vol_ma = _get(row, "vol_ma_20")
    if not (math.isfinite(vol_q) and math.isfinite(vol_ma) and vol_ma > 0):
        return False
    return vol_q >= multiplier * vol_ma


def gate_volume_above_ma_proxy(row: Any, multiplier: float = 1.5) -> bool:
    """Pragmatic variant: compare current-bar base volume (recovered as
    ``expm1(vol_log)``) against ``vol_ma_20``. Same intent as the
    spec-literal gate but actually has signal on the current dataset.

    Justification: the upstream pipeline computes ``vol_ma_20`` from the
    base-currency ``volume`` column, and ``vol_log = log1p(volume)``.
    So the proper comparison on this dataset is ``expm1(vol_log) >=
    multiplier * vol_ma_20``.
    """
    vol_log = _get(row, "vol_log")
    vol_ma = _get(row, "vol_ma_20")
    if not (math.isfinite(vol_log) and math.isfinite(vol_ma) and vol_ma > 0):
        return False
    vol_now = math.expm1(vol_log)
    return vol_now >= multiplier * vol_ma


def gate_atr_not_extreme(
    row: Any,
    *,
    cap: float | None = None,
    symbol: str | None = None,
    percentile: int = 80,
) -> bool:
    """Require ``atrp_14 <= cap``.

    ``cap`` is the absolute threshold (e.g. 0.0008 = 8 bps). If ``cap`` is
    None and a per-symbol cap exists in ``ATR_PERCENTILE_CAPS[symbol][
    percentile]``, use that. Otherwise the gate returns True (no-op) --
    fail-open so the caller can tell the gate is uncalibrated rather than
    silently rejecting everything.

    The intent: avoid news-spike / wild-vol regimes where the model has
    less edge.
    """
    atrp = _get(row, "atrp_14")
    if not math.isfinite(atrp):
        return False  # bad data -> skip
    if cap is None and symbol is not None:
        cap = ATR_PERCENTILE_CAPS.get(symbol, {}).get(percentile)
    if cap is None:
        return True  # no cap configured -> fail-open
    return atrp <= cap


def gate_trend_align(row: Any) -> bool:
    """Require price > EMA(50): trade-with-trend bias.

    Uses ``close_over_ema_50`` if available (this dataset's pre-computed
    relative distance to EMA50: ``close / ema_50 - 1``). The raw ``close``
    column is dropped during feature engineering, so ``close_over_ema_50``
    is the canonical way to express this gate on the 135-feature row.

    Returns True iff ``close_over_ema_50 > 0`` (price above EMA50).
    """
    rel = _get(row, "close_over_ema_50")
    if not math.isfinite(rel):
        return False
    return rel > 0.0


def gate_spread_ok(row: Any, max_spread_bps: float = 2.0) -> bool:
    """Require effective bid/ask spread <= ``max_spread_bps`` basis points.

    ``spread_pct`` in this dataset is the *fractional* spread
    (``spread_abs / mid``), so 1 bp = 0.0001 in those units. We convert
    by multiplying by 10000.

    Caveat: ``spread_pct`` is identically 0 in the current 1m datasets
    (no L1 book data in the OHLCV backfill). The gate therefore trivially
    passes; it only has teeth once the dataset is rebuilt with quote/book
    columns populated.
    """
    spread = _get(row, "spread_pct")
    if not math.isfinite(spread):
        return False
    return spread * 10000.0 <= max_spread_bps


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


FilterFn = Callable[[Any], bool]


def gate_all(row: Any, filters: Iterable[FilterFn]) -> bool:
    """Short-circuit AND over a list of gate callables. Returns True iff
    every gate accepts the row.

    Empty ``filters`` -> True (vacuously). Any gate that raises is
    treated as a reject (we don't want a bad gate to silently let
    everything through).
    """
    for f in filters:
        try:
            ok = f(row)
        except Exception:  # noqa: BLE001 -- defensive: reject on error
            return False
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Vectorised helpers (used by the probe to score 19k+ rows efficiently)
# ---------------------------------------------------------------------------


def vectorised_volume_above_ma(df, multiplier: float = 1.5) -> np.ndarray:
    """Vectorised ``gate_volume_above_ma`` over a DataFrame. Returns bool array."""
    if "volume_quote" not in df.columns or "vol_ma_20" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    vq = df["volume_quote"].to_numpy(dtype=np.float64, copy=False)
    vm = df["vol_ma_20"].to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(vq) & np.isfinite(vm) & (vm > 0.0) & (vq >= multiplier * vm)
    return ok


def vectorised_volume_above_ma_proxy(df, multiplier: float = 1.5) -> np.ndarray:
    """Vectorised proxy variant using ``expm1(vol_log)``."""
    if "vol_log" not in df.columns or "vol_ma_20" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    vl = df["vol_log"].to_numpy(dtype=np.float64, copy=False)
    vm = df["vol_ma_20"].to_numpy(dtype=np.float64, copy=False)
    vn = np.expm1(vl)
    ok = np.isfinite(vn) & np.isfinite(vm) & (vm > 0.0) & (vn >= multiplier * vm)
    return ok


def vectorised_atr_not_extreme(df, cap: float) -> np.ndarray:
    """Vectorised ``gate_atr_not_extreme`` with an absolute cap."""
    if "atrp_14" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    a = df["atrp_14"].to_numpy(dtype=np.float64, copy=False)
    return np.isfinite(a) & (a <= cap)


def vectorised_trend_align(df) -> np.ndarray:
    """Vectorised ``gate_trend_align`` using ``close_over_ema_50``."""
    if "close_over_ema_50" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    r = df["close_over_ema_50"].to_numpy(dtype=np.float64, copy=False)
    return np.isfinite(r) & (r > 0.0)


def vectorised_spread_ok(df, max_spread_bps: float = 2.0) -> np.ndarray:
    """Vectorised ``gate_spread_ok``."""
    if "spread_pct" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    s = df["spread_pct"].to_numpy(dtype=np.float64, copy=False)
    return np.isfinite(s) & ((s * 10000.0) <= max_spread_bps)
