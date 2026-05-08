"""Cartesian-product correlation miner over (feature_a, feature_b, horizon).

Phase 4 / E2 (CEO plan). The miner pulls feature time series from a list of
:class:`FeatureSource` adapters, then iterates the cartesian product of
``(feature_a, feature_b, horizon)`` tuples and computes a rank-IC (Spearman
correlation) between ``feature_a`` at time ``t`` and the forward-return of
``feature_b`` over ``horizon_bars`` bars.

Rank-IC is the standard "information coefficient" used in quant research —
the Spearman correlation of a signal at ``t`` with realized returns at
``t+h``. It's preferred over Pearson because it's robust to outliers and
non-linear monotonic relationships, both of which dominate in noisy
cross-asset data.

The skeleton is intentionally hermetic: no network IO inside :class:`CorrelationMiner`
— callers pass in :class:`FeatureSource` instances that own their own data
plumbing. This keeps the miner unit-testable with synthetic DataFrames and
lets the production wire-up live in :mod:`alpha_lab.feature_sources` (whose
real implementations are a future PR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Default horizons (in bars). Multi-horizon mining lets the gate identify
# whether a candidate signal predicts short-term moves (5 bars), medium
# (15 bars), or longer (60 bars). Override via the constructor when the
# caller knows their bar size + target latency.
DEFAULT_HORIZON_BARS: Tuple[int, ...] = (5, 15, 60)


if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    # The feature-source asset class enum reuses ``protocols.tradeable.AssetClass``
    # so the rest of the stack speaks the same taxonomy. Imported lazily inside
    # the source adapters; the miner accepts any string for portability.
    from protocols.tradeable import AssetClass  # noqa: F401


__all__ = [
    "CorrelationMiner",
    "CorrelationResult",
    "DEFAULT_HORIZON_BARS",
    "FeaturePair",
    "FeatureSource",
]


@dataclass(frozen=True)
class FeaturePair:
    """A single (feature_a -> feature_b @ horizon) tuple to evaluate.

    Frozen + hashable so call sites can use it as a dict key when accumulating
    rolling rank-IC history per pair.

    Attributes:
        feature_a: column name on the source-A DataFrame (the "signal").
        feature_b: column name on the source-B DataFrame (the "target" for
            forward-return computation).
        horizon_bars: number of bars to look ahead when computing the
            forward return on ``feature_b``.
        asset_class_a: high-level taxonomy label for source A (e.g.
            ``"spot_crypto"`` or ``"prediction_binary"``). Stored as a string
            rather than an enum so the dataclass is trivially serializable
            and survives a JSON round-trip into the audit log.
        asset_class_b: same, for source B.
    """

    feature_a: str
    feature_b: str
    horizon_bars: int
    asset_class_a: str
    asset_class_b: str

    def stable_id(self) -> str:
        """Stable, filesystem-safe identifier — used as a Redis hash key.

        Concatenates the asset_class + feature names + horizon with ``::`` so
        the same pair always maps to the same key across runs. The ordering
        of (a, b) is preserved (i.e. swapping features yields a different
        pair, because the miner treats them asymmetrically — feature_a is
        the signal, feature_b's forward-return is the target).
        """
        return (
            f"{self.asset_class_a}::{self.feature_a}"
            f"->{self.asset_class_b}::{self.feature_b}"
            f"@h={self.horizon_bars}"
        )


@dataclass(frozen=True)
class CorrelationResult:
    """Output of a single ``(pair, window)`` rank-IC computation.

    Attributes:
        pair: the :class:`FeaturePair` evaluated.
        rank_ic: signed Spearman rank correlation in ``[-1, 1]``. ``0.0``
            when the input has fewer than 3 valid (non-NaN, non-constant)
            samples.
        n_samples: number of paired ``(signal_t, forward_return_t+h)``
            observations used. Excludes NaNs introduced by the forward-return
            shift at the tail of the series.
        pvalue: two-sided p-value reported by SciPy. ``1.0`` when the input
            is degenerate (n < 3 or constant series).
        computed_at_utc: ISO-8601 timestamp the result was computed.
    """

    pair: FeaturePair
    rank_ic: float
    n_samples: int
    pvalue: float
    computed_at_utc: str


@runtime_checkable
class FeatureSource(Protocol):
    """Adapter interface — implemented by crypto / Polymarket / etc. sources.

    The miner reads from a list of :class:`FeatureSource` instances; each
    source owns its own plumbing (parquet reads, REST calls, etc.). The
    Protocol is :func:`runtime_checkable`, so call sites can guard with
    ``isinstance(obj, FeatureSource)`` for tests and dynamic config paths.

    Implementations live in :mod:`alpha_lab.feature_sources`.
    """

    @property
    def asset_class(self) -> Any:  # AssetClass enum or string
        ...

    @property
    def name(self) -> str:
        """Short label used to disambiguate sources in audit logs."""
        ...

    def fetch_window(
        self, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Return a DataFrame keyed by UTC timestamp with feature columns.

        Index: monotonic ``DatetimeIndex`` in UTC. Columns: arbitrary numeric
        features. The miner drops NaNs and constant columns before computing
        rank-IC, so sources are free to emit ragged or partially-populated
        frames.
        """
        ...


def _asset_class_label(value: Any) -> str:
    """Coerce an enum / string asset-class value into a stable string label.

    Accepts ``AssetClass.SPOT_CRYPTO`` (returns ``"spot_crypto"``) or a raw
    string (returned as-is). Used so the miner can stay agnostic of whether
    a source uses the protocols enum.
    """
    if value is None:
        return "unknown"
    # Enum-like: prefer .value over .name so we get the snake_case string.
    if hasattr(value, "value") and not isinstance(value, str):
        return str(value.value)
    return str(value)


def _spearman_rank_ic(
    signal: pd.Series, forward_return: pd.Series
) -> Tuple[float, int, float]:
    """Compute the (rank-IC, n_samples, pvalue) triple for a paired series.

    Returns ``(0.0, 0, 1.0)`` when the input is degenerate:
      * fewer than 3 valid pairs after NaN drop, or
      * either column is constant (zero variance ⇒ Spearman is undefined).

    SciPy is the source of truth for the p-value; we don't try to derive it
    by hand. SciPy's deprecation of ``scipy.stats.spearmanr`` for the
    ``correlation`` attribute is handled by reading ``.statistic`` first
    with a fallback to ``.correlation``.
    """
    paired = pd.concat([signal, forward_return], axis=1).dropna()
    if len(paired) < 3:
        return 0.0, len(paired), 1.0

    a_col = paired.iloc[:, 0]
    b_col = paired.iloc[:, 1]
    if a_col.nunique(dropna=True) < 2 or b_col.nunique(dropna=True) < 2:
        return 0.0, len(paired), 1.0

    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - scipy is in requirements.txt
        # Numpy fallback — rank-correlate manually. Loses the p-value.
        a_rank = a_col.rank()
        b_rank = b_col.rank()
        denom = a_rank.std() * b_rank.std()
        if denom == 0:
            return 0.0, len(paired), 1.0
        rank_ic = float(((a_rank - a_rank.mean()) * (b_rank - b_rank.mean())).mean() / denom)
        return rank_ic, len(paired), 1.0

    res = spearmanr(a_col.to_numpy(), b_col.to_numpy())
    # SciPy >= 1.9 uses ``.statistic``; older releases used ``.correlation``.
    rank_ic = float(getattr(res, "statistic", getattr(res, "correlation", float("nan"))))
    pvalue = float(getattr(res, "pvalue", 1.0))
    if not np.isfinite(rank_ic):
        return 0.0, len(paired), 1.0
    return rank_ic, len(paired), pvalue


def _forward_return(series: pd.Series, horizon_bars: int) -> pd.Series:
    """Return the per-bar forward return of ``series`` over ``horizon_bars``.

    ``series.shift(-horizon)`` looks ``horizon`` bars into the future, so the
    last ``horizon`` rows become NaN — they're dropped by ``_spearman_rank_ic``
    when paired with the signal column.

    For features that aren't price-like (e.g. orderbook depth), the
    "forward return" is just the percent change of the feature itself
    over the horizon. The miner is agnostic of whether feature_b is a
    price or a generic feature; the rank-IC measures co-movement either
    way.
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    shifted = series.shift(-horizon_bars)
    # pct_change-style return; safe against zero-baseline divides.
    base = series.replace(0, np.nan)
    return (shifted - series) / base


class CorrelationMiner:
    """Cartesian-product rank-IC miner across a list of FeatureSources.

    The miner is hermetic: it does not perform any IO of its own. Caller
    constructs one :class:`FeatureSource` per asset-class (or finer granularity
    if desired), passes them in, and calls :meth:`mine`. The miner pulls a
    fixed window from each source, then iterates the full cross-product of
    ``(source_a.features, source_b.features, horizon_bars)``.

    Asymmetry note: ``feature_a`` is treated as the *signal* (used at time
    ``t``), and ``feature_b``'s *forward return* over ``horizon_bars`` is the
    target. The cartesian product therefore includes both ``(A, B)`` and
    ``(B, A)`` orderings — they are distinct hypotheses ("does A predict B?"
    vs. "does B predict A?").

    Self-pairs (``feature_a == feature_b`` on the same source) are skipped to
    avoid trivial autocorrelation results dominating the ranking.
    """

    def __init__(
        self,
        feature_sources: Sequence[FeatureSource],
        horizon_bars_options: Sequence[int] = DEFAULT_HORIZON_BARS,
    ) -> None:
        if not feature_sources:
            raise ValueError("feature_sources must be non-empty")
        if not horizon_bars_options:
            raise ValueError("horizon_bars_options must be non-empty")
        for h in horizon_bars_options:
            if h <= 0:
                raise ValueError(f"horizon_bars must all be positive (got {h})")

        self.feature_sources: List[FeatureSource] = list(feature_sources)
        self.horizon_bars_options: List[int] = list(horizon_bars_options)

    def mine(
        self,
        window_days: int = 30,
        *,
        end_utc: Optional[datetime] = None,
    ) -> List[CorrelationResult]:
        """Pull windows, compute the cartesian product, return ranked results.

        Args:
            window_days: how far back to fetch from each source. The miner
                requests ``[end_utc - window_days, end_utc]`` from every
                source uniformly; per-source clamping is the source's job.
            end_utc: anchor for the right edge of the window. Defaults to
                ``datetime.now(timezone.utc)``. Tests inject a fixed value
                for determinism.

        Returns:
            A list of :class:`CorrelationResult`, sorted by ``|rank_ic|``
            descending (most-correlated pairs first). Empty list when no
            source returned data, or when there are no valid feature columns.
        """
        if window_days <= 0:
            raise ValueError("window_days must be positive")

        anchor = end_utc or datetime.now(timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        start = anchor - timedelta(days=window_days)

        # 1. Fetch every source's window once. A source that errors out is
        #    skipped with a warning rather than aborting the whole run — a
        #    nightly job should degrade gracefully when one venue is down.
        source_data: List[Tuple[FeatureSource, pd.DataFrame]] = []
        for source in self.feature_sources:
            try:
                df = source.fetch_window(start, anchor)
            except Exception as exc:  # noqa: BLE001 - hermetic best-effort
                LOGGER.warning(
                    "alpha_lab: source %r raised during fetch_window: %s",
                    getattr(source, "name", repr(source)),
                    exc,
                )
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            # Normalize index to a sorted DatetimeIndex so paired alignment
            # via pd.concat below joins on identical labels.
            if not isinstance(df.index, pd.DatetimeIndex):
                # Tolerate sources that emit a "timestamp" column instead of
                # an index — common with parquet exports.
                if "timestamp" in df.columns:
                    df.index = pd.to_datetime(df["timestamp"], utc=True)
                    df = df.drop(columns=["timestamp"])
                else:
                    LOGGER.warning(
                        "alpha_lab: source %r DataFrame has no DatetimeIndex; "
                        "skipping",
                        getattr(source, "name", repr(source)),
                    )
                    continue
            df = df.sort_index()
            # Drop non-numeric columns (the rank-IC is undefined on strings)
            # and columns that are entirely NaN.
            numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
            if numeric_df.empty:
                continue
            source_data.append((source, numeric_df))

        if not source_data:
            return []

        # 2. Cartesian product over (source_a, source_b, feature_a, feature_b,
        #    horizon). Note: this is the upper bound; same-source/same-feature
        #    pairs are skipped inside the loop.
        computed_at = datetime.now(timezone.utc).isoformat()
        results: List[CorrelationResult] = []
        for (src_a, df_a), (src_b, df_b) in product(source_data, repeat=2):
            class_a = _asset_class_label(getattr(src_a, "asset_class", "unknown"))
            class_b = _asset_class_label(getattr(src_b, "asset_class", "unknown"))
            # Align on the intersection of timestamps so paired Spearman
            # operates on a single index. Sources at different bar sizes
            # will collapse to whatever they share; for production, callers
            # should resample upstream to a common bar size.
            joined_index = df_a.index.intersection(df_b.index)
            if len(joined_index) < 4:
                # Need at least 4 overlapping bars to even form a single
                # (signal_t, forward_return_t+h) pair at horizon=1.
                continue
            df_a_aligned = df_a.loc[joined_index]
            df_b_aligned = df_b.loc[joined_index]

            for feat_a, feat_b, horizon in product(
                df_a_aligned.columns, df_b_aligned.columns, self.horizon_bars_options
            ):
                # Skip same-source autocorrelation (same column on same source).
                if src_a is src_b and feat_a == feat_b:
                    continue
                signal = df_a_aligned[feat_a]
                target = _forward_return(df_b_aligned[feat_b], horizon)
                rank_ic, n_samples, pvalue = _spearman_rank_ic(signal, target)
                pair = FeaturePair(
                    feature_a=str(feat_a),
                    feature_b=str(feat_b),
                    horizon_bars=int(horizon),
                    asset_class_a=class_a,
                    asset_class_b=class_b,
                )
                results.append(
                    CorrelationResult(
                        pair=pair,
                        rank_ic=float(rank_ic),
                        n_samples=int(n_samples),
                        pvalue=float(pvalue),
                        computed_at_utc=computed_at,
                    )
                )

        results.sort(key=lambda r: abs(r.rank_ic), reverse=True)
        return results
