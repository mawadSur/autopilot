"""Regime window encoder (E4 v0).

Defines the data record (:class:`RegimeWindow`) and the encoding strategy
(:class:`RegimeEncoder`) that turns a ``window_size`` slice of a feature
DataFrame into a fixed-dim float vector for k-NN regime lookup.

Encoding strategy (v0)
----------------------
For each feature column we compute four summary statistics over the window:

* ``mean`` — central tendency
* ``std``  — within-window dispersion
* ``last`` — most recent bar value (carries the "where are we now" signal
  that pure mean/std would smooth away)
* ``pct_change`` — ``(last - first) / (|first| + eps)``, a coarse drift /
  trend proxy

The 4 stats are concatenated in a fixed order (``mean, std, last,
pct_change``) per column, then the columns are concatenated in the order
the caller passes (default: dataframe column order). For 36 features × 4
stats this gives a 144-dim embedding — small enough that even the numpy
brute-force store works at sub-millisecond on tens of thousands of windows.

The choice is deliberately simple. A learned encoder (autoencoder /
contrastive learning trained to make windows with similar future-return
distributions cluster together) is the planned v1 upgrade — the public API
of :meth:`RegimeEncoder.encode_features` is the seam that swap will live
behind. Until then the stats above are interpretable, deterministic, and
have no training dependency.

NaN handling
------------
Pandas' rolling features pad early bars with NaN. We fill NaN values with
their **column mean** before computing the summary stats so a single bad
column doesn't poison the whole vector. If an entire column is NaN we fall
back to ``0.0`` — that column carries no regime information for this
window. The choice (column-mean fill, then 0.0) trades off vs. dropping
NaN columns entirely; we pick fill so the embedding dimension stays fixed
across windows. Without that, downstream FAISS / numpy stores would have
to special-case ragged inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --- module constants -------------------------------------------------------

#: Names of the per-column summary statistics, in the order they appear in the
#: concatenated embedding. Bumping this list is a backward-incompatible change
#: to the embedding format — bump :attr:`RegimeEncoder.VERSION` accordingly.
SUMMARY_STAT_NAMES: tuple[str, ...] = ("mean", "std", "last", "pct_change")

#: A small constant added to denominators to avoid divide-by-zero. Same magnitude
#: as ``utils.compute_features`` uses for analogous guards.
_EPS: float = 1e-12


# --- data record ------------------------------------------------------------


@dataclass(frozen=True)
class RegimeWindow:
    """A single embedded historical window with looked-up parameter metadata.

    Frozen so instances are hashable (lookups can dedupe by identity if a
    caller wants). The ``embedding`` and ``metadata`` fields are stored as
    plain Python types (``list``, ``dict``) for clean JSON serialization in
    persistence layers.

    Attributes
    ----------
    symbol : str
        Trading symbol the window came from (e.g. ``"ETH-USD"``). Carried so
        downstream consumers can filter neighbors to same-symbol regimes.
    window_end_utc : str
        ISO-8601 UTC timestamp of the *last* bar in the window. The encoder
        does not validate the format — callers are responsible for passing a
        parseable string.
    bars : int
        Number of bars in the window the embedding was computed over.
    embedding : list[float]
        The concatenated summary-stat vector. Length = ``n_features * 4``.
    metadata : dict[str, float]
        Parameters resolved post-hoc from this window — typically
        ``optimal_threshold``, ``kelly_size_pct``, ``regime_label``,
        ``realized_sharpe``. The :class:`RegimeLookup` similarity-weighted-
        averages these across the k nearest neighbors at inference.
    """

    symbol: str
    window_end_utc: str
    bars: int
    embedding: List[float] = field(hash=False)
    metadata: Dict[str, float] = field(hash=False)

    def __hash__(self) -> int:
        # ``embedding`` and ``metadata`` are mutable container types so they
        # can't be hashed directly. The (symbol, window_end_utc, bars) triple
        # is unique per window in practice (a single symbol can't have two
        # windows ending at the same instant with the same length), so we
        # hash on it and skip the embedding for hash purposes.
        return hash((self.symbol, self.window_end_utc, self.bars))


# --- encoder ----------------------------------------------------------------


class RegimeEncoder:
    """Turn a feature DataFrame window into a fixed-dim embedding vector.

    Parameters
    ----------
    feature_cols : Sequence[str], optional
        Subset of columns to encode. If ``None``, every column in the input
        DataFrame is used (in DataFrame order). Pinning the column list at
        construction time is the recommended way to guarantee deterministic
        embedding dimensionality across calls — without it, two calls with
        different DataFrames would silently produce different-length vectors
        and the FAISS store would reject one of them.
    """

    #: Bump this on any backward-incompatible change to the embedding format
    #: (different stats, different ordering, different NaN policy, etc.).
    #: Stored alongside the FAISS index to detect cross-version collisions.
    VERSION: str = "v0"

    def __init__(self, feature_cols: Optional[Sequence[str]] = None) -> None:
        self.feature_cols: Optional[List[str]] = (
            list(feature_cols) if feature_cols is not None else None
        )

    # -- public API ----------------------------------------------------------

    def encode_features(
        self,
        features: pd.DataFrame,
        window_size: int = 60,
    ) -> List[float]:
        """Encode the *trailing* ``window_size`` rows of ``features``.

        If the DataFrame is longer than ``window_size`` only the tail is used
        (matches the inference-time pattern in :class:`predictor.LegacyPredictor`
        where a rolling buffer is kept and the most recent window is sliced
        off the end). If shorter, every row is used.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame of engineered features. Numeric columns only — non-
            numeric columns will raise during stat computation.
        window_size : int, default 60
            Number of trailing bars to include. Must be >= 1.

        Returns
        -------
        list[float]
            The concatenated summary-stat vector. Length is deterministic
            for a given ``feature_cols`` setting:
            ``len(feature_cols) * len(SUMMARY_STAT_NAMES)``.

        Raises
        ------
        ValueError
            If ``features`` is empty or ``window_size < 1``.
        """
        if window_size < 1:
            raise ValueError(
                f"window_size must be >= 1, got {window_size!r}"
            )
        if features is None or len(features) == 0:
            raise ValueError(
                "encode_features requires a non-empty DataFrame; got "
                f"{0 if features is None else len(features)} rows"
            )

        cols = self.feature_cols or list(features.columns)
        # Slice trailing window; if features has fewer rows than window_size,
        # use everything we have.
        window = features.tail(window_size)
        # Project to the configured columns. Missing columns become all-NaN
        # so they still occupy their slot in the output dim — this preserves
        # vector dimensionality even when the caller passes a DataFrame that
        # is missing some columns.
        projected = pd.DataFrame(
            {c: window[c] if c in window.columns else np.nan for c in cols},
            index=window.index,
        )

        # Replace inf with NaN so the same fill strategy handles both.
        projected = projected.replace([np.inf, -np.inf], np.nan)

        # Per-column mean fill. Where the entire column is NaN this returns
        # NaN — we then replace those with 0.0 below.
        col_means = projected.mean(axis=0, skipna=True)
        filled = projected.fillna(col_means)
        filled = filled.fillna(0.0)

        # Compute stats. Using numpy is faster than pandas .agg for small
        # frames and guarantees consistent dtype semantics.
        arr = filled.to_numpy(dtype=np.float64, copy=False)

        if arr.shape[0] == 0:
            # Defensive: caller passed a DataFrame whose tail is empty after
            # projection. We treated len(features) == 0 above already, so
            # this shouldn't fire, but is cheap insurance.
            raise ValueError("encoded window is empty after projection")

        means = np.nanmean(arr, axis=0)
        # ddof=0 so a single-row window doesn't produce NaN std.
        stds = np.nanstd(arr, axis=0, ddof=0)
        last = arr[-1]
        first = arr[0]
        pct_change = (last - first) / (np.abs(first) + _EPS)

        # Replace any residual NaN/inf (can happen if the entire column was
        # NaN AND mean fill couldn't recover it; means/stds will be NaN).
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
        last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)
        pct_change = np.nan_to_num(pct_change, nan=0.0, posinf=0.0, neginf=0.0)

        # Stack as (4, n_features) and flatten in (stat, feature) order so
        # the layout is [feat0_mean, feat0_std, feat0_last, feat0_pct,
        # feat1_mean, ...]. Iterating per-feature is cleaner than per-stat
        # for code that wants to reason about a single feature's slice.
        per_feat = np.stack([means, stds, last, pct_change], axis=0)  # (4, F)
        embedding = per_feat.T.reshape(-1)  # (F * 4,)

        return embedding.astype(float).tolist()

    # -- introspection -------------------------------------------------------

    def expected_dim(self, n_features: Optional[int] = None) -> int:
        """Return the embedding dimension this encoder produces.

        If ``feature_cols`` was set at construction, the dim is fully
        determined and ``n_features`` is ignored. Otherwise the caller must
        pass the column count of the DataFrames they intend to encode.
        """
        if self.feature_cols is not None:
            n = len(self.feature_cols)
        else:
            if n_features is None:
                raise ValueError(
                    "expected_dim requires n_features when encoder was "
                    "constructed without feature_cols"
                )
            n = int(n_features)
        return n * len(SUMMARY_STAT_NAMES)


__all__ = [
    "SUMMARY_STAT_NAMES",
    "RegimeEncoder",
    "RegimeWindow",
]
