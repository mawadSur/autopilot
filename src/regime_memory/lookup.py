"""Regime parameter lookup: encoder + store + defaults → resolved params dict.

This is the integration seam that lets a live predictor (or risk manager)
ask: "given the current feature window, what threshold / Kelly fraction did
we use during the most-similar historical regimes, and how confident are we
in that match?"

The resolution recipe is intentionally simple so it can be reasoned about
without re-reading FAISS internals:

1. Encode the current feature window via :class:`RegimeEncoder`.
2. Query the store for the top-``k`` cosine-similarity neighbors.
3. For each metadata field (``optimal_threshold``, ``kelly_size_pct``,
   ``regime_label``, ``realized_sharpe``, …) compute a similarity-weighted
   average across the neighbors that *have* that field. Fields that are
   absent from every neighbor fall back to the matching key in
   ``defaults``.
4. Return the resolved dict with an extra ``_regime_confidence`` field set
   to the mean cosine similarity over the neighbors actually used (range
   ``[0, 1]`` after we clip negative similarities to 0 — anti-correlated
   regimes provide no signal here).

Confidence semantics
--------------------
``_regime_confidence`` is the consumer-facing knob for trusting the
resolved params:

* ``0.0`` — the store was empty, the query failed, or every neighbor was
  anti-correlated. Consumers should fall back to static / default params.
* ``< 0.5`` — weak match. The :mod:`INTEGRATION.md` recommends using the
  resolved params as a soft prior at most.
* ``>= 0.5`` — strong match. Safe to override the static threshold /
  Kelly fraction directly.

Negative-similarity weights are dropped (clipped to zero) before averaging
because a regime that's the *opposite* of the current state shouldn't pull
the resolved threshold one way or the other — it should just be ignored.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from regime_memory.encoder import RegimeEncoder
from regime_memory.store import NaiveRegimeStore, RegimeStore


# Fields whose similarity-weighted average we always compute (when present).
# Other numeric metadata fields are also resolved — this list is just the
# documented "expected" set so callers know what defaults to provide.
_DEFAULT_RESOLVED_FIELDS: Tuple[str, ...] = (
    "optimal_threshold",
    "kelly_size_pct",
    "realized_sharpe",
    "regime_label",
)


_StoreT = Union[NaiveRegimeStore, RegimeStore]


class RegimeLookup:
    """Resolve runtime params from a feature window via nearest historical regimes.

    Parameters
    ----------
    store : :class:`NaiveRegimeStore` | :class:`RegimeStore`
        The vector store to query. Both backends are supported transparently
        because they share an identical public API.
    encoder : :class:`RegimeEncoder`
        The encoder used to embed the current feature window. Should match
        whatever encoder produced the windows in ``store`` — otherwise the
        cosine similarities are meaningless. Callers are responsible for
        keeping these aligned (we don't have a way to verify it from the
        encoder alone — that's a v1 upgrade).
    defaults : Mapping[str, float]
        Static fallback values for the resolved fields. Returned verbatim
        (with ``_regime_confidence = 0.0``) when the store is empty or the
        query yields no usable neighbors. Any metadata field that *no*
        neighbor exposes also falls back to the matching key here.
    """

    def __init__(
        self,
        store: _StoreT,
        encoder: RegimeEncoder,
        defaults: Mapping[str, float],
    ) -> None:
        self.store = store
        self.encoder = encoder
        # Copy so callers can mutate their own dict later without affecting
        # the lookup's fallback values.
        self.defaults: Dict[str, float] = dict(defaults)

    # -- public API ----------------------------------------------------------

    def resolve_params(
        self,
        features: pd.DataFrame,
        *,
        k: int = 10,
        window_size: int = 60,
    ) -> Dict[str, float]:
        """Return a dict of resolved params for the current window.

        Always includes ``_regime_confidence``. Other keys come from the
        union of (a) keys present in any neighbor's ``metadata`` and (b)
        keys present in ``defaults``. Keys absent from every neighbor fall
        back to the value in ``defaults`` (and ``0.0`` if not in defaults
        either — this branch is mostly defensive).

        Parameters
        ----------
        features : pd.DataFrame
            Feature window to encode. Passed through to
            :meth:`RegimeEncoder.encode_features`.
        k : int, default 10
            Number of neighbors to consult.
        window_size : int, default 60
            Trailing window size for encoding. Forwarded to the encoder.
        """

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")

        # Empty store: short-circuit to defaults so we don't pay the encoder
        # cost on a query that can't return anything.
        if len(self.store) == 0:
            return self._defaults_with_zero_confidence()

        embedding = self.encoder.encode_features(features, window_size=window_size)
        neighbors: List[Tuple[object, float]] = self.store.query(embedding, k=k)
        if not neighbors:
            return self._defaults_with_zero_confidence()

        # Drop neighbors whose similarity is non-positive — they provide no
        # constructive evidence. If everyone is non-positive, fall back.
        usable = [(w, sim) for w, sim in neighbors if sim > 0.0]
        if not usable:
            return self._defaults_with_zero_confidence()

        # Confidence = mean of usable similarities, clipped to [0, 1].
        confidence = sum(sim for _, sim in usable) / len(usable)
        confidence = max(0.0, min(1.0, confidence))

        # Determine the union of metadata keys present anywhere in the
        # neighbor set, plus the keys we have static defaults for. This way
        # a default-only field (no neighbor knows it) still appears in the
        # output, and a neighbor-only field surfaces too.
        all_keys: set[str] = set(self.defaults.keys())
        for w, _ in usable:
            all_keys.update(getattr(w, "metadata", {}).keys())

        resolved: Dict[str, float] = {}
        for key in all_keys:
            weighted_sum = 0.0
            weight_total = 0.0
            for w, sim in usable:
                meta = getattr(w, "metadata", {})
                if key in meta:
                    weighted_sum += float(meta[key]) * sim
                    weight_total += sim
            if weight_total > 0.0:
                resolved[key] = weighted_sum / weight_total
            else:
                # No neighbor exposed this field; fall back.
                resolved[key] = float(self.defaults.get(key, 0.0))

        resolved["_regime_confidence"] = float(confidence)
        return resolved

    # -- helpers -------------------------------------------------------------

    def _defaults_with_zero_confidence(self) -> Dict[str, float]:
        out = {k: float(v) for k, v in self.defaults.items()}
        out["_regime_confidence"] = 0.0
        return out


__all__ = [
    "RegimeLookup",
]
