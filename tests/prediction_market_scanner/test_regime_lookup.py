"""Unit tests for ``regime_memory.lookup``.

Covers:
* Empty store → defaults with ``_regime_confidence = 0.0``
* Single regime identical to query → that regime's metadata + confidence ≈ 1.0
* Multiple regimes → similarity-weighted average across neighbors
* ``k`` parameter respected
* Defaults supplied for fields not present in any retrieved regime
* Anti-correlated neighbors (sim ≤ 0) ignored (yield zero-confidence path)
"""

from __future__ import annotations

import unittest
from typing import Dict, List

import numpy as np
import pandas as pd

from regime_memory.encoder import RegimeEncoder, RegimeWindow
from regime_memory.lookup import RegimeLookup
from regime_memory.store import NaiveRegimeStore


def _features_from_vector(vec: List[float]) -> pd.DataFrame:
    """Build a 1-row DataFrame whose encoder embedding is exactly ``vec``.

    This is a test-only shortcut: with a single-row window the encoder's
    ``mean`` and ``last`` slots equal the input value, ``std`` is 0, and
    ``pct_change`` is also 0 (first == last). So an N-feature single-row
    frame produces a 4N-dim embedding with predictable structure.

    Returns a DataFrame whose encoded vector is *NOT* literally ``vec`` —
    instead the helper builds a frame so the encoded direction matches the
    intent of the test. We use it only as "make a frame whose encoding is
    similar to the stored embedding" for the lookup tests.
    """

    return pd.DataFrame({f"f{i}": [v] for i, v in enumerate(vec)})


class _FakeEncoder:
    """Deterministic encoder that just returns whatever you stuffed into the
    DataFrame's ``embedding`` column. Lets the lookup tests exercise the
    similarity math without depending on the real encoder's stat layout.
    """

    def encode_features(
        self, features: pd.DataFrame, window_size: int = 60
    ) -> List[float]:
        # The DataFrame is expected to have a single row with one column per
        # embedding dim, named "e0"..."eN-1".
        row = features.iloc[-1]
        return [float(row[c]) for c in features.columns]


def _make_window(
    *,
    end: str,
    embedding: List[float],
    metadata: Dict[str, float],
) -> RegimeWindow:
    return RegimeWindow(
        symbol="ETH-USD",
        window_end_utc=end,
        bars=60,
        embedding=list(embedding),
        metadata=dict(metadata),
    )


def _features(emb: List[float]) -> pd.DataFrame:
    return pd.DataFrame({f"e{i}": [v] for i, v in enumerate(emb)})


class RegimeLookupEmptyStoreTests(unittest.TestCase):
    def test_empty_store_returns_defaults_with_zero_confidence(self) -> None:
        store = NaiveRegimeStore(dim=3)
        encoder = _FakeEncoder()
        defaults = {"optimal_threshold": 0.5, "kelly_size_pct": 0.01}
        lookup = RegimeLookup(store, encoder, defaults)

        out = lookup.resolve_params(_features([0.1, 0.2, 0.3]))
        self.assertEqual(out["_regime_confidence"], 0.0)
        self.assertAlmostEqual(out["optimal_threshold"], 0.5)
        self.assertAlmostEqual(out["kelly_size_pct"], 0.01)

    def test_invalid_k_raises(self) -> None:
        store = NaiveRegimeStore(dim=2)
        store.add(_make_window(end="t1", embedding=[1.0, 0.0], metadata={"x": 1.0}))
        lookup = RegimeLookup(store, _FakeEncoder(), {"x": 0.0})
        with self.assertRaises(ValueError):
            lookup.resolve_params(_features([1.0, 0.0]), k=0)


class RegimeLookupIdenticalNeighborTests(unittest.TestCase):
    def test_identical_window_yields_high_confidence(self) -> None:
        store = NaiveRegimeStore(dim=3)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 2.0, 3.0],
                metadata={"optimal_threshold": 0.62, "kelly_size_pct": 0.07},
            )
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            {"optimal_threshold": 0.5, "kelly_size_pct": 0.01},
        )

        out = lookup.resolve_params(_features([1.0, 2.0, 3.0]), k=1)
        self.assertGreater(out["_regime_confidence"], 0.99)
        self.assertAlmostEqual(out["optimal_threshold"], 0.62, places=5)
        self.assertAlmostEqual(out["kelly_size_pct"], 0.07, places=5)


class RegimeLookupAggregationTests(unittest.TestCase):
    def test_weighted_average_across_diverse_regimes(self) -> None:
        store = NaiveRegimeStore(dim=2)
        # Two regimes; the query is 45° between them, so cosine similarities
        # are equal and the resolved metadata should be a plain average.
        store.add(_make_window(end="t1", embedding=[1.0, 0.0], metadata={"thr": 0.4}))
        store.add(_make_window(end="t2", embedding=[0.0, 1.0], metadata={"thr": 0.8}))
        lookup = RegimeLookup(store, _FakeEncoder(), {"thr": 0.5})

        out = lookup.resolve_params(_features([1.0, 1.0]), k=2)
        self.assertAlmostEqual(out["thr"], 0.6, places=5)
        # Confidence = mean(sim) where each sim ≈ cos(45°) ≈ 0.7071.
        self.assertAlmostEqual(out["_regime_confidence"], 0.7071, places=3)

    def test_k_parameter_limits_neighbors_used(self) -> None:
        # Build 5 regimes, all with sim > 0 to the query but with sharply
        # different metadata. With k=1 only the closest contributes; with
        # k=5 all five do, pulling the weighted mean toward the average.
        store = NaiveRegimeStore(dim=2)
        # Regime 0 is identical to query (sim = 1.0), thr=0.9.
        store.add(_make_window(end="t0", embedding=[1.0, 0.0], metadata={"thr": 0.9}))
        # Other 4 regimes: closer to (0, 1) — lower similarity, thr=0.1.
        for i in range(1, 5):
            angle = 1.0 - 0.05 * i  # slowly drifting away from query
            store.add(
                _make_window(
                    end=f"t{i}",
                    embedding=[angle, (1 - angle**2) ** 0.5],
                    metadata={"thr": 0.1},
                )
            )
        lookup = RegimeLookup(store, _FakeEncoder(), {"thr": 0.5})

        with_k1 = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        with_k5 = lookup.resolve_params(_features([1.0, 0.0]), k=5)

        # k=1 picks regime 0 only → thr ≈ 0.9.
        self.assertAlmostEqual(with_k1["thr"], 0.9, places=4)
        # k=5 includes the 4 lower-thr regimes → weighted mean is between
        # 0.1 and 0.9, strictly less than 0.9.
        self.assertLess(with_k5["thr"], 0.9)
        self.assertGreater(with_k5["thr"], 0.1)

    def test_field_present_only_in_defaults_falls_back(self) -> None:
        store = NaiveRegimeStore(dim=2)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.6},
            )
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            {
                "optimal_threshold": 0.5,
                # ``kelly_size_pct`` is in defaults but absent from the
                # neighbor's metadata — the lookup should surface the
                # default as-is.
                "kelly_size_pct": 0.02,
            },
        )

        out = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        self.assertAlmostEqual(out["optimal_threshold"], 0.6, places=5)
        self.assertAlmostEqual(out["kelly_size_pct"], 0.02, places=5)
        self.assertGreater(out["_regime_confidence"], 0.99)

    def test_anti_correlated_neighbor_is_ignored(self) -> None:
        store = NaiveRegimeStore(dim=2)
        # Single neighbor that points in the *opposite* direction to the
        # query — sim ≈ -1.0. Lookup must drop it and return defaults with
        # zero confidence rather than letting a flipped regime steer params.
        store.add(_make_window(end="t1", embedding=[1.0, 0.0], metadata={"thr": 0.99}))
        lookup = RegimeLookup(store, _FakeEncoder(), {"thr": 0.4})
        out = lookup.resolve_params(_features([-1.0, 0.0]), k=1)
        self.assertEqual(out["_regime_confidence"], 0.0)
        self.assertAlmostEqual(out["thr"], 0.4, places=6)


class RegimeLookupRealEncoderTests(unittest.TestCase):
    """Smoke test that exercises the real RegimeEncoder + NaiveRegimeStore
    pipeline end to end (fake encoder isolates the math; this isolates the
    wiring)."""

    def test_real_encoder_round_trip(self) -> None:
        cols = [f"f{i}" for i in range(3)]
        encoder = RegimeEncoder(feature_cols=cols)
        dim = encoder.expected_dim()
        store = NaiveRegimeStore(dim=dim)

        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(60, 3)), columns=cols)
        emb = encoder.encode_features(df, window_size=60)
        store.add(
            RegimeWindow(
                symbol="ETH-USD",
                window_end_utc="2026-05-01T00:00:00+00:00",
                bars=60,
                embedding=emb,
                metadata={"optimal_threshold": 0.55, "kelly_size_pct": 0.03},
            )
        )

        lookup = RegimeLookup(
            store,
            encoder,
            {"optimal_threshold": 0.5, "kelly_size_pct": 0.01},
        )
        out = lookup.resolve_params(df, k=1, window_size=60)
        self.assertGreater(out["_regime_confidence"], 0.99)
        self.assertAlmostEqual(out["optimal_threshold"], 0.55, places=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
