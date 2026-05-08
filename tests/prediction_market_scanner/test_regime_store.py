"""Unit tests for ``regime_memory.store``.

Covers:
* :class:`NaiveRegimeStore` round-trip: add → query → similarity ranking
* Empty store query returns ``[]`` (no crash)
* Mismatched embedding dim on add raises a clear :class:`ValueError`
* Cosine similarity values are bounded in ``[-1.0, 1.0]``
* Save → load round-trip preserves window count and metadata
* :func:`make_regime_store` picks the right backend
* Without FAISS, :class:`RegimeStore` constructor raises :class:`ImportError`
* With FAISS (skipped otherwise), the FAISS-backed store passes the same
  correctness contract as the naive store
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

from regime_memory.encoder import RegimeWindow
from regime_memory.store import (
    NaiveRegimeStore,
    RegimeStore,
    make_regime_store,
)
from regime_memory import store as store_module


def _make_window(
    *,
    symbol: str = "ETH-USD",
    end: str = "2026-05-01T12:00:00+00:00",
    bars: int = 60,
    embedding: List[float],
    metadata: dict | None = None,
) -> RegimeWindow:
    return RegimeWindow(
        symbol=symbol,
        window_end_utc=end,
        bars=bars,
        embedding=list(embedding),
        metadata=dict(metadata or {"optimal_threshold": 0.6, "kelly_size_pct": 0.05}),
    )


class NaiveRegimeStoreTests(unittest.TestCase):
    def test_add_and_query_returns_nearest_first(self) -> None:
        store = NaiveRegimeStore(dim=4)
        # Three deliberately-spread embeddings; the query is closest to the
        # third.
        store.add(_make_window(end="t1", embedding=[1.0, 0.0, 0.0, 0.0]))
        store.add(_make_window(end="t2", embedding=[0.0, 1.0, 0.0, 0.0]))
        store.add(_make_window(end="t3", embedding=[0.6, 0.8, 0.0, 0.0]))

        results = store.query([0.6, 0.8, 0.0, 0.0], k=3)
        self.assertEqual(len(results), 3)
        # Top match should be the (0.6, 0.8, ...) window with sim ≈ 1.0.
        top_window, top_sim = results[0]
        self.assertEqual(top_window.window_end_utc, "t3")
        self.assertAlmostEqual(top_sim, 1.0, places=5)
        # Sims must be monotonically non-increasing.
        sims = [s for _, s in results]
        for a, b in zip(sims, sims[1:]):
            self.assertGreaterEqual(a, b)

    def test_query_on_empty_store_returns_empty_list(self) -> None:
        store = NaiveRegimeStore(dim=4)
        self.assertEqual(store.query([0.1, 0.2, 0.3, 0.4], k=5), [])

    def test_add_many_matches_one_by_one(self) -> None:
        a = NaiveRegimeStore(dim=3)
        b = NaiveRegimeStore(dim=3)
        windows = [
            _make_window(end=f"t{i}", embedding=[float(i), 1.0, 0.0])
            for i in range(5)
        ]
        for w in windows:
            a.add(w)
        b.add_many(windows)
        self.assertEqual(len(a), len(b))
        ra = a.query([1.0, 1.0, 0.0], k=3)
        rb = b.query([1.0, 1.0, 0.0], k=3)
        self.assertEqual(
            [w.window_end_utc for w, _ in ra],
            [w.window_end_utc for w, _ in rb],
        )

    def test_add_with_wrong_dim_raises_value_error(self) -> None:
        store = NaiveRegimeStore(dim=4)
        with self.assertRaises(ValueError) as ctx:
            store.add(_make_window(embedding=[1.0, 0.0]))
        msg = str(ctx.exception)
        self.assertIn("dim mismatch", msg)
        self.assertIn("4", msg)

    def test_query_with_wrong_dim_raises_value_error(self) -> None:
        store = NaiveRegimeStore(dim=4)
        store.add(_make_window(embedding=[0.5] * 4))
        with self.assertRaises(ValueError):
            store.query([0.5, 0.5], k=1)

    def test_cosine_similarity_bounded(self) -> None:
        store = NaiveRegimeStore(dim=4)
        rng = np.random.default_rng(7)
        for i in range(50):
            v = rng.normal(size=4).astype(float).tolist()
            store.add(_make_window(end=f"t{i}", embedding=v))
        # Query with various directions including the negation of one stored
        # vector — the resulting similarity must lie in [-1, 1].
        q = rng.normal(size=4).astype(float).tolist()
        results = store.query(q, k=10)
        for _, sim in results:
            self.assertGreaterEqual(sim, -1.0 - 1e-6)
            self.assertLessEqual(sim, 1.0 + 1e-6)

    def test_zero_vector_does_not_propagate_nan(self) -> None:
        store = NaiveRegimeStore(dim=3)
        store.add(_make_window(end="zero", embedding=[0.0, 0.0, 0.0]))
        store.add(_make_window(end="real", embedding=[1.0, 0.0, 0.0]))
        results = store.query([1.0, 0.0, 0.0], k=2)
        self.assertEqual(len(results), 2)
        for _, sim in results:
            self.assertFalse(math.isnan(sim))

    def test_save_load_round_trip_preserves_data(self) -> None:
        store = NaiveRegimeStore(dim=3)
        store.add(
            _make_window(
                symbol="ETH-USD",
                end="t1",
                embedding=[1.0, 2.0, 3.0],
                metadata={"optimal_threshold": 0.55, "kelly_size_pct": 0.02},
            )
        )
        store.add(
            _make_window(
                symbol="BTC-USD",
                end="t2",
                embedding=[4.0, 5.0, 6.0],
                metadata={"optimal_threshold": 0.65, "kelly_size_pct": 0.04},
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.npz"
            store.save(path)
            self.assertTrue(path.exists())

            loaded = NaiveRegimeStore.load(path)
            self.assertEqual(len(loaded), 2)

            # Querying for the second stored window should rank it first.
            results = loaded.query([4.0, 5.0, 6.0], k=2)
            top_window, top_sim = results[0]
            self.assertEqual(top_window.symbol, "BTC-USD")
            self.assertEqual(top_window.window_end_utc, "t2")
            self.assertAlmostEqual(top_sim, 1.0, places=5)
            # Metadata survived the round trip.
            self.assertAlmostEqual(top_window.metadata["optimal_threshold"], 0.65, places=6)
            self.assertAlmostEqual(top_window.metadata["kelly_size_pct"], 0.04, places=6)

    def test_save_load_empty_store(self) -> None:
        store = NaiveRegimeStore(dim=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.npz"
            store.save(path)
            loaded = NaiveRegimeStore.load(path)
            self.assertEqual(len(loaded), 0)
            self.assertEqual(loaded.query([0.0, 0.0, 0.0, 0.0], k=3), [])

    def test_constructor_rejects_bad_dim(self) -> None:
        with self.assertRaises(ValueError):
            NaiveRegimeStore(dim=0)
        with self.assertRaises(ValueError):
            NaiveRegimeStore(dim=-3)

    def test_query_k_must_be_positive(self) -> None:
        store = NaiveRegimeStore(dim=2)
        store.add(_make_window(embedding=[1.0, 0.0]))
        with self.assertRaises(ValueError):
            store.query([1.0, 0.0], k=0)


class FactoryTests(unittest.TestCase):
    def test_factory_returns_naive_when_faiss_unavailable(self) -> None:
        # When FAISS isn't installed (the project's default), the factory
        # must hand back the numpy-backed store regardless of preference.
        if store_module._HAS_FAISS:
            self.skipTest("FAISS is installed; covered by the FAISS test")
        s = make_regime_store(dim=8)
        self.assertIsInstance(s, NaiveRegimeStore)

    def test_factory_respects_prefer_faiss_false(self) -> None:
        # Even when FAISS is available, prefer_faiss=False must yield the
        # numpy store — useful for tests that want to exercise both paths.
        s = make_regime_store(dim=8, prefer_faiss=False)
        self.assertIsInstance(s, NaiveRegimeStore)

    def test_regime_store_constructor_raises_without_faiss(self) -> None:
        if store_module._HAS_FAISS:
            self.skipTest("FAISS is installed; the no-FAISS branch can't fire")
        with self.assertRaises(ImportError) as ctx:
            RegimeStore(dim=4)
        self.assertIn("faiss", str(ctx.exception).lower())


@unittest.skipUnless(store_module._HAS_FAISS, "faiss-cpu not installed")
class FaissRegimeStoreTests(unittest.TestCase):
    """Mirror the naive-store correctness contract for the FAISS backend."""

    def test_faiss_round_trip(self) -> None:
        s = RegimeStore(dim=4)
        s.add(_make_window(end="t1", embedding=[1.0, 0.0, 0.0, 0.0]))
        s.add(_make_window(end="t2", embedding=[0.6, 0.8, 0.0, 0.0]))
        results = s.query([0.6, 0.8, 0.0, 0.0], k=2)
        self.assertEqual(results[0][0].window_end_utc, "t2")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

    def test_faiss_save_load(self) -> None:
        s = RegimeStore(dim=3)
        s.add_many(
            [
                _make_window(end="t1", embedding=[1.0, 2.0, 3.0]),
                _make_window(end="t2", embedding=[3.0, 2.0, 1.0]),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.npz"
            s.save(path)
            loaded = RegimeStore.load(path)
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
