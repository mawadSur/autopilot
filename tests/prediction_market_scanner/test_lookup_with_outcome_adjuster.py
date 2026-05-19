"""Integration tests for :class:`RegimeLookup` + :class:`OutcomeAdjuster`.

Covers:
* New observable keys (``_outcome_adjustment_delta``, ``_regime_label``)
  appear when an adjuster is wired.
* When NO adjuster is wired, the resolved dict has EXACTLY the same keys
  as today (backward compatibility — Sprint 2 #6 hard constraint).
* The resolved ``optimal_threshold`` equals neighbor-avg + delta,
  clipped to [0, 1].
* A ``redis.exceptions.RedisError`` from the adjuster degrades to
  delta=0.0 with a WARN log (no crash, no silent skip).
"""

from __future__ import annotations

import unittest
from typing import Dict, List
from unittest.mock import MagicMock

import fakeredis
import pandas as pd
import redis.exceptions

from regime_memory.encoder import RegimeWindow
from regime_memory.lookup import RegimeLookup
from regime_memory.outcome_adjuster import OutcomeAdjuster
from regime_memory.store import NaiveRegimeStore


class _FakeEncoder:
    """Echo the input frame's columns straight into the embedding."""

    def encode_features(
        self, features: pd.DataFrame, window_size: int = 60
    ) -> List[float]:
        row = features.iloc[-1]
        return [float(row[c]) for c in features.columns]


def _features(emb: List[float]) -> pd.DataFrame:
    return pd.DataFrame({f"e{i}": [v] for i, v in enumerate(emb)})


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


def _make_adjuster(client) -> OutcomeAdjuster:
    return OutcomeAdjuster(
        client,
        namespace="test-lookup-adj",
        max_adjustment=0.05,
        losses_to_raise=3,
        wins_to_relax=5,
        per_event_delta=0.01,
    )


class BackwardCompatTests(unittest.TestCase):
    """Without an adjuster wired, behavior is identical to today."""

    def test_no_adjuster_no_new_keys(self) -> None:
        store = NaiveRegimeStore(dim=2)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.6, "regime_label": 2.0},
            )
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            defaults={"optimal_threshold": 0.5},
        )
        out = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        self.assertNotIn("_outcome_adjustment_delta", out)
        self.assertNotIn("_regime_label", out)
        self.assertAlmostEqual(out["optimal_threshold"], 0.6, places=5)


class AdjusterAppliedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.adjuster = _make_adjuster(self.client)

    def test_resolved_keys_appear_when_adjuster_wired(self) -> None:
        store = NaiveRegimeStore(dim=2)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.6, "regime_label": 2.0},
            )
        )
        # Seed an adjustment for the trend_up label (regime_label=2.0).
        self.client.hset(
            self.adjuster.full_hash_key, "trend_up", "0.020000"
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            defaults={"optimal_threshold": 0.5},
            outcome_adjuster=self.adjuster,
        )
        out = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        self.assertIn("_outcome_adjustment_delta", out)
        self.assertIn("_regime_label", out)
        self.assertEqual(out["_regime_label"], "trend_up")
        self.assertAlmostEqual(out["_outcome_adjustment_delta"], 0.02, places=6)
        # 0.6 base + 0.02 adjustment
        self.assertAlmostEqual(out["optimal_threshold"], 0.62, places=6)

    def test_no_adjustment_for_unseen_label(self) -> None:
        store = NaiveRegimeStore(dim=2)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.6, "regime_label": 1.0},
            )
        )
        # Hash is empty -> delta should be 0.0, threshold unchanged.
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            defaults={"optimal_threshold": 0.5},
            outcome_adjuster=self.adjuster,
        )
        out = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        self.assertEqual(out["_regime_label"], "chop")
        self.assertAlmostEqual(out["_outcome_adjustment_delta"], 0.0, places=6)
        self.assertAlmostEqual(out["optimal_threshold"], 0.6, places=6)

    def test_threshold_clipped_to_unit_interval(self) -> None:
        # Plant a deliberately corrupt 0.99 value (bypasses the adjuster's
        # write-side clip). current_adjustment clips to +0.05, then the
        # lookup adds +0.05 to a base 0.97 -> 1.02 which must clip to 1.0.
        self.client.hset(
            self.adjuster.full_hash_key, "trend_up", "0.99"
        )
        store = NaiveRegimeStore(dim=2)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.97, "regime_label": 2.0},
            )
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            defaults={"optimal_threshold": 0.5},
            outcome_adjuster=self.adjuster,
        )
        out = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        self.assertEqual(out["_regime_label"], "trend_up")
        # current_adjustment clips to +0.05; base 0.97 + 0.05 = 1.02 -> 1.0
        self.assertAlmostEqual(out["optimal_threshold"], 1.0, places=6)


class RedisErrorTests(unittest.TestCase):
    def test_redis_error_falls_back_to_zero_delta(self) -> None:
        # A mock adjuster whose current_adjustment raises RedisError.
        bad_adj = MagicMock(spec=OutcomeAdjuster)
        bad_adj.current_adjustment.side_effect = redis.exceptions.ConnectionError(
            "kaboom"
        )

        store = NaiveRegimeStore(dim=2)
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.6, "regime_label": 2.0},
            )
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            defaults={"optimal_threshold": 0.5},
            outcome_adjuster=bad_adj,
        )
        with self.assertLogs("regime_memory.lookup", level="WARNING") as cm:
            out = lookup.resolve_params(_features([1.0, 0.0]), k=1)
        self.assertAlmostEqual(out["_outcome_adjustment_delta"], 0.0, places=6)
        self.assertAlmostEqual(out["optimal_threshold"], 0.6, places=6)
        self.assertTrue(
            any("outcome_adjuster read failed" in line for line in cm.output)
        )


class ClosestNeighborSelectionTests(unittest.TestCase):
    """The label applied is the CLOSEST neighbor's, not a weighted one."""

    def test_closest_neighbor_label_used(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        adjuster = _make_adjuster(client)
        # Seed: trend_up gets +0.03, trend_down gets +0.02.
        client.hset(adjuster.full_hash_key, "trend_up", "0.030000")
        client.hset(adjuster.full_hash_key, "trend_down", "0.020000")

        store = NaiveRegimeStore(dim=2)
        # Query [1, 0]: identical to first neighbor (sim=1), closer than
        # the second neighbor (sim < 1).
        store.add(
            _make_window(
                end="t1",
                embedding=[1.0, 0.0],
                metadata={"optimal_threshold": 0.5, "regime_label": 2.0},
            )
        )
        store.add(
            _make_window(
                end="t2",
                embedding=[0.5, 0.5],
                metadata={"optimal_threshold": 0.5, "regime_label": 0.0},
            )
        )
        lookup = RegimeLookup(
            store,
            _FakeEncoder(),
            defaults={"optimal_threshold": 0.5},
            outcome_adjuster=adjuster,
        )
        out = lookup.resolve_params(_features([1.0, 0.0]), k=2)
        # Closest neighbor is trend_up -> delta = 0.03.
        self.assertEqual(out["_regime_label"], "trend_up")
        self.assertAlmostEqual(out["_outcome_adjustment_delta"], 0.03, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
