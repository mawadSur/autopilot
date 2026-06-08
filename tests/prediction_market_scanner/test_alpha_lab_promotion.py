"""Tests for alpha_lab.auto_promotion_gate.

Hermetic: in-memory store + tmpdir for the JSONL queue. No Redis.
A separate test exercises the Redis store via fakeredis if available.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from alpha_lab.auto_promotion_gate import (
    AutoPromotionGate,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_THRESHOLD,
    PromotionCandidate,
    _InMemoryHistoryStore,
    _RedisHistoryStore,
)
from alpha_lab.correlation_miner import CorrelationResult, FeaturePair


def _make_pair(suffix: str = "1") -> FeaturePair:
    return FeaturePair(
        feature_a=f"feat_a_{suffix}",
        feature_b=f"feat_b_{suffix}",
        horizon_bars=15,
        asset_class_a="spot_crypto",
        asset_class_b="prediction_binary",
    )


def _make_result(pair: FeaturePair, rank_ic: float, day_offset: int = 0) -> CorrelationResult:
    ts = datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(days=day_offset)
    return CorrelationResult(
        pair=pair,
        rank_ic=rank_ic,
        n_samples=200,
        pvalue=0.01,
        computed_at_utc=ts.isoformat(),
    )


class GateThresholdLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = Path(self.tmp.name) / "promotion_queue.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _gate(self, **kw) -> AutoPromotionGate:
        return AutoPromotionGate(promotion_queue_path=self.queue, **kw)

    def test_thirty_results_at_007_emit_one_candidate(self) -> None:
        gate = self._gate()
        pair = _make_pair()
        for i in range(DEFAULT_MIN_SAMPLES):
            gate.record(_make_result(pair, 0.07, day_offset=i))
        candidates = gate.check_promotions()
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertEqual(cand.pair, pair)
        self.assertAlmostEqual(cand.rank_ic_30d_avg, 0.07)
        self.assertEqual(cand.rank_ic_30d_count, DEFAULT_MIN_SAMPLES)
        # Audit log appended.
        self.assertTrue(self.queue.exists())
        lines = self.queue.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["pair"]["feature_a"], pair.feature_a)
        self.assertAlmostEqual(payload["rank_ic_30d_avg"], 0.07)

    def test_thirty_results_at_003_emit_no_candidate(self) -> None:
        gate = self._gate()
        pair = _make_pair()
        for i in range(DEFAULT_MIN_SAMPLES):
            gate.record(_make_result(pair, 0.03, day_offset=i))
        self.assertEqual(gate.check_promotions(), [])
        # No queue file written when no promotions emitted.
        self.assertFalse(self.queue.exists())

    def test_twentynine_results_at_010_emit_no_candidate_below_min_samples(self) -> None:
        gate = self._gate()
        pair = _make_pair()
        for i in range(DEFAULT_MIN_SAMPLES - 1):
            gate.record(_make_result(pair, 0.10, day_offset=i))
        self.assertEqual(gate.check_promotions(), [])

    def test_negative_rank_ic_above_threshold_emits_candidate(self) -> None:
        """The threshold is on |rank_ic|; anti-correlations are surfaced too,
        with the sign preserved on rank_ic_30d_avg so the operator can decide.
        """
        gate = self._gate()
        pair = _make_pair("neg")
        for i in range(DEFAULT_MIN_SAMPLES):
            gate.record(_make_result(pair, -0.08, day_offset=i))
        candidates = gate.check_promotions()
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].rank_ic_30d_avg, -0.08)

    def test_re_recording_extends_window_correctly(self) -> None:
        gate = self._gate(min_samples=5, threshold_rank_ic=0.05)
        pair = _make_pair()
        # First batch: small rank_ic, below threshold but enough samples.
        for i in range(3):
            gate.record(_make_result(pair, 0.01, day_offset=i))
        self.assertEqual(gate.check_promotions(), [])
        # Top up with high rank_ic samples — total of 5 samples, mean climbs.
        for i in range(2):
            gate.record(_make_result(pair, 0.20, day_offset=10 + i))
        candidates = gate.check_promotions()
        self.assertEqual(len(candidates), 1)
        # Mean of [0.01, 0.01, 0.01, 0.20, 0.20] = 0.086
        self.assertAlmostEqual(candidates[0].rank_ic_30d_avg, (0.03 + 0.40) / 5)
        self.assertEqual(candidates[0].rank_ic_30d_count, 5)

    def test_first_and_last_seen_set_from_history(self) -> None:
        gate = self._gate(min_samples=3, threshold_rank_ic=0.05)
        pair = _make_pair()
        # Record out of order to confirm first/last are derived from values,
        # not insertion order.
        for offset in [5, 0, 10]:
            gate.record(_make_result(pair, 0.10, day_offset=offset))
        cand = gate.check_promotions()[0]
        self.assertEqual(
            cand.first_seen_utc, "2026-05-01T00:00:00+00:00"
        )
        self.assertEqual(
            cand.last_seen_utc, "2026-05-11T00:00:00+00:00"
        )

    def test_multiple_pairs_evaluated_independently(self) -> None:
        gate = self._gate(min_samples=5, threshold_rank_ic=0.05)
        pair_hot = _make_pair("hot")
        pair_cold = _make_pair("cold")
        for i in range(5):
            gate.record(_make_result(pair_hot, 0.10, day_offset=i))
            gate.record(_make_result(pair_cold, 0.01, day_offset=i))
        candidates = gate.check_promotions()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].pair, pair_hot)


class GateAuditQueueTests(unittest.TestCase):
    """Verify the JSONL queue is appended atomically + idempotently."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = Path(self.tmp.name) / "promotion_queue.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_jsonl_append_to_tmpdir(self) -> None:
        gate = AutoPromotionGate(
            promotion_queue_path=self.queue,
            threshold_rank_ic=0.05,
            min_samples=3,
        )
        pair = _make_pair()
        for i in range(3):
            gate.record(_make_result(pair, 0.10, day_offset=i))
        gate.check_promotions()
        # Re-call: should append, not overwrite.
        gate.check_promotions()
        lines = self.queue.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            payload = json.loads(line)
            self.assertIn("emitted_at_utc", payload)
            self.assertIn("pair", payload)
            self.assertIn("stable_id", payload["pair"])

    def test_no_promotions_means_queue_not_created(self) -> None:
        gate = AutoPromotionGate(
            promotion_queue_path=self.queue,
            threshold_rank_ic=0.05,
            min_samples=30,
        )
        pair = _make_pair()
        for i in range(5):
            gate.record(_make_result(pair, 0.99, day_offset=i))
        # Below min_samples, no promotion -> no queue file.
        self.assertEqual(gate.check_promotions(), [])
        self.assertFalse(self.queue.exists())

    def test_promotion_payload_round_trips_to_dict(self) -> None:
        pair = _make_pair()
        cand = PromotionCandidate(
            pair=pair,
            rank_ic_30d_avg=0.07,
            rank_ic_30d_count=30,
            first_seen_utc="2026-05-01T00:00:00+00:00",
            last_seen_utc="2026-05-30T00:00:00+00:00",
        )
        d = cand.to_dict()
        self.assertEqual(d["pair"]["feature_a"], pair.feature_a)
        self.assertEqual(d["pair"]["stable_id"], pair.stable_id())
        # JSON round-trip works.
        s = json.dumps(d, sort_keys=True)
        parsed = json.loads(s)
        self.assertAlmostEqual(parsed["rank_ic_30d_avg"], 0.07)


class GateConstructorTests(unittest.TestCase):
    def test_negative_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AutoPromotionGate(threshold_rank_ic=-0.01)

    def test_nonpositive_min_samples_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AutoPromotionGate(min_samples=0)

    def test_record_rejects_wrong_type(self) -> None:
        gate = AutoPromotionGate()
        with self.assertRaises(TypeError):
            gate.record({"pair": "x", "rank_ic": 0.1})  # type: ignore[arg-type]

    def test_default_threshold_matches_ceo_plan(self) -> None:
        # Sanity: 0.05 is the threshold called out in the CEO plan.
        self.assertEqual(DEFAULT_THRESHOLD, 0.05)
        self.assertEqual(DEFAULT_MIN_SAMPLES, 30)


class InMemoryStoreTests(unittest.TestCase):
    def test_history_capped_at_maxlen(self) -> None:
        store = _InMemoryHistoryStore(maxlen=3)
        pair = _make_pair()
        for i in range(10):
            store.append(pair, float(i), f"2026-05-{i+1:02d}T00:00:00+00:00")
        pairs = store.all_pairs()
        self.assertEqual(len(pairs), 1)
        # Only the last 3 entries retained.
        _, history = pairs[0]
        self.assertEqual([r for r, _ts in history], [7.0, 8.0, 9.0])

    def test_empty_store_yields_no_pairs(self) -> None:
        store = _InMemoryHistoryStore()
        self.assertEqual(store.all_pairs(), [])


class RedisStoreSmokeTests(unittest.TestCase):
    """Smoke-test the Redis-backed store using fakeredis when available."""

    def test_redis_store_round_trip_with_fakeredis(self) -> None:
        try:
            import fakeredis  # type: ignore[import-not-found]
        except ImportError:
            self.skipTest("fakeredis not installed")
        client = fakeredis.FakeRedis()
        store = _RedisHistoryStore(client)
        pair = _make_pair()
        for i in range(5):
            store.append(pair, 0.1 * (i + 1), f"2026-05-{i+1:02d}T00:00:00+00:00")
        pairs = store.all_pairs()
        self.assertEqual(len(pairs), 1)
        recovered_pair, history = pairs[0]
        self.assertEqual(recovered_pair, pair)
        self.assertEqual(len(history), 5)


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
