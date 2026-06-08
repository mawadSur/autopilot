"""Unit tests for the Redis-backed :class:`TradeContextStore` (Lane E E1).

All tests use ``fakeredis.FakeRedis()``. The store reuses the same Redis
client convention as ``state.position_store``, so these tests mirror the
``test_position_store.py`` shape.
"""

from __future__ import annotations

import math
import unittest

import fakeredis

from state.trade_context_store import (
    DEFAULT_TTL_SECONDS,
    PostmortemQueue,
    RedisPostmortemQueue,
    TradeContextSnapshot,
    TradeContextStore,
)


def _new_snapshot(
    *,
    trade_id: str = "trade-1",
    phase: str = "signal",
    symbol: str = "ETH/USD",
    feature_buffer: dict | None = None,
) -> TradeContextSnapshot:
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc="2026-05-08T12:00:00+00:00",
        phase=phase,  # type: ignore[arg-type]
        feature_buffer=feature_buffer or {"rsi_14": 65.0, "ema_diff": -0.0021},
        feature_window=[{"rsi_14": 64.0}, {"rsi_14": 65.0}],
        model_probs={"long": 0.62, "short": 0.18, "flat": 0.20},
        model_confidence=0.62,
        risk_metrics_input={"kelly_input": 0.04, "win_prob": 0.62},
        risk_metrics_output={"sized_quote_usd": 250.0},
        breaker_context={"daily_pnl_usd": -12.0},
        ticker_buffer=[
            {"bid": 1999.5, "ask": 2000.5},
            {"bid": 1999.7, "ask": 2000.7},
        ],
        notes=None,
    )


class TradeContextSnapshotTests(unittest.TestCase):
    def test_to_dict_round_trips_through_json(self) -> None:
        snap = _new_snapshot()
        blob = snap.to_json()
        revived = TradeContextSnapshot.from_json(blob)
        self.assertEqual(revived.trade_id, snap.trade_id)
        self.assertEqual(revived.symbol, snap.symbol)
        self.assertEqual(revived.phase, snap.phase)
        self.assertEqual(revived.feature_buffer, snap.feature_buffer)
        self.assertEqual(revived.model_probs, snap.model_probs)

    def test_nan_in_feature_buffer_round_trips_as_none(self) -> None:
        snap = _new_snapshot(
            feature_buffer={
                "ok": 1.5,
                "broken_nan": float("nan"),
                "broken_inf": float("inf"),
                "broken_ninf": float("-inf"),
            }
        )
        blob = snap.to_json()
        revived = TradeContextSnapshot.from_json(blob)
        self.assertEqual(revived.feature_buffer["ok"], 1.5)
        # NaN/Inf must come back as None to keep JSON valid + give
        # downstream agents a clear "feature was non-finite" signal.
        self.assertIsNone(revived.feature_buffer["broken_nan"])
        self.assertIsNone(revived.feature_buffer["broken_inf"])
        self.assertIsNone(revived.feature_buffer["broken_ninf"])


class TradeContextStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.store = TradeContextStore(
            redis_client=self.fake, namespace="test"
        )

    # --- round-trip ----------------------------------------------------
    def test_record_and_get_snapshot_round_trips_through_redis(self) -> None:
        snap = _new_snapshot()
        self.store.record_snapshot(snap)
        revived = self.store.get_signal_snapshot(snap.trade_id)
        assert revived is not None
        self.assertEqual(revived.trade_id, snap.trade_id)
        self.assertEqual(revived.phase, "signal")
        self.assertAlmostEqual(revived.model_confidence, 0.62)

    # --- multi-phase ---------------------------------------------------
    def test_multiple_phases_for_same_trade_id(self) -> None:
        sig = _new_snapshot(trade_id="multi", phase="signal")
        fill = _new_snapshot(trade_id="multi", phase="fill")
        breaker = _new_snapshot(trade_id="multi", phase="breaker")
        self.store.record_snapshot(sig)
        self.store.record_snapshot(fill)
        self.store.record_snapshot(breaker)

        snaps = self.store.get_snapshots("multi")
        self.assertEqual(set(snaps.keys()), {"signal", "fill", "breaker"})
        self.assertEqual(snaps["fill"].phase, "fill")
        # Convenience accessors agree with the dict.
        self.assertIsNotNone(self.store.get_fill_snapshot("multi"))
        self.assertIsNotNone(self.store.get_signal_snapshot("multi"))

    # --- missing -------------------------------------------------------
    def test_missing_trade_id_returns_none_and_empty_dict(self) -> None:
        self.assertIsNone(self.store.get_signal_snapshot("never-stored"))
        self.assertEqual(self.store.get_snapshots("never-stored"), {})

    # --- NaN -----------------------------------------------------------
    def test_nan_in_feature_buffer_round_trips_through_redis(self) -> None:
        snap = _new_snapshot(
            feature_buffer={
                "good": 0.5,
                "bad_nan": float("nan"),
            }
        )
        # Must not raise.
        self.store.record_snapshot(snap)
        revived = self.store.get_signal_snapshot(snap.trade_id)
        assert revived is not None
        self.assertAlmostEqual(revived.feature_buffer["good"], 0.5)
        self.assertIsNone(revived.feature_buffer["bad_nan"])
        self.assertFalse(
            any(
                isinstance(v, float) and math.isnan(v)
                for v in revived.feature_buffer.values()
            )
        )

    # --- TTL -----------------------------------------------------------
    def test_ttl_is_set_on_record_snapshot(self) -> None:
        snap = _new_snapshot()
        self.store.record_snapshot(snap)
        ttl = self.store.get_ttl(snap.trade_id, "signal")
        # fakeredis honours EXPIRE: TTL should be > 0 and ≤ default.
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, DEFAULT_TTL_SECONDS)

    def test_get_ttl_returns_negative_two_for_missing(self) -> None:
        self.assertEqual(self.store.get_ttl("ghost", "signal"), -2)

    # --- delete --------------------------------------------------------
    def test_delete_snapshots_removes_all_phases(self) -> None:
        for phase in ("signal", "fill", "breaker"):
            self.store.record_snapshot(
                _new_snapshot(trade_id="goner", phase=phase)
            )
        # Confirm all three phases stored.
        self.assertEqual(len(self.store.get_snapshots("goner")), 3)

        deleted = self.store.delete_snapshots("goner")
        self.assertEqual(deleted, 3)
        self.assertEqual(self.store.get_snapshots("goner"), {})

    def test_delete_snapshots_returns_zero_for_missing(self) -> None:
        self.assertEqual(self.store.delete_snapshots("never-existed"), 0)

    # --- validation ----------------------------------------------------
    def test_record_snapshot_rejects_invalid_phase(self) -> None:
        snap = _new_snapshot()
        snap.phase = "garbage"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            self.store.record_snapshot(snap)

    def test_record_snapshot_rejects_empty_trade_id(self) -> None:
        snap = _new_snapshot(trade_id="")
        with self.assertRaises(ValueError):
            self.store.record_snapshot(snap)

    # --- key namespacing -----------------------------------------------
    def test_keys_use_trade_ctx_prefix_and_namespace(self) -> None:
        snap = _new_snapshot(trade_id="ns-check", phase="signal")
        self.store.record_snapshot(snap)
        # Must be at the documented key path — other Redis namespaces
        # (positions:, open_set, closed:*, errors:*) should NOT be touched.
        keys = list(self.fake.scan_iter(match="*"))
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], "test:trade_ctx:ns-check:signal")


class CanonicalBreakerFieldsTests(unittest.TestCase):
    """Phase-16 canonical fields on ``TradeContextSnapshot``.

    A5 ProcessIntegrity prefers these typed fields over substring-matching
    the ``breaker_context`` dict / ``notes`` string. Tests assert: round
    trip works, legacy snapshots without the fields deserialize cleanly
    (defaults to None), and the new fields survive Redis storage.
    """

    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.store = TradeContextStore(
            redis_client=self.fake, namespace="test-canonical"
        )

    def test_new_fields_round_trip_through_to_json_and_from_json(self) -> None:
        snap = TradeContextSnapshot(
            trade_id="trade-canon",
            symbol="BTC/USD",
            captured_at_utc="2026-05-08T12:00:00+00:00",
            phase="breaker",
            kill_switch_reason="daily_loss_limit",
            stop_loss_trigger_price=98.5,
            breaker_decision="force_flat",
        )
        blob = snap.to_json()
        revived = TradeContextSnapshot.from_json(blob)
        self.assertEqual(revived.kill_switch_reason, "daily_loss_limit")
        self.assertAlmostEqual(revived.stop_loss_trigger_price, 98.5)
        self.assertEqual(revived.breaker_decision, "force_flat")

    def test_legacy_blob_without_new_fields_defaults_to_none(self) -> None:
        """Snapshots stored before Phase-16 must round-trip without the new keys."""
        import json as _json

        legacy = _json.dumps(
            {
                "trade_id": "legacy",
                "symbol": "BTC/USD",
                "captured_at_utc": "2026-05-01T12:00:00+00:00",
                "phase": "breaker",
                "feature_buffer": {},
                "feature_window": None,
                "model_probs": {},
                "model_confidence": 0.0,
                "risk_metrics_input": {},
                "risk_metrics_output": {},
                "breaker_context": {"reason": "kill_switch_file_present"},
                "ticker_buffer": [],
                "notes": "kill_switch_file_present",
            }
        )
        revived = TradeContextSnapshot.from_json(legacy)
        self.assertIsNone(revived.kill_switch_reason)
        self.assertIsNone(revived.stop_loss_trigger_price)
        self.assertIsNone(revived.breaker_decision)

    def test_new_fields_survive_round_trip_through_redis(self) -> None:
        snap = TradeContextSnapshot(
            trade_id="trade-redis-canon",
            symbol="ETH/USD",
            captured_at_utc="2026-05-08T12:00:00+00:00",
            phase="breaker",
            kill_switch_reason="manual",
            stop_loss_trigger_price=2000.0,
            breaker_decision="force_flat",
        )
        self.store.record_snapshot(snap)
        revived = self.store.get_snapshot("trade-redis-canon", "breaker")
        assert revived is not None
        self.assertEqual(revived.kill_switch_reason, "manual")
        self.assertAlmostEqual(revived.stop_loss_trigger_price, 2000.0)
        self.assertEqual(revived.breaker_decision, "force_flat")


class RedisPostmortemQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.queue = RedisPostmortemQueue(
            redis_client=self.fake, namespace="test"
        )

    def test_enqueue_pushes_trade_id_onto_redis_list(self) -> None:
        self.queue.enqueue("trade-1")
        self.queue.enqueue("trade-2")
        self.assertEqual(self.queue.queue_length(), 2)
        # LIST ordering: LPUSH means later entries are at the head.
        # We don't enforce ordering as part of the API contract beyond
        # "all enqueued trades are present", but verify both made it in.
        contents = list(self.fake.lrange("test:postmortem:queue", 0, -1))
        self.assertEqual(set(contents), {"trade-1", "trade-2"})

    def test_enqueue_rejects_empty_trade_id(self) -> None:
        with self.assertRaises(ValueError):
            self.queue.enqueue("")

    def test_protocol_compatibility(self) -> None:
        """RedisPostmortemQueue must satisfy PostmortemQueue Protocol."""
        # ``Protocol`` is structural; isinstance checks need
        # @runtime_checkable. Verify the required method exists instead.
        self.assertTrue(hasattr(self.queue, "enqueue"))
        self.assertTrue(callable(self.queue.enqueue))
        # Sanity: the type hint annotation accepts our impl.
        q: PostmortemQueue = self.queue  # noqa: F841 - type-check only


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
