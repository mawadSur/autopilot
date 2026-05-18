"""Unit tests for the Redis-backed :class:`PositionStore`.

Every test uses ``fakeredis.FakeRedis()`` as the underlying client — none of
these tests require a live Redis instance.
"""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timedelta, timezone

import fakeredis
from pydantic import ValidationError

from state.position_store import (
    PENDING_ORPHAN_AGE,
    POSTMORTEM_LOSS_PCT_THRESHOLD,
    Position,
    PositionStore,
)


class _StubPostmortemQueue:
    """In-memory PostmortemQueue stub for trigger-gate tests."""

    def __init__(self, *, raise_on_enqueue: bool = False) -> None:
        self.enqueued: list[str] = []
        self._raise = raise_on_enqueue

    def enqueue(self, trade_id: str) -> None:
        if self._raise:
            raise RuntimeError("simulated queue failure")
        self.enqueued.append(trade_id)


def _new_position(
    *,
    side: str = "long",
    status: str = "open",
    entry_price: float = 100.0,
    base_size: float = 1.0,
    symbol: str = "ETH/USDT",
    exchange: str = "coinbase",
    opened_at_utc: str | None = None,
    entry_order_id: str | None = "order-abc",
    fees_usd: float = 0.0,
) -> Position:
    return Position(
        position_id=str(uuid.uuid4()),
        exchange=exchange,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        entry_price=entry_price,
        entry_quote_usd=entry_price * base_size,
        base_size=base_size,
        opened_at_utc=opened_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        entry_order_id=entry_order_id,
        fees_usd=fees_usd,
    )


class _StubExchange:
    """Stub ``exchange`` for :meth:`PositionStore.reconcile` tests.

    Tests construct one with the order ids they want to be "still open" on the
    exchange. Every call to ``get_open_orders`` returns those as dicts.
    """

    def __init__(self, open_order_ids: list[str] | None = None) -> None:
        self._ids = list(open_order_ids or [])
        self.calls: list[dict] = []

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        self.calls.append({"symbol": symbol})
        return [{"order_id": oid} for oid in self._ids]


class PositionSchemaTests(unittest.TestCase):
    def test_position_minimal_construction(self) -> None:
        position = _new_position()
        self.assertEqual(position.side, "long")
        self.assertEqual(position.status, "open")
        self.assertEqual(position.fees_usd, 0.0)
        self.assertIsNone(position.exit_price)
        self.assertEqual(position.model_meta, {})

    def test_position_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValidationError):
            Position(
                position_id="x",
                exchange="coinbase",
                symbol="ETH/USDT",
                side="long",
                status="open",
                entry_price=1.0,
                entry_quote_usd=1.0,
                base_size=1.0,
                opened_at_utc=datetime.now(timezone.utc).isoformat(),
                bogus_extra="nope",
            )

    def test_position_rejects_invalid_side(self) -> None:
        with self.assertRaises(ValidationError):
            Position(
                position_id="x",
                exchange="coinbase",
                symbol="ETH/USDT",
                side="sideways",  # type: ignore[arg-type]
                status="open",
                entry_price=1.0,
                entry_quote_usd=1.0,
                base_size=1.0,
                opened_at_utc=datetime.now(timezone.utc).isoformat(),
            )

    def test_position_default_bars_held_and_high_water_mark(self) -> None:
        """Sprint 1 Wave 2: ``bars_held`` defaults to 0 and
        ``high_water_mark`` defaults to None. Legacy positions in Redis
        that predate these fields still validate cleanly thanks to the
        defaults — this is the backward-compat guarantee for Wave 2."""
        position = Position(
            position_id="legacy",
            exchange="coinbase",
            symbol="ETH/USDT",
            side="long",
            status="open",
            entry_price=100.0,
            entry_quote_usd=100.0,
            base_size=1.0,
            opened_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.assertEqual(position.bars_held, 0)
        self.assertIsNone(position.high_water_mark)
        # Round-trip via model_dump_json → model_validate_json must
        # preserve the defaults exactly so a Redis re-read does the right
        # thing for a legacy blob.
        roundtrip = Position.model_validate_json(position.model_dump_json())
        self.assertEqual(roundtrip.bars_held, 0)
        self.assertIsNone(roundtrip.high_water_mark)


class PositionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.store = PositionStore(redis_client=self.fake, namespace="test")

    # --- writes -------------------------------------------------------
    def test_record_open_persists_position_and_open_set(self) -> None:
        pos = _new_position()
        recorded = self.store.record_open(pos)

        self.assertEqual(recorded.status, "open")
        # Position blob exists.
        self.assertIsNotNone(self.fake.get(f"test:positions:{pos.position_id}"))
        # ID is in the open_set.
        self.assertIn(pos.position_id, self.fake.smembers("test:open_set"))

    def test_get_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(self.store.get("nope"))

    def test_list_open_returns_only_open_or_pending(self) -> None:
        live = self.store.record_open(_new_position(symbol="ETH/USDT"))
        pending = self.store.record_pending(_new_position(symbol="BTC/USDT"))
        # Manually inject a "closed" position into the open_set to simulate
        # a stale entry — list_open must filter it out.
        stale_closed = _new_position(symbol="SOL/USDT")
        stale_closed = stale_closed.model_copy(update={"status": "closed"})
        self.fake.set(
            f"test:positions:{stale_closed.position_id}",
            stale_closed.model_dump_json(),
        )
        self.fake.sadd("test:open_set", stale_closed.position_id)

        ids = {p.position_id for p in self.store.list_open()}
        self.assertEqual(ids, {live.position_id, pending.position_id})

    def test_record_close_moves_to_closed_today(self) -> None:
        pos = self.store.record_open(_new_position(entry_price=100.0, base_size=1.0))
        closed = self.store.record_close(
            pos.position_id, exit_price=110.0, exit_quote_usd=110.0
        )

        self.assertEqual(closed.status, "closed")
        self.assertNotIn(pos.position_id, self.fake.smembers("test:open_set"))
        # Closed-today set contains it (UTC date).
        date_key = (
            "test:closed:" + datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        self.assertIn(pos.position_id, self.fake.smembers(date_key))

    def test_record_close_computes_long_pnl(self) -> None:
        pos = self.store.record_open(_new_position(entry_price=100.0, base_size=2.0))
        closed = self.store.record_close(
            pos.position_id, exit_price=110.0, exit_quote_usd=220.0
        )
        # 2 * (110 - 100) - 0 = 20
        self.assertAlmostEqual(closed.realized_pnl_usd or 0.0, 20.0)

    def test_record_close_computes_short_pnl(self) -> None:
        pos = self.store.record_open(
            _new_position(side="short", entry_price=100.0, base_size=2.0)
        )
        closed = self.store.record_close(
            pos.position_id, exit_price=90.0, exit_quote_usd=180.0
        )
        # 2 * (100 - 90) - 0 = 20
        self.assertAlmostEqual(closed.realized_pnl_usd or 0.0, 20.0)

    def test_record_close_subtracts_fees(self) -> None:
        # Entry-side fees applied first via mark_filled-style flow.
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0, fees_usd=0.5)
        )
        closed = self.store.record_close(
            pos.position_id, exit_price=110.0, exit_quote_usd=110.0, fees_usd=0.7
        )
        # gross 10 - (0.5 + 0.7) = 8.8
        self.assertAlmostEqual(closed.realized_pnl_usd or 0.0, 8.8)
        self.assertAlmostEqual(closed.fees_usd, 1.2)

    # --- aggregations -------------------------------------------------
    def test_open_notional_usd_sums_open_positions(self) -> None:
        self.store.record_open(_new_position(entry_price=100.0, base_size=1.0))
        self.store.record_open(_new_position(entry_price=200.0, base_size=2.0))
        # Total = 100 + 400 = 500
        self.assertAlmostEqual(self.store.open_notional_usd(), 500.0)

    def test_open_notional_for_symbol_filters(self) -> None:
        self.store.record_open(
            _new_position(symbol="ETH/USDT", entry_price=100.0, base_size=1.0)
        )
        self.store.record_open(
            _new_position(symbol="BTC/USDT", entry_price=200.0, base_size=2.0)
        )
        self.assertAlmostEqual(
            self.store.open_notional_for_symbol("ETH/USDT"), 100.0
        )
        self.assertAlmostEqual(
            self.store.open_notional_for_symbol("BTC/USDT"), 400.0
        )
        self.assertAlmostEqual(
            self.store.open_notional_for_symbol("DOGE/USDT"), 0.0
        )

    def test_daily_realized_pnl_usd_sums_closed_today(self) -> None:
        a = self.store.record_open(_new_position(entry_price=100.0, base_size=1.0))
        b = self.store.record_open(_new_position(entry_price=100.0, base_size=2.0))
        self.store.record_close(a.position_id, exit_price=110.0, exit_quote_usd=110.0)
        self.store.record_close(b.position_id, exit_price=95.0, exit_quote_usd=190.0)
        # 10 + (-10) = 0
        self.assertAlmostEqual(self.store.daily_realized_pnl_usd(), 0.0)

    def test_daily_realized_pnl_usd_excludes_yesterday(self) -> None:
        pos = self.store.record_open(_new_position(entry_price=100.0, base_size=1.0))
        closed = self.store.record_close(
            pos.position_id, exit_price=110.0, exit_quote_usd=110.0
        )
        # Patch the position blob's closed_at_utc to yesterday and migrate the
        # set membership to yesterday's bucket — simulates a position that
        # closed across midnight.
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        moved = closed.model_copy(update={"closed_at_utc": yesterday.isoformat()})
        self.fake.set(
            f"test:positions:{closed.position_id}", moved.model_dump_json()
        )
        today_key = (
            "test:closed:" + datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        yesterday_key = "test:closed:" + yesterday.strftime("%Y-%m-%d")
        self.fake.srem(today_key, closed.position_id)
        self.fake.sadd(yesterday_key, closed.position_id)

        self.assertAlmostEqual(self.store.daily_realized_pnl_usd(), 0.0)

    # --- reconcile ----------------------------------------------------
    def test_reconcile_drops_orphan_pending_after_1_hour(self) -> None:
        old_open = (
            datetime.now(timezone.utc) - PENDING_ORPHAN_AGE - timedelta(minutes=5)
        ).replace(microsecond=0)
        pos = self.store.record_pending(
            _new_position(
                opened_at_utc=old_open.isoformat(),
                entry_order_id="ghost-order",
            )
        )
        exchange = _StubExchange(open_order_ids=[])  # exchange knows nothing

        result = self.store.reconcile(exchange)

        self.assertEqual(result["dropped"], 1)
        self.assertEqual(result["warnings"], [])
        # Removed from open_set.
        self.assertNotIn(pos.position_id, self.fake.smembers("test:open_set"))
        # Marked closed with the reconciled-orphan note.
        rehydrated = self.store.get(pos.position_id)
        assert rehydrated is not None
        self.assertEqual(rehydrated.status, "closed")
        self.assertEqual(rehydrated.notes, "reconciled-orphan")
        self.assertEqual(rehydrated.realized_pnl_usd, 0.0)

    def test_reconcile_keeps_pending_under_1_hour(self) -> None:
        recent = (datetime.now(timezone.utc) - timedelta(minutes=30)).replace(
            microsecond=0
        )
        pos = self.store.record_pending(
            _new_position(
                opened_at_utc=recent.isoformat(),
                entry_order_id="not-yet-orphaned",
            )
        )
        exchange = _StubExchange(open_order_ids=[])

        result = self.store.reconcile(exchange)

        self.assertEqual(result["dropped"], 0)
        self.assertEqual(result["reconciled"], 1)
        # Still pending and still in the open_set.
        self.assertIn(pos.position_id, self.fake.smembers("test:open_set"))
        rehydrated = self.store.get(pos.position_id)
        assert rehydrated is not None
        self.assertEqual(rehydrated.status, "pending")

    def test_reconcile_keeps_pending_when_exchange_confirms_order(self) -> None:
        # Even with an "ancient" opened_at, an exchange-confirmed order is not
        # an orphan.
        ancient = (
            datetime.now(timezone.utc) - PENDING_ORPHAN_AGE - timedelta(hours=12)
        ).replace(microsecond=0)
        pos = self.store.record_pending(
            _new_position(
                opened_at_utc=ancient.isoformat(),
                entry_order_id="exchange-knows-this",
            )
        )
        exchange = _StubExchange(open_order_ids=["exchange-knows-this"])

        result = self.store.reconcile(exchange)

        self.assertEqual(result["dropped"], 0)
        self.assertEqual(result["reconciled"], 1)
        self.assertIn(pos.position_id, self.fake.smembers("test:open_set"))
        rehydrated = self.store.get(pos.position_id)
        assert rehydrated is not None
        self.assertEqual(rehydrated.status, "pending")

    # --- admin --------------------------------------------------------
    def test_clear_namespace_deletes_all_keys(self) -> None:
        self.store.record_open(_new_position())
        self.store.record_open(_new_position())
        self.store.record_pending(_new_position())
        self.assertGreater(len(list(self.fake.scan_iter(match="test:*"))), 0)

        deleted = self.store.clear_namespace()

        self.assertGreater(deleted, 0)
        self.assertEqual(list(self.fake.scan_iter(match="test:*")), [])

    # --- error counter (Lane A P0 #3) --------------------------------
    def test_increment_error_returns_running_count(self) -> None:
        self.assertEqual(self.store.increment_error("ETH/USD"), 1)
        self.assertEqual(self.store.increment_error("ETH/USD"), 2)
        self.assertEqual(self.store.errors_today("ETH/USD"), 2)
        # Different symbol increments independently.
        self.assertEqual(self.store.increment_error("BTC/USD"), 1)
        self.assertEqual(self.store.errors_today("BTC/USD"), 1)
        self.assertEqual(self.store.errors_today("ETH/USD"), 2)

    def test_errors_today_returns_zero_for_unset_symbol(self) -> None:
        self.assertEqual(self.store.errors_today("UNKNOWN/USD"), 0)

    def test_errors_today_all_returns_full_map(self) -> None:
        self.store.increment_error("ETH/USD")
        self.store.increment_error("ETH/USD")
        self.store.increment_error("BTC/USD")
        all_counts = self.store.errors_today_all()
        self.assertEqual(all_counts, {"ETH/USD": 2, "BTC/USD": 1})

    def test_reset_errors_for_day_clears_hash(self) -> None:
        self.store.increment_error("ETH/USD")
        self.store.increment_error("BTC/USD")
        self.assertEqual(self.store.reset_errors_for_day(), 1)
        self.assertEqual(self.store.errors_today("ETH/USD"), 0)
        self.assertEqual(self.store.errors_today("BTC/USD"), 0)
        # Calling again with no key returns 0 (idempotent).
        self.assertEqual(self.store.reset_errors_for_day(), 0)

    def test_concurrent_increments_via_two_stores_share_counter(self) -> None:
        """Two PositionStore instances over one fakeredis converge on the
        same per-symbol count — the contract that justifies moving the
        counter out of process memory in the first place (P0 #3)."""
        store_b = PositionStore(redis_client=self.fake, namespace="test")
        for _ in range(5):
            self.store.increment_error("ETH/USD")
        for _ in range(3):
            store_b.increment_error("ETH/USD")
        self.assertEqual(self.store.errors_today("ETH/USD"), 8)
        self.assertEqual(store_b.errors_today("ETH/USD"), 8)

    # ------------------------------------------------------------------
    # Sprint 1 Wave 2: update_runtime_fields round-trip
    # ------------------------------------------------------------------
    def test_update_runtime_fields_persists_bars_held_and_hwm(self) -> None:
        """``update_runtime_fields`` is the per-tick seam ExitPolicy uses
        to keep ``bars_held`` + ``high_water_mark`` fresh across ticks and
        across supervisor restarts. The Redis blob must reflect the patch
        atomically — a re-read via ``get`` returns the new values."""
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0)
        )
        # Initial values reflect the schema defaults.
        self.assertEqual(pos.bars_held, 0)
        self.assertIsNone(pos.high_water_mark)
        updated = self.store.update_runtime_fields(
            pos.position_id, bars_held=5, high_water_mark=110.0
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated.bars_held, 5)
        self.assertEqual(updated.high_water_mark, 110.0)
        # Round-trip via get() — proves the write hit Redis, not just the
        # in-memory local.
        roundtrip = self.store.get(pos.position_id)
        self.assertIsNotNone(roundtrip)
        self.assertEqual(roundtrip.bars_held, 5)
        self.assertEqual(roundtrip.high_water_mark, 110.0)

    def test_update_runtime_fields_returns_none_for_unknown_id(self) -> None:
        result = self.store.update_runtime_fields(
            "ghost", bars_held=1, high_water_mark=1.0
        )
        self.assertIsNone(result)

    def test_update_runtime_fields_noop_when_both_args_none(self) -> None:
        pos = self.store.record_open(_new_position(entry_price=100.0))
        # Calling with no patch returns the existing position unchanged.
        result = self.store.update_runtime_fields(pos.position_id)
        self.assertEqual(result.position_id, pos.position_id)
        self.assertEqual(result.bars_held, 0)


class PostmortemTriggerGateTests(unittest.TestCase):
    """Lane E E2 trigger gate (D5).

    The trigger fires on ``record_close`` when:
      realized_pnl_usd < 0 AND
        (|loss| >= 0.005 * bankroll OR forced_flat-style notes)
    """

    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.queue = _StubPostmortemQueue()
        # Provider returns a fixed bankroll so the threshold is deterministic.
        self.bankroll_usd = 10_000.0
        self.store = PositionStore(
            redis_client=self.fake,
            namespace="test",
            postmortem_queue=self.queue,
            bankroll_provider=lambda: self.bankroll_usd,
        )

    # --- (a) loss above threshold ------------------------------------
    def test_trigger_fires_when_loss_meets_threshold(self) -> None:
        # Loss of 0.5% * 10_000 = 50 USD threshold. Position long 100 → 95
        # for base_size=10 yields -50 USD exactly.
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=10.0)
        )
        closed = self.store.record_close(
            pos.position_id, exit_price=95.0, exit_quote_usd=950.0
        )
        self.assertLess(closed.realized_pnl_usd or 0.0, 0)
        self.assertEqual(self.queue.enqueued, [pos.position_id])

    def test_trigger_fires_when_loss_clearly_above_threshold(self) -> None:
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=10.0)
        )
        # 100 USD loss > 50 USD threshold.
        self.store.record_close(
            pos.position_id, exit_price=90.0, exit_quote_usd=900.0
        )
        self.assertEqual(self.queue.enqueued, [pos.position_id])

    # --- (b) forced_flat path ----------------------------------------
    def test_trigger_fires_when_forced_flat_regardless_of_loss_size(self) -> None:
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0)
        )
        # Manually rewrite notes to simulate a force-flatted close — the loss
        # is tiny (1 USD) but force-flat marks it as worth investigating.
        with_notes = pos.model_copy(update={"notes": "force_flat:daily_loss"})
        self.fake.set(
            f"test:positions:{pos.position_id}",
            with_notes.model_dump_json(),
        )
        self.store.record_close(
            pos.position_id, exit_price=99.0, exit_quote_usd=99.0
        )
        self.assertEqual(self.queue.enqueued, [pos.position_id])

    def test_forced_flat_detection_handles_kill_switch_note(self) -> None:
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0)
        )
        with_notes = pos.model_copy(
            update={"notes": "kill_switch trip — manual halt"}
        )
        self.fake.set(
            f"test:positions:{pos.position_id}",
            with_notes.model_dump_json(),
        )
        self.store.record_close(
            pos.position_id, exit_price=99.0, exit_quote_usd=99.0
        )
        self.assertEqual(self.queue.enqueued, [pos.position_id])

    def test_forced_flat_detection_handles_daily_loss_limit_note(self) -> None:
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0)
        )
        with_notes = pos.model_copy(
            update={"notes": "halt_new_entries: daily_loss_limit reached"}
        )
        self.fake.set(
            f"test:positions:{pos.position_id}",
            with_notes.model_dump_json(),
        )
        self.store.record_close(
            pos.position_id, exit_price=99.5, exit_quote_usd=99.5
        )
        self.assertEqual(self.queue.enqueued, [pos.position_id])

    # --- (c) profitable closes -----------------------------------------
    def test_trigger_does_not_fire_on_profitable_close(self) -> None:
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=10.0)
        )
        closed = self.store.record_close(
            pos.position_id, exit_price=110.0, exit_quote_usd=1100.0
        )
        self.assertGreater(closed.realized_pnl_usd or 0.0, 0)
        self.assertEqual(self.queue.enqueued, [])

    def test_trigger_does_not_fire_on_breakeven_close(self) -> None:
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0)
        )
        self.store.record_close(
            pos.position_id, exit_price=100.0, exit_quote_usd=100.0
        )
        self.assertEqual(self.queue.enqueued, [])

    # --- (d) loss below threshold and not forced --------------------
    def test_trigger_does_not_fire_on_small_non_forced_loss(self) -> None:
        # Tiny loss: -1 USD on 10_000 bankroll = 0.01% << 0.5% threshold.
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=1.0)
        )
        closed = self.store.record_close(
            pos.position_id, exit_price=99.0, exit_quote_usd=99.0
        )
        self.assertLess(closed.realized_pnl_usd or 0.0, 0)
        self.assertEqual(self.queue.enqueued, [])

    # --- (e) no queue -> no error -----------------------------------
    def test_no_queue_wired_does_not_raise_on_qualifying_loss(self) -> None:
        store = PositionStore(
            redis_client=fakeredis.FakeRedis(decode_responses=True),
            namespace="noqueue",
            postmortem_queue=None,
            bankroll_provider=lambda: 10_000.0,
        )
        pos = store.record_open(
            _new_position(entry_price=100.0, base_size=10.0)
        )
        # Must not raise — debug log only.
        closed = store.record_close(
            pos.position_id, exit_price=90.0, exit_quote_usd=900.0
        )
        self.assertLess(closed.realized_pnl_usd or 0.0, 0)

    def test_explicit_bankroll_kwarg_overrides_provider(self) -> None:
        # Tiny bankroll override → 100 USD loss exceeds 0.5% threshold.
        pos = self.store.record_open(
            _new_position(entry_price=100.0, base_size=10.0)
        )
        self.store.record_close(
            pos.position_id,
            exit_price=99.0,
            exit_quote_usd=990.0,
            bankroll_usd=1_000.0,
        )
        # Loss is 10 USD; threshold = 0.5% * 1000 = 5 USD → triggers.
        self.assertEqual(self.queue.enqueued, [pos.position_id])

    def test_queue_enqueue_failure_does_not_break_record_close(self) -> None:
        flaky_queue = _StubPostmortemQueue(raise_on_enqueue=True)
        store = PositionStore(
            redis_client=fakeredis.FakeRedis(decode_responses=True),
            namespace="flaky",
            postmortem_queue=flaky_queue,
            bankroll_provider=lambda: 10_000.0,
        )
        pos = store.record_open(
            _new_position(entry_price=100.0, base_size=10.0)
        )
        # The close itself must succeed even though the queue throws.
        closed = store.record_close(
            pos.position_id, exit_price=90.0, exit_quote_usd=900.0
        )
        self.assertEqual(closed.status, "closed")

    def test_threshold_is_documented_constant(self) -> None:
        # The 0.005 (0.5%) threshold is locked by D5 — pin it as a regression
        # guard so a future refactor can't silently change the trigger
        # sensitivity.
        self.assertEqual(POSTMORTEM_LOSS_PCT_THRESHOLD, 0.005)


# ---------------------------------------------------------------------------
# Task 4: orphan-position telemetry
# ---------------------------------------------------------------------------


class OrphanCountTests(unittest.TestCase):
    """``PositionStore.orphan_count()`` exposes drift surfaces for metrics."""

    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.store = PositionStore(redis_client=self.fake, namespace="test")

    def test_orphan_count_zero_when_no_open_positions(self) -> None:
        self.assertEqual(self.store.orphan_count(), 0)

    def test_orphan_count_zero_for_clean_open_positions(self) -> None:
        self.store.record_open(_new_position())
        self.store.record_open(_new_position(symbol="BTC/USDT"))
        self.assertEqual(self.store.orphan_count(), 0)

    def test_orphan_count_flags_pending_positions_older_than_orphan_age(
        self,
    ) -> None:
        # Pending position with an opened_at_utc older than PENDING_ORPHAN_AGE
        # is treated as an orphan candidate so the operator dashboard
        # surfaces the drift before the next reconciliation pass.
        old_iso = (
            datetime.now(timezone.utc) - PENDING_ORPHAN_AGE - timedelta(minutes=1)
        ).replace(microsecond=0).isoformat()
        self.store.record_pending(
            _new_position(status="pending", opened_at_utc=old_iso)
        )
        self.assertEqual(self.store.orphan_count(), 1)

    def test_orphan_count_does_not_flag_fresh_pending_positions(self) -> None:
        # A pending position younger than the orphan age is NOT an orphan;
        # the fill confirmation may still arrive.
        fresh_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.store.record_pending(
            _new_position(status="pending", opened_at_utc=fresh_iso)
        )
        self.assertEqual(self.store.orphan_count(), 0)

    def test_orphan_count_flags_open_positions_with_orphan_notes(self) -> None:
        # An open position whose notes reference an orphan condition
        # (eg. a manual orphan tag) is counted.
        position = _new_position()
        position = position.model_copy(update={"notes": "reconciled-orphan"})
        # Persist directly so we can inject the note.
        self.store._redis.set(
            self.store._position_key(position.position_id),
            position.model_dump_json(),
        )
        self.store._redis.sadd(
            self.store._open_set_key, position.position_id
        )
        self.assertEqual(self.store.orphan_count(), 1)


class StructuredFillMetadataTests(unittest.TestCase):
    """Phase-16: structured ``partial_fills`` / ``rejection_reason`` /
    ``stop_trigger_price`` fields on Position. A2 ExecutionForensics reads
    these directly instead of regex-scraping notes."""

    def setUp(self) -> None:
        self.fake = fakeredis.FakeRedis(decode_responses=True)
        self.store = PositionStore(redis_client=self.fake, namespace="test-fill")

    def test_position_with_structured_fields_round_trips_through_redis(self) -> None:
        partial_fills = [
            {"size": 0.05, "price": 100.0, "filled_at_utc": "2026-05-08T12:00:00+00:00"},
            {"size": 0.03, "price": 100.2, "filled_at_utc": "2026-05-08T12:00:01+00:00"},
        ]
        position = Position(
            position_id="trade-fill",
            exchange="coinbase",
            symbol="BTC/USD",
            side="long",
            status="open",
            entry_price=100.0,
            entry_quote_usd=8.0,
            base_size=0.08,
            opened_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            partial_fills=partial_fills,
            rejection_reason=None,
            stop_trigger_price=98.0,
        )
        self.store.record_open(position)
        revived = self.store.get("trade-fill")
        assert revived is not None
        self.assertEqual(revived.partial_fills, partial_fills)
        self.assertIsNone(revived.rejection_reason)
        self.assertAlmostEqual(revived.stop_trigger_price, 98.0)

    def test_position_with_rejection_reason_round_trips(self) -> None:
        position = Position(
            position_id="trade-rej",
            exchange="coinbase",
            symbol="BTC/USD",
            side="long",
            status="open",
            entry_price=100.0,
            entry_quote_usd=10.0,
            base_size=0.1,
            opened_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            rejection_reason="insufficient_funds",
        )
        self.store.record_open(position)
        revived = self.store.get("trade-rej")
        assert revived is not None
        self.assertEqual(revived.rejection_reason, "insufficient_funds")
        self.assertIsNone(revived.partial_fills)
        self.assertIsNone(revived.stop_trigger_price)

    def test_legacy_position_without_new_fields_deserializes_with_defaults(
        self,
    ) -> None:
        """Backward compat: a position blob written before Phase-16 (no
        partial_fills / rejection_reason / stop_trigger_price keys) must
        deserialize cleanly with all three defaulted to None."""
        import json as _json

        legacy_blob = _json.dumps(
            {
                "position_id": "legacy-1",
                "exchange": "coinbase",
                "symbol": "ETH/USD",
                "side": "long",
                "status": "open",
                "entry_price": 2000.0,
                "entry_quote_usd": 100.0,
                "base_size": 0.05,
                "opened_at_utc": "2026-04-01T00:00:00+00:00",
                "fees_usd": 0.5,
                "model_meta": {},
                "notes": "some-legacy-note",
            }
        )
        # Manually persist the legacy shape.
        self.fake.set(self.store._position_key("legacy-1"), legacy_blob)
        self.fake.sadd(self.store._open_set_key, "legacy-1")
        revived = self.store.get("legacy-1")
        assert revived is not None
        self.assertIsNone(revived.partial_fills)
        self.assertIsNone(revived.rejection_reason)
        self.assertIsNone(revived.stop_trigger_price)
        self.assertEqual(revived.notes, "some-legacy-note")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
