"""Unit tests for :mod:`loss_postmortem.execution_forensics` (Lane E A2).

Five scenarios mirror the brief:

1. Slippage > 50 bps → primary_cause.
2. Signal→fill latency > 10s → primary_cause.
3. Stale ticker + partial fill in notes → contributing.
4. Healthy fill (no flags) → innocent.
5. Missing fill snapshot AND missing position → unknown ("missing inputs").

Plus a few extras for the safety wrappers + stop-loss drift coverage.
"""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timedelta, timezone

import fakeredis

from loss_postmortem.execution_forensics import (
    ExecutionForensicsAgent,
    SLIPPAGE_PRIMARY_BPS,
)
from state.position_store import Position, PositionStore
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
)


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


def _stores() -> tuple[TradeContextStore, PositionStore]:
    """Two stores, one fakeredis backend each, isolated namespace."""
    rc1 = fakeredis.FakeRedis(decode_responses=True)
    rc2 = fakeredis.FakeRedis(decode_responses=True)
    ctx = TradeContextStore(redis_client=rc1, namespace="test-exec")
    pos = PositionStore(redis_client=rc2, namespace="test-exec")
    return ctx, pos


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat()


def _signal_snapshot(
    *,
    trade_id: str,
    captured_at: datetime,
    mid: float,
    ticker_age_s: float = 0.0,
    symbol: str = "BTC/USD",
) -> TradeContextSnapshot:
    """Snapshot at signal phase with one ticker entry at the given mid.

    ``ticker_age_s`` controls how old the ticker is *relative to the capture
    time* — lets the staleness check be exercised.
    """
    bid = mid - 0.5
    ask = mid + 0.5
    tick_at = captured_at - timedelta(seconds=ticker_age_s)
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=_iso(captured_at),
        phase="signal",
        ticker_buffer=[
            {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "last": mid,
                "volume_24h_base": 1000.0,
                "as_of_utc": _iso(tick_at),
            }
        ],
        model_confidence=0.7,
    )


def _fill_snapshot(
    *,
    trade_id: str,
    captured_at: datetime,
    mid: float,
    symbol: str = "BTC/USD",
) -> TradeContextSnapshot:
    bid = mid - 0.5
    ask = mid + 0.5
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=_iso(captured_at),
        phase="fill",
        ticker_buffer=[
            {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "last": mid,
                "volume_24h_base": 1000.0,
                "as_of_utc": _iso(captured_at),
            }
        ],
    )


def _position(
    *,
    trade_id: str,
    entry_price: float,
    exit_price: float | None = None,
    notes: str | None = None,
    side: str = "long",
    base_size: float = 0.1,
    symbol: str = "BTC/USD",
    model_meta: dict | None = None,
) -> Position:
    return Position(
        position_id=trade_id,
        exchange="coinbase",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status="closed",
        entry_price=entry_price,
        entry_quote_usd=entry_price * base_size,
        base_size=base_size,
        exit_price=exit_price,
        opened_at_utc=_iso(datetime.now(timezone.utc)),
        notes=notes,
        model_meta=model_meta or {},
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class ExecutionForensicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.pos = _stores()

    # ---- 1. Slippage primary cause -----------------------------------------
    def test_slippage_above_primary_threshold_yields_primary_cause(self) -> None:
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        # Signal-mid = 30000, entry filled at 30200 → 200/30000 * 10000 = ~66.7 bps
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=30000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=1.0),
                mid=30200.0,
            )
        )
        position = _position(
            trade_id=trade_id, entry_price=30200.0, exit_price=29900.0
        )
        self.pos.record_open(position)
        # Stamp closed-state in store too (record_close needs an existing pos).
        self.pos.record_close(
            trade_id, exit_price=29900.0, exit_quote_usd=29900.0 * 0.1
        )

        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "primary_cause")
        # Slippage bullet should mention bps explicitly.
        self.assertTrue(
            any("slippage" in e.lower() and "bps" in e.lower() for e in finding.evidence),
            finding.evidence,
        )
        # Should suggest tightening paper-slippage assumption.
        self.assertIsNotNone(finding.suggested_action)
        # Confidence well above mid given a primary trigger.
        self.assertGreater(finding.confidence, 0.7)
        # Threshold const is exposed for callers — sanity check it's the one used.
        self.assertGreaterEqual(SLIPPAGE_PRIMARY_BPS, 50.0)

    # ---- 2. Latency primary cause ------------------------------------------
    def test_signal_fill_latency_above_10s_yields_primary_cause(self) -> None:
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        # 12 seconds between signal and fill — over the 10s primary threshold.
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=2500.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=12.0),
                mid=2501.0,
            )
        )
        # Tiny slippage so only latency triggers.
        position = _position(
            trade_id=trade_id, entry_price=2501.0, exit_price=2480.0
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=2480.0, exit_quote_usd=2480.0 * 0.1
        )

        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertTrue(
            any("latency" in e.lower() for e in finding.evidence), finding.evidence
        )

    # ---- 3. Yellow flags → contributing ------------------------------------
    def test_stale_ticker_plus_partial_fill_yields_contributing(self) -> None:
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        # Ticker is 6s stale (threshold is 3s) AND notes mention partial fill.
        self.ctx.record_snapshot(
            _signal_snapshot(
                trade_id=trade_id,
                captured_at=signal_t,
                mid=3000.0,
                ticker_age_s=6.0,
            )
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=1.0),
                mid=3001.0,
            )
        )
        position = _position(
            trade_id=trade_id,
            entry_price=3001.0,  # tiny slippage, well under contributing
            exit_price=2980.0,
            notes="partial_fill at 0.05; remainder cancelled",
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=2980.0, exit_quote_usd=2980.0 * 0.1
        )

        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "contributing")
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("stale ticker", joined)
        self.assertIn("partial", joined)
        # Multiple yellows ⇒ severity at least 2.
        self.assertGreaterEqual(finding.severity, 2)

    # ---- 4. Innocent --------------------------------------------------------
    def test_clean_fill_yields_innocent(self) -> None:
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        # Fast fill, fresh ticker, ~0 bps slippage, no rejection/partial notes.
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=4000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=0.3),
                mid=4000.5,
            )
        )
        position = _position(
            trade_id=trade_id, entry_price=4000.0, exit_price=3990.0
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=3990.0, exit_quote_usd=3990.0 * 0.1
        )

        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "innocent")
        self.assertGreater(finding.confidence, 0.0)

    # ---- 5. Missing inputs → unknown ---------------------------------------
    def test_missing_snapshots_and_position_yields_unknown(self) -> None:
        trade_id = "nonexistent-trade"
        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.error, "missing_inputs")
        # Evidence bullet must explain the limitation.
        self.assertTrue(
            any("nothing to inspect" in e.lower() for e in finding.evidence),
            finding.evidence,
        )

    # ---- bonus: fill snapshot missing but position present -----------------
    def test_fill_missing_emits_limitation_bullet(self) -> None:
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=5000.0)
        )
        # No fill snapshot recorded.
        position = _position(
            trade_id=trade_id, entry_price=5000.0, exit_price=4995.0
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=4995.0, exit_quote_usd=4995.0 * 0.1
        )

        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        # Must emit the canonical limitation bullet.
        self.assertTrue(
            any("fill snapshot missing" in e.lower() for e in finding.evidence),
            finding.evidence,
        )
        # Without red flags the verdict should not be primary_cause.
        self.assertNotEqual(finding.verdict, "primary_cause")

    # ---- bonus: stop-loss drift flagged ------------------------------------
    def test_stop_loss_drift_flags_contributing(self) -> None:
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=2000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=0.5),
                mid=2000.0,
            )
        )
        # Stop trigger 1980, but actual exit drifted ~1.0% lower → 100 bps drift.
        position = _position(
            trade_id=trade_id,
            entry_price=2000.0,
            exit_price=1960.0,
            notes="stop_loss closed @ stop_price=1980",
            model_meta={"closed_via_stop": True},
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=1960.0, exit_quote_usd=1960.0 * 0.1
        )

        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "contributing")
        self.assertTrue(
            any("stop-loss" in e.lower() for e in finding.evidence),
            finding.evidence,
        )

    # ---- Phase-16: A2 prefers canonical Position fields over notes -----
    def test_canonical_partial_fills_field_wins_over_notes(self) -> None:
        """A2 reads ``Position.partial_fills`` directly when populated."""
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=2000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=0.5),
                mid=2000.5,
            )
        )
        # Position has structured partial_fills, NO partial-marker in notes.
        position = _position(
            trade_id=trade_id,
            entry_price=2000.5,
            exit_price=1990.0,
            notes=None,
        )
        position = position.model_copy(
            update={
                "partial_fills": [
                    {"size": 0.04, "price": 2000.0, "filled_at_utc": "2026-05-08T12:00:00+00:00"},
                    {"size": 0.06, "price": 2001.0, "filled_at_utc": "2026-05-08T12:00:01+00:00"},
                ],
            }
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=1990.0, exit_quote_usd=1990.0 * 0.1
        )
        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        # The structured field path produces evidence mentioning fills count.
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("partial fills detected on position record", joined)
        self.assertEqual(finding.verdict, "contributing")

    def test_canonical_rejection_reason_field_wins_over_notes(self) -> None:
        """A2 reads ``Position.rejection_reason`` directly when populated."""
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=2000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=0.5),
                mid=2000.5,
            )
        )
        position = _position(
            trade_id=trade_id,
            entry_price=2000.5,
            exit_price=1990.0,
            notes=None,
        )
        position = position.model_copy(
            update={"rejection_reason": "insufficient_funds"}
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=1990.0, exit_quote_usd=1990.0 * 0.1
        )
        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("rejection on position record", joined)
        self.assertIn("insufficient_funds", joined)

    def test_canonical_stop_trigger_price_field_wins_over_notes(self) -> None:
        """A2 reads ``Position.stop_trigger_price`` directly when populated."""
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=2000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=0.5),
                mid=2000.0,
            )
        )
        # Trigger 1980 via canonical field (not via notes); exit drifted to 1960.
        position = _position(
            trade_id=trade_id,
            entry_price=2000.0,
            exit_price=1960.0,
            notes="stop_loss closed",
            model_meta={"closed_via_stop": True},
        )
        position = position.model_copy(
            update={"stop_trigger_price": 1980.0}
        )
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=1960.0, exit_quote_usd=1960.0 * 0.1
        )
        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        self.assertEqual(finding.verdict, "contributing")
        self.assertTrue(
            any("stop-loss" in e.lower() for e in finding.evidence),
            finding.evidence,
        )

    def test_legacy_position_without_canonical_fields_uses_notes_fallback(
        self,
    ) -> None:
        """A position with no structured fields (legacy) still works via the
        old notes-scan path."""
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"
        signal_t = datetime.now(timezone.utc)
        self.ctx.record_snapshot(
            _signal_snapshot(trade_id=trade_id, captured_at=signal_t, mid=4000.0)
        )
        self.ctx.record_snapshot(
            _fill_snapshot(
                trade_id=trade_id,
                captured_at=signal_t + timedelta(seconds=0.5),
                mid=4001.0,
            )
        )
        # All structured fields are None (the default). Notes carry the
        # partial-fill marker the legacy path scans for.
        position = _position(
            trade_id=trade_id,
            entry_price=4001.0,
            exit_price=3990.0,
            notes="partial_fill at 0.05; remainder cancelled",
        )
        # Sanity: structured fields default None.
        self.assertIsNone(position.partial_fills)
        self.assertIsNone(position.rejection_reason)
        self.assertIsNone(position.stop_trigger_price)
        self.pos.record_open(position)
        self.pos.record_close(
            trade_id, exit_price=3990.0, exit_quote_usd=3990.0 * 0.1
        )
        agent = ExecutionForensicsAgent(
            context_store=self.ctx, position_store=self.pos
        )
        finding = agent.investigate(trade_id)
        # The notes scan still flags the partial — no regression.
        joined = " | ".join(finding.evidence).lower()
        self.assertIn("partial fill detected in position.notes", joined)

    # ---- bonus: safe_investigate timeout path ------------------------------
    def test_safe_investigate_returns_unknown_on_timeout(self) -> None:
        # Subclass that just sleeps so we exercise the BaseForensicsAgent
        # timeout wrapper that ExecutionForensicsAgent inherits.
        import time as _time

        class _SlowExec(ExecutionForensicsAgent):
            def investigate(self, trade_id: str):  # type: ignore[override]
                _time.sleep(0.5)
                # never reached
                return super().investigate(trade_id)  # pragma: no cover

        agent = _SlowExec(
            context_store=self.ctx, position_store=self.pos, timeout_s=0.05
        )
        finding = agent.safe_investigate("anything")
        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.error, "timeout")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
