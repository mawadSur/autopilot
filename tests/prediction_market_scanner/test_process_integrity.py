"""Unit tests for ``src/loss_postmortem/process_integrity.py`` (Lane E A5).

Five scenarios:

1. Healthy snapshots, no contradictions → verdict="innocent".
2. Stop-loss exit drifted materially from trigger → verdict="primary_cause".
3. Paper-vs-live divergence across snapshots → verdict="primary_cause".
4. Kill-switch referenced in notes but breaker_context didn't record it →
   verdict="primary_cause".
5. Race-condition cluster (>= threshold concurrent error increments) →
   verdict="contributing".

A 6th sanity test asserts the agent crashes safely (via ``safe_investigate``)
when the snapshot store is broken.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

import fakeredis

from loss_postmortem.process_integrity import (
    _RACE_CONCURRENCY_THRESHOLD,
    ProcessIntegrityAgent,
)
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _store(redis_client: Optional[Any] = None) -> TradeContextStore:
    return TradeContextStore(
        redis_client=redis_client or fakeredis.FakeRedis(decode_responses=True),
        namespace="test",
    )


def _signal_snap(
    trade_id: str = "trade-1",
    *,
    symbol: str = "BTC/USD",
    notes: Optional[str] = None,
    proposed_notional: float = 100.0,
) -> TradeContextSnapshot:
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=utc_now_iso(),
        phase="signal",
        feature_buffer={},
        feature_window=None,
        model_probs={},
        model_confidence=0.7,
        risk_metrics_input={
            "side": "buy",
            "proposed_notional_usd": float(proposed_notional),
            "current_open_notional_usd": 0.0,
            "current_per_symbol_notional_usd": 0.0,
            "daily_realized_pnl_usd": 0.0,
            "equity_peak_usd": 1000.0,
            "equity_current_usd": 990.0,
        },
        risk_metrics_output={},
        breaker_context={},
        ticker_buffer=[
            {"bid": 100.0, "ask": 100.05, "last": 100.02, "mid": 100.025},
        ],
        notes=notes,
    )


def _fill_snap(
    trade_id: str = "trade-1",
    *,
    symbol: str = "BTC/USD",
    exchange: str = "coinbase-paper",
    notes: Optional[str] = "paper-deferred-fill",
) -> TradeContextSnapshot:
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=utc_now_iso(),
        phase="fill",
        feature_buffer={},
        feature_window=None,
        model_probs={},
        model_confidence=0.0,
        risk_metrics_input={},
        risk_metrics_output={
            "position_id": trade_id,
            "side": "long",
            "fill_price": 100.0,
            "fill_size": 1.0,
            "fill_quote_usd": 100.0,
            "fees_usd": 0.0,
            "exchange": exchange,
            "status": "open",
        },
        breaker_context={},
        ticker_buffer=[],
        notes=notes,
    )


def _breaker_snap(
    trade_id: str = "trade-1",
    *,
    symbol: str = "BTC/USD",
    breaker_context: Optional[Dict[str, Any]] = None,
    risk_input: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = "kill_switch_file_present",
) -> TradeContextSnapshot:
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=utc_now_iso(),
        phase="breaker",
        feature_buffer={},
        feature_window=None,
        model_probs={},
        model_confidence=0.0,
        risk_metrics_input=risk_input or {},
        risk_metrics_output={},
        breaker_context=breaker_context or {},
        ticker_buffer=[],
        notes=notes,
    )


class _StubPosition:
    """Lightweight position stand-in (avoids hauling in pydantic for tests)."""

    def __init__(
        self,
        *,
        position_id: str,
        exchange: str = "coinbase-paper",
        exit_price: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.position_id = position_id
        self.exchange = exchange
        self.exit_price = exit_price
        self.notes = notes


class _StubPositionStore:
    def __init__(self, position: Optional[_StubPosition]) -> None:
        self._pos = position

    def get(self, position_id: str) -> Optional[_StubPosition]:
        if self._pos is None or self._pos.position_id != position_id:
            return None
        return self._pos


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


class ProcessIntegrityAgentTests(unittest.TestCase):
    # --------------------------------------------------------------- 1
    def test_healthy_snapshots_yield_innocent(self) -> None:
        store = _store()
        store.record_snapshot(_signal_snap())
        store.record_snapshot(_fill_snap())

        position_store = _StubPositionStore(
            _StubPosition(
                position_id="trade-1",
                exchange="coinbase-paper",
                exit_price=99.0,
                notes="paper-deferred-fill",
            )
        )
        agent = ProcessIntegrityAgent(
            context_store=store, position_store=position_store
        )
        finding = agent.investigate("trade-1")
        self.assertEqual(finding.verdict, "innocent")
        self.assertEqual(finding.agent, "process")

    # --------------------------------------------------------------- 2
    def test_stoploss_drift_primary_cause(self) -> None:
        store = _store()
        store.record_snapshot(_signal_snap(notes=None))
        store.record_snapshot(_fill_snap())
        # Trigger price 100, exit went off at 95 (5% drift) — that's a bug.
        store.record_snapshot(
            _breaker_snap(
                breaker_context={
                    "tripped": ["stop_loss"],
                    "reason": "stop_loss",
                    "recommended_action": "force_flat",
                    "details": {},
                    "trigger_price": 100.0,
                },
                notes="stop_loss force_flat",
            )
        )
        position_store = _StubPositionStore(
            _StubPosition(
                position_id="trade-1",
                exchange="coinbase-paper",
                exit_price=95.0,
                notes="paper-deferred-fill stop_loss force_flat",
            )
        )
        agent = ProcessIntegrityAgent(
            context_store=store, position_store=position_store
        )
        finding = agent.investigate("trade-1")
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertGreaterEqual(finding.confidence, 0.6)
        self.assertTrue(
            any("stop-loss" in e for e in finding.evidence),
            f"expected stop-loss evidence, got {finding.evidence!r}",
        )

    # --------------------------------------------------------------- 3
    def test_paper_live_divergence(self) -> None:
        store = _store()
        # Signal says paper.
        store.record_snapshot(_signal_snap(notes="paper-deferred-fill"))
        # Fill says live exchange.
        fill = _fill_snap(exchange="coinbase", notes=None)
        store.record_snapshot(fill)
        # Breaker not present — only two declared sources, but we'll
        # also wire the position to lean live so we get 3 declared
        # sources and the primary-cause path triggers.
        position_store = _StubPositionStore(
            _StubPosition(
                position_id="trade-1",
                exchange="coinbase",
                exit_price=99.0,
                notes=None,
            )
        )
        agent = ProcessIntegrityAgent(
            context_store=store, position_store=position_store
        )
        finding = agent.investigate("trade-1")
        # With three declared sources split (signal=paper, fill=live,
        # position=live), the agent escalates to primary_cause.
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertTrue(
            any("paper-vs-live" in e for e in finding.evidence),
            f"expected paper-vs-live evidence, got {finding.evidence!r}",
        )

    # --------------------------------------------------------------- 4
    def test_kill_switch_supposed_to_trip_but_didnt(self) -> None:
        store = _store()
        # Signal-time notes mention kill_switch.
        store.record_snapshot(_signal_snap(notes="kill_switch_pending"))
        store.record_snapshot(_fill_snap())
        # Breaker context did NOT record kill_switch as tripped.
        store.record_snapshot(
            _breaker_snap(
                breaker_context={
                    "tripped": ["daily_loss"],  # kill_switch missing
                    "reason": "daily_loss",
                    "recommended_action": "halt_new_entries",
                    "details": {},
                },
                notes=None,
            )
        )
        agent = ProcessIntegrityAgent(context_store=store)
        finding = agent.investigate("trade-1")
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertTrue(
            any("kill_switch" in e for e in finding.evidence),
            f"expected kill_switch evidence, got {finding.evidence!r}",
        )

    # --------------------------------------------------------------- 5
    def test_race_condition_trail_yields_contributing(self) -> None:
        # Live Redis stub used for both snapshot store AND error counter.
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        store = _store(redis_client=redis_client)
        signal = _signal_snap(symbol="BTC/USD", notes=None)
        store.record_snapshot(signal)
        store.record_snapshot(_fill_snap(symbol="BTC/USD", notes="paper-deferred-fill"))

        # Seed the per-symbol error counter for the trade's date with a
        # cluster of increments — exceeds the contention threshold.
        date_part = signal.captured_at_utc[:10]
        err_key = f"test:errors:by_symbol:{date_part}"
        redis_client.hset(err_key, "BTC/USD", _RACE_CONCURRENCY_THRESHOLD + 2)

        agent = ProcessIntegrityAgent(
            context_store=store,
            redis_client=redis_client,
            namespace="test",
        )
        finding = agent.investigate("trade-1")
        self.assertEqual(finding.verdict, "contributing")
        self.assertTrue(
            any("error counter" in e for e in finding.evidence),
            f"expected error-counter evidence, got {finding.evidence!r}",
        )

    # --------------------------------------------------------------- 6 (defense-in-depth)
    def test_missing_snapshots_yield_innocent_with_evidence(self) -> None:
        store = _store()  # entirely empty
        agent = ProcessIntegrityAgent(context_store=store)
        finding = agent.investigate("trade-missing")
        # No flags fire → innocent, but evidence records the gap.
        self.assertEqual(finding.verdict, "innocent")
        self.assertTrue(
            any("no snapshots" in e for e in finding.evidence),
            f"expected snapshot-gap evidence, got {finding.evidence!r}",
        )

    # --------------------------------------------------------------- 7
    # Phase-16: A5 prefers canonical TradeContextSnapshot fields over
    # substring-matching breaker_context / notes.
    def test_canonical_kill_switch_reason_satisfies_breaker_check(self) -> None:
        """A breaker snapshot with ``kill_switch_reason`` set must satisfy the
        kill-switch consistency check even without ``breaker_context.tripped``
        carrying ``"kill_switch"``."""
        store = _store()
        store.record_snapshot(_signal_snap(notes="kill_switch_pending"))
        store.record_snapshot(_fill_snap())
        # No "kill_switch" in tripped, but the canonical field IS set —
        # the legacy probe would flag this as primary, but with the
        # canonical field, A5 must accept it.
        breaker = _breaker_snap(
            breaker_context={
                "tripped": ["daily_loss"],
                "reason": "daily_loss",
                "recommended_action": "force_flat",
                "details": {},
            },
            notes=None,
        )
        breaker.kill_switch_reason = "kill_switch"
        store.record_snapshot(breaker)
        agent = ProcessIntegrityAgent(context_store=store)
        finding = agent.investigate("trade-1")
        # No primary kill_switch flag because the canonical field
        # confirms the trip even though the legacy substring isn't there.
        # Could land as innocent or contributing depending on the
        # other paths; the load-bearing assertion is that the
        # "supposed to trip but didn't" primary flag is NOT raised.
        self.assertNotEqual(finding.verdict, "primary_cause")
        self.assertFalse(
            any(
                "kill_switch" in e and "supposed to trip" in e.lower()
                for e in finding.evidence
            ),
            f"unexpected supposed-to-trip evidence: {finding.evidence!r}",
        )

    def test_canonical_stop_loss_trigger_price_used(self) -> None:
        """A5 prefers the canonical ``stop_loss_trigger_price`` over the
        legacy ``risk_metrics_input.stop_loss_trigger`` probe."""
        store = _store()
        store.record_snapshot(_signal_snap(notes=None))
        store.record_snapshot(_fill_snap())
        # No legacy keys at all — only the canonical field carries the
        # trigger price. With exit at 95 vs trigger at 100 the drift is
        # 5% which is well above the primary threshold.
        breaker = _breaker_snap(
            breaker_context={
                "tripped": ["stop_loss"],
                "reason": "stop_loss",
                "recommended_action": "force_flat",
                "details": {},
            },
            notes="stop_loss force_flat",
        )
        breaker.stop_loss_trigger_price = 100.0
        store.record_snapshot(breaker)
        position_store = _StubPositionStore(
            _StubPosition(
                position_id="trade-1",
                exchange="coinbase-paper",
                exit_price=95.0,
                notes="paper-deferred-fill stop_loss force_flat",
            )
        )
        agent = ProcessIntegrityAgent(
            context_store=store, position_store=position_store
        )
        finding = agent.investigate("trade-1")
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertTrue(
            any("stop-loss" in e for e in finding.evidence),
            f"expected stop-loss evidence, got {finding.evidence!r}",
        )

    def test_legacy_snapshot_without_canonical_fields_still_works(self) -> None:
        """Backward-compat: A5 must keep working on snapshots captured before
        Phase-16 (where ``stop_loss_trigger_price`` is None)."""
        store = _store()
        store.record_snapshot(_signal_snap(notes=None))
        store.record_snapshot(_fill_snap())
        breaker = _breaker_snap(
            breaker_context={
                "tripped": ["stop_loss"],
                "reason": "stop_loss",
                "recommended_action": "force_flat",
                "details": {},
                # Legacy place where the trigger price was stuffed.
                "trigger_price": 100.0,
            },
            notes="stop_loss force_flat",
        )
        # Canonical fields stay None — the legacy fall-through must fire.
        self.assertIsNone(breaker.stop_loss_trigger_price)
        store.record_snapshot(breaker)
        position_store = _StubPositionStore(
            _StubPosition(
                position_id="trade-1",
                exchange="coinbase-paper",
                exit_price=95.0,
                notes="paper-deferred-fill stop_loss force_flat",
            )
        )
        agent = ProcessIntegrityAgent(
            context_store=store, position_store=position_store
        )
        finding = agent.investigate("trade-1")
        # The legacy trigger_price probe still works — primary cause.
        self.assertEqual(finding.verdict, "primary_cause")

    def test_safe_investigate_swallows_store_failure(self) -> None:
        class _ExplodingStore:
            namespace = "test"
            ttl_seconds = 100

            def get_snapshots(self, trade_id: str) -> Dict[str, Any]:
                raise RuntimeError("boom")

        agent = ProcessIntegrityAgent(context_store=_ExplodingStore())  # type: ignore[arg-type]
        finding = agent.safe_investigate("trade-1")
        self.assertEqual(finding.verdict, "unknown")
        self.assertIsNotNone(finding.error)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
