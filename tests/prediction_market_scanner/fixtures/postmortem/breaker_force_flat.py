"""Fixture 3: Breaker force-flat loss.

A losing trade where the kill switch was tripped mid-position. Position
notes contain ``force_flat`` and ``kill_switch`` markers; the breaker
snapshot's ``breaker_context.tripped`` list, however, did NOT record
``kill_switch`` — only ``daily_loss``. This is a classic "supposed to
trip but didn't surface in structured logging" bug.

Expected verdict (A5 ProcessIntegrityAgent):
- ``primary_cause`` because notes imply kill_switch but breaker_context
  doesn't record it as tripped (`_check_kill_switch_consistency` flag).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


SYMBOL = "SOL/USD"
TRADE_ID = "fixture-breaker-force-flat-trade-1"


def build_fixture(
    *,
    context_store: TradeContextStore,
    position_store: Optional[PositionStore] = None,
    meta_base_dir: Optional[str] = None,
    trade_id: str = TRADE_ID,
    symbol: str = SYMBOL,
) -> str:
    captured_at = datetime(2026, 5, 8, 14, 0, 0, tzinfo=timezone.utc)

    # ----- healthy meta so signal forensics stays clean -----
    if meta_base_dir is not None:
        slug = (
            symbol.lower()
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        slug_dir = os.path.join(meta_base_dir, slug)
        os.makedirs(slug_dir, exist_ok=True)
        meta_payload = {
            "optimal_threshold": 0.5,
            "feature_means": {"return_1": 0.0, "atr_14": 1.0},
            "feature_stds": {"return_1": 1.0, "atr_14": 0.5},
            "metrics_test": {"reliability_slope": 0.85},
        }
        with open(os.path.join(slug_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh)

    # ----- signal snapshot — notes mention kill_switch_pending -----
    signal_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.isoformat(),
        phase="signal",
        feature_buffer={"return_1": 0.02, "atr_14": 1.05},
        model_probs={"long": 0.7, "short": 0.3},
        model_confidence=0.7,
        risk_metrics_input={
            "side": "long",
            "proposed_notional_usd": 50.0,
            "bankroll": 10_000.0,
        },
        risk_metrics_output={
            "adjusted_position_size_pct": 0.5,
            "expected_value_estimate": 0.2,
        },
        breaker_context={"recommended_action": "allow", "tripped": []},
        ticker_buffer=[
            {
                "symbol": symbol,
                "bid": 99.5,
                "ask": 100.5,
                "last": 100.0,
                "as_of_utc": captured_at.isoformat(),
            }
        ],
        # The signal-time notes hint at kill_switch — A5's coherence
        # check looks for this.
        notes="kill_switch_pending",
    )
    context_store.record_snapshot(signal_snap)

    # ----- fill snapshot — clean fill -----
    fill_t = captured_at + timedelta(seconds=1)
    fill_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=fill_t.isoformat(),
        phase="fill",
        feature_buffer={},
        model_confidence=0.0,
        risk_metrics_output={
            "fill_price": 100.0,
            "fill_size": 0.5,
            "exchange": "coinbase",
            "status": "open",
        },
        ticker_buffer=[],
        notes=None,
    )
    context_store.record_snapshot(fill_snap)

    # ----- breaker snapshot: kill_switch_referenced in notes but tripped list omits it -----
    breaker_t = captured_at + timedelta(minutes=5)
    breaker_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=breaker_t.isoformat(),
        phase="breaker",
        feature_buffer={},
        model_confidence=0.0,
        risk_metrics_input={},
        risk_metrics_output={},
        breaker_context={
            "tripped": ["daily_loss"],  # kill_switch missing — the bug A5 detects
            "reason": "daily_loss",
            "recommended_action": "halt_new_entries",
            "details": {},
        },
        ticker_buffer=[],
        notes="kill_switch force_flat halt_new_entries",
    )
    context_store.record_snapshot(breaker_snap)

    # ----- position record -----
    if position_store is not None:
        position = Position(
            position_id=trade_id,
            exchange="coinbase",
            symbol=symbol,
            side="long",
            status="closed",
            entry_price=100.0,
            entry_quote_usd=100.0 * 0.5,
            base_size=0.5,
            exit_price=98.0,
            exit_quote_usd=98.0 * 0.5,
            realized_pnl_usd=-1.0,
            opened_at_utc=captured_at.isoformat(),
            closed_at_utc=breaker_t.isoformat(),
            notes="force_flat kill_switch halt_new_entries",
            model_meta={},
        )
        position_store.record_open(position)
        position_store.record_close(
            trade_id, exit_price=98.0, exit_quote_usd=98.0 * 0.5
        )

    return trade_id
