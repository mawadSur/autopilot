"""Fixture 2: Slippage-driven loss.

A losing trade where the model signal was healthy but the live fill
landed 60-70 bps worse than the signal-time mid. ExecutionForensicsAgent
should classify this as ``primary_cause`` with a slippage evidence
bullet; SignalForensicsAgent should remain ``innocent`` (or at most
``contributing``) because the features and confidence were healthy.

Constants:
- Signal-time mid: 30_000.0
- Entry fill price: 30_210.0  → 210/30000 * 10_000 = 70 bps slippage
- Threshold for ``primary_cause`` is 50 bps (SLIPPAGE_PRIMARY_BPS).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


SYMBOL = "ETH/USD"
TRADE_ID = "fixture-slippage-trade-1"


def build_fixture(
    *,
    context_store: TradeContextStore,
    position_store: Optional[PositionStore] = None,
    meta_base_dir: Optional[str] = None,
    trade_id: str = TRADE_ID,
    symbol: str = SYMBOL,
) -> str:
    captured_at = datetime(2026, 5, 8, 13, 0, 0, tzinfo=timezone.utc)

    # ----- write a healthy model meta so SignalForensicsAgent stays innocent -----
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
            "metrics_test": {"reliability_slope": 0.85},  # well-calibrated
            "threshold_metrics": {
                "0.5": {"reliability_slope": 0.85},
                "0.6": {"reliability_slope": 0.9},
            },
        }
        with open(os.path.join(slug_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh)

    # ----- signal snapshot: healthy features, comfortable confidence -----
    signal_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.isoformat(),
        phase="signal",
        feature_buffer={"return_1": 0.05, "atr_14": 1.05},
        feature_window=[
            {"regime": "trend", "return_1": 0.04},
            {"regime": "trend", "return_1": 0.05},
            {"regime": "trend", "return_1": 0.05},
            {"regime": "trend", "return_1": 0.05},
        ],
        model_probs={"long": 0.85, "short": 0.15},
        model_confidence=0.85,  # well above threshold 0.5
        risk_metrics_input={
            "side": "long",
            "proposed_notional_usd": 100.0,
            "bankroll": 10_000.0,
        },
        risk_metrics_output={
            "adjusted_position_size_pct": 1.0,
            "expected_value_estimate": 0.5,
        },
        breaker_context={"recommended_action": "allow", "tripped": []},
        ticker_buffer=[
            {
                "symbol": symbol,
                "bid": 2_999.5,
                "ask": 3_000.5,
                "last": 3_000.0,
                "as_of_utc": captured_at.isoformat(),
            }
        ],
        notes=None,
    )
    context_store.record_snapshot(signal_snap)

    # ----- fill snapshot: 1 second after signal (no latency flag),
    #       fill mid drifted but the actual entry is what matters for slippage -----
    fill_t = captured_at + timedelta(seconds=1)
    fill_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=fill_t.isoformat(),
        phase="fill",
        feature_buffer={},
        model_confidence=0.0,
        risk_metrics_output={
            "fill_price": 3_021.0,  # ~70 bps above the 3000 signal-mid
            "fill_size": 0.033,
            "exchange": "coinbase",
            "status": "open",
        },
        ticker_buffer=[
            {
                "symbol": symbol,
                "bid": 3_020.5,
                "ask": 3_021.5,
                "last": 3_021.0,
                "as_of_utc": fill_t.isoformat(),
            }
        ],
        notes=None,
    )
    context_store.record_snapshot(fill_snap)

    # ----- position record: entry at 3021 vs signal-mid 3000 → 70 bps slip -----
    if position_store is not None:
        position = Position(
            position_id=trade_id,
            exchange="coinbase",
            symbol=symbol,
            side="long",
            status="closed",
            entry_price=3_021.0,
            entry_quote_usd=3_021.0 * 0.033,
            base_size=0.033,
            exit_price=2_980.0,  # closed lower → loss
            exit_quote_usd=2_980.0 * 0.033,
            realized_pnl_usd=-1.353,
            opened_at_utc=captured_at.isoformat(),
            closed_at_utc=(captured_at + timedelta(minutes=10)).isoformat(),
            notes=None,
            model_meta={},
        )
        position_store.record_open(position)
        position_store.record_close(
            trade_id, exit_price=2_980.0, exit_quote_usd=2_980.0 * 0.033
        )

    return trade_id
