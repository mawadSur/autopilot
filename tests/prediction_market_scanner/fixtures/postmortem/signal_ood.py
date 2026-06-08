"""Fixture 1: Signal-OOD loss.

A losing trade where the model's signal fired with features ~6σ from the
training distribution. The fixture seeds a TradeContextSnapshot at the
``signal`` phase whose ``feature_buffer`` is well outside the training
mean/std distribution recorded in the per-symbol ``model_crypto/<slug>/meta.json``
file (which the fixture writes to a tmpdir the test redirects A1 to).

Combined red flags this fixture deliberately stages:
1. Confidence within 5 % of the optimal threshold (thin margin).
2. Mahalanobis distance > 3σ (extreme OOD across many features).
3. Anti-calibrated reliability slope at the active probability bin.
4. Regime shift in the last 3 bars before signal.

The signal-forensics agent's verdict ladder counts 3+ red flags as
``primary_cause``. This fixture stages 4 distinct flags so the verdict
is robust to any single check skipping due to a data shape it doesn't
recognise.

Returned: ``trade_id``. The caller is responsible for instantiating the
SignalForensicsAgent with ``meta_base_dir=<the meta_base_dir argument>``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


SYMBOL = "BTC/USD"
TRADE_ID = "fixture-signal-ood-trade-1"


def build_fixture(
    *,
    context_store: TradeContextStore,
    position_store: Optional[PositionStore] = None,
    meta_base_dir: Optional[str] = None,
    trade_id: str = TRADE_ID,
    symbol: str = SYMBOL,
) -> str:
    """Populate stores + meta dir for a Signal-OOD scenario.

    Returns the ``trade_id``. Tests can pass this to the synthesizer's
    ``process_one(trade_id)`` once they have an in-process A1 wired with
    ``meta_base_dir`` pointing at the same tmpdir that this function
    writes to.
    """

    captured_at = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)

    # ----- model meta with training distribution + anti-calibrated slope -----
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
            "optimal_threshold": 0.6,
            # Mean=0, std=1 across many features so a feature value of 6.0
            # produces a per-feature z-score of 6 → well above the 2σ cap.
            "feature_means": {f"f{i}": 0.0 for i in range(8)},
            "feature_stds": {f"f{i}": 1.0 for i in range(8)},
            # Anti-calibrated reliability_slope at the bin closest to
            # the model's active probability triggers check 4.
            "metrics_test": {"reliability_slope": -0.7},
            "threshold_metrics": {
                "0.55": {"reliability_slope": -0.6},
                "0.60": {"reliability_slope": -0.5},
            },
        }
        with open(os.path.join(slug_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh)

    # ----- signal snapshot: thin margin + OOD features + regime shift -----
    feature_buffer = {f"f{i}": 6.0 for i in range(8)}
    feature_window = [
        # Regime shift inside the last 3 bars (range vs trend).
        {"regime": "range", "f0": 5.5},
        {"regime": "range", "f0": 5.7},
        {"regime": "range", "f0": 5.8},
        {"regime": "trend", "f0": 6.0},  # most recent — different regime
    ]
    signal_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.isoformat(),
        phase="signal",
        feature_buffer=feature_buffer,
        feature_window=feature_window,
        # model_probs left empty so the agent falls back to model_confidence
        # for the reliability bin lookup, exercising the documented fallback
        # path.
        model_probs={},
        model_confidence=0.61,  # 0.61 vs threshold 0.60 → thin margin red flag
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
                "bid": 30_000.0,
                "ask": 30_001.0,
                "last": 30_000.5,
                "as_of_utc": captured_at.isoformat(),
            }
        ],
        notes="paper-deferred-fill",
    )
    context_store.record_snapshot(signal_snap)

    # ----- fill snapshot: clean (no execution issues) -----
    fill_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.replace(second=1).isoformat(),
        phase="fill",
        feature_buffer={},
        model_confidence=0.0,
        risk_metrics_output={
            "fill_price": 30_001.0,
            "fill_size": 0.003,
            "exchange": "coinbase-paper",
            "status": "open",
        },
        ticker_buffer=[],
        notes="paper-deferred-fill",
    )
    context_store.record_snapshot(fill_snap)

    # ----- position record -----
    if position_store is not None:
        position = Position(
            position_id=trade_id,
            exchange="coinbase-paper",
            symbol=symbol,
            side="long",
            status="closed",
            entry_price=30_001.0,
            entry_quote_usd=30_001.0 * 0.003,
            base_size=0.003,
            exit_price=29_700.0,
            exit_quote_usd=29_700.0 * 0.003,
            realized_pnl_usd=-0.9,
            opened_at_utc=captured_at.isoformat(),
            closed_at_utc=captured_at.replace(minute=10).isoformat(),
            notes="paper-deferred-fill exit_via_signal_reverse",
            model_meta={},
        )
        position_store.record_open(position)
        position_store.record_close(
            trade_id, exit_price=29_700.0, exit_quote_usd=29_700.0 * 0.003
        )

    return trade_id
