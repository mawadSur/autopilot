"""Fixture 5: Race-condition loss.

A losing trade where the per-symbol error counter is elevated, implying
concurrent error writes on the same trade date. ProcessIntegrityAgent's
``_check_race_condition_trail`` reads the Redis HASH
``{ns}:errors:by_symbol:{date}`` and, when the per-symbol counter is at
or above ``_RACE_CONCURRENCY_THRESHOLD`` (5), emits a ``contributing``
verdict with a race-related evidence bullet.

To exercise that path we:
1. Populate a fakeredis client's ``test:errors:by_symbol:2026-05-08``
   hash with ``{symbol: "12"}`` (well above the threshold of 5).
2. Hand the same fakeredis client to the ProcessIntegrityAgent through
   the ``redis_client`` constructor arg.
3. Wire a TradeContextSnapshot for the signal phase so A5 has a date +
   symbol anchor; nothing else needs to be inconsistent — the race
   trail alone gives at least ``contributing``.

Returned: trade_id (the test owns the redis_client / namespace and
threads them into the agent factory itself).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


SYMBOL = "ETH/USD"
TRADE_ID = "fixture-race-condition-trade-1"
ERROR_COUNTER_VALUE = 12  # >> _RACE_CONCURRENCY_THRESHOLD (5)


def build_fixture(
    *,
    context_store: TradeContextStore,
    position_store: Optional[PositionStore] = None,
    meta_base_dir: Optional[str] = None,
    trade_id: str = TRADE_ID,
    symbol: str = SYMBOL,
    redis_client=None,
    namespace: str = "autopilot",
) -> str:
    """Populate stores + Redis error counter for a race-condition scenario.

    The caller MUST pass ``redis_client`` (e.g. the same fakeredis client
    that backs ``context_store``) and the same ``namespace`` it will use
    when constructing the ProcessIntegrityAgent — otherwise the hash key
    won't line up and the race check won't fire.
    """

    captured_at = datetime(2026, 5, 8, 16, 0, 0, tzinfo=timezone.utc)

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
            "threshold_metrics": {"0.5": {"reliability_slope": 0.85}},
        }
        with open(os.path.join(slug_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh)

    # ----- populate the per-symbol error counter so A5 sees contention -----
    # Process A5's `_check_race_condition_trail` looks at
    # `{ns}:errors:by_symbol:{date_part}` (date taken from the snapshot's
    # captured_at_utc) and HGETs `symbol`. We set it well above the
    # threshold to ensure the contributing verdict.
    if redis_client is not None:
        date_part = captured_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
        key = f"{namespace}:errors:by_symbol:{date_part}"
        try:
            redis_client.hset(key, symbol, str(ERROR_COUNTER_VALUE))
        except Exception:  # noqa: BLE001 - tolerate odd client mocks
            pass

    # ----- signal snapshot: clean features so only A5 fires -----
    signal_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.isoformat(),
        phase="signal",
        feature_buffer={"return_1": 0.02, "atr_14": 1.05},
        feature_window=[
            {"regime": "trend", "return_1": 0.02},
            {"regime": "trend", "return_1": 0.02},
            {"regime": "trend", "return_1": 0.02},
            {"regime": "trend", "return_1": 0.02},
        ],
        model_probs={"long": 0.65, "short": 0.35},
        model_confidence=0.65,
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
                "bid": 2_999.5,
                "ask": 3_000.5,
                "last": 3_000.0,
                "as_of_utc": captured_at.isoformat(),
            }
        ],
        notes=None,
    )
    context_store.record_snapshot(signal_snap)

    # ----- fill snapshot: clean -----
    fill_t = captured_at + timedelta(seconds=1)
    fill_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=fill_t.isoformat(),
        phase="fill",
        feature_buffer={},
        model_confidence=0.0,
        risk_metrics_output={
            "fill_price": 3_000.5,
            "fill_size": 0.016,
            "exchange": "coinbase",
            "status": "open",
        },
        ticker_buffer=[],
        notes=None,
    )
    context_store.record_snapshot(fill_snap)

    # ----- position record (loss) -----
    if position_store is not None:
        position = Position(
            position_id=trade_id,
            exchange="coinbase",
            symbol=symbol,
            side="long",
            status="closed",
            entry_price=3_000.5,
            entry_quote_usd=3_000.5 * 0.016,
            base_size=0.016,
            exit_price=2_975.0,
            exit_quote_usd=2_975.0 * 0.016,
            realized_pnl_usd=-0.408,
            opened_at_utc=captured_at.isoformat(),
            closed_at_utc=(captured_at + timedelta(minutes=10)).isoformat(),
            notes=None,
            model_meta={},
        )
        position_store.record_open(position)
        position_store.record_close(
            trade_id, exit_price=2_975.0, exit_quote_usd=2_975.0 * 0.016
        )

    return trade_id
