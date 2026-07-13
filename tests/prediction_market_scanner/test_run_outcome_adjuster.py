"""Tests for ``scripts/run_outcome_adjuster.py``.

Same loading + fixture pattern as ``test_diagnose_calibration_drift.py``:
script loaded via importlib because ``scripts/`` isn't on sys.path under
the prediction-market-scanner runner.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import fakeredis

from regime_memory.outcome_adjuster import OutcomeAdjuster
from state.position_store import Position, PositionStore
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_outcome_adjuster.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_run_outcome_adjuster_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_run_outcome_adjuster_under_test"] = module
    spec.loader.exec_module(module)
    return module


roa = _load_script_module()


NAMESPACE = "test-roa"
TARGET_DATE = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc).date()


def _fresh_stack():
    client = fakeredis.FakeRedis(decode_responses=True)
    pstore = PositionStore(redis_client=client, namespace=NAMESPACE)
    cstore = TradeContextStore(redis_client=client, namespace=NAMESPACE)
    adjuster = OutcomeAdjuster(
        client,
        namespace=NAMESPACE,
        max_adjustment=0.05,
        losses_to_raise=3,
        wins_to_relax=5,
        per_event_delta=0.01,
    )
    return client, pstore, cstore, adjuster


def _seed_position_with_label(
    pstore: PositionStore,
    cstore: TradeContextStore,
    *,
    position_id: str,
    pnl: float,
    label_value: float,
    symbol: str = "ETH/USD",
    closed_at: Optional[datetime] = None,
    use_snapshot: bool = True,
) -> Position:
    """Seed a closed position; attach the regime label via signal snapshot.

    ``use_snapshot=True`` writes the label into the signal-snapshot's
    risk_metrics_input dict (matching the resolver's primary path).
    """
    opened_at = datetime(
        TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day, 9, 0, 0,
        tzinfo=timezone.utc,
    )
    closed_at = closed_at or (opened_at + timedelta(minutes=10))

    position = Position(
        position_id=position_id,
        exchange="coinbase-paper",
        symbol=symbol,
        side="long",  # type: ignore[arg-type]
        status="closed",
        entry_price=2000.0,
        entry_quote_usd=50.0,
        base_size=0.025,
        exit_price=2010.0 if pnl > 0 else 1990.0,
        exit_quote_usd=50.0 + pnl,
        realized_pnl_usd=float(pnl),
        fees_usd=0.0,
        opened_at_utc=opened_at.isoformat(),
        closed_at_utc=closed_at.isoformat(),
    )
    # Persist via the same channel the script reads from (closed-set).
    blob_key = pstore._position_key(position_id)
    closed_key = pstore._closed_set_key(closed_at)
    pstore._redis.set(blob_key, position.model_dump_json())
    pstore._redis.sadd(closed_key, position_id)

    if use_snapshot:
        snap = TradeContextSnapshot(
            trade_id=position_id,
            symbol=symbol,
            captured_at_utc=utc_now_iso(),
            phase="signal",
            feature_buffer={},
            feature_window=None,
            model_probs={},
            model_confidence=0.6,
            risk_metrics_input={"regime_label": float(label_value)},
            risk_metrics_output={},
            breaker_context={},
            ticker_buffer=[],
            notes=None,
        )
        cstore.record_snapshot(snap)
    return position


def _make_args(**overrides) -> SimpleNamespace:
    defaults = {
        "date": TARGET_DATE.isoformat(),
        "position_store_url": None,
        "namespace": NAMESPACE,
        "symbol": None,
        "dry_run": False,
        "alert": False,
        "reset": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class HappyPathTests(unittest.TestCase):
    def test_two_regimes_writes_expected_adjustments(self) -> None:
        client, pstore, cstore, adjuster = _fresh_stack()
        # 6 losses in trend_up (label=2.0), 6 losses in chop (label=1.0).
        # Both should bump to +0.02 (two crossings of 3).
        for i in range(1, 7):
            _seed_position_with_label(
                pstore, cstore,
                position_id=f"U-{i}",
                pnl=-1.0,
                label_value=2.0,
                closed_at=datetime(
                    TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                    9, i, 0, tzinfo=timezone.utc,
                ),
            )
        for i in range(1, 7):
            _seed_position_with_label(
                pstore, cstore,
                position_id=f"C-{i}",
                pnl=-1.0,
                label_value=1.0,
                closed_at=datetime(
                    TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                    10, i, 0, tzinfo=timezone.utc,
                ),
            )
        args = _make_args()
        exit_code, report, blob = roa.run(
            args=args,
            store=pstore,
            trade_ctx_store=cstore,
            adjuster=adjuster,
            now_utc=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day + 1,
                0, 10, 0, tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(blob["n_positions"], 12)
        self.assertEqual(blob["n_resolved_labels"], 12)
        self.assertEqual(blob["n_skipped"], 0)
        self.assertAlmostEqual(
            blob["new_adjustments"]["trend_up"], 0.02, places=6
        )
        self.assertAlmostEqual(
            blob["new_adjustments"]["chop"], 0.02, places=6
        )
        # Adjuster state matches.
        self.assertAlmostEqual(adjuster.current_adjustment("trend_up"), 0.02, places=6)
        self.assertAlmostEqual(adjuster.current_adjustment("chop"), 0.02, places=6)


class DryRunTests(unittest.TestCase):
    def test_dry_run_writes_nothing(self) -> None:
        client, pstore, cstore, adjuster = _fresh_stack()
        for i in range(1, 4):
            _seed_position_with_label(
                pstore, cstore,
                position_id=f"U-{i}",
                pnl=-1.0,
                label_value=2.0,
                closed_at=datetime(
                    TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                    9, i, 0, tzinfo=timezone.utc,
                ),
            )
        args = _make_args(dry_run=True)
        exit_code, report, blob = roa.run(
            args=args,
            store=pstore,
            trade_ctx_store=cstore,
            adjuster=adjuster,
            now_utc=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day + 1,
                0, 10, 0, tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("DRY-RUN", report)
        self.assertIn("no Redis writes", report)
        # Simulated new state shows what it WOULD have been...
        self.assertAlmostEqual(
            blob["new_adjustments"].get("trend_up", 0.0), 0.01, places=6
        )
        # ...but the persisted state is unchanged.
        self.assertEqual(adjuster.current_adjustment("trend_up"), 0.0)


class SkippedLabelTests(unittest.TestCase):
    def test_positions_without_label_are_counted_skipped(self) -> None:
        client, pstore, cstore, adjuster = _fresh_stack()
        # Two with label, two without (no snapshot written).
        _seed_position_with_label(
            pstore, cstore, position_id="U-1", pnl=-1.0, label_value=2.0,
            closed_at=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                9, 1, 0, tzinfo=timezone.utc,
            ),
        )
        _seed_position_with_label(
            pstore, cstore, position_id="U-2", pnl=-1.0, label_value=2.0,
            closed_at=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                9, 2, 0, tzinfo=timezone.utc,
            ),
        )
        # Two without snapshot — no resolvable label.
        _seed_position_with_label(
            pstore, cstore, position_id="N-1", pnl=-1.0, label_value=0.0,
            closed_at=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                9, 3, 0, tzinfo=timezone.utc,
            ),
            use_snapshot=False,
        )
        _seed_position_with_label(
            pstore, cstore, position_id="N-2", pnl=-1.0, label_value=0.0,
            closed_at=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
                9, 4, 0, tzinfo=timezone.utc,
            ),
            use_snapshot=False,
        )
        args = _make_args()
        exit_code, report, blob = roa.run(
            args=args,
            store=pstore,
            trade_ctx_store=cstore,
            adjuster=adjuster,
            now_utc=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day + 1,
                0, 10, 0, tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(blob["n_positions"], 4)
        self.assertEqual(blob["n_resolved_labels"], 2)
        self.assertEqual(blob["n_skipped"], 2)


class ResetFlagTests(unittest.TestCase):
    def test_reset_all_clears_hash(self) -> None:
        client, pstore, cstore, adjuster = _fresh_stack()
        client.hset(adjuster.full_hash_key, "trend_up", "0.020000")
        client.hset(adjuster.full_hash_key, "chop", "0.010000")
        args = _make_args(reset="all")
        exit_code, msg, blob = roa.run(
            args=args,
            store=pstore,
            trade_ctx_store=cstore,
            adjuster=adjuster,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(adjuster.all_adjustments(), {})

    def test_reset_single_label(self) -> None:
        client, pstore, cstore, adjuster = _fresh_stack()
        client.hset(adjuster.full_hash_key, "trend_up", "0.020000")
        client.hset(adjuster.full_hash_key, "chop", "0.010000")
        args = _make_args(reset="trend_up")
        exit_code, msg, blob = roa.run(
            args=args,
            store=pstore,
            trade_ctx_store=cstore,
            adjuster=adjuster,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(adjuster.current_adjustment("trend_up"), 0.0)
        self.assertAlmostEqual(
            adjuster.current_adjustment("chop"), 0.01, places=6
        )


class LabelResolverTests(unittest.TestCase):
    def test_label_from_signal_snapshot_risk_metrics_input(self) -> None:
        _, pstore, cstore, adjuster = _fresh_stack()
        pos = _seed_position_with_label(
            pstore, cstore, position_id="X", pnl=-1.0, label_value=2.0,
        )
        label = roa.resolve_regime_label(pos, trade_ctx_store=cstore)
        self.assertEqual(label, "trend_up")

    def test_label_from_model_meta(self) -> None:
        _, pstore, cstore, adjuster = _fresh_stack()
        # No snapshot, but model_meta carries the label directly.
        opened_at = datetime(
            TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
            9, 0, 0, tzinfo=timezone.utc,
        )
        pos = Position(
            position_id="META-1",
            exchange="coinbase-paper",
            symbol="ETH/USD",
            side="long",  # type: ignore[arg-type]
            status="closed",
            entry_price=2000.0,
            entry_quote_usd=50.0,
            base_size=0.025,
            exit_price=1990.0,
            exit_quote_usd=49.0,
            realized_pnl_usd=-1.0,
            fees_usd=0.0,
            opened_at_utc=opened_at.isoformat(),
            closed_at_utc=(opened_at + timedelta(minutes=5)).isoformat(),
            model_meta={"regime_label": 1.0},
        )
        label = roa.resolve_regime_label(pos, trade_ctx_store=cstore)
        self.assertEqual(label, "chop")

    def test_no_label_anywhere_returns_none(self) -> None:
        _, pstore, cstore, adjuster = _fresh_stack()
        opened_at = datetime(
            TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day,
            9, 0, 0, tzinfo=timezone.utc,
        )
        pos = Position(
            position_id="EMPTY-1",
            exchange="coinbase-paper",
            symbol="ETH/USD",
            side="long",  # type: ignore[arg-type]
            status="closed",
            entry_price=2000.0,
            entry_quote_usd=50.0,
            base_size=0.025,
            exit_price=1990.0,
            exit_quote_usd=49.0,
            realized_pnl_usd=-1.0,
            fees_usd=0.0,
            opened_at_utc=opened_at.isoformat(),
            closed_at_utc=(opened_at + timedelta(minutes=5)).isoformat(),
        )
        label = roa.resolve_regime_label(pos, trade_ctx_store=cstore)
        self.assertIsNone(label)


class EmptyDayTests(unittest.TestCase):
    def test_no_positions_for_date_runs_clean(self) -> None:
        _, pstore, cstore, adjuster = _fresh_stack()
        args = _make_args()
        exit_code, report, blob = roa.run(
            args=args,
            store=pstore,
            trade_ctx_store=cstore,
            adjuster=adjuster,
            now_utc=datetime(
                TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day + 1,
                0, 10, 0, tzinfo=timezone.utc,
            ),
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(blob["n_positions"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
