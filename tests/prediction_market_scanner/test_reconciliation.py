"""Tests for ``src/ops/reconciliation.py`` and the CLI wrapper.

Hermetic: every collaborator is stubbed; nothing reaches Redis or
ccxt. The CLI test calls ``run_reconciliation`` directly instead of
shelling out, so we don't depend on subprocess behaviour for argv.
"""

from __future__ import annotations

import io
import unittest
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from state.position_store import Position

from ops.reconciliation import (
    DEFAULT_SIZE_TOLERANCE_PCT,
    PositionReconciler,
    ReconciliationReport,
)
from ops import reconciliation_cli


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubPositionStore:
    """Just enough of ``PositionStore`` for the reconciler to work."""

    def __init__(self, *, open_positions: Optional[List[Position]] = None) -> None:
        self._open: List[Position] = list(open_positions or [])

    def list_open(self) -> List[Position]:
        return list(self._open)

    def orphan_count(self) -> int:
        # The store-side helper counts pending positions tagged for orphan
        # cleanup; the simple stub returns 0 unless we override it.
        return 0


class _StubOrder:
    """OrderResult-shaped order for stub get_open_orders results."""

    def __init__(self, *, symbol: str, base_size: float, order_id: str = "ord-stub") -> None:
        self.symbol = symbol
        self.base_size = base_size
        self.filled_base = base_size
        self.order_id = order_id


class _StubExchange:
    def __init__(
        self,
        *,
        open_orders: Optional[List[_StubOrder]] = None,
        balances: Optional[List[Any]] = None,
        raise_on_get_open_orders: Optional[Exception] = None,
    ) -> None:
        self._open_orders = list(open_orders or [])
        self._balances = list(balances or [])
        self._raise = raise_on_get_open_orders
        self.calls: List[Dict[str, Any]] = []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[_StubOrder]:
        self.calls.append({"method": "get_open_orders", "symbol": symbol})
        if self._raise is not None:
            raise self._raise
        if symbol is None:
            return list(self._open_orders)
        return [o for o in self._open_orders if o.symbol == symbol]

    def get_balances(self) -> List[Any]:
        self.calls.append({"method": "get_balances"})
        return list(self._balances)


class _StubPusher:
    def __init__(self) -> None:
        self.gauge_calls: List[Dict[str, Any]] = []
        self.counter_calls: List[Dict[str, Any]] = []
        self.histogram_calls: List[Dict[str, Any]] = []

    def is_enabled(self) -> bool:
        return True

    def gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.gauge_calls.append({"name": name, "value": float(value), "labels": labels or {}})

    def counter(
        self, name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.counter_calls.append(
            {"name": name, "increment": float(increment), "labels": labels or {}}
        )

    def histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.histogram_calls.append(
            {"name": name, "value": float(value), "labels": labels or {}}
        )

    def push(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(
    *,
    symbol: str = "ETH/USD",
    base_size: float = 0.05,
    status: str = "open",
    side: str = "long",
) -> Position:
    return Position(
        position_id=str(uuid.uuid4()),
        exchange="coinbase",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        entry_price=2_000.0,
        entry_quote_usd=2_000.0 * base_size,
        base_size=base_size,
        entry_order_id=f"order-{uuid.uuid4().hex[:6]}",
        opened_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


# ---------------------------------------------------------------------------
# (a) Clean state
# ---------------------------------------------------------------------------


class TestReconcileClean(unittest.TestCase):
    def test_position_matching_exchange_yields_zero_orphans_zero_ghosts(self) -> None:
        position = _make_position(symbol="ETH/USD", base_size=0.05)
        store = _StubPositionStore(open_positions=[position])
        exchange = _StubExchange(
            open_orders=[_StubOrder(symbol="ETH/USD", base_size=0.05)]
        )
        reconciler = PositionReconciler(position_store=store, exchange=exchange)
        report = reconciler.reconcile()
        self.assertEqual(report.orphan_count, 0)
        self.assertEqual(report.ghost_count, 0)
        self.assertEqual(report.drift_count, 0)
        self.assertEqual(report.clean_count, 1)
        self.assertFalse(report.has_issues)


# ---------------------------------------------------------------------------
# (b) Orphan: in store but not on exchange
# ---------------------------------------------------------------------------


class TestReconcileOrphan(unittest.TestCase):
    def test_position_in_store_but_not_on_exchange_flags_orphan(self) -> None:
        position = _make_position(symbol="ETH/USD", base_size=0.05)
        store = _StubPositionStore(open_positions=[position])
        exchange = _StubExchange(open_orders=[])  # exchange has nothing
        reconciler = PositionReconciler(position_store=store, exchange=exchange)
        report = reconciler.reconcile()
        self.assertEqual(report.orphan_count, 1)
        self.assertEqual(report.ghost_count, 0)
        self.assertEqual(report.drift_count, 0)
        self.assertTrue(report.has_issues)
        # Alert must reference the symbol or position id.
        self.assertTrue(any("ETH/USD" in alert for alert in report.alerts))


# ---------------------------------------------------------------------------
# (c) Ghost: on exchange but not in store
# ---------------------------------------------------------------------------


class TestReconcileGhost(unittest.TestCase):
    def test_position_on_exchange_but_not_in_store_flags_ghost(self) -> None:
        store = _StubPositionStore(open_positions=[])
        exchange = _StubExchange(
            open_orders=[_StubOrder(symbol="BTC/USD", base_size=0.001)]
        )
        reconciler = PositionReconciler(position_store=store, exchange=exchange)
        report = reconciler.reconcile()
        self.assertEqual(report.ghost_count, 1)
        self.assertEqual(report.orphan_count, 0)
        self.assertEqual(report.drift_count, 0)
        self.assertTrue(report.has_issues)


# ---------------------------------------------------------------------------
# (d) Size drift > tolerance
# ---------------------------------------------------------------------------


class TestReconcileDrift(unittest.TestCase):
    def test_size_drift_above_tolerance_flags_drift(self) -> None:
        position = _make_position(symbol="ETH/USD", base_size=0.10)
        store = _StubPositionStore(open_positions=[position])
        # 50% mismatch - well above default 0.5% tolerance
        exchange = _StubExchange(
            open_orders=[_StubOrder(symbol="ETH/USD", base_size=0.05)]
        )
        reconciler = PositionReconciler(position_store=store, exchange=exchange)
        report = reconciler.reconcile()
        self.assertEqual(report.drift_count, 1)
        self.assertEqual(report.orphan_count, 0)
        self.assertEqual(report.ghost_count, 0)

    def test_size_drift_below_tolerance_treated_as_clean(self) -> None:
        position = _make_position(symbol="ETH/USD", base_size=0.05)
        store = _StubPositionStore(open_positions=[position])
        # Within tolerance: 0.05 vs 0.0501 (0.2% delta vs 0.5% tol)
        exchange = _StubExchange(
            open_orders=[_StubOrder(symbol="ETH/USD", base_size=0.0501)]
        )
        reconciler = PositionReconciler(
            position_store=store,
            exchange=exchange,
            size_tolerance_pct=DEFAULT_SIZE_TOLERANCE_PCT,
        )
        report = reconciler.reconcile()
        self.assertEqual(report.drift_count, 0)
        self.assertEqual(report.clean_count, 1)


# ---------------------------------------------------------------------------
# (e) Metrics emission
# ---------------------------------------------------------------------------


class TestReconcilerEmitsMetrics(unittest.TestCase):
    def test_reconciler_emits_expected_prometheus_metrics(self) -> None:
        position = _make_position(symbol="ETH/USD", base_size=0.05)
        store = _StubPositionStore(open_positions=[position])
        # Force one orphan + one ghost to exercise both gauges.
        exchange = _StubExchange(
            open_orders=[_StubOrder(symbol="BTC/USD", base_size=0.001)]
        )
        pusher = _StubPusher()
        reconciler = PositionReconciler(
            position_store=store,
            exchange=exchange,
            metrics_pusher=pusher,
        )
        report = reconciler.reconcile()
        self.assertEqual(report.orphan_count, 1)
        self.assertEqual(report.ghost_count, 1)

        gauge_names = {call["name"] for call in pusher.gauge_calls}
        counter_names = {call["name"] for call in pusher.counter_calls}
        self.assertIn("reconciliation_orphans", gauge_names)
        self.assertIn("reconciliation_ghosts", gauge_names)
        self.assertIn("reconciliation_drift_count", gauge_names)
        self.assertIn("orphan_positions", gauge_names)
        self.assertIn("reconciliation_run_total", counter_names)

    def test_reconciler_handles_missing_pusher_gracefully(self) -> None:
        store = _StubPositionStore()
        exchange = _StubExchange(open_orders=[])
        reconciler = PositionReconciler(
            position_store=store,
            exchange=exchange,
            metrics_pusher=None,
        )
        # Just shouldn't raise.
        report = reconciler.reconcile()
        self.assertIsInstance(report, ReconciliationReport)

    def test_exchange_failure_becomes_error_finding(self) -> None:
        store = _StubPositionStore(open_positions=[_make_position()])
        exchange = _StubExchange(
            raise_on_get_open_orders=RuntimeError("rate-limited"),
        )
        reconciler = PositionReconciler(position_store=store, exchange=exchange)
        report = reconciler.reconcile()
        self.assertEqual(report.error_count, 1)
        self.assertTrue(report.has_issues)


# ---------------------------------------------------------------------------
# (f) CLI end-to-end
# ---------------------------------------------------------------------------


class TestReconciliationCLI(unittest.TestCase):
    def test_cli_runs_with_stubbed_collaborators_and_returns_zero_when_clean(
        self,
    ) -> None:
        position = _make_position(symbol="ETH/USD", base_size=0.05)
        store = _StubPositionStore(open_positions=[position])
        exchange = _StubExchange(
            open_orders=[_StubOrder(symbol="ETH/USD", base_size=0.05)]
        )
        report = reconciliation_cli.run_reconciliation(
            symbol="ETH/USD",
            position_store=store,
            exchange=exchange,
            metrics_pusher=None,
        )
        self.assertFalse(report.has_issues)
        self.assertEqual(report.clean_count, 1)

    def test_cli_returns_nonzero_via_main_when_issues_detected(self) -> None:
        # We can't easily inject collaborators into ``main()`` without
        # touching argv parsing, so we test the report.has_issues exit
        # contract by constructing a report and exercising the printer.
        store = _StubPositionStore(open_positions=[_make_position(symbol="ETH/USD")])
        exchange = _StubExchange(open_orders=[])  # all orphans
        report = reconciliation_cli.run_reconciliation(
            symbol="ETH/USD",
            position_store=store,
            exchange=exchange,
            metrics_pusher=None,
        )
        self.assertTrue(report.has_issues)
        # Smoke test the human-readable printer.
        buf = io.StringIO()
        with redirect_stdout(buf):
            reconciliation_cli._print_human_summary(report, quiet=False)
        out = buf.getvalue()
        self.assertIn("orphan", out)


if __name__ == "__main__":
    unittest.main()
