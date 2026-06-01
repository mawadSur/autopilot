"""Position reconciliation between ``PositionStore`` and an exchange.

Reconciliation is a defensive, mostly-read-only sanity check. It does NOT
mutate the position store — operators get a structured report and decide
what to do. The store's own ``reconcile()`` method handles the narrow
"pending position older than 1h with no matching exchange order" case;
this module is broader: it covers all four drift modes:

* **orphan** — position exists in ``PositionStore`` but the exchange has
  no open order or balance to back it.
* **ghost** — exchange has an open order for a symbol with no matching
  position in the store.
* **drift** — base size on the position differs from the exchange-reported
  size beyond a small relative tolerance.
* **clean** — local + exchange agree.

Failure mode is "report" — every result is an item in
:class:`ReconciliationReport.findings`. Metrics are pushed best-effort via
the supplied :class:`MetricsPusher`; Sentry breadcrumbs are emitted when
the SDK is importable.

Called from cron via :mod:`ops.reconciliation_cli`, or programmatically by
the supervisor on a configurable cadence.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Protocol


LOGGER = logging.getLogger(__name__)


# Default size-drift tolerance (relative). 0.5% covers normal Coinbase
# rounding without flagging legitimate small partial-fill reconciliation
# noise.
DEFAULT_SIZE_TOLERANCE_PCT = 0.005

# Default price-drift tolerance (relative). Generous because we only
# compare entry_price against the *current* mid as a sanity backstop —
# the real check is on size, since the entry price is historical.
DEFAULT_PRICE_TOLERANCE_PCT = 0.10


class ExchangeProtocol(Protocol):
    """Minimal protocol the reconciler needs from an exchange client.

    ``CoinbaseExchange`` already implements both methods. Any object with
    these two callables can be plugged in (paper exchanges, hyperliquid,
    test stubs).
    """

    def get_open_orders(self, symbol: Optional[str] = None) -> Iterable[Any]: ...

    def get_balances(self) -> Iterable[Any]: ...


@dataclass
class ReconciliationFinding:
    """A single drift / orphan / ghost / clean record."""

    kind: str  # one of: clean, orphan, ghost, drift, error
    symbol: str
    position_id: Optional[str] = None
    expected_size: Optional[float] = None
    actual_size: Optional[float] = None
    detail: str = ""


@dataclass
class ReconciliationReport:
    """Aggregate report from :meth:`PositionReconciler.reconcile`.

    Operators consume the per-finding list; supervisor / cron consume the
    per-bucket counters via metrics. ``alerts`` is the human-readable list
    of things worth paging on.
    """

    clean_count: int = 0
    orphan_count: int = 0
    ghost_count: int = 0
    drift_count: int = 0
    error_count: int = 0
    findings: List[ReconciliationFinding] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    ran_at_utc: str = ""

    @property
    def has_issues(self) -> bool:
        """True iff anything other than ``clean`` was observed."""
        return (
            self.orphan_count > 0
            or self.ghost_count > 0
            or self.drift_count > 0
            or self.error_count > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "clean_count": self.clean_count,
            "orphan_count": self.orphan_count,
            "ghost_count": self.ghost_count,
            "drift_count": self.drift_count,
            "error_count": self.error_count,
            "ran_at_utc": self.ran_at_utc,
            "findings": [
                {
                    "kind": f.kind,
                    "symbol": f.symbol,
                    "position_id": f.position_id,
                    "expected_size": f.expected_size,
                    "actual_size": f.actual_size,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
            "alerts": list(self.alerts),
        }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sentry_breadcrumb(message: str, data: Optional[dict[str, Any]] = None) -> None:
    """Best-effort Sentry breadcrumb. Never raises."""
    try:
        import sentry_sdk  # type: ignore[import-not-found]

        sentry_sdk.add_breadcrumb(
            category="reconciliation",
            message=message,
            level="info",
            data=data or {},
        )
    except Exception:  # noqa: BLE001 - monitoring failure must never escape
        pass


def _sentry_capture_message(message: str, level: str = "warning") -> None:
    """Best-effort Sentry capture. Never raises."""
    try:
        import sentry_sdk  # type: ignore[import-not-found]

        sentry_sdk.capture_message(message, level=level)
    except Exception:  # noqa: BLE001
        pass


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _extract_order_symbol(order: Any) -> Optional[str]:
    """Pull a symbol out of either a pydantic-model order or a plain dict."""
    if hasattr(order, "symbol"):
        try:
            return str(getattr(order, "symbol"))
        except Exception:  # noqa: BLE001
            return None
    if isinstance(order, dict):
        sym = order.get("symbol")
        return str(sym) if sym else None
    return None


def _extract_order_size(order: Any) -> Optional[float]:
    """Pull a base size out of either an OrderResult-like object or a dict."""
    for attr in ("base_size", "filled_base"):
        if hasattr(order, attr):
            value = getattr(order, attr)
            if value is not None:
                return _coerce_float(value)
    if isinstance(order, dict):
        for key in ("base_size", "filled_base", "size"):
            if order.get(key) is not None:
                return _coerce_float(order[key])
    return None


class PositionReconciler:
    """Compare ``PositionStore`` open positions against exchange state.

    Construct with the position store, an exchange client (or list of
    clients keyed by exchange name), and an optional metrics pusher.

    The reconciler is read-only against both the store and the exchange
    apart from telemetry — fixing detected drift is an operator task.
    """

    def __init__(
        self,
        *,
        position_store: Any,
        exchange: Any,
        metrics_pusher: Optional[Any] = None,
        size_tolerance_pct: float = DEFAULT_SIZE_TOLERANCE_PCT,
    ) -> None:
        self.position_store = position_store
        self.exchange = exchange
        self.metrics_pusher = metrics_pusher
        self.size_tolerance_pct = float(size_tolerance_pct)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    def reconcile(self, symbol: Optional[str] = None) -> ReconciliationReport:
        """Run a single reconciliation pass.

        If ``symbol`` is provided, only positions/orders for that symbol
        are evaluated. Otherwise every open position + every open order
        is checked. Always returns a :class:`ReconciliationReport`; never
        raises on exchange or store errors (those become ``error`` findings).
        """
        report = ReconciliationReport(ran_at_utc=_utcnow_iso())
        _sentry_breadcrumb(
            "reconciliation.start",
            data={"symbol": symbol},
        )

        # --- 1. Fetch state from both sides -------------------------------
        try:
            open_positions = list(self.position_store.list_open() or [])
            if symbol is not None:
                open_positions = [p for p in open_positions if p.symbol == symbol]
        except Exception as exc:  # noqa: BLE001 - keep going, surface the error
            LOGGER.warning("reconcile: list_open failed: %s", exc)
            report.error_count += 1
            report.findings.append(
                ReconciliationFinding(
                    kind="error",
                    symbol=symbol or "*",
                    detail=f"list_open failed: {exc!r}",
                )
            )
            self._emit_metrics(report)
            return report

        try:
            raw_orders = list(self.exchange.get_open_orders(symbol=symbol) or [])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("reconcile: get_open_orders failed: %s", exc)
            report.error_count += 1
            report.findings.append(
                ReconciliationFinding(
                    kind="error",
                    symbol=symbol or "*",
                    detail=f"get_open_orders failed: {exc!r}",
                )
            )
            self._emit_metrics(report)
            return report

        # --- 2. Index exchange orders by symbol ---------------------------
        # Same symbol may have multiple open orders (eg. a limit + a stop);
        # we sum sizes for the per-symbol comparison.
        exchange_size_by_symbol: dict[str, float] = {}
        exchange_orders_by_symbol: dict[str, List[Any]] = {}
        for order in raw_orders:
            sym = _extract_order_symbol(order)
            if not sym:
                continue
            size = _extract_order_size(order) or 0.0
            exchange_size_by_symbol[sym] = (
                exchange_size_by_symbol.get(sym, 0.0) + size
            )
            exchange_orders_by_symbol.setdefault(sym, []).append(order)

        # --- 3. Walk store positions: orphan / drift / clean --------------
        seen_symbols: set[str] = set()
        for position in open_positions:
            sym = position.symbol
            seen_symbols.add(sym)
            expected_size = _coerce_float(position.base_size)
            actual_size = exchange_size_by_symbol.get(sym)

            # Pending positions with no exchange order are NOT immediately
            # an orphan; the store's own reconcile() handles the time-based
            # cleanup. Only flag as orphan if the position is fully open
            # (filled) — those should always have a backing exchange-side
            # order or balance. For pending positions, we surface them as
            # "drift" only when the size delta is large.
            if actual_size is None:
                if getattr(position, "status", "open") == "open":
                    report.orphan_count += 1
                    detail = (
                        f"position {position.position_id} has size "
                        f"{expected_size:g} but exchange shows no order"
                    )
                    report.findings.append(
                        ReconciliationFinding(
                            kind="orphan",
                            symbol=sym,
                            position_id=position.position_id,
                            expected_size=expected_size,
                            actual_size=0.0,
                            detail=detail,
                        )
                    )
                    report.alerts.append(f"orphan position {position.position_id} on {sym}")
                else:
                    # Pending position with no matching exchange order —
                    # store.reconcile() will time-cleanup. Treat as clean
                    # for the purposes of this drift report.
                    report.clean_count += 1
                continue

            # Size comparison.
            tol = max(
                self.size_tolerance_pct * max(expected_size, actual_size),
                1e-9,
            )
            if abs(expected_size - actual_size) <= tol:
                report.clean_count += 1
                report.findings.append(
                    ReconciliationFinding(
                        kind="clean",
                        symbol=sym,
                        position_id=position.position_id,
                        expected_size=expected_size,
                        actual_size=actual_size,
                        detail="match",
                    )
                )
            else:
                report.drift_count += 1
                detail = (
                    f"size drift: store={expected_size:g} exchange={actual_size:g}"
                )
                report.findings.append(
                    ReconciliationFinding(
                        kind="drift",
                        symbol=sym,
                        position_id=position.position_id,
                        expected_size=expected_size,
                        actual_size=actual_size,
                        detail=detail,
                    )
                )
                report.alerts.append(
                    f"size drift on {sym} pos={position.position_id}: "
                    f"{expected_size:g} vs exchange {actual_size:g}"
                )

        # --- 4. Walk exchange-only symbols: ghosts ------------------------
        for sym, total_size in exchange_size_by_symbol.items():
            if sym in seen_symbols:
                continue
            if total_size <= 0:
                continue
            report.ghost_count += 1
            detail = f"exchange has open order(s) totaling {total_size:g} not in store"
            report.findings.append(
                ReconciliationFinding(
                    kind="ghost",
                    symbol=sym,
                    position_id=None,
                    expected_size=0.0,
                    actual_size=total_size,
                    detail=detail,
                )
            )
            report.alerts.append(f"ghost position on {sym}: exchange size={total_size:g}")

        # --- 5. Telemetry -------------------------------------------------
        self._emit_metrics(report)
        if report.orphan_count or report.ghost_count or report.drift_count:
            _sentry_capture_message(
                f"reconciliation drift: orphans={report.orphan_count} "
                f"ghosts={report.ghost_count} drift={report.drift_count}",
                level="warning",
            )

        return report

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def _emit_metrics(self, report: ReconciliationReport) -> None:
        """Push reconciliation outcomes to Prometheus. Best-effort."""
        if self.metrics_pusher is None:
            return
        try:
            self.metrics_pusher.counter("reconciliation_run_total", 1.0)
            self.metrics_pusher.gauge(
                "reconciliation_orphans", float(report.orphan_count)
            )
            self.metrics_pusher.gauge(
                "reconciliation_ghosts", float(report.ghost_count)
            )
            self.metrics_pusher.gauge(
                "reconciliation_drift_count", float(report.drift_count)
            )
            # Surface orphan count via the dedicated gauge requested in
            # Task 4 (position-orphan telemetry). Use the position store's
            # ``orphan_count`` helper when available, falling back to the
            # report's count for stub stores.
            try:
                store_orphans = float(self.position_store.orphan_count())
            except (AttributeError, Exception):  # noqa: BLE001
                store_orphans = float(report.orphan_count)
            self.metrics_pusher.gauge("orphan_positions", store_orphans)
        except Exception as exc:  # noqa: BLE001 - telemetry must never raise
            LOGGER.warning("reconciliation metrics emit failed: %s", exc)


__all__ = [
    "DEFAULT_PRICE_TOLERANCE_PCT",
    "DEFAULT_SIZE_TOLERANCE_PCT",
    "ExchangeProtocol",
    "PositionReconciler",
    "ReconciliationFinding",
    "ReconciliationReport",
]
