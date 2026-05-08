"""CLI entry point for position reconciliation.

Run from cron or on demand:

    python -m ops.reconciliation_cli --symbol ETH/USD
    python -m ops.reconciliation_cli           # reconcile every symbol

Exit codes:
    0  — clean (no orphans / ghosts / drift / errors)
    1  — issues detected (drift / orphan / ghost / error in any bucket)
    2  — fatal: could not construct position store or exchange client

Defaults are read from the environment so this works under cron without
extra args. Construct your own client + store and call
:func:`run_reconciliation` directly for embedded use cases.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

# Mirror the sys.path shim used by main.py / orchestrator.py / live_supervisor.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ops.reconciliation import PositionReconciler, ReconciliationReport


LOGGER = logging.getLogger("ops.reconciliation_cli")


def _build_default_position_store() -> Any:
    """Lazy-build the production ``PositionStore`` keyed off ``REDIS_URL``."""
    from state.position_store import get_default_store

    return get_default_store()


def _build_default_exchange() -> Any:
    """Lazy-build a sandbox ``CoinbaseExchange`` for read-only reconciliation.

    Reconciliation only calls ``get_open_orders`` + ``get_balances``, so even
    a sandbox client gives a reasonable answer for non-production runs.
    """
    from exchanges.coinbase import CoinbaseExchange

    return CoinbaseExchange()


def _build_default_metrics_pusher() -> Optional[Any]:
    """Construct a :class:`MetricsPusher` if PROMETHEUS_PUSH_URL is set."""
    try:
        from observability.monitoring import MetricsPusher

        pusher = MetricsPusher()
        return pusher if pusher.is_enabled() else None
    except Exception as exc:  # noqa: BLE001 - never crash on telemetry init
        LOGGER.warning("MetricsPusher init failed: %s", exc)
        return None


def run_reconciliation(
    *,
    symbol: Optional[str] = None,
    position_store: Any = None,
    exchange: Any = None,
    metrics_pusher: Any = None,
) -> ReconciliationReport:
    """Run reconciliation against the supplied (or default) collaborators."""
    store = position_store if position_store is not None else _build_default_position_store()
    exch = exchange if exchange is not None else _build_default_exchange()
    pusher = metrics_pusher if metrics_pusher is not None else _build_default_metrics_pusher()

    reconciler = PositionReconciler(
        position_store=store,
        exchange=exch,
        metrics_pusher=pusher,
    )
    return reconciler.reconcile(symbol=symbol)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile PositionStore against the exchange.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Reconcile a single symbol (e.g. ETH/USD). Defaults to all.",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit the report as JSON on stdout instead of a human summary.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-finding output; print the counts only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        report = run_reconciliation(symbol=args.symbol)
    except Exception as exc:  # noqa: BLE001 - CLI must surface a clean exit code
        LOGGER.exception("Reconciliation failed to start: %s", exc)
        return 2

    if args.emit_json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        _print_human_summary(report, quiet=args.quiet)

    return 1 if report.has_issues else 0


def _print_human_summary(report: ReconciliationReport, *, quiet: bool) -> None:
    print(
        f"reconciliation @ {report.ran_at_utc} | "
        f"clean={report.clean_count} orphan={report.orphan_count} "
        f"ghost={report.ghost_count} drift={report.drift_count} "
        f"error={report.error_count}"
    )
    if quiet:
        return
    for finding in report.findings:
        if finding.kind == "clean":
            continue
        print(
            f"  [{finding.kind:6s}] {finding.symbol} "
            f"pos={finding.position_id or '-'} "
            f"expected={finding.expected_size} actual={finding.actual_size} "
            f"-- {finding.detail}"
        )
    for alert in report.alerts:
        print(f"  ALERT: {alert}")


if __name__ == "__main__":
    sys.exit(main())
