"""CLI to sync ``trade_execution_*.json`` + ``performance_audit.json`` into SQLite.

The JSON files remain canonical. This script builds / updates a secondary
SQLite mirror that the FastAPI ``/trades`` and ``/postmortems`` endpoints
(and ``calibration_agent.build_dataset``) can opt into reading from.

Usage::

    python src/storage/sync.py [--trade-store-dir DIR] [--audit-path PATH] [--db-path PATH]

Defaults:
    --trade-store-dir : ``$AUTOPILOT_TRADE_STORE`` env var, else the repo root.
    --audit-path      : ``<trade-store-dir>/performance_audit.json``.
    --db-path         : ``$AUTOPILOT_SQLITE_PATH`` env var, else ``<trade-store-dir>/autopilot.sqlite``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# sys.path shim mirrors build_dataset.py so this CLI runs without PYTHONPATH set.
_SRC_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))

from storage.sqlite_store import (  # noqa: E402  (after sys.path shim)
    SQLITE_PATH_ENV_VAR,
    SQLiteStore,
    sync_audit_to_sqlite,
    sync_trade_logs_to_sqlite,
)

LOGGER = logging.getLogger(__name__)
TRADE_STORE_ENV_VAR = "AUTOPILOT_TRADE_STORE"


def _default_trade_store_dir() -> Path:
    raw = (os.environ.get(TRADE_STORE_ENV_VAR) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _REPO_ROOT


def _default_db_path(trade_store_dir: Path) -> Path:
    raw = (os.environ.get(SQLITE_PATH_ENV_VAR) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return trade_store_dir / "autopilot.sqlite"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync trade_execution_<id>.json files and performance_audit.json into "
            "an additive SQLite mirror. JSON files remain canonical."
        )
    )
    parser.add_argument(
        "--trade-store-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing trade_execution_*.json files. "
            f"Defaults to ${TRADE_STORE_ENV_VAR} or the repo root."
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help="Path to performance_audit.json. Defaults to <trade-store-dir>/performance_audit.json.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help=(
            "Output SQLite db path. "
            f"Defaults to ${SQLITE_PATH_ENV_VAR} or <trade-store-dir>/autopilot.sqlite."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip syncing performance_audit.json (only mirror trade logs).",
    )
    parser.add_argument(
        "--skip-trades",
        action="store_true",
        help="Skip syncing trade_execution_*.json (only mirror audit reviews).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    trade_store_dir = (args.trade_store_dir or _default_trade_store_dir()).expanduser().resolve()
    audit_path = (args.audit_path or trade_store_dir / "performance_audit.json").expanduser().resolve()
    db_path = (args.db_path or _default_db_path(trade_store_dir)).expanduser().resolve()

    print(f"Trade store dir: {trade_store_dir}", file=sys.stderr)
    print(f"Audit file:      {audit_path}", file=sys.stderr)
    print(f"SQLite db:       {db_path}", file=sys.stderr)

    store = SQLiteStore(db_path)
    try:
        if not args.skip_trades:
            trade_result = sync_trade_logs_to_sqlite(trade_store_dir, store=store)
            print(
                f"Trades synced: {trade_result['synced']} "
                f"(errors={len(trade_result['errors'])})",
                file=sys.stderr,
            )
            for error in trade_result["errors"]:
                print(f"  trade error: {error}", file=sys.stderr)

        if not args.skip_audit:
            if audit_path.is_file():
                audit_result = sync_audit_to_sqlite(audit_path, store=store)
                print(
                    f"Reviews synced: {audit_result['synced']} "
                    f"(errors={len(audit_result['errors'])})",
                    file=sys.stderr,
                )
                for error in audit_result["errors"]:
                    print(f"  review error: {error}", file=sys.stderr)
            else:
                print(f"Audit file not found, skipping: {audit_path}", file=sys.stderr)
    finally:
        store.close()

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
