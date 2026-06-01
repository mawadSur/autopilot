"""Additive SQLite storage layer for the prediction-market pipeline.

JSON files (``trade_execution_<id>.json`` + ``performance_audit.json``) remain
the canonical write path. This package provides a *secondary* SQLite index
that is built from those JSON files for fast querying.

Toggled by the ``AUTOPILOT_SQLITE_PATH`` environment variable. When unset,
:func:`get_default_store` returns ``None`` and existing call sites continue
to read from JSON.

Public API::

    from storage import (
        SQLiteStore,
        sync_trade_logs_to_sqlite,
        sync_audit_to_sqlite,
        is_sqlite_enabled,
        get_default_store,
    )
"""

from __future__ import annotations

from .sqlite_store import (
    SQLITE_PATH_ENV_VAR,
    SQLiteStore,
    get_default_store,
    is_sqlite_enabled,
    reset_default_store,
    sync_audit_to_sqlite,
    sync_trade_logs_to_sqlite,
)

__all__ = [
    "SQLITE_PATH_ENV_VAR",
    "SQLiteStore",
    "get_default_store",
    "is_sqlite_enabled",
    "reset_default_store",
    "sync_audit_to_sqlite",
    "sync_trade_logs_to_sqlite",
]
