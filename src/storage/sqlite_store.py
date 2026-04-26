"""Additive SQLite mirror of trade execution logs + performance audit reviews.

The JSON files written by ``orchestrator._write_trade_execution_log`` and
``outcome_review_agent.logger.PerformanceTracker`` remain the canonical
storage. This module provides a *secondary* SQLite index that is built from
those JSON files for fast querying.

Schema lives in code (``CREATE TABLE IF NOT EXISTS`` at module load time);
no Alembic, no migrations directory. Schema versioning is recorded in the
``schema_meta`` table — bump ``SCHEMA_VERSION`` here if you change a column.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

SQLITE_PATH_ENV_VAR = "AUTOPILOT_SQLITE_PATH"
SCHEMA_VERSION = "1"

# ``CREATE TABLE IF NOT EXISTS`` keeps reopens safe; bump SCHEMA_VERSION above
# when any column changes so callers can detect incompatible mirrors.
SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        status TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'orchestrator',
        created_at_utc TEXT,
        settled_at TEXT,
        final_outcome INTEGER,
        market_outcome INTEGER,
        entry_price REAL,
        exit_price REAL,
        position_size_usd REAL,
        realized_pnl_usd REAL,
        max_loss_usd REAL,
        market_title TEXT,
        market_category TEXT,
        implied_prob REAL,
        volume_24h REAL,
        features_window_json TEXT,
        research_json TEXT,
        calibration_json TEXT,
        risk_json TEXT,
        notes TEXT,
        source_file TEXT,
        last_synced_at_utc TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
    "CREATE INDEX IF NOT EXISTS idx_trades_source ON trades(source)",
    "CREATE INDEX IF NOT EXISTS idx_trades_settled_at ON trades(settled_at)",
    """
    CREATE TABLE IF NOT EXISTS reviews (
        trade_key TEXT PRIMARY KEY,
        trade_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        settled_at TEXT,
        reviewed_at TEXT,
        final_outcome INTEGER,
        matrix_classification TEXT,
        outcome_review_json TEXT NOT NULL,
        additional_reviews_json TEXT,
        last_synced_at_utc TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reviews_matrix ON reviews(matrix_classification)",
    "CREATE INDEX IF NOT EXISTS idx_reviews_settled_at ON reviews(settled_at)",
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)


# Reserved keys that PerformanceTracker writes alongside agent reviews — these
# are normalized into dedicated columns so they don't get bucketed into the
# ``additional_reviews_json`` blob.
_RESERVED_AUDIT_KEYS = frozenset(
    {
        "trade_id",
        "source_file",
        "trade_key",
        "settled_at",
        "reviewed_at",
        "outcome_review",
        "final_outcome",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_optional_int(value: Any) -> Optional[int]:
    """Tri-state bool/None → 0/1/None for SQLite INTEGER columns."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_blob(value: Any) -> Optional[str]:
    """Serialize a structured field for storage. Returns None for missing data."""

    if value is None:
        return None
    if isinstance(value, str):
        # Pre-serialized payloads pass through unchanged.
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def _decode_blob(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


class SQLiteStore:
    """Thin wrapper around a sqlite3 connection.

    The connection is opened with ``check_same_thread=False`` so a FastAPI
    worker can share a single instance across requests. All writes are wrapped
    in ``with self._connection:`` blocks for implicit transactions.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets FastAPI workers share the singleton;
        # ``_lock`` serializes writes so concurrent upserts don't interleave.
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._initialize_schema()

    # ------------------------------------------------------------------
    # schema bootstrap
    # ------------------------------------------------------------------
    def _initialize_schema(self) -> None:
        with self._lock, self._connection:
            cursor = self._connection.cursor()
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def upsert_trade(self, payload: Dict[str, Any], *, source_file: Path | str) -> None:
        """Insert or update a single trade row from an event_payload dict."""

        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        trade_id = _coerce_optional_str(payload.get("trade_id") or payload.get("event_id"))
        if not trade_id:
            raise ValueError("event_payload must include 'trade_id' or 'event_id'")
        event_id = _coerce_optional_str(payload.get("event_id")) or trade_id
        status = _coerce_optional_str(payload.get("status")) or "open"
        source = _coerce_optional_str(payload.get("source")) or "orchestrator"

        scanner = payload.get("scanner") if isinstance(payload.get("scanner"), dict) else {}
        market_title = _coerce_optional_str(
            (scanner.get("title") if isinstance(scanner, dict) else None)
            or payload.get("market_title")
        )
        market_category = _coerce_optional_str(
            (scanner.get("category") if isinstance(scanner, dict) else None)
            or payload.get("market_category")
        )
        implied_prob = _coerce_optional_float(
            (scanner.get("implied_prob") if isinstance(scanner, dict) else None)
        )
        volume_24h = _coerce_optional_float(
            (scanner.get("volume_24h") if isinstance(scanner, dict) else None)
        )

        row = {
            "trade_id": trade_id,
            "event_id": event_id,
            "status": status,
            "source": source,
            "created_at_utc": _coerce_optional_str(payload.get("created_at_utc")),
            "settled_at": _coerce_optional_str(payload.get("settled_at")),
            "final_outcome": _coerce_optional_int(payload.get("final_outcome")),
            "market_outcome": _coerce_optional_int(payload.get("market_outcome")),
            "entry_price": _coerce_optional_float(payload.get("entry_price")),
            "exit_price": _coerce_optional_float(payload.get("exit_price")),
            "position_size_usd": _coerce_optional_float(payload.get("position_size_usd")),
            "realized_pnl_usd": _coerce_optional_float(payload.get("realized_pnl_usd")),
            "max_loss_usd": _coerce_optional_float(payload.get("max_loss_usd")),
            "market_title": market_title,
            "market_category": market_category,
            "implied_prob": implied_prob,
            "volume_24h": volume_24h,
            "features_window_json": _json_blob(payload.get("features_window")),
            "research_json": _json_blob(payload.get("research")),
            "calibration_json": _json_blob(payload.get("calibration")),
            "risk_json": _json_blob(payload.get("risk")),
            "notes": _coerce_optional_str(payload.get("notes")),
            "source_file": str(Path(source_file).resolve()) if source_file else None,
            "last_synced_at_utc": _utc_now_iso(),
        }

        columns = list(row.keys())
        placeholders = ", ".join(f":{name}" for name in columns)
        update_assignments = ", ".join(
            f"{name} = excluded.{name}" for name in columns if name != "trade_id"
        )
        sql = (
            f"INSERT INTO trades ({', '.join(columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(trade_id) DO UPDATE SET {update_assignments}"
        )
        with self._lock, self._connection:
            self._connection.execute(sql, row)

    def upsert_review(self, audit_entry: Dict[str, Any]) -> None:
        """Insert or update a review row from a PerformanceTracker audit entry."""

        if not isinstance(audit_entry, dict):
            raise TypeError("audit_entry must be a dict")

        trade_id = _coerce_optional_str(audit_entry.get("trade_id"))
        source_file = _coerce_optional_str(audit_entry.get("source_file"))
        if not trade_id or not source_file:
            raise ValueError(
                "audit_entry must include 'trade_id' and 'source_file' (PerformanceTracker shape)"
            )
        # Match PerformanceTracker._trade_key — defaults to "<source_file>:<trade_id>"
        # but honor the precomputed key when present so we round-trip identically.
        trade_key = _coerce_optional_str(audit_entry.get("trade_key")) or f"{source_file}:{trade_id}"

        outcome_review = audit_entry.get("outcome_review") or {}
        if not isinstance(outcome_review, dict):
            outcome_review = {"review": outcome_review}
        matrix_classification = _coerce_optional_str(outcome_review.get("matrix_classification"))

        additional = {
            key: value
            for key, value in audit_entry.items()
            if key not in _RESERVED_AUDIT_KEYS
        }
        additional_blob = _json_blob(additional) if additional else None

        row = {
            "trade_key": trade_key,
            "trade_id": trade_id,
            "source_file": source_file,
            "settled_at": _coerce_optional_str(audit_entry.get("settled_at")),
            "reviewed_at": _coerce_optional_str(audit_entry.get("reviewed_at")),
            "final_outcome": _coerce_optional_int(audit_entry.get("final_outcome")),
            "matrix_classification": matrix_classification,
            "outcome_review_json": _json_blob(outcome_review) or "{}",
            "additional_reviews_json": additional_blob,
            "last_synced_at_utc": _utc_now_iso(),
        }

        columns = list(row.keys())
        placeholders = ", ".join(f":{name}" for name in columns)
        update_assignments = ", ".join(
            f"{name} = excluded.{name}" for name in columns if name != "trade_key"
        )
        sql = (
            f"INSERT INTO reviews ({', '.join(columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(trade_key) DO UPDATE SET {update_assignments}"
        )
        with self._lock, self._connection:
            self._connection.execute(sql, row)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def list_trades(
        self,
        *,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        if status is not None:
            clauses.append("status = :status")
            params["status"] = status
        if source is not None:
            clauses.append("source = :source")
            params["source"] = source
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params["limit"] = max(1, int(limit))
        sql = (
            "SELECT * FROM trades "
            f"{where} "
            "ORDER BY COALESCE(settled_at, created_at_utc, last_synced_at_utc) DESC, trade_id ASC "
            "LIMIT :limit"
        )
        with self._lock:
            cursor = self._connection.execute(sql, params)
            rows = cursor.fetchall()
        return [self._row_to_trade(row) for row in rows]

    def list_reviews(
        self,
        *,
        matrix_classification: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        if matrix_classification is not None:
            clauses.append("matrix_classification = :matrix_classification")
            params["matrix_classification"] = matrix_classification
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params["limit"] = max(1, int(limit))
        sql = (
            "SELECT * FROM reviews "
            f"{where} "
            "ORDER BY COALESCE(reviewed_at, settled_at, last_synced_at_utc) DESC, trade_key ASC "
            "LIMIT :limit"
        )
        with self._lock:
            cursor = self._connection.execute(sql, params)
            rows = cursor.fetchall()
        return [self._row_to_review(row) for row in rows]

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            )
            row = cursor.fetchone()
        return self._row_to_trade(row) if row is not None else None

    def get_review(self, trade_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM reviews WHERE trade_key = ?", (trade_key,)
            )
            row = cursor.fetchone()
        return self._row_to_review(row) if row is not None else None

    def schema_version(self) -> Optional[str]:
        with self._lock:
            cursor = self._connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
        return row["value"] if row is not None else None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            try:
                self._connection.close()
            except sqlite3.Error:  # pragma: no cover - best-effort
                LOGGER.exception("Failed to close SQLite connection at %s", self.db_path)

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # row → dict helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        for blob_field in (
            "features_window_json",
            "research_json",
            "calibration_json",
            "risk_json",
        ):
            decoded_key = blob_field[: -len("_json")]  # features_window_json → features_window
            data[decoded_key] = _decode_blob(data.pop(blob_field))
        return data

    @staticmethod
    def _row_to_review(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["outcome_review"] = _decode_blob(data.pop("outcome_review_json")) or {}
        data["additional_reviews"] = _decode_blob(data.pop("additional_reviews_json")) or {}
        return data


# ----------------------------------------------------------------------
# module-level helpers
# ----------------------------------------------------------------------
_DEFAULT_STORE: Optional[SQLiteStore] = None
_DEFAULT_STORE_PATH: Optional[Path] = None
_DEFAULT_STORE_LOCK = threading.Lock()


def is_sqlite_enabled() -> bool:
    """True when ``AUTOPILOT_SQLITE_PATH`` is set to a non-empty value."""

    return bool((os.environ.get(SQLITE_PATH_ENV_VAR) or "").strip())


def _resolve_default_path() -> Optional[Path]:
    raw = (os.environ.get(SQLITE_PATH_ENV_VAR) or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def get_default_store() -> Optional[SQLiteStore]:
    """Return a process-wide singleton ``SQLiteStore`` when enabled.

    When the env var is unset, returns ``None`` so callers can fall back to
    the JSON path. Cached per resolved path; if the env var changes between
    calls the singleton is rebuilt against the new location.
    """

    global _DEFAULT_STORE, _DEFAULT_STORE_PATH

    target = _resolve_default_path()
    if target is None:
        return None

    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is not None and _DEFAULT_STORE_PATH == target:
            return _DEFAULT_STORE
        if _DEFAULT_STORE is not None:
            try:
                _DEFAULT_STORE.close()
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("Could not close stale default SQLite store")
        _DEFAULT_STORE = SQLiteStore(target)
        _DEFAULT_STORE_PATH = target
        return _DEFAULT_STORE


def reset_default_store() -> None:
    """Close & forget the cached default store. Used by tests for isolation."""

    global _DEFAULT_STORE, _DEFAULT_STORE_PATH

    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is not None:
            try:
                _DEFAULT_STORE.close()
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("Could not close default SQLite store during reset")
        _DEFAULT_STORE = None
        _DEFAULT_STORE_PATH = None


def sync_trade_logs_to_sqlite(
    trade_store_dir: Path | str,
    *,
    store: Optional[SQLiteStore] = None,
    glob_pattern: str = "trade_execution_*.json",
) -> Dict[str, Any]:
    """Walk ``trade_execution_*.json`` files and upsert each into SQLite.

    Returns ``{"synced": N, "errors": [...], "store_path": <str>}``. The store
    parameter falls back to :func:`get_default_store`; raises ``RuntimeError``
    when no store is wired in (caller must opt in via env var or argument).
    """

    target_store = store or get_default_store()
    if target_store is None:
        raise RuntimeError(
            f"SQLite store is not enabled; set {SQLITE_PATH_ENV_VAR} or pass store=..."
        )

    directory = Path(trade_store_dir)
    errors: List[Dict[str, str]] = []
    synced = 0

    if not directory.is_dir():
        return {"synced": 0, "errors": [{"path": str(directory), "error": "not a directory"}], "store_path": str(target_store.db_path)}

    for file_path in sorted(directory.glob(glob_pattern)):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(file_path), "error": f"read failed: {exc}"})
            continue
        if not isinstance(payload, dict):
            errors.append({"path": str(file_path), "error": "non-object payload"})
            continue
        try:
            target_store.upsert_trade(payload, source_file=file_path)
            synced += 1
        except (ValueError, TypeError, sqlite3.Error) as exc:
            errors.append({"path": str(file_path), "error": f"upsert failed: {exc}"})

    return {"synced": synced, "errors": errors, "store_path": str(target_store.db_path)}


def sync_audit_to_sqlite(
    audit_path: Path | str,
    *,
    store: Optional[SQLiteStore] = None,
) -> Dict[str, Any]:
    """Read ``performance_audit.json`` and upsert each ``reviews[]`` entry."""

    target_store = store or get_default_store()
    if target_store is None:
        raise RuntimeError(
            f"SQLite store is not enabled; set {SQLITE_PATH_ENV_VAR} or pass store=..."
        )

    path = Path(audit_path)
    if not path.is_file():
        return {"synced": 0, "errors": [{"path": str(path), "error": "audit file missing"}], "store_path": str(target_store.db_path)}

    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"synced": 0, "errors": [{"path": str(path), "error": f"read failed: {exc}"}], "store_path": str(target_store.db_path)}

    reviews = audit.get("reviews") if isinstance(audit, dict) else None
    if not isinstance(reviews, list):
        return {"synced": 0, "errors": [{"path": str(path), "error": "no reviews[] array"}], "store_path": str(target_store.db_path)}

    errors: List[Dict[str, str]] = []
    synced = 0
    for index, entry in enumerate(reviews):
        if not isinstance(entry, dict):
            errors.append({"path": f"{path}[{index}]", "error": "non-object review"})
            continue
        try:
            target_store.upsert_review(entry)
            synced += 1
        except (ValueError, TypeError, sqlite3.Error) as exc:
            errors.append({"path": f"{path}[{index}]", "error": f"upsert failed: {exc}"})

    return {"synced": synced, "errors": errors, "store_path": str(target_store.db_path)}


__all__ = [
    "SCHEMA_STATEMENTS",
    "SCHEMA_VERSION",
    "SQLITE_PATH_ENV_VAR",
    "SQLiteStore",
    "get_default_store",
    "is_sqlite_enabled",
    "reset_default_store",
    "sync_audit_to_sqlite",
    "sync_trade_logs_to_sqlite",
]
