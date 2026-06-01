"""Unit tests for the additive SQLite storage layer.

These tests do not touch any other part of the system — they exercise
``SQLiteStore`` directly against an in-memory tmp database file and verify
the sync helpers correctly walk JSON inputs into the SQLite mirror.

The ``AUTOPILOT_SQLITE_PATH`` env var is left unset for most tests; the
two sync tests pass an explicit store rather than relying on the singleton
so we don't accidentally pollute other test cases.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from storage import (
    SQLITE_PATH_ENV_VAR,
    SQLiteStore,
    get_default_store,
    is_sqlite_enabled,
    reset_default_store,
    sync_audit_to_sqlite,
    sync_trade_logs_to_sqlite,
)


def _make_trade_payload(
    trade_id: str,
    *,
    status: str = "open",
    source: str = "orchestrator",
    final_outcome: Any = None,
    market_outcome: Any = None,
    title: str = "Sample market title",
    category: str = "Politics",
) -> Dict[str, Any]:
    return {
        "event_id": trade_id,
        "trade_id": trade_id,
        "status": status,
        "source": source,
        "created_at_utc": "2026-04-20T12:00:00+00:00",
        "settled_at": "2026-04-22T12:00:00+00:00" if status == "settled" else None,
        "final_outcome": final_outcome,
        "market_outcome": market_outcome,
        "scanner": {
            "market_id": trade_id,
            "title": title,
            "category": category,
            "implied_prob": 0.45,
            "volume_24h": 12_345.0,
        },
        "features_window": {"market_implied_prob": 0.45, "open_interest": 30_000.0},
        "research": {
            "reddit_query": "demo reddit",
            "news_query": "demo news",
        },
        "calibration": {
            "calibrated_true_prob": 0.55,
            "edge_vs_market": 0.10,
            "action": "paper-trade candidate",
        },
        "risk": {
            "risk_metrics": {"adjusted_position_size_pct": 1.2},
            "risk_assessment": {"allow_trade": True, "final_recommendation": "small"},
        },
        "entry_price": 0.46,
        "exit_price": None,
        "position_size_usd": 230.0,
        "realized_pnl_usd": None,
        "max_loss_usd": 230.0,
        "notes": None,
    }


def _make_audit_entry(
    trade_id: str,
    *,
    matrix_classification: str = "Deserved Success",
    final_outcome: bool = True,
    source_file: str = "/tmp/trade_execution_demo.json",
) -> Dict[str, Any]:
    return {
        "trade_id": trade_id,
        "source_file": source_file,
        "trade_key": f"{source_file}:{trade_id}",
        "settled_at": "2026-04-22T12:00:00+00:00",
        "reviewed_at": "2026-04-23T06:00:00+00:00",
        "outcome_review": {
            "matrix_classification": matrix_classification,
            "thesis_held": True,
            "good_process": True,
            "good_outcome": True,
        },
        "final_outcome": final_outcome,
        "data_quality_review": {"flags": []},
    }


class _StoreTestCase(unittest.TestCase):
    """Provides a fresh tmp-dir + ``SQLiteStore`` for every test method."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)
        self.db_path = self.tmp_dir / "autopilot.sqlite"
        self.store = SQLiteStore(self.db_path)
        self.addCleanup(self.store.close)


class SchemaInitTests(_StoreTestCase):
    def test_schema_created_on_init(self) -> None:
        # Open a parallel read-only connection to introspect sqlite_master so we
        # don't depend on internal state of ``SQLiteStore``.
        conn = sqlite3.connect(str(self.db_path))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
        finally:
            conn.close()

        self.assertIn("trades", tables)
        self.assertIn("reviews", tables)
        self.assertIn("schema_meta", tables)
        self.assertIn("idx_trades_status", indexes)
        self.assertIn("idx_trades_source", indexes)
        self.assertIn("idx_reviews_matrix", indexes)
        self.assertEqual(self.store.schema_version(), "1")


class UpsertTradeTests(_StoreTestCase):
    def test_upsert_trade_inserts_then_updates(self) -> None:
        payload = _make_trade_payload("mkt-1", status="open")
        source_file = self.tmp_dir / "trade_execution_mkt-1.json"
        self.store.upsert_trade(payload, source_file=source_file)

        rows = self.store.list_trades()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "open")
        self.assertEqual(rows[0]["source"], "orchestrator")
        self.assertIsNone(rows[0]["final_outcome"])
        self.assertEqual(rows[0]["market_title"], "Sample market title")

        # Upsert the same trade after settling — must update in place, not insert.
        settled_payload = _make_trade_payload(
            "mkt-1",
            status="settled",
            final_outcome=True,
            market_outcome=True,
        )
        settled_payload["exit_price"] = 0.78
        settled_payload["realized_pnl_usd"] = 120.0
        self.store.upsert_trade(settled_payload, source_file=source_file)

        updated_rows = self.store.list_trades()
        self.assertEqual(len(updated_rows), 1)
        self.assertEqual(updated_rows[0]["status"], "settled")
        self.assertEqual(updated_rows[0]["final_outcome"], 1)
        self.assertEqual(updated_rows[0]["market_outcome"], 1)
        self.assertAlmostEqual(updated_rows[0]["exit_price"], 0.78)
        self.assertAlmostEqual(updated_rows[0]["realized_pnl_usd"], 120.0)

        single = self.store.get_trade("mkt-1")
        self.assertIsNotNone(single)
        self.assertEqual(single["calibration"]["action"], "paper-trade candidate")


class UpsertReviewTests(_StoreTestCase):
    def test_upsert_review_inserts(self) -> None:
        entry = _make_audit_entry("mkt-1", matrix_classification="Deserved Success")
        self.store.upsert_review(entry)

        rows = self.store.list_reviews()
        self.assertEqual(len(rows), 1)
        review = rows[0]
        self.assertEqual(review["trade_id"], "mkt-1")
        self.assertEqual(review["matrix_classification"], "Deserved Success")
        self.assertEqual(review["final_outcome"], 1)
        self.assertEqual(review["outcome_review"]["thesis_held"], True)
        self.assertIn("data_quality_review", review["additional_reviews"])


class ListTradesFilterTests(_StoreTestCase):
    def _seed_mixed(self) -> None:
        for trade_id, status, source in [
            ("mkt-open", "open", "orchestrator"),
            ("mkt-settled", "settled", "orchestrator"),
            ("mkt-shadow", "open", "shadow"),
            ("mkt-backfill", "settled", "backfill"),
        ]:
            payload = _make_trade_payload(trade_id, status=status, source=source)
            self.store.upsert_trade(payload, source_file=self.tmp_dir / f"trade_execution_{trade_id}.json")

    def test_list_trades_filters_by_status(self) -> None:
        self._seed_mixed()
        settled = self.store.list_trades(status="settled")
        self.assertEqual({row["trade_id"] for row in settled}, {"mkt-settled", "mkt-backfill"})
        self.assertTrue(all(row["status"] == "settled" for row in settled))

    def test_list_trades_filters_by_source(self) -> None:
        self._seed_mixed()
        backfill = self.store.list_trades(source="backfill")
        self.assertEqual({row["trade_id"] for row in backfill}, {"mkt-backfill"})
        self.assertTrue(all(row["source"] == "backfill" for row in backfill))


class ListReviewsFilterTests(_StoreTestCase):
    def test_list_reviews_filters_by_matrix_classification(self) -> None:
        for trade_id, classification in [
            ("mkt-1", "Deserved Success"),
            ("mkt-2", "Good Failure"),
            ("mkt-3", "Dumb Luck"),
            ("mkt-4", "Good Failure"),
        ]:
            entry = _make_audit_entry(
                trade_id,
                matrix_classification=classification,
                source_file=f"/tmp/trade_execution_{trade_id}.json",
            )
            self.store.upsert_review(entry)

        good_failures = self.store.list_reviews(matrix_classification="Good Failure")
        self.assertEqual({row["trade_id"] for row in good_failures}, {"mkt-2", "mkt-4"})
        self.assertTrue(all(row["matrix_classification"] == "Good Failure" for row in good_failures))


class SyncTradeLogsTests(_StoreTestCase):
    def test_sync_trade_logs_to_sqlite_walks_directory(self) -> None:
        for trade_id, status in [("mkt-a", "open"), ("mkt-b", "settled"), ("mkt-c", "open")]:
            payload = _make_trade_payload(trade_id, status=status)
            (self.tmp_dir / f"trade_execution_{trade_id}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

        result = sync_trade_logs_to_sqlite(self.tmp_dir, store=self.store)
        self.assertEqual(result["synced"], 3)
        self.assertEqual(result["errors"], [])

        rows = self.store.list_trades(limit=10)
        self.assertEqual({row["trade_id"] for row in rows}, {"mkt-a", "mkt-b", "mkt-c"})


class SyncAuditTests(_StoreTestCase):
    def test_sync_audit_to_sqlite_iterates_reviews_array(self) -> None:
        audit = {
            "reviews": [
                _make_audit_entry("mkt-1", matrix_classification="Deserved Success"),
                _make_audit_entry("mkt-2", matrix_classification="Good Failure"),
            ],
            "aggregates": {"review_count": 2, "process_health_pct": 100.0, "win_rate_pct": 50.0},
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
        }
        audit_path = self.tmp_dir / "performance_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")

        result = sync_audit_to_sqlite(audit_path, store=self.store)
        self.assertEqual(result["synced"], 2)
        self.assertEqual(result["errors"], [])

        rows = self.store.list_reviews(limit=10)
        self.assertEqual({row["trade_id"] for row in rows}, {"mkt-1", "mkt-2"})


class EnvHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        # Always start from a clean default-store cache so other tests don't bleed in.
        reset_default_store()
        self.addCleanup(reset_default_store)

    def test_is_sqlite_enabled_reads_env_var(self) -> None:
        with patch.dict(os.environ, {SQLITE_PATH_ENV_VAR: ""}, clear=False):
            self.assertFalse(is_sqlite_enabled())
        with patch.dict(os.environ, {SQLITE_PATH_ENV_VAR: "/tmp/some.sqlite"}, clear=False):
            self.assertTrue(is_sqlite_enabled())
        # Whitespace-only counts as unset.
        with patch.dict(os.environ, {SQLITE_PATH_ENV_VAR: "   "}, clear=False):
            self.assertFalse(is_sqlite_enabled())

    def test_get_default_store_returns_none_when_disabled(self) -> None:
        env_overrides = {SQLITE_PATH_ENV_VAR: ""}
        with patch.dict(os.environ, env_overrides, clear=False):
            # Defensive: pop the var entirely so an outer test setUp can't inject a path.
            os.environ.pop(SQLITE_PATH_ENV_VAR, None)
            self.assertIsNone(get_default_store())


if __name__ == "__main__":
    unittest.main()
