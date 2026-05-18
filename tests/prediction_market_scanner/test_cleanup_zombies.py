"""Tests for ``scripts/cleanup_zombies.py``.

Uses fakeredis + a real PositionStore so the open_set / position-key
membership invariants are exercised exactly as production does. The
script is loaded via importlib because ``scripts/`` is not on
``sys.path`` in the standard prediction-market-scanner test invocation.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis

from state.position_store import Position, PositionStore


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "cleanup_zombies.py"


def _load_cleanup_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_cleanup_zombies_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_cleanup_zombies_under_test"] = module
    spec.loader.exec_module(module)
    return module


cleanup = _load_cleanup_module()


def _make_position(
    *,
    position_id: str,
    exchange: str,
    opened_at: datetime,
    symbol: str = "ETH/USDT",
    side: str = "long",
) -> Position:
    return Position(
        position_id=position_id,
        exchange=exchange,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status="open",
        entry_price=2500.0,
        entry_quote_usd=250.0,
        base_size=0.1,
        opened_at_utc=opened_at.isoformat(),
    )


def _fresh_store() -> PositionStore:
    client = fakeredis.FakeRedis(decode_responses=True)
    return PositionStore(redis_client=client, namespace="test-zombies")


class CleanupZombiesCoreLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
        self.store = _fresh_store()

    def _seed(self, *positions: Position) -> None:
        for pos in positions:
            self.store.record_open(pos)

    def test_dry_run_prints_table_but_does_not_delete(self) -> None:
        old = _make_position(
            position_id="paper-old-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=48),
        )
        self._seed(old)

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = cleanup.cleanup_zombies(
                store=self.store, hours=24, write=False, now=self.now
            )

        # No deletes happened.
        self.assertEqual(result["purged"], [])
        self.assertEqual(len(result["preserved"]), 1)
        self.assertIsNotNone(self.store.get("paper-old-1"))
        self.assertEqual(
            [p.position_id for p in self.store.list_open()],
            ["paper-old-1"],
        )

        # Table row marks the candidate WOULD-PURGE.
        row = result["rows"][0]
        self.assertEqual(row["verdict"], "purge")
        self.assertEqual(row["action"], "WOULD-PURGE")
        self.assertEqual(row["tag"], "coinbase-paper")

    def test_write_deletes_paper_positions_older_than_threshold(self) -> None:
        old1 = _make_position(
            position_id="paper-old-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=30),
        )
        old2 = _make_position(
            position_id="paper-old-2",
            exchange="hyperliquid-paper",
            opened_at=self.now - timedelta(hours=25),
        )
        self._seed(old1, old2)

        result = cleanup.cleanup_zombies(
            store=self.store, hours=24, write=True, now=self.now
        )

        self.assertEqual(sorted(result["purged"]), ["paper-old-1", "paper-old-2"])
        self.assertEqual(result["preserved"], [])
        self.assertIsNone(self.store.get("paper-old-1"))
        self.assertIsNone(self.store.get("paper-old-2"))
        self.assertEqual(self.store.list_open(), [])

    def test_positions_newer_than_threshold_are_preserved(self) -> None:
        old = _make_position(
            position_id="paper-old-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=48),
        )
        fresh = _make_position(
            position_id="paper-fresh-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=2),
        )
        # Exactly at threshold (24h) should also be preserved — only
        # strictly older positions are purged.
        boundary = _make_position(
            position_id="paper-boundary-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=24),
        )
        self._seed(old, fresh, boundary)

        result = cleanup.cleanup_zombies(
            store=self.store, hours=24, write=True, now=self.now
        )

        self.assertEqual(result["purged"], ["paper-old-1"])
        # The two preserved are still in the store.
        self.assertIsNotNone(self.store.get("paper-fresh-1"))
        self.assertIsNotNone(self.store.get("paper-boundary-1"))
        open_ids = {p.position_id for p in self.store.list_open()}
        self.assertEqual(open_ids, {"paper-fresh-1", "paper-boundary-1"})

        # Row verdicts.
        verdicts_by_id = {r["full_id"]: r["verdict"] for r in result["rows"]}
        self.assertEqual(verdicts_by_id["paper-old-1"], "purge")
        self.assertEqual(verdicts_by_id["paper-fresh-1"], "too_young")
        self.assertEqual(verdicts_by_id["paper-boundary-1"], "too_young")

    def test_non_paper_positions_are_never_touched(self) -> None:
        live_old = _make_position(
            position_id="live-old-1",
            exchange="coinbase",
            opened_at=self.now - timedelta(hours=72),
        )
        paper_old = _make_position(
            position_id="paper-old-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=72),
        )
        self._seed(live_old, paper_old)

        result = cleanup.cleanup_zombies(
            store=self.store, hours=24, write=True, now=self.now
        )

        # Only the paper one was purged.
        self.assertEqual(result["purged"], ["paper-old-1"])
        self.assertIsNotNone(self.store.get("live-old-1"))
        self.assertIsNone(self.store.get("paper-old-1"))

        verdicts_by_id = {r["full_id"]: r["verdict"] for r in result["rows"]}
        self.assertEqual(verdicts_by_id["live-old-1"], "not_paper")
        self.assertEqual(verdicts_by_id["paper-old-1"], "purge")
        # SKIP-LIVE action label flag for the operator.
        actions_by_id = {r["full_id"]: r["action"] for r in result["rows"]}
        self.assertEqual(actions_by_id["live-old-1"], "SKIP-LIVE")

    def test_dry_run_is_default_in_argv(self) -> None:
        # Sanity-check the CLI parser: --write is opt-in.
        parser = cleanup.build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.write)
        self.assertEqual(args.hours, cleanup.DEFAULT_HOURS)

        args2 = parser.parse_args(["--write", "--hours", "6"])
        self.assertTrue(args2.write)
        self.assertEqual(args2.hours, 6.0)

    def test_bad_opened_at_is_skipped_not_purged(self) -> None:
        bad = _make_position(
            position_id="paper-bad-1",
            exchange="coinbase-paper",
            opened_at=self.now - timedelta(hours=48),
        )
        # Corrupt the timestamp deliberately via a model_copy.
        broken = bad.model_copy(update={"opened_at_utc": "not-a-timestamp"})
        # Persist through the same atomic path record_open uses.
        self.store._redis.set(  # noqa: SLF001 - test-only seed
            f"{self.store.namespace}:positions:{broken.position_id}",
            broken.model_dump_json(),
        )
        self.store._redis.sadd(  # noqa: SLF001 - test-only seed
            f"{self.store.namespace}:open_set", broken.position_id
        )

        result = cleanup.cleanup_zombies(
            store=self.store, hours=24, write=True, now=self.now
        )

        self.assertEqual(result["purged"], [])
        # Position remains in the store untouched.
        self.assertIsNotNone(self.store.get("paper-bad-1"))
        self.assertEqual(result["rows"][0]["verdict"], "bad_opened_at")
        self.assertEqual(result["rows"][0]["action"], "SKIP-BAD-TS")


if __name__ == "__main__":
    unittest.main()
