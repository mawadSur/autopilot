"""Tests for ``scripts/run_postmortem.py``.

The runner orchestrates the Lane E forensic swarm over a day's closed
positions and emits a daily digest. Tests use:

* ``fakeredis`` + a real :class:`PositionStore` so the date-scoped read
  path is exercised exactly as production does.
* A stub synthesizer with canned :class:`PostmortemReport`s so we don't
  fan out a multiprocessing pool inside unit tests.
* A stub :class:`requests.Session` injected into a real :class:`Notifier`
  to assert publish payloads without touching the network.

The script lives in ``scripts/`` which is not on the standard test
``sys.path``; we load it via ``importlib`` exactly as
``test_cleanup_zombies.py`` does.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import fakeredis
import redis

from alerts.notifier import Notifier
from loss_postmortem.base import ForensicsFinding
from loss_postmortem.synthesizer import PostmortemReport
from state.position_store import Position, PositionStore


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_postmortem.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_run_postmortem_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_run_postmortem_under_test"] = module
    spec.loader.exec_module(module)
    return module


run_postmortem = _load_module()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fresh_store() -> PositionStore:
    client = fakeredis.FakeRedis(decode_responses=True)
    return PositionStore(redis_client=client, namespace="test-postmortem")


def _make_closed(
    *,
    position_id: str,
    closed_at: _dt.datetime,
    symbol: str = "ETH/USD",
    realized: float = -10.0,
) -> Position:
    opened = closed_at - _dt.timedelta(minutes=5)
    return Position(
        position_id=position_id,
        exchange="coinbase-paper",
        symbol=symbol,
        side="long",
        status="open",
        entry_price=2500.0,
        entry_quote_usd=250.0,
        base_size=0.1,
        opened_at_utc=opened.isoformat(),
    )


def _seed_closed(
    store: PositionStore,
    positions: List[Position],
    *,
    closed_at: _dt.datetime,
    realized_pnl_usd: float = -10.0,
) -> None:
    for pos in positions:
        store.record_open(pos)
        store.record_close(
            pos.position_id,
            exit_price=pos.entry_price * 0.98,
            exit_quote_usd=pos.base_size * pos.entry_price * 0.98,
        )
        # record_close stamps a fresh closed_at_utc; rewrite to land on
        # the target date for the test.
        existing = store.get(pos.position_id)
        assert existing is not None
        rewritten = existing.model_copy(
            update={
                "closed_at_utc": closed_at.isoformat(),
                "realized_pnl_usd": realized_pnl_usd,
            }
        )
        # Persist + ensure the closed-set membership matches the target day.
        store._redis.set(  # type: ignore[attr-defined]
            store._position_key(pos.position_id),  # type: ignore[attr-defined]
            rewritten.model_dump_json(),
        )
        # The default record_close also adds the position to the closed-set
        # for "today" (when the test ran). For deterministic tests we want
        # it in the target-date closed-set too; re-add explicitly.
        store._redis.sadd(  # type: ignore[attr-defined]
            store._closed_set_key(closed_at),  # type: ignore[attr-defined]
            pos.position_id,
        )


def _canned_findings() -> List[ForensicsFinding]:
    return [
        ForensicsFinding(
            agent="signal",
            verdict="primary_cause",
            confidence=0.85,
            evidence=["signal_low_margin"],
            severity=3,
        ),
        ForensicsFinding(
            agent="execution",
            verdict="innocent",
            confidence=0.4,
            evidence=["fill_within_slippage_band"],
        ),
        ForensicsFinding(
            agent="sizing",
            verdict="innocent",
            confidence=0.4,
        ),
        ForensicsFinding(
            agent="context",
            verdict="contributing",
            confidence=0.5,
            evidence=["news_window_active"],
        ),
        ForensicsFinding(
            agent="process",
            verdict="innocent",
            confidence=0.3,
        ),
    ]


def _canned_report(*, trade_id: str, symbol: str, loss_usd: float) -> PostmortemReport:
    return PostmortemReport(
        trade_id=trade_id,
        symbol=symbol,
        root_cause="Signal",
        summary="Root cause: Signal. Primary: signal. Contributing: context.",
        findings=_canned_findings(),
        weight_delta=-0.05,
        loss_usd=loss_usd,
        loss_pct=loss_usd / 250.0,
        duration_s=0.01,
        triggered_at_utc="2026-05-18T20:25:34Z",
        actions=[],
    )


class _StubSynthesizer:
    """Returns canned reports; records process_one invocations."""

    def __init__(
        self,
        *,
        reports_by_id: Dict[str, PostmortemReport],
        raise_for: Optional[Dict[str, BaseException]] = None,
    ) -> None:
        self.reports_by_id = dict(reports_by_id)
        self.raise_for = dict(raise_for or {})
        self.calls: List[str] = []

    def process_one(self, trade_id: str) -> PostmortemReport:
        self.calls.append(trade_id)
        if trade_id in self.raise_for:
            raise self.raise_for[trade_id]
        return self.reports_by_id[trade_id]


class _FakeResponse:
    def __init__(self, status_code: int = 204) -> None:
        self.status_code = status_code
        self.text = ""

    def json(self) -> Dict[str, Any]:
        return {}


class _FakeSession:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.headers: Dict[str, str] = {}

    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **_: Any,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(status_code=204)


def _make_notifier_with_stub() -> tuple[Notifier, _FakeSession]:
    """Build a Discord-only Notifier with HTTP session stubbed.

    The Notifier constructor falls back to env vars when its kwargs are
    None/empty; a stray ``TELEGRAM_BOT_TOKEN`` (set by an unrelated test
    or in the operator's shell) would silently turn this into a
    two-channel notifier and double the post count under
    ``unittest discover``. Explicitly zero those fields after construction
    so the test stays hermetic regardless of the ambient environment.
    """
    sess = _FakeSession()
    n = Notifier(
        discord_webhook_url="https://discord.test/webhook",
        telegram_bot_token=None,
        telegram_chat_id=None,
        session=sess,  # type: ignore[arg-type]
    )
    n.telegram_bot_token = None
    n.telegram_chat_id = None
    return n, sess


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class RunPostmortemHappyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = _dt.date(2026, 5, 18)
        self.closed_at = _dt.datetime(2026, 5, 18, 20, 25, 34, tzinfo=_dt.timezone.utc)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "digests"

        self.store = _fresh_store()
        self.positions = [
            _make_closed(position_id="trade-a", closed_at=self.closed_at),
            _make_closed(position_id="trade-b", closed_at=self.closed_at, symbol="BTC/USD"),
            _make_closed(position_id="trade-c", closed_at=self.closed_at, symbol="SOL/USD"),
        ]
        _seed_closed(self.store, self.positions, closed_at=self.closed_at)

    def test_writes_digest_for_each_closed_position(self) -> None:
        reports = {
            "trade-a": _canned_report(trade_id="trade-a", symbol="ETH/USD", loss_usd=-12.5),
            "trade-b": _canned_report(trade_id="trade-b", symbol="BTC/USD", loss_usd=-25.0),
            "trade-c": _canned_report(trade_id="trade-c", symbol="SOL/USD", loss_usd=-5.0),
        }
        synth = _StubSynthesizer(reports_by_id=reports)
        notifier, sess = _make_notifier_with_stub()

        code = run_postmortem.run(
            target=self.target,
            position_store_url=None,
            out_dir=self.out_dir,
            dry_run=False,
            no_publish=False,
            position_store=self.store,
            synthesizer=synth,
            notifier=notifier,
        )

        self.assertEqual(code, 0)
        # All three closed positions were inspected, in some order.
        self.assertEqual(sorted(synth.calls), ["trade-a", "trade-b", "trade-c"])

        digest_path = self.out_dir / "2026-05-18.md"
        self.assertTrue(digest_path.exists(), f"digest not written: {digest_path}")
        text = digest_path.read_text(encoding="utf-8")
        self.assertIn("Loss Postmortem Digest — 2026-05-18", text)
        self.assertIn("Closed positions inspected: 3", text)
        self.assertIn("Postmortems produced: 3", text)
        # Root-cause distribution should mention Signal=3.
        self.assertIn("Signal=3", text)
        # Per-trade sections present.
        for short_id in ("trade-a", "trade-b", "trade-c"):
            self.assertIn(short_id, text)

        # Notifier was called exactly once (Discord-only, no telegram).
        self.assertEqual(len(sess.calls), 1)
        body = sess.calls[0]["json"]
        embed = body["embeds"][0]
        # Title contains the one-line summary.
        self.assertIn("Postmortem digest 2026-05-18", embed["title"])

    def test_dry_run_writes_nothing_and_posts_nothing(self) -> None:
        reports = {
            "trade-a": _canned_report(trade_id="trade-a", symbol="ETH/USD", loss_usd=-12.5),
            "trade-b": _canned_report(trade_id="trade-b", symbol="BTC/USD", loss_usd=-25.0),
            "trade-c": _canned_report(trade_id="trade-c", symbol="SOL/USD", loss_usd=-5.0),
        }
        synth = _StubSynthesizer(reports_by_id=reports)
        notifier, sess = _make_notifier_with_stub()

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_postmortem.run(
                target=self.target,
                position_store_url=None,
                out_dir=self.out_dir,
                dry_run=True,
                no_publish=False,
                position_store=self.store,
                synthesizer=synth,
                notifier=notifier,
            )

        self.assertEqual(code, 0)
        # Nothing on disk, nothing posted.
        self.assertFalse((self.out_dir / "2026-05-18.md").exists())
        self.assertEqual(sess.calls, [])

        # Stdout has the digest.
        out = buf.getvalue()
        self.assertIn("Loss Postmortem Digest — 2026-05-18", out)
        self.assertIn("Closed positions inspected: 3", out)

    def test_no_publish_writes_digest_but_skips_notifier(self) -> None:
        reports = {
            "trade-a": _canned_report(trade_id="trade-a", symbol="ETH/USD", loss_usd=-12.5),
            "trade-b": _canned_report(trade_id="trade-b", symbol="BTC/USD", loss_usd=-25.0),
            "trade-c": _canned_report(trade_id="trade-c", symbol="SOL/USD", loss_usd=-5.0),
        }
        synth = _StubSynthesizer(reports_by_id=reports)
        notifier, sess = _make_notifier_with_stub()

        code = run_postmortem.run(
            target=self.target,
            position_store_url=None,
            out_dir=self.out_dir,
            dry_run=False,
            no_publish=True,
            position_store=self.store,
            synthesizer=synth,
            notifier=notifier,
        )
        self.assertEqual(code, 0)
        self.assertTrue((self.out_dir / "2026-05-18.md").exists())
        # Notifier was never invoked.
        self.assertEqual(sess.calls, [])


class RunPostmortemSpecialistFailureTests(unittest.TestCase):
    """Specialist failures (synthesizer.process_one raising) must isolate.

    The synthesizer's normal path already converts per-agent failures to
    ``verdict="unknown"`` findings. But if the SYNTHESIZER itself raises
    a named exception (e.g. mid-run Redis loss while reading a context
    snapshot), the runner records the failure and keeps going.
    """

    def setUp(self) -> None:
        self.target = _dt.date(2026, 5, 18)
        self.closed_at = _dt.datetime(2026, 5, 18, 20, 25, 34, tzinfo=_dt.timezone.utc)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "digests"

        self.store = _fresh_store()
        self.positions = [
            _make_closed(position_id="trade-good-1", closed_at=self.closed_at),
            _make_closed(position_id="trade-bad", closed_at=self.closed_at),
            _make_closed(position_id="trade-good-2", closed_at=self.closed_at),
        ]
        _seed_closed(self.store, self.positions, closed_at=self.closed_at)

    def test_one_specialist_failure_does_not_block_others(self) -> None:
        reports = {
            "trade-good-1": _canned_report(
                trade_id="trade-good-1", symbol="ETH/USD", loss_usd=-12.0
            ),
            "trade-good-2": _canned_report(
                trade_id="trade-good-2", symbol="ETH/USD", loss_usd=-9.0
            ),
        }
        synth = _StubSynthesizer(
            reports_by_id=reports,
            raise_for={
                "trade-bad": redis.exceptions.ConnectionError(
                    "context_store rpop failed"
                ),
            },
        )
        notifier, _ = _make_notifier_with_stub()

        code = run_postmortem.run(
            target=self.target,
            position_store_url=None,
            out_dir=self.out_dir,
            dry_run=False,
            no_publish=True,
            position_store=self.store,
            synthesizer=synth,
            notifier=notifier,
        )
        self.assertEqual(code, 0)

        text = (self.out_dir / "2026-05-18.md").read_text(encoding="utf-8")
        # Two reports landed.
        self.assertIn("Postmortems produced: 2", text)
        # The failing trade is recorded as a Specialist errored: section.
        self.assertIn("## Specialist errored: redis_disconnect", text)
        # The two good reports are still rendered.
        self.assertIn("trade-good-1", text)
        self.assertIn("trade-good-2", text)


class RunPostmortemRedisDownTests(unittest.TestCase):
    """When the position-store read itself fails, write empty digest, exit 0."""

    def setUp(self) -> None:
        self.target = _dt.date(2026, 5, 18)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "digests"

    def test_redis_unreachable_writes_empty_digest_and_exits_zero(self) -> None:
        notifier, sess = _make_notifier_with_stub()

        store = _fresh_store()

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise redis.exceptions.ConnectionError("Connection refused")

        with patch.object(store, "list_closed_today", side_effect=_boom):
            code = run_postmortem.run(
                target=self.target,
                position_store_url=None,
                out_dir=self.out_dir,
                dry_run=False,
                no_publish=True,
                position_store=store,
                # Synthesizer must NOT be reached.
                synthesizer=_StubSynthesizer(reports_by_id={}),
                notifier=notifier,
            )
        self.assertEqual(code, 0)
        digest_path = self.out_dir / "2026-05-18.md"
        self.assertTrue(digest_path.exists())
        text = digest_path.read_text(encoding="utf-8")
        self.assertIn("Redis unreachable, no data analyzed today", text)
        self.assertIn("No closed positions were analysed.", text)
        # No publish (we passed --no-publish).
        self.assertEqual(sess.calls, [])

    def test_redis_unreachable_dry_run_prints_empty_digest(self) -> None:
        store = _fresh_store()

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise redis.exceptions.ConnectionError("Connection refused")

        buf = io.StringIO()
        with patch.object(store, "list_closed_today", side_effect=_boom):
            with redirect_stdout(buf):
                code = run_postmortem.run(
                    target=self.target,
                    position_store_url=None,
                    out_dir=self.out_dir,
                    dry_run=True,
                    no_publish=False,
                    position_store=store,
                    synthesizer=_StubSynthesizer(reports_by_id={}),
                    notifier=None,
                )
        self.assertEqual(code, 0)
        self.assertFalse((self.out_dir / "2026-05-18.md").exists())
        self.assertIn("Redis unreachable", buf.getvalue())


class RunPostmortemArgParsingTests(unittest.TestCase):
    def test_default_date_is_yesterday_utc(self) -> None:
        ns = run_postmortem._parse_args(["--no-publish"])
        # date is unresolved at parse time; verify the resolver lands on
        # exactly one day before "now" in UTC.
        target = run_postmortem._resolve_target_date(ns.date)
        expected = (
            _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=1)
        ).date()
        # Tolerate the parse straddling a UTC date boundary mid-test.
        self.assertIn(target, {expected, expected + _dt.timedelta(days=1)})

    def test_bad_date_format_raises_systemexit(self) -> None:
        with self.assertRaises(SystemExit):
            run_postmortem._resolve_target_date("nope")

    def test_explicit_date_resolves(self) -> None:
        target = run_postmortem._resolve_target_date("2026-05-18")
        self.assertEqual(target, _dt.date(2026, 5, 18))


if __name__ == "__main__":
    unittest.main()
