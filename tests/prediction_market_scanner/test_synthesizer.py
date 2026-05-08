"""Tests for :class:`loss_postmortem.synthesizer.LossPostmortemSynthesizer`.

Scope:

  (a) All five agents return primary_cause for "signal"  → root_cause = "Signal"
  (b) Two primary_cause across categories (signal + sizing) → root_cause = "Mixed"
  (c) All agents return verdict="unknown"                → root_cause = "Unknown"
  (d) Weight delta is bounded for any combination        → |delta| <= 0.10
  (e) ``runs/postmortems/{trade_id}.json`` and ``.md`` are written
  (f) Retrain queue triggered when 3+ Signal-cause postmortems on same symbol < 24h
  (g) Risk recommender NOT triggered with 4 Sizing-cause; triggered at 5
  (h) Daily digest produces a string AND calls notifier (mock notifier)
  (i) Agent timeout — slow A1; verdict="unknown", error contains "timeout",
      swarm still completes
  (j) Multiprocessing path uses ``multiprocessing.get_context("spawn")``

All tests use ``tempfile.TemporaryDirectory`` for ``runs_dir`` and
constructor-injected mock agents to avoid real Redis / multiprocessing
where it isn't being explicitly tested.
"""

from __future__ import annotations

import datetime as _dt
import json
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

from loss_postmortem.base import (
    BaseForensicsAgent,
    ForensicsFinding,
)
from loss_postmortem.synthesizer import (
    LossPostmortemSynthesizer,
    PostmortemReport,
    SIGNAL_CLUSTER_COUNT,
    SIZING_CLUSTER_COUNT,
    Synthesizer,
    WEIGHT_DELTA_MAX_ABS,
    _bound_weight_delta,
    _classify_root_cause,
)


# ---------------------------------------------------------------------------
# Stub agents
# ---------------------------------------------------------------------------


class _StubAgent(BaseForensicsAgent):
    """In-process stub agent that returns a pre-canned finding."""

    def __init__(
        self,
        *,
        agent_name: str,
        finding: ForensicsFinding,
        delay_s: float = 0.0,
        timeout_s: float = 1.0,
    ) -> None:
        # Purposely DON'T call super().__init__ — we don't need a real
        # context_store. Tests don't exercise the base wrappers' Redis
        # dependency.
        self.agent_name = agent_name  # type: ignore[assignment]
        self.timeout_s = timeout_s
        self.context_store = None  # type: ignore[assignment]
        self._finding = finding
        self._delay_s = float(delay_s)

    def investigate(self, trade_id: str) -> ForensicsFinding:  # noqa: D401
        if self._delay_s:
            time.sleep(self._delay_s)
        return self._finding


def _finding(
    agent: str,
    verdict: str,
    *,
    confidence: float = 0.5,
    evidence: Optional[List[str]] = None,
    suggested_action: Optional[Dict[str, Any]] = None,
) -> ForensicsFinding:
    return ForensicsFinding(
        agent=agent,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=list(evidence or [f"{agent} stub bullet"]),
        suggested_action=suggested_action,
    )


def _factories(findings_by_agent: Dict[str, ForensicsFinding]) -> Dict[str, Callable[[], BaseForensicsAgent]]:
    """Return a ``agent_factories`` dict the synthesizer accepts."""
    out: Dict[str, Callable[[], BaseForensicsAgent]] = {}
    for name in ("signal", "execution", "sizing", "context", "process"):
        finding = findings_by_agent.get(
            name,
            _finding(name, "innocent"),
        )
        # Capture name + finding by closure correctly with default args.
        def _make(n: str = name, f: ForensicsFinding = finding) -> BaseForensicsAgent:
            return _StubAgent(agent_name=n, finding=f)
        out[name] = _make
    return out


# ---------------------------------------------------------------------------
# Classification (a)(b)(c)
# ---------------------------------------------------------------------------


class ClassificationTests(unittest.TestCase):
    def test_a_signal_primary_classifies_as_signal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            findings_by_agent = {
                "signal": _finding("signal", "primary_cause", confidence=0.9),
                "execution": _finding("execution", "innocent"),
                "sizing": _finding("sizing", "innocent"),
                "context": _finding("context", "innocent"),
                "process": _finding("process", "innocent"),
            }
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                agent_factories=_factories(findings_by_agent),
            )
            report = s.process_one("trade-A")
        self.assertEqual(report.root_cause, "Signal")
        self.assertEqual(report.weight_delta, -0.05)

    def test_b_two_primaries_across_categories_classifies_as_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            findings_by_agent = {
                "signal": _finding("signal", "primary_cause", confidence=0.8),
                "execution": _finding("execution", "innocent"),
                "sizing": _finding("sizing", "primary_cause", confidence=0.7),
                "context": _finding("context", "innocent"),
                "process": _finding("process", "innocent"),
            }
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                agent_factories=_factories(findings_by_agent),
            )
            report = s.process_one("trade-B")
        self.assertEqual(report.root_cause, "Mixed")
        self.assertAlmostEqual(report.weight_delta, -0.03, places=6)

    def test_c_all_unknown_classifies_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            findings_by_agent = {
                name: ForensicsFinding.unknown(
                    agent=name, error="stub_unknown",  # type: ignore[arg-type]
                )
                for name in ("signal", "execution", "sizing", "context", "process")
            }
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                agent_factories=_factories(findings_by_agent),
            )
            report = s.process_one("trade-C")
        self.assertEqual(report.root_cause, "Unknown")
        self.assertEqual(report.weight_delta, 0.0)

    def test_classify_root_cause_helper_directly(self) -> None:
        # Single primary in execution → Execution.
        findings = [
            _finding("execution", "primary_cause"),
            _finding("signal", "innocent"),
        ]
        self.assertEqual(_classify_root_cause(findings), "Execution")

        # Empty list → Unknown.
        self.assertEqual(_classify_root_cause([]), "Unknown")

        # Only contributing → Unknown (we never promote contributing).
        self.assertEqual(
            _classify_root_cause([_finding("signal", "contributing")]),
            "Unknown",
        )


# ---------------------------------------------------------------------------
# Weight-delta bounds (d)
# ---------------------------------------------------------------------------


class WeightDeltaBoundsTests(unittest.TestCase):
    def test_d_weight_delta_bounded_for_any_root_cause(self) -> None:
        for root_cause in (
            "Signal", "Sizing", "Execution", "Context", "Process",
            "Mixed", "Unknown",
        ):
            from loss_postmortem.synthesizer import _WEIGHT_DELTA_BY_ROOT_CAUSE
            delta = _bound_weight_delta(_WEIGHT_DELTA_BY_ROOT_CAUSE.get(root_cause, 0.0))
            self.assertLessEqual(abs(delta), WEIGHT_DELTA_MAX_ABS)

    def test_d_bound_helper_clips_extreme_values(self) -> None:
        self.assertEqual(_bound_weight_delta(1.5), WEIGHT_DELTA_MAX_ABS)
        self.assertEqual(_bound_weight_delta(-2.0), -WEIGHT_DELTA_MAX_ABS)
        self.assertEqual(_bound_weight_delta(float("nan")), 0.0)

    def test_d_apply_weight_delta_calls_adjuster(self) -> None:
        adjuster = MagicMock()
        adjuster.apply_postmortem_delta = MagicMock(return_value=1.0)
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                outcome_weight_adjuster=adjuster,
                agent_factories=_factories({
                    "signal": _finding("signal", "primary_cause"),
                }),
            )
            s.process_one("trade-D")
        adjuster.apply_postmortem_delta.assert_called_once()
        call_arg = adjuster.apply_postmortem_delta.call_args[0][0]
        self.assertAlmostEqual(call_arg, -0.05)

    def test_d_apply_weight_delta_skipped_when_zero(self) -> None:
        # Unknown root cause → zero delta → adjuster NOT called.
        adjuster = MagicMock()
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                outcome_weight_adjuster=adjuster,
                agent_factories=_factories({
                    name: ForensicsFinding.unknown(
                        agent=name, error="x",  # type: ignore[arg-type]
                    )
                    for name in ("signal", "execution", "sizing", "context", "process")
                }),
            )
            s.process_one("trade-D2")
        adjuster.apply_postmortem_delta.assert_not_called()

    def test_d_no_adjuster_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                outcome_weight_adjuster=None,
                agent_factories=_factories({
                    "signal": _finding("signal", "primary_cause"),
                }),
            )
            report = s.process_one("trade-D3")
        self.assertEqual(report.root_cause, "Signal")


# ---------------------------------------------------------------------------
# Output writing (e)
# ---------------------------------------------------------------------------


class OutputWritingTests(unittest.TestCase):
    def test_e_json_and_md_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                agent_factories=_factories({
                    "signal": _finding(
                        "signal",
                        "primary_cause",
                        suggested_action={"type": "raise_floor", "from": 0.6, "to": 0.65},
                    ),
                }),
            )
            report = s.process_one("trade-E")

            json_path = Path(td) / "postmortems" / "trade-E.json"
            md_path = Path(td) / "postmortems" / "trade-E.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["trade_id"], "trade-E")
            self.assertEqual(data["synthesizer"]["root_cause"], "Signal")
            self.assertAlmostEqual(
                data["synthesizer"]["weight_adjustment_for_16"], -0.05
            )
            # Findings preserved.
            self.assertEqual(len(data["findings"]), 5)

            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("trade-E", md_text)
            self.assertIn("Signal", md_text)

    def test_e_runs_dir_created_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            nested = Path(td) / "deep" / "runs"
            self.assertFalse(nested.exists())
            s = LossPostmortemSynthesizer(
                runs_dir=nested,
                use_multiprocessing=False,
                agent_factories=_factories({
                    "signal": _finding("signal", "innocent"),
                }),
            )
            s.process_one("trade-E2")
            self.assertTrue((nested / "postmortems" / "trade-E2.json").exists())


# ---------------------------------------------------------------------------
# Retrain queue (f)
# ---------------------------------------------------------------------------


class RetrainQueueTests(unittest.TestCase):
    def _seed_signal_postmortems(
        self,
        runs_dir: Path,
        *,
        symbol: str,
        count: int,
        within_24h: bool = True,
    ) -> None:
        pm_dir = runs_dir / "postmortems"
        pm_dir.mkdir(parents=True, exist_ok=True)
        if within_24h:
            ts = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2)
        else:
            ts = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=7)
        for i in range(count):
            tid = f"{symbol.replace('/', '-')}-seed-{i}"
            entry = {
                "trade_id": tid,
                "symbol": symbol,
                "loss_usd": -25.0,
                "loss_pct": -0.005,
                "triggered_at_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_s": 1.0,
                "findings": [],
                "synthesizer": {
                    "root_cause": "Signal",
                    "summary": "...",
                    "actions": [],
                    "weight_adjustment_for_16": -0.05,
                },
            }
            (pm_dir / f"{tid}.json").write_text(
                json.dumps(entry, sort_keys=True), encoding="utf-8"
            )

    def test_f_retrain_appended_when_3_plus_signal_in_24h(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            self._seed_signal_postmortems(runs, symbol="BTC/USD", count=2)
            # Now run the synthesizer on a 3rd Signal-cause trade for BTC/USD.
            s = LossPostmortemSynthesizer(
                runs_dir=runs,
                use_multiprocessing=False,
                position_store=_StubPositionStore("trade-F", symbol="BTC/USD"),
                agent_factories=_factories({
                    "signal": _finding("signal", "primary_cause"),
                }),
            )
            s.process_one("trade-F")
            queue_path = runs / "retrain_queue.jsonl"
            self.assertTrue(queue_path.exists())
            lines = queue_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["symbol"], "BTC/USD")
            self.assertEqual(entry["reason"], "signal_cause_cluster")
            # Includes seed IDs + current trade.
            self.assertIn("trade-F", entry["postmortem_ids"])
            self.assertGreaterEqual(len(entry["postmortem_ids"]), SIGNAL_CLUSTER_COUNT)

    def test_f_retrain_NOT_appended_when_only_2_signal_in_24h(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            # Seed only 1 — current trade makes 2 total → below threshold.
            self._seed_signal_postmortems(runs, symbol="ETH/USD", count=1)
            s = LossPostmortemSynthesizer(
                runs_dir=runs,
                use_multiprocessing=False,
                position_store=_StubPositionStore("trade-F2", symbol="ETH/USD"),
                agent_factories=_factories({
                    "signal": _finding("signal", "primary_cause"),
                }),
            )
            s.process_one("trade-F2")
            self.assertFalse((runs / "retrain_queue.jsonl").exists())

    def test_f_retrain_NOT_appended_for_old_postmortems(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            # 5 OLD signals (>24h ago) shouldn't count.
            self._seed_signal_postmortems(
                runs, symbol="SOL/USD", count=5, within_24h=False,
            )
            s = LossPostmortemSynthesizer(
                runs_dir=runs,
                use_multiprocessing=False,
                position_store=_StubPositionStore("trade-F3", symbol="SOL/USD"),
                agent_factories=_factories({
                    "signal": _finding("signal", "primary_cause"),
                }),
            )
            s.process_one("trade-F3")
            self.assertFalse((runs / "retrain_queue.jsonl").exists())


# ---------------------------------------------------------------------------
# Risk recommender (g)
# ---------------------------------------------------------------------------


class RiskRecommenderTests(unittest.TestCase):
    def _seed_sizing_postmortems(
        self,
        runs_dir: Path,
        *,
        symbol: str,
        count: int,
    ) -> None:
        pm_dir = runs_dir / "postmortems"
        pm_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2)
        for i in range(count):
            tid = f"sz-seed-{i}"
            entry = {
                "trade_id": tid,
                "symbol": symbol,
                "loss_usd": -50.0,
                "loss_pct": -0.01,
                "triggered_at_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_s": 1.0,
                "findings": [],
                "synthesizer": {
                    "root_cause": "Sizing",
                    "summary": "...",
                    "actions": [],
                    "weight_adjustment_for_16": -0.05,
                },
            }
            (pm_dir / f"{tid}.json").write_text(
                json.dumps(entry, sort_keys=True), encoding="utf-8"
            )

    def test_g_risk_recommender_NOT_triggered_at_4_sizing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            self._seed_sizing_postmortems(runs, symbol="BTC/USD", count=3)
            s = LossPostmortemSynthesizer(
                runs_dir=runs,
                use_multiprocessing=False,
                position_store=_StubPositionStore("trade-G1", symbol="BTC/USD"),
                agent_factories=_factories({
                    "sizing": _finding("sizing", "primary_cause"),
                }),
            )
            s.process_one("trade-G1")
            # 3 seed + 1 current = 4 → below threshold.
            self.assertFalse((runs / "risk_recommendations.jsonl").exists())

    def test_g_risk_recommender_triggered_at_5_sizing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            self._seed_sizing_postmortems(runs, symbol="BTC/USD", count=4)
            s = LossPostmortemSynthesizer(
                runs_dir=runs,
                use_multiprocessing=False,
                position_store=_StubPositionStore("trade-G2", symbol="BTC/USD"),
                agent_factories=_factories({
                    "sizing": _finding("sizing", "primary_cause"),
                }),
            )
            s.process_one("trade-G2")
            # 4 seed + 1 current = 5 → triggered.
            risk_path = runs / "risk_recommendations.jsonl"
            self.assertTrue(risk_path.exists())
            entries = [
                json.loads(line) for line in
                risk_path.read_text(encoding="utf-8").strip().splitlines()
            ]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["symbol"], "BTC/USD")
            self.assertGreaterEqual(
                len(entries[0]["evidence"]), SIZING_CLUSTER_COUNT
            )


# ---------------------------------------------------------------------------
# Daily digest (h)
# ---------------------------------------------------------------------------


class DailyDigestTests(unittest.TestCase):
    def test_h_digest_returns_string_and_calls_notifier(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            pm_dir = runs / "postmortems"
            pm_dir.mkdir(parents=True, exist_ok=True)
            now = _dt.datetime.now(_dt.timezone.utc)
            losses = [
                ("trade-X", "BTC/USD", -150.0, "Signal"),
                ("trade-Y", "ETH/USD", -75.0, "Sizing"),
                ("trade-Z", "SOL/USD", -40.0, "Execution"),
                ("trade-W", "BTC/USD", -25.0, "Context"),
            ]
            for tid, sym, loss, rc in losses:
                entry = {
                    "trade_id": tid,
                    "symbol": sym,
                    "loss_usd": loss,
                    "loss_pct": -0.01,
                    "triggered_at_utc": (
                        now - _dt.timedelta(hours=3)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "duration_s": 1.0,
                    "findings": [],
                    "synthesizer": {
                        "root_cause": rc,
                        "summary": "...",
                        "actions": [],
                        "weight_adjustment_for_16": 0.0,
                    },
                }
                (pm_dir / f"{tid}.json").write_text(
                    json.dumps(entry, sort_keys=True), encoding="utf-8"
                )

            notifier = MagicMock()
            notifier.alert = MagicMock(return_value=True)
            s = LossPostmortemSynthesizer(
                runs_dir=runs,
                use_multiprocessing=False,
                notifier=notifier,
            )
            digest = s.daily_digest()
            self.assertIsInstance(digest, str)
            self.assertIn("Loss Postmortem Daily Digest", digest)
            self.assertIn("trade-X", digest)
            # Top-3 shows the 3 biggest losses by USD (most negative).
            self.assertIn("-150", digest.replace(",", ""))
            self.assertIn("Signal=1", digest)
            notifier.alert.assert_called_once()

    def test_h_digest_no_notifier_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                notifier=None,
            )
            digest = s.daily_digest()
            self.assertIsInstance(digest, str)
            self.assertIn("Loss Postmortem Daily Digest", digest)

    def test_h_digest_notifier_crash_does_not_propagate(self) -> None:
        notifier = MagicMock()
        notifier.alert = MagicMock(side_effect=RuntimeError("boom"))
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                notifier=notifier,
            )
            digest = s.daily_digest()
            self.assertIsInstance(digest, str)
            notifier.alert.assert_called_once()


# ---------------------------------------------------------------------------
# Agent timeout (i)
# ---------------------------------------------------------------------------


class AgentTimeoutTests(unittest.TestCase):
    def test_i_slow_agent_times_out_swarm_completes(self) -> None:
        # Build a slow signal agent (delay > timeout). The other 4 succeed.
        slow_signal = _StubAgent(
            agent_name="signal",
            finding=_finding("signal", "innocent"),
            delay_s=0.30,
            timeout_s=0.05,  # 50ms — much shorter than 300ms delay
        )
        factories: Dict[str, Callable[[], BaseForensicsAgent]] = {
            "signal": lambda: slow_signal,
        }
        # Other 4 — innocent.
        for name in ("execution", "sizing", "context", "process"):
            factories[name] = (
                lambda n=name: _StubAgent(
                    agent_name=n,
                    finding=_finding(n, "innocent"),
                )
            )
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                agent_factories=factories,
            )
            report = s.process_one("trade-I")
        # Find the signal finding in the report.
        signal_finding = next(f for f in report.findings if f.agent == "signal")
        self.assertEqual(signal_finding.verdict, "unknown")
        self.assertIsNotNone(signal_finding.error)
        self.assertIn("timeout", (signal_finding.error or "").lower())
        # Swarm still completed: 5 findings total.
        self.assertEqual(len(report.findings), 5)


# ---------------------------------------------------------------------------
# Multiprocessing path (j)
# ---------------------------------------------------------------------------


class MultiprocessingPathTests(unittest.TestCase):
    def test_j_uses_spawn_context_for_pool(self) -> None:
        # Patch get_context to verify "spawn" is requested. Then return a
        # mock context whose .Pool yields a fake pool that returns 5
        # innocent findings.

        spawn_ctx = MagicMock()
        fake_pool = MagicMock()
        fake_pool.__enter__ = MagicMock(return_value=fake_pool)
        fake_pool.__exit__ = MagicMock(return_value=False)

        def _async_for(name: str) -> Any:
            ar = MagicMock()
            ar.get = MagicMock(return_value=ForensicsFinding(
                agent=name,  # type: ignore[arg-type]
                verdict="innocent",
                evidence=[f"{name} mp stub"],
            ).to_dict())
            return ar

        # apply_async receives (worker_callable, ((agent_name, trade_id, cfg),))
        def _apply_async(callable_, args=(), kwds=None) -> Any:
            (payload,) = args
            agent_name, _trade_id, _cfg = payload
            return _async_for(agent_name)

        fake_pool.apply_async = _apply_async
        spawn_ctx.Pool = MagicMock(return_value=fake_pool)

        with patch(
            "loss_postmortem.synthesizer.multiprocessing.get_context",
            return_value=spawn_ctx,
        ) as mock_get_context:
            with tempfile.TemporaryDirectory() as td:
                s = LossPostmortemSynthesizer(
                    runs_dir=td,
                    use_multiprocessing=True,
                )
                report = s.process_one("trade-J")
        mock_get_context.assert_called_with("spawn")
        # Pool was constructed with processes=5 (one per agent).
        spawn_ctx.Pool.assert_called_once()
        kwargs = spawn_ctx.Pool.call_args.kwargs
        self.assertEqual(kwargs.get("processes"), 5)
        self.assertEqual(report.root_cause, "Unknown")  # all innocent → no primary
        self.assertEqual(len(report.findings), 5)

    def test_j_pool_failure_returns_unknown_findings(self) -> None:
        # If get_context raises, the synthesizer falls back to all-unknown.
        with patch(
            "loss_postmortem.synthesizer.multiprocessing.get_context",
            side_effect=RuntimeError("no spawn for you"),
        ):
            with tempfile.TemporaryDirectory() as td:
                s = LossPostmortemSynthesizer(
                    runs_dir=td,
                    use_multiprocessing=True,
                )
                report = s.process_one("trade-J2")
        self.assertEqual(report.root_cause, "Unknown")
        self.assertEqual(len(report.findings), 5)
        for f in report.findings:
            self.assertEqual(f.verdict, "unknown")


# ---------------------------------------------------------------------------
# Drain queue
# ---------------------------------------------------------------------------


class _FakeRedisQueue:
    """Minimal stand-in for a Redis client supporting LPUSH/RPOP semantics."""

    def __init__(self, ids: List[str]) -> None:
        # ``ids`` represents what's already been LPUSHed in order (left to right).
        # LPUSH puts new items on the LEFT (index 0). RPOP yields from the
        # RIGHT, giving FIFO order: ids[-1] pops first.
        self._list: List[str] = list(ids)

    def rpop(self, key: str) -> Optional[str]:
        if not self._list:
            return None
        return self._list.pop(-1)

    def lpush(self, key: str, value: str) -> None:
        self._list.insert(0, value)


class DrainTests(unittest.TestCase):
    def test_drain_processes_queue_in_fifo_order(self) -> None:
        # LPUSH order: t1 first, t2 next, t3 last (so list is [t3, t2, t1]).
        # RPOP yields t1, t2, t3 — FIFO.
        fake = _FakeRedisQueue(["t3", "t2", "t1"])
        with tempfile.TemporaryDirectory() as td:
            calls: List[str] = []

            def _factory_for(name: str) -> Callable[[], BaseForensicsAgent]:
                def _make() -> BaseForensicsAgent:
                    return _StubAgent(
                        agent_name=name,
                        finding=_finding(name, "innocent"),
                    )
                return _make

            factories = {n: _factory_for(n) for n in (
                "signal", "execution", "sizing", "context", "process",
            )}
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                redis_client=fake,
                agent_factories=factories,
            )
            # Wrap process_one to record call order.
            orig = s.process_one

            def _spy(trade_id: str) -> Any:
                calls.append(trade_id)
                return orig(trade_id)

            s.process_one = _spy  # type: ignore[method-assign]
            n = s.drain(max_items=10)
        self.assertEqual(n, 3)
        self.assertEqual(calls, ["t1", "t2", "t3"])

    def test_drain_empty_queue_returns_zero(self) -> None:
        fake = _FakeRedisQueue([])
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                redis_client=fake,
            )
            self.assertEqual(s.drain(max_items=10), 0)

    def test_drain_no_redis_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                redis_client=None,
                context_store=None,
            )
            self.assertEqual(s.drain(max_items=10), 0)

    def test_drain_continues_when_one_trade_fails(self) -> None:
        fake = _FakeRedisQueue(["bad", "good"])
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                redis_client=fake,
            )
            orig = s.process_one
            calls: List[str] = []

            def _maybe_fail(trade_id: str) -> Any:
                calls.append(trade_id)
                if trade_id == "bad":
                    raise RuntimeError("simulated process_one crash")
                return orig(trade_id)

            s.process_one = _maybe_fail  # type: ignore[method-assign]
            n = s.drain(max_items=10)
        # 1 succeeded ("good"), 1 crashed ("bad" did NOT increment).
        self.assertEqual(n, 1)
        self.assertEqual(calls, ["good", "bad"])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class _StubPositionStore:
    """Stub :class:`PositionStore` returning a minimal Position-like object."""

    def __init__(self, trade_id: str, *, symbol: str, pnl: float = -42.0) -> None:
        self.trade_id = trade_id
        self.symbol = symbol
        self.pnl = pnl

    def get(self, position_id: str) -> Any:
        if position_id != self.trade_id:
            return None
        class _Pos:
            pass
        p = _Pos()
        p.symbol = self.symbol  # type: ignore[attr-defined]
        p.realized_pnl_usd = self.pnl  # type: ignore[attr-defined]
        p.entry_quote_usd = 1000.0  # type: ignore[attr-defined]
        return p


class MiscTests(unittest.TestCase):
    def test_synthesizer_alias_is_loss_postmortem_synthesizer(self) -> None:
        self.assertIs(Synthesizer, LossPostmortemSynthesizer)

    def test_position_metadata_populated_in_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                position_store=_StubPositionStore("trade-M", symbol="BTC/USD", pnl=-99.5),
                agent_factories=_factories({
                    "signal": _finding("signal", "primary_cause"),
                }),
            )
            report = s.process_one("trade-M")
        self.assertEqual(report.symbol, "BTC/USD")
        self.assertAlmostEqual(report.loss_usd, -99.5)
        self.assertAlmostEqual(report.loss_pct, -99.5 / 1000.0, places=6)

    def test_actions_aggregated_from_findings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
                agent_factories=_factories({
                    "signal": _finding(
                        "signal",
                        "primary_cause",
                        suggested_action={"type": "raise_floor", "from": 0.6},
                    ),
                    "sizing": _finding(
                        "sizing",
                        "contributing",
                        suggested_action={"type": "lower_kelly", "from": 0.5},
                    ),
                }),
            )
            report = s.process_one("trade-M2")
        types = sorted(a.get("type") for a in report.actions)
        self.assertEqual(types, ["lower_kelly", "raise_floor"])

    def test_invalid_trade_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s = LossPostmortemSynthesizer(
                runs_dir=td,
                use_multiprocessing=False,
            )
            with self.assertRaises(ValueError):
                s.process_one("")


if __name__ == "__main__":
    unittest.main()
