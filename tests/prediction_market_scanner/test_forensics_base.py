"""Unit tests for ``src/loss_postmortem/base.py`` (Lane E E3).

Covers:
- ForensicsFinding construction + validation
- ForensicsFinding.to_dict round-trips through JSON
- BaseForensicsAgent subclassing → returns finding
- _safe_run / safe_investigate catches subclass exceptions
- _with_timeout / safe_investigate catches a hanging subclass
"""

from __future__ import annotations

import json
import time
import unittest

import fakeredis

from loss_postmortem.base import (
    EVIDENCE_MAX_LEN,
    BaseForensicsAgent,
    ForensicsFinding,
)
from state.trade_context_store import TradeContextStore


def _store() -> TradeContextStore:
    return TradeContextStore(
        redis_client=fakeredis.FakeRedis(decode_responses=True),
        namespace="test",
    )


# ---------------------------------------------------------------------------
# ForensicsFinding tests
# ---------------------------------------------------------------------------


class ForensicsFindingTests(unittest.TestCase):
    def test_valid_construction_succeeds(self) -> None:
        f = ForensicsFinding(
            agent="signal",
            verdict="primary_cause",
            confidence=0.85,
            evidence=["model OOD", "regime mismatch"],
            suggested_action={
                "type": "raise_floor",
                "from": 0.6,
                "to": 0.65,
            },
            severity=4,
        )
        self.assertEqual(f.agent, "signal")
        self.assertEqual(f.verdict, "primary_cause")
        self.assertAlmostEqual(f.confidence, 0.85)
        self.assertEqual(f.severity, 4)
        self.assertIsNone(f.error)

    def test_invalid_verdict_raises(self) -> None:
        with self.assertRaises(ValueError):
            ForensicsFinding(agent="signal", verdict="GUILTY")  # type: ignore[arg-type]

    def test_invalid_agent_raises(self) -> None:
        with self.assertRaises(ValueError):
            ForensicsFinding(agent="kraken", verdict="innocent")  # type: ignore[arg-type]

    def test_confidence_clamped_to_unit_interval(self) -> None:
        too_high = ForensicsFinding(
            agent="signal", verdict="contributing", confidence=2.5
        )
        self.assertEqual(too_high.confidence, 1.0)
        too_low = ForensicsFinding(
            agent="signal", verdict="contributing", confidence=-0.5
        )
        self.assertEqual(too_low.confidence, 0.0)

    def test_nan_confidence_becomes_zero(self) -> None:
        f = ForensicsFinding(
            agent="signal", verdict="unknown", confidence=float("nan")
        )
        self.assertEqual(f.confidence, 0.0)

    def test_severity_clamped_to_one_through_five(self) -> None:
        self.assertEqual(
            ForensicsFinding(
                agent="signal", verdict="innocent", severity=99
            ).severity,
            5,
        )
        self.assertEqual(
            ForensicsFinding(
                agent="signal", verdict="innocent", severity=-3
            ).severity,
            1,
        )

    def test_evidence_string_truncation(self) -> None:
        long_string = "x" * (EVIDENCE_MAX_LEN * 3)
        f = ForensicsFinding(
            agent="execution",
            verdict="contributing",
            evidence=[long_string, "short one"],
        )
        self.assertEqual(len(f.evidence), 2)
        self.assertLessEqual(len(f.evidence[0]), EVIDENCE_MAX_LEN)
        self.assertEqual(f.evidence[1], "short one")

    def test_to_dict_round_trips_through_json(self) -> None:
        f = ForensicsFinding(
            agent="sizing",
            verdict="contributing",
            confidence=0.6,
            evidence=["kelly capped"],
            suggested_action={"type": "lower_kelly_cap", "to": 0.25},
            severity=3,
            runtime_s=0.04,
        )
        blob = json.dumps(f.to_dict())
        revived = json.loads(blob)
        self.assertEqual(revived["agent"], "sizing")
        self.assertEqual(revived["verdict"], "contributing")
        self.assertAlmostEqual(revived["confidence"], 0.6)
        self.assertEqual(revived["suggested_action"]["type"], "lower_kelly_cap")
        self.assertEqual(revived["severity"], 3)
        self.assertEqual(revived["evidence"], ["kelly capped"])

    def test_unknown_factory(self) -> None:
        f = ForensicsFinding.unknown(
            agent="context", error="timeout", runtime_s=60.5
        )
        self.assertEqual(f.verdict, "unknown")
        self.assertEqual(f.agent, "context")
        self.assertEqual(f.error, "timeout")
        self.assertEqual(f.confidence, 0.0)


# ---------------------------------------------------------------------------
# BaseForensicsAgent tests
# ---------------------------------------------------------------------------


class _GoodAgent(BaseForensicsAgent):
    agent_name = "signal"

    def investigate(self, trade_id: str) -> ForensicsFinding:
        return ForensicsFinding(
            agent="signal",
            verdict="contributing",
            confidence=0.7,
            evidence=[f"investigated {trade_id}"],
            severity=2,
        )


class _CrashingAgent(BaseForensicsAgent):
    agent_name = "execution"

    def investigate(self, trade_id: str) -> ForensicsFinding:
        raise RuntimeError("simulated agent crash")


class _HangingAgent(BaseForensicsAgent):
    agent_name = "process"

    def investigate(self, trade_id: str) -> ForensicsFinding:
        time.sleep(2.0)
        return ForensicsFinding(  # pragma: no cover - never reached
            agent="process", verdict="innocent"
        )


class _NonsenseAgent(BaseForensicsAgent):
    agent_name = "context"

    def investigate(self, trade_id: str) -> ForensicsFinding:
        return "this is not a finding"  # type: ignore[return-value]


class BaseForensicsAgentTests(unittest.TestCase):
    def test_good_agent_returns_finding_via_safe_investigate(self) -> None:
        agent = _GoodAgent(context_store=_store())
        finding = agent.safe_investigate("trade-1")
        self.assertEqual(finding.agent, "signal")
        self.assertEqual(finding.verdict, "contributing")
        self.assertAlmostEqual(finding.confidence, 0.7)
        # Runtime is stamped by safe_investigate even when subclass didn't.
        self.assertGreaterEqual(finding.runtime_s, 0.0)
        self.assertIsNone(finding.error)

    def test_crashing_agent_yields_unknown_with_error(self) -> None:
        agent = _CrashingAgent(context_store=_store())
        finding = agent.safe_investigate("trade-x")
        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.agent, "execution")
        self.assertIsNotNone(finding.error)
        self.assertIn("simulated agent crash", finding.error or "")

    def test_hanging_agent_times_out(self) -> None:
        # Use a tight timeout so the test stays fast.
        agent = _HangingAgent(context_store=_store(), timeout_s=0.1)
        start = time.monotonic()
        finding = agent.safe_investigate("trade-y")
        elapsed = time.monotonic() - start
        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.error, "timeout")
        # safe_investigate should return well before the agent's 2s sleep.
        self.assertLess(elapsed, 1.5)

    def test_nonsense_return_yields_unknown(self) -> None:
        agent = _NonsenseAgent(context_store=_store())
        finding = agent.safe_investigate("trade-z")
        self.assertEqual(finding.verdict, "unknown")
        self.assertIsNotNone(finding.error)

    def test_default_investigate_raises_not_implemented(self) -> None:
        bare = BaseForensicsAgent(context_store=_store())
        # Direct call surfaces the abstract contract.
        with self.assertRaises(NotImplementedError):
            bare.investigate("trade-1")

    def test_safe_investigate_on_bare_base_returns_unknown(self) -> None:
        bare = BaseForensicsAgent(context_store=_store())
        finding = bare.safe_investigate("trade-1")
        self.assertEqual(finding.verdict, "unknown")
        self.assertIsNotNone(finding.error)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
