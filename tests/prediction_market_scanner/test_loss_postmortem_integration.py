"""Lane E end-to-end integration tests for the Loss Postmortem swarm.

These tests exercise the full :class:`LossPostmortemSynthesizer` pipeline
against five hand-built loss scenarios under
``tests/prediction_market_scanner/fixtures/postmortem/``. Each fixture
populates a :class:`TradeContextStore` (backed by ``fakeredis``) plus a
:class:`PositionStore` with a synthetic loss whose root cause maps to a
specific specialist agent (A1-A5).

Implementation notes
--------------------
- We DO NOT spawn real subprocesses. The synthesizer accepts an
  ``agent_factories`` injection + ``use_multiprocessing=False``; the
  swarm's multiprocessing path is already covered by ``test_synthesizer.py``.
- Each agent factory builds the actual specialist class (Signal,
  Execution, Sizing, Context, Process) with the in-memory stores
  injected. A4 (context) is wired with the canned news/markets/Gemini
  callables the news_event_miss fixture returns.
- Verdict assertions use SET membership for ``root_cause`` and range
  bounds for ``weight_delta`` since real swarm verdicts are
  probabilistic across multiple agents.
- Output paths are redirected with ``tempfile.TemporaryDirectory`` so
  no test ever writes outside its tmpdir.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import fakeredis

from loss_postmortem.base import BaseForensicsAgent, ForensicsFinding
from loss_postmortem.context_forensics import ContextForensicsAgent
from loss_postmortem.execution_forensics import ExecutionForensicsAgent
from loss_postmortem.process_integrity import ProcessIntegrityAgent
from loss_postmortem.signal_forensics import SignalForensicsAgent
from loss_postmortem.sizing_forensics import SizingForensicsAgent
from loss_postmortem.synthesizer import LossPostmortemSynthesizer
from state.position_store import PositionStore
from state.trade_context_store import TradeContextStore

from fixtures.postmortem import (
    breaker_force_flat,
    news_event_miss,
    race_condition,
    signal_ood,
    slippage_driven,
)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


def _make_stores(
    namespace: str = "test",
) -> tuple[fakeredis.FakeRedis, TradeContextStore, PositionStore]:
    """Build a fakeredis client + the two stores that share it."""

    fake = fakeredis.FakeRedis(decode_responses=True)
    ctx_store = TradeContextStore(redis_client=fake, namespace=namespace)
    pos_store = PositionStore(redis_client=fake, namespace=namespace)
    return fake, ctx_store, pos_store


class _EmptyNewsFetcher:
    """Stand-in news fetcher that always returns no headlines.

    Used as the default A4 news fetcher for fixtures that don't care
    about news context — ensures we never make real Google News RSS
    calls during integration tests.
    """

    def fetch_news(self) -> list:  # noqa: D401
        return []


def _empty_news_factory(_query: str) -> _EmptyNewsFetcher:
    return _EmptyNewsFetcher()


def _empty_markets_fetcher() -> list:
    return []


def _build_factories(
    *,
    ctx_store: TradeContextStore,
    pos_store: PositionStore,
    redis_client: Any,
    namespace: str,
    meta_base_dir: str,
    a4_overrides: Optional[Dict[str, Callable[..., Any]]] = None,
    crash_agent: Optional[str] = None,
) -> Dict[str, Callable[[], BaseForensicsAgent]]:
    """Build the five agent factories the synthesizer fans out to.

    ``a4_overrides`` injects the news_event_miss fixture's canned
    fetchers so the Context agent runs deterministically without the
    network. When unspecified, A4 is wired to empty stubs so it never
    contacts Google News / Polymarket / Gemini. ``crash_agent``, if set
    to a known name, swaps that agent for one whose ``investigate``
    raises — used to assert the swarm keeps the other four agents alive.
    """

    a4_overrides = a4_overrides or {}
    news_factory = a4_overrides.get("news_fetcher_factory") or _empty_news_factory
    markets_fetcher = a4_overrides.get("markets_fetcher") or _empty_markets_fetcher
    # Even an empty headline+market list shouldn't trigger the LLM call
    # path inside ContextForensicsAgent (it gates on having results), but
    # we wire a benign caller anyway to be safe across refactors.
    gemini_caller = a4_overrides.get("gemini_caller") or (lambda _p: "")

    def _signal_factory() -> BaseForensicsAgent:
        return SignalForensicsAgent(
            context_store=ctx_store,
            meta_base_dir=meta_base_dir,
            timeout_s=10.0,
        )

    def _execution_factory() -> BaseForensicsAgent:
        return ExecutionForensicsAgent(
            context_store=ctx_store,
            position_store=pos_store,
            timeout_s=10.0,
        )

    def _sizing_factory() -> BaseForensicsAgent:
        return SizingForensicsAgent(
            context_store=ctx_store,
            position_store=pos_store,
            timeout_s=10.0,
        )

    def _context_factory() -> BaseForensicsAgent:
        return ContextForensicsAgent(
            context_store=ctx_store,
            timeout_s=10.0,
            news_fetcher_factory=news_factory,
            markets_fetcher=markets_fetcher,
            gemini_caller=gemini_caller,
        )

    def _process_factory() -> BaseForensicsAgent:
        return ProcessIntegrityAgent(
            context_store=ctx_store,
            position_store=pos_store,
            redis_client=redis_client,
            namespace=namespace,
            timeout_s=10.0,
        )

    factories: Dict[str, Callable[[], BaseForensicsAgent]] = {
        "signal": _signal_factory,
        "execution": _execution_factory,
        "sizing": _sizing_factory,
        "context": _context_factory,
        "process": _process_factory,
    }

    if crash_agent and crash_agent in factories:
        # Swap the named factory for one that raises during investigate.
        # ``safe_investigate`` should catch and surface verdict="unknown"
        # with an error string.
        original = factories[crash_agent]

        def _crashing_factory() -> BaseForensicsAgent:
            agent = original()

            def _boom(_trade_id: str) -> ForensicsFinding:  # noqa: ANN001
                raise RuntimeError("simulated_agent_crash")

            agent.investigate = _boom  # type: ignore[method-assign]
            return agent

        factories[crash_agent] = _crashing_factory

    return factories


def _run_swarm(
    *,
    runs_dir: str,
    factories: Dict[str, Callable[[], BaseForensicsAgent]],
    ctx_store: TradeContextStore,
    pos_store: PositionStore,
    trade_id: str,
):
    synth = LossPostmortemSynthesizer(
        runs_dir=runs_dir,
        use_multiprocessing=False,
        agent_factories=factories,
        context_store=ctx_store,
        position_store=pos_store,
    )
    return synth, synth.process_one(trade_id)


def _findings_by_agent(report) -> Dict[str, ForensicsFinding]:
    return {f.agent: f for f in report.findings}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class LossPostmortemIntegrationTests(unittest.TestCase):
    """End-to-end swarm runs against the five fixture loss scenarios."""

    def test_signal_ood_classified_as_signal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_dir = str(Path(td) / "runs")
            meta_dir = str(Path(td) / "model_crypto")
            redis_client, ctx_store, pos_store = _make_stores()

            trade_id = signal_ood.build_fixture(
                context_store=ctx_store,
                position_store=pos_store,
                meta_base_dir=meta_dir,
            )
            factories = _build_factories(
                ctx_store=ctx_store,
                pos_store=pos_store,
                redis_client=redis_client,
                namespace="test",
                meta_base_dir=meta_dir,
            )
            _, report = _run_swarm(
                runs_dir=runs_dir,
                factories=factories,
                ctx_store=ctx_store,
                pos_store=pos_store,
                trade_id=trade_id,
            )

        self.assertIn(
            report.root_cause,
            {"Signal", "Mixed"},
            msg=(
                f"signal_ood expected root_cause Signal or Mixed; "
                f"got {report.root_cause!r}. Findings: "
                f"{[(f.agent, f.verdict) for f in report.findings]}"
            ),
        )
        self.assertGreaterEqual(report.weight_delta, -0.10)
        self.assertLessEqual(report.weight_delta, -0.03)

    def test_slippage_classified_as_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_dir = str(Path(td) / "runs")
            meta_dir = str(Path(td) / "model_crypto")
            redis_client, ctx_store, pos_store = _make_stores()

            trade_id = slippage_driven.build_fixture(
                context_store=ctx_store,
                position_store=pos_store,
                meta_base_dir=meta_dir,
            )
            factories = _build_factories(
                ctx_store=ctx_store,
                pos_store=pos_store,
                redis_client=redis_client,
                namespace="test",
                meta_base_dir=meta_dir,
            )
            _, report = _run_swarm(
                runs_dir=runs_dir,
                factories=factories,
                ctx_store=ctx_store,
                pos_store=pos_store,
                trade_id=trade_id,
            )

        self.assertIn(
            report.root_cause,
            {"Execution", "Mixed"},
            msg=(
                f"slippage expected Execution or Mixed; "
                f"got {report.root_cause!r}. Findings: "
                f"{[(f.agent, f.verdict) for f in report.findings]}"
            ),
        )
        self.assertGreaterEqual(report.weight_delta, -0.10)
        self.assertLessEqual(report.weight_delta, 0.0)

    def test_breaker_force_flat_classified_as_process(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_dir = str(Path(td) / "runs")
            meta_dir = str(Path(td) / "model_crypto")
            redis_client, ctx_store, pos_store = _make_stores()

            trade_id = breaker_force_flat.build_fixture(
                context_store=ctx_store,
                position_store=pos_store,
                meta_base_dir=meta_dir,
            )
            factories = _build_factories(
                ctx_store=ctx_store,
                pos_store=pos_store,
                redis_client=redis_client,
                namespace="test",
                meta_base_dir=meta_dir,
            )
            _, report = _run_swarm(
                runs_dir=runs_dir,
                factories=factories,
                ctx_store=ctx_store,
                pos_store=pos_store,
                trade_id=trade_id,
            )

        # Process must show up in the root cause label (Process or Mixed).
        self.assertIn(
            "Process",
            report.root_cause,
            msg=(
                f"breaker_force_flat expected Process in root_cause; "
                f"got {report.root_cause!r}. Findings: "
                f"{[(f.agent, f.verdict) for f in report.findings]}"
            ),
        )

    def test_news_event_miss_increases_context_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_dir = str(Path(td) / "runs")
            meta_dir = str(Path(td) / "model_crypto")
            redis_client, ctx_store, pos_store = _make_stores()

            info = news_event_miss.build_fixture(
                context_store=ctx_store,
                position_store=pos_store,
                meta_base_dir=meta_dir,
            )
            trade_id = info["trade_id"]
            factories = _build_factories(
                ctx_store=ctx_store,
                pos_store=pos_store,
                redis_client=redis_client,
                namespace="test",
                meta_base_dir=meta_dir,
                a4_overrides={
                    "news_fetcher_factory": info["news_fetcher_factory"],
                    "markets_fetcher": info["markets_fetcher"],
                    "gemini_caller": info["gemini_caller"],
                },
            )
            _, report = _run_swarm(
                runs_dir=runs_dir,
                factories=factories,
                ctx_store=ctx_store,
                pos_store=pos_store,
                trade_id=trade_id,
            )

        findings = _findings_by_agent(report)
        ctx_finding = findings.get("context")
        self.assertIsNotNone(ctx_finding, "context finding missing")
        # Verdict must reflect that the news density red flag fired.
        self.assertIn(
            ctx_finding.verdict,
            {"contributing", "primary_cause"},
            msg=(
                f"context expected contributing/primary_cause; got "
                f"{ctx_finding.verdict!r}. Evidence: {ctx_finding.evidence}"
            ),
        )
        # At least one evidence bullet must mention news / headline.
        joined = " ".join(ctx_finding.evidence).lower()
        self.assertTrue(
            "news" in joined or "headline" in joined,
            msg=f"no news/headline evidence bullet; got {ctx_finding.evidence!r}",
        )

    def test_race_condition_classified_as_process(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_dir = str(Path(td) / "runs")
            meta_dir = str(Path(td) / "model_crypto")
            redis_client, ctx_store, pos_store = _make_stores()

            trade_id = race_condition.build_fixture(
                context_store=ctx_store,
                position_store=pos_store,
                meta_base_dir=meta_dir,
                redis_client=redis_client,
                namespace="test",
            )
            factories = _build_factories(
                ctx_store=ctx_store,
                pos_store=pos_store,
                redis_client=redis_client,
                namespace="test",
                meta_base_dir=meta_dir,
            )
            _, report = _run_swarm(
                runs_dir=runs_dir,
                factories=factories,
                ctx_store=ctx_store,
                pos_store=pos_store,
                trade_id=trade_id,
            )

        findings = _findings_by_agent(report)
        proc_finding = findings.get("process")
        self.assertIsNotNone(proc_finding, "process finding missing")
        # With the W1A verdict-ladder tightening (counter >= 15 promotes
        # to primary_cause) the canonical fixture seeds at the very-high
        # tier so A5 returns primary_cause and the swarm's root_cause
        # label includes "Process".
        self.assertEqual(
            proc_finding.verdict,
            "primary_cause",
            msg=(
                f"process expected primary_cause from very-high race "
                f"cluster; got {proc_finding.verdict!r}. Evidence: "
                f"{proc_finding.evidence}"
            ),
        )
        self.assertIn(
            "Process",
            report.root_cause,
            msg=(
                f"race_condition expected Process in root_cause; "
                f"got {report.root_cause!r}. Findings: "
                f"{[(f.agent, f.verdict) for f in report.findings]}"
            ),
        )
        # Verify the evidence references the race / error counter.
        joined = " ".join(proc_finding.evidence).lower()
        self.assertTrue(
            "error counter" in joined or "concurrent" in joined or "race" in joined
            or "extreme error contention" in joined,
            msg=f"no race-related evidence bullet; got {proc_finding.evidence!r}",
        )

    def test_postmortem_outputs_written_for_all_fixtures(self) -> None:
        """Each fixture writes both ``{trade_id}.json`` and ``{trade_id}.md``."""

        scenarios = [
            ("signal_ood", signal_ood, {}),
            ("slippage", slippage_driven, {}),
            ("breaker", breaker_force_flat, {}),
            ("news", news_event_miss, {"is_dict_return": True}),
            ("race", race_condition, {"needs_redis_args": True}),
        ]

        for label, module, opts in scenarios:
            with self.subTest(scenario=label):
                with tempfile.TemporaryDirectory() as td:
                    runs_dir = str(Path(td) / "runs")
                    meta_dir = str(Path(td) / "model_crypto")
                    redis_client, ctx_store, pos_store = _make_stores()

                    a4_overrides: Dict[str, Callable[..., Any]] = {}
                    if opts.get("is_dict_return"):
                        info = module.build_fixture(
                            context_store=ctx_store,
                            position_store=pos_store,
                            meta_base_dir=meta_dir,
                        )
                        trade_id = info["trade_id"]
                        a4_overrides = {
                            "news_fetcher_factory": info["news_fetcher_factory"],
                            "markets_fetcher": info["markets_fetcher"],
                            "gemini_caller": info["gemini_caller"],
                        }
                    elif opts.get("needs_redis_args"):
                        trade_id = module.build_fixture(
                            context_store=ctx_store,
                            position_store=pos_store,
                            meta_base_dir=meta_dir,
                            redis_client=redis_client,
                            namespace="test",
                        )
                    else:
                        trade_id = module.build_fixture(
                            context_store=ctx_store,
                            position_store=pos_store,
                            meta_base_dir=meta_dir,
                        )

                    factories = _build_factories(
                        ctx_store=ctx_store,
                        pos_store=pos_store,
                        redis_client=redis_client,
                        namespace="test",
                        meta_base_dir=meta_dir,
                        a4_overrides=a4_overrides,
                    )
                    _, report = _run_swarm(
                        runs_dir=runs_dir,
                        factories=factories,
                        ctx_store=ctx_store,
                        pos_store=pos_store,
                        trade_id=trade_id,
                    )

                    json_path = Path(runs_dir) / "postmortems" / f"{trade_id}.json"
                    md_path = Path(runs_dir) / "postmortems" / f"{trade_id}.md"
                    self.assertTrue(
                        json_path.exists(),
                        f"json output missing for {label}: {json_path}",
                    )
                    self.assertTrue(
                        md_path.exists(),
                        f"md output missing for {label}: {md_path}",
                    )
                    # Sanity-check the JSON shape.
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    self.assertEqual(payload.get("trade_id"), trade_id)
                    self.assertIn("synthesizer", payload)
                    self.assertIn("root_cause", payload["synthesizer"])

    def test_one_agent_crash_does_not_block_swarm(self) -> None:
        """Crashing one specialist must not poison the other four."""

        with tempfile.TemporaryDirectory() as td:
            runs_dir = str(Path(td) / "runs")
            meta_dir = str(Path(td) / "model_crypto")
            redis_client, ctx_store, pos_store = _make_stores()

            trade_id = signal_ood.build_fixture(
                context_store=ctx_store,
                position_store=pos_store,
                meta_base_dir=meta_dir,
            )
            factories = _build_factories(
                ctx_store=ctx_store,
                pos_store=pos_store,
                redis_client=redis_client,
                namespace="test",
                meta_base_dir=meta_dir,
                crash_agent="execution",
            )
            _, report = _run_swarm(
                runs_dir=runs_dir,
                factories=factories,
                ctx_store=ctx_store,
                pos_store=pos_store,
                trade_id=trade_id,
            )

        findings = _findings_by_agent(report)
        # All five agents must produce a finding (the crashed one as
        # verdict="unknown" with an error string).
        self.assertEqual(
            set(findings.keys()),
            {"signal", "execution", "sizing", "context", "process"},
        )
        crashed = findings["execution"]
        self.assertEqual(crashed.verdict, "unknown")
        self.assertIsNotNone(crashed.error)
        self.assertIn("simulated_agent_crash", str(crashed.error))

        # The other four must NOT be ``unknown`` because of the crash —
        # at least one should still produce a meaningful verdict
        # (innocent / contributing / primary_cause).
        non_crashed = [
            f for name, f in findings.items() if name != "execution"
        ]
        self.assertTrue(
            any(f.verdict != "unknown" for f in non_crashed),
            msg=(
                "expected at least one of the other four agents to return "
                f"a non-unknown verdict; got "
                f"{[(f.agent, f.verdict, f.error) for f in non_crashed]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
