"""LossPostmortemSynthesizer — closing agent of the Loss Postmortem Swarm (Lane E, E9).

The synthesizer is the orchestrator + classifier + feedback driver. It does
NOT do its own forensics; it spawns the five specialists (A1-A5), collects
their findings, classifies a single root cause, persists the report, and
fires up to four feedback channels.

Concurrency model
-----------------
Five specialists run in parallel via :class:`multiprocessing.Pool`. We
explicitly use the ``"spawn"`` start method so the workers do NOT inherit
already-loaded ML models (Lane A's predictor caches XGBoost boosters at
import time) — a fork-based pool would copy all that state into every
child and double or quadruple memory usage.

Per-agent timeout: 60 s wall-clock (already enforced inside
:class:`BaseForensicsAgent.safe_investigate`). The synthesizer adds an
outer timeout of ``per_agent_timeout_s + 5`` so a wedged worker process
(not just a wedged ``investigate()`` call) cannot stall the pool.

Pickling: each child worker receives only primitives (``trade_id``, a
config dict). The worker constructs its own :class:`TradeContextStore`
and agent instance — we never pickle a live Redis client or an agent
object across the process boundary.

Testing path
------------
Production: :meth:`process_one` constructs a multiprocessing pool and
fans out. For unit tests, callers can inject pre-built agents via the
``agent_factories`` constructor arg AND set
``use_multiprocessing=False`` — the in-process path runs the five
specialists serially and is fully mockable.

Root-cause classification
-------------------------
1. Count agents with ``verdict="primary_cause"``.
2. Exactly one agent in primary_cause → use that agent's category.
3. Two or more agents in primary_cause across DIFFERENT categories →
   ``"Mixed"``.
4. Zero in primary_cause AND zero non-unknown findings → ``"Unknown"``.
5. Zero in primary_cause but at least one ``"contributing"`` →
   ``"Unknown"`` (we deliberately don't promote contributing to root
   cause without a primary vote; the digest still surfaces them).

Feedback channels (D6 — all four)
---------------------------------
1. **OutcomeWeightAdjuster delta** — root-cause-driven nudge to the
   calibration weight. Mapping: Signal=-0.05, Sizing=-0.05, Mixed=-0.03,
   everything else 0. Bounded to ``|delta| ≤ 0.10``.
2. **Retrain queue** (``runs/retrain_queue.jsonl``) — append when 3+
   Signal-cause postmortems for the same symbol within the last 24 h.
3. **Risk-param recommender** (``runs/risk_recommendations.jsonl``) —
   append when 5+ Sizing-cause postmortems for the same symbol within
   the last 7 d. NEVER auto-applied; this file is for human review.
4. **Daily digest** (``daily_digest()``) — separate method called by
   ``live_supervisor.daily_close()``. Scans the past 24 h of
   postmortems, dispatches via ``alerts/notifier.py``.

File-side-effect contract
-------------------------
All file paths are configurable via the ``runs_dir`` constructor arg so
tests can redirect everything to a tmpdir. Defaults to ``./runs``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from loss_postmortem.base import (
    DEFAULT_AGENT_TIMEOUT_S,
    AgentName,
    BaseForensicsAgent,
    ForensicsFinding,
)

LOGGER = logging.getLogger(__name__)

# Mapping of agent_name -> RootCause label used in the synthesizer report.
_AGENT_TO_ROOT_CAUSE: Dict[str, str] = {
    "signal": "Signal",
    "execution": "Execution",
    "sizing": "Sizing",
    "context": "Context",
    "process": "Process",
}

# Weight-delta mapping per root cause. Operational issues (Execution,
# Context, Process) are explicitly 0 — they don't reflect on signal
# quality so the calibration weight should not move because of them.
_WEIGHT_DELTA_BY_ROOT_CAUSE: Dict[str, float] = {
    "Signal": -0.05,
    "Sizing": -0.05,
    "Execution": 0.0,
    "Context": 0.0,
    "Process": 0.0,
    "Mixed": -0.03,
    "Unknown": 0.0,
}

# Hard cap on weight delta — the test suite enforces this; we enforce it too.
WEIGHT_DELTA_MAX_ABS = 0.10

# Trigger thresholds for the retrain queue and risk recommender.
SIGNAL_CLUSTER_COUNT = 3
SIGNAL_CLUSTER_WINDOW_S = 24 * 3600  # 24h
SIZING_CLUSTER_COUNT = 5
SIZING_CLUSTER_WINDOW_S = 7 * 24 * 3600  # 7d

# Outer per-agent timeout buffer: adds 5s on top of the agent's own
# safe_investigate timeout to catch wedged worker processes (not just
# wedged investigate calls).
WORKER_TIMEOUT_BUFFER_S = 5.0


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------


@dataclass
class PostmortemReport:
    """Structured result of one postmortem investigation.

    Mirrors the schema in ``autopilot_loss_postmortem_swarm_plan_2026_05_08.md``
    plus a few operational fields the digest needs.
    """

    trade_id: str
    symbol: Optional[str]
    root_cause: str
    summary: str
    findings: List[ForensicsFinding]
    weight_delta: float
    loss_usd: Optional[float] = None
    loss_pct: Optional[float] = None
    duration_s: float = 0.0
    triggered_at_utc: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "loss_usd": self.loss_usd,
            "loss_pct": self.loss_pct,
            "triggered_at_utc": self.triggered_at_utc,
            "duration_s": self.duration_s,
            "findings": [f.to_dict() for f in self.findings],
            "synthesizer": {
                "root_cause": self.root_cause,
                "summary": self.summary,
                "actions": list(self.actions),
                "weight_adjustment_for_16": self.weight_delta,
            },
        }


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> Optional[_dt.datetime]:
    """Best-effort ISO-8601 parse. Returns ``None`` on failure."""
    if not ts or not isinstance(ts, str):
        return None
    # tolerate a trailing 'Z'
    raw = ts.rstrip("Z")
    try:
        if "T" not in raw:
            return None
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt
    except Exception:  # noqa: BLE001 - tolerate any malformed string
        return None


def _bound_weight_delta(delta: float) -> float:
    """Clip ``delta`` to ``[-WEIGHT_DELTA_MAX_ABS, +WEIGHT_DELTA_MAX_ABS]``."""
    try:
        d = float(delta)
    except (TypeError, ValueError):
        return 0.0
    if d != d:  # NaN
        return 0.0
    return max(-WEIGHT_DELTA_MAX_ABS, min(WEIGHT_DELTA_MAX_ABS, d))


def _classify_root_cause(findings: Sequence[ForensicsFinding]) -> str:
    """Deterministic root-cause classification.

    Algorithm:
    1. If every finding is verdict='unknown' → "Unknown".
    2. Collect agents with verdict='primary_cause'.
       - 0 → "Unknown" (contributing alone never promotes to root).
       - 1 → that agent's category.
       - >=2 across distinct categories → "Mixed".
       - >=2 but all the same category → that category (defensive — shouldn't
         happen since each agent runs once, but cheap to check).
    """
    if not findings:
        return "Unknown"

    # All unknown → Unknown.
    if all(f.verdict == "unknown" for f in findings):
        return "Unknown"

    primaries = [f for f in findings if f.verdict == "primary_cause"]
    if not primaries:
        return "Unknown"

    categories = {_AGENT_TO_ROOT_CAUSE.get(f.agent, "Unknown") for f in primaries}
    if len(categories) == 1:
        return next(iter(categories))
    return "Mixed"


def _summarize(
    *,
    root_cause: str,
    findings: Sequence[ForensicsFinding],
) -> str:
    """One-line plain-English summary for the JSON report and digest."""
    if root_cause == "Unknown":
        return "All agents returned unknown verdicts or no primary cause was identified."
    primaries = [f for f in findings if f.verdict == "primary_cause"]
    contributing = [f for f in findings if f.verdict == "contributing"]
    bits = [f"Root cause: {root_cause}."]
    if primaries:
        names = ", ".join(sorted({f.agent for f in primaries}))
        bits.append(f"Primary: {names}.")
    if contributing:
        names = ", ".join(sorted({f.agent for f in contributing}))
        bits.append(f"Contributing: {names}.")
    return " ".join(bits)


def _gather_actions(findings: Sequence[ForensicsFinding]) -> List[Dict[str, Any]]:
    """Pull every non-null suggested_action across findings.

    The synthesizer doesn't try to dedupe semantically — that's an audit
    decision a human should make. We just collect them for the report.
    """
    actions: List[Dict[str, Any]] = []
    for f in findings:
        if f.suggested_action:
            try:
                # Defensive copy; tests shouldn't get aliasing surprises.
                actions.append(dict(f.suggested_action))
            except (TypeError, ValueError):
                continue
    return actions


def _markdown_for_report(report: PostmortemReport) -> str:
    """Human-readable markdown digest of a single postmortem."""
    lines: List[str] = []
    lines.append(f"# Loss Postmortem — `{report.trade_id}`")
    lines.append("")
    lines.append(f"- **Symbol**: {report.symbol or '?'}")
    if report.loss_usd is not None:
        lines.append(f"- **Loss USD**: ${report.loss_usd:,.2f}")
    if report.loss_pct is not None:
        lines.append(f"- **Loss %**: {report.loss_pct * 100:.2f}%")
    lines.append(f"- **Triggered**: {report.triggered_at_utc}")
    lines.append(f"- **Investigation duration**: {report.duration_s:.2f}s")
    lines.append(f"- **Root cause**: {report.root_cause}")
    lines.append(f"- **Weight delta**: {report.weight_delta:+.4f}")
    lines.append("")
    lines.append(f"_{report.summary}_")
    lines.append("")
    lines.append("## Findings")
    for f in report.findings:
        lines.append("")
        lines.append(f"### {f.agent} — {f.verdict} (confidence {f.confidence:.2f})")
        if f.error:
            lines.append(f"_Error_: `{f.error}`")
        if f.evidence:
            for bullet in f.evidence:
                lines.append(f"- {bullet}")
        else:
            lines.append("_no evidence bullets emitted_")
        if f.suggested_action:
            lines.append("")
            lines.append(
                f"Suggested action: `{json.dumps(f.suggested_action, sort_keys=True)}`"
            )
    if report.actions:
        lines.append("")
        lines.append("## Aggregated actions")
        for a in report.actions:
            lines.append(f"- `{json.dumps(a, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# multiprocessing worker
# ---------------------------------------------------------------------------


def _default_agent_worker(payload: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """Top-level (picklable) worker that the spawn pool invokes.

    Receives ``(agent_name, trade_id, config)``; constructs the relevant
    agent in the child process (so we never pickle live Redis clients or
    ML-loaded objects), runs ``safe_investigate``, returns the finding as
    a dict (also picklable).

    ``config`` is a small dict used to rebuild a TradeContextStore in the
    child. Currently supports:
      - ``redis_url`` (str, optional)
      - ``namespace`` (str, optional)
      - ``timeout_s`` (float, optional)

    The default worker is intentionally minimal; production deployments
    that need richer agents (e.g., ContextForensicsAgent with a live
    Gemini caller) should subclass :class:`LossPostmortemSynthesizer`
    and override :attr:`worker_callable`.
    """
    agent_name, trade_id, config = payload
    try:
        from state.trade_context_store import TradeContextStore

        store = TradeContextStore(
            redis_url=config.get("redis_url") or "redis://localhost:6379/0",
            namespace=config.get("namespace") or "autopilot",
        )
        timeout_s = float(config.get("timeout_s") or DEFAULT_AGENT_TIMEOUT_S)

        if agent_name == "signal":
            from loss_postmortem.signal_forensics import SignalForensicsAgent
            agent = SignalForensicsAgent(context_store=store, timeout_s=timeout_s)
        elif agent_name == "execution":
            from loss_postmortem.execution_forensics import ExecutionForensicsAgent
            agent = ExecutionForensicsAgent(context_store=store, timeout_s=timeout_s)
        elif agent_name == "sizing":
            from loss_postmortem.sizing_forensics import SizingForensicsAgent
            agent = SizingForensicsAgent(context_store=store, timeout_s=timeout_s)
        elif agent_name == "context":
            from loss_postmortem.context_forensics import ContextForensicsAgent
            agent = ContextForensicsAgent(context_store=store, timeout_s=timeout_s)
        elif agent_name == "process":
            from loss_postmortem.process_integrity import ProcessIntegrityAgent
            agent = ProcessIntegrityAgent(context_store=store, timeout_s=timeout_s)
        else:
            return ForensicsFinding.unknown(
                agent="signal",  # placeholder; will be overwritten by caller
                error=f"unknown_agent_name:{agent_name!r}",
            ).to_dict()

        finding = agent.safe_investigate(trade_id)
        return finding.to_dict()
    except Exception as exc:  # noqa: BLE001 - report rather than crash the pool
        return ForensicsFinding.unknown(
            agent=agent_name if agent_name in {"signal", "execution", "sizing", "context", "process"} else "process",
            error=f"worker_crash:{exc!r}",
        ).to_dict()


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class LossPostmortemSynthesizer:
    """Closing orchestrator of the Loss Postmortem Swarm."""

    # Allow subclasses to override the worker callable.
    worker_callable: Callable[[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]] = staticmethod(
        _default_agent_worker
    )

    def __init__(
        self,
        *,
        runs_dir: Optional[str | Path] = None,
        outcome_weight_adjuster: Any = None,
        notifier: Any = None,
        position_store: Any = None,
        context_store: Any = None,
        redis_client: Any = None,
        worker_config: Optional[Mapping[str, Any]] = None,
        per_agent_timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        use_multiprocessing: bool = True,
        agent_factories: Optional[Mapping[str, Callable[[], BaseForensicsAgent]]] = None,
        pool_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else Path("runs")
        self.postmortem_dir = self.runs_dir / "postmortems"
        self.retrain_queue_path = self.runs_dir / "retrain_queue.jsonl"
        self.risk_recommendations_path = self.runs_dir / "risk_recommendations.jsonl"

        self.outcome_weight_adjuster = outcome_weight_adjuster
        self.notifier = notifier
        self.position_store = position_store
        self.context_store = context_store
        self.redis_client = redis_client

        self.worker_config: Dict[str, Any] = dict(worker_config or {})
        self.per_agent_timeout_s = float(per_agent_timeout_s)
        self.use_multiprocessing = bool(use_multiprocessing)

        # Optional in-process agent factories. Keyed by agent_name
        # ("signal","execution","sizing","context","process"). Each factory
        # returns a fresh agent instance; the synthesizer calls
        # ``safe_investigate`` on it. Used by tests to inject mocks.
        self.agent_factories: Dict[str, Callable[[], BaseForensicsAgent]] = (
            dict(agent_factories) if agent_factories else {}
        )

        # Optional pool factory for tests that want to assert the spawn
        # context was used. Defaults to multiprocessing.get_context("spawn").Pool.
        self._pool_factory = pool_factory

    # ------------------------------------------------------------------
    # queue draining
    # ------------------------------------------------------------------
    def drain(self, max_items: int = 10) -> int:
        """Drain up to ``max_items`` trade IDs from the postmortem queue.

        Reads from Redis list ``{ns}:postmortem:queue`` via RPOP (the
        producer LPUSHes, so RPOP yields FIFO order). Returns the number
        of postmortems processed.

        If no Redis client is wired or the queue is empty, returns 0
        with no side effects.
        """
        if max_items <= 0:
            return 0

        redis_client = self._resolve_redis_client()
        if redis_client is None:
            LOGGER.debug("synthesizer.drain: no redis client; skipping")
            return 0

        ns = self.worker_config.get("namespace") or "autopilot"
        key = f"{ns}:postmortem:queue"
        processed = 0
        for _ in range(max_items):
            try:
                trade_id = redis_client.rpop(key)
            except Exception as exc:  # noqa: BLE001 - queue is best-effort
                LOGGER.warning("postmortem queue rpop failed: %r", exc)
                break
            if trade_id is None:
                break
            if isinstance(trade_id, bytes):
                trade_id = trade_id.decode("utf-8", errors="replace")
            try:
                self.process_one(str(trade_id))
                processed += 1
            except Exception as exc:  # noqa: BLE001 - never let one bad trade halt the drain
                LOGGER.error(
                    "synthesizer.process_one(%r) crashed: %r",
                    trade_id,
                    exc,
                )
                continue
        return processed

    def _resolve_redis_client(self) -> Any:
        """Return a usable Redis client or None.

        Prefers the explicit ``redis_client`` constructor arg. Falls back
        to ``self.context_store._redis`` if a context store was injected.
        """
        if self.redis_client is not None:
            return self.redis_client
        store = self.context_store
        if store is not None:
            client = getattr(store, "_redis", None)
            if client is not None:
                return client
        return None

    # ------------------------------------------------------------------
    # single-trade processing
    # ------------------------------------------------------------------
    def process_one(self, trade_id: str) -> PostmortemReport:
        """Run the full swarm for ``trade_id`` and persist the report."""
        if not trade_id:
            raise ValueError("trade_id must be a non-empty string")

        start = time.monotonic()
        triggered_at = _utcnow_iso()

        findings = self._run_agents(trade_id)
        root_cause = _classify_root_cause(findings)
        summary = _summarize(root_cause=root_cause, findings=findings)
        actions = _gather_actions(findings)
        weight_delta = _bound_weight_delta(_WEIGHT_DELTA_BY_ROOT_CAUSE.get(root_cause, 0.0))

        symbol, loss_usd, loss_pct = self._lookup_position_metadata(trade_id)
        duration_s = time.monotonic() - start

        report = PostmortemReport(
            trade_id=trade_id,
            symbol=symbol,
            root_cause=root_cause,
            summary=summary,
            findings=list(findings),
            weight_delta=weight_delta,
            loss_usd=loss_usd,
            loss_pct=loss_pct,
            duration_s=duration_s,
            triggered_at_utc=triggered_at,
            actions=actions,
        )

        # Persist report (json + md). Best-effort — exceptions logged, not raised.
        self._write_report(report)

        # Drive feedback channels.
        self._apply_weight_delta(weight_delta)
        if root_cause == "Signal" and symbol:
            self._maybe_append_retrain_queue(symbol=symbol)
        if root_cause == "Sizing" and symbol:
            self._maybe_append_risk_recommendation(symbol=symbol)

        return report

    # ------------------------------------------------------------------
    # agent execution
    # ------------------------------------------------------------------
    def _run_agents(self, trade_id: str) -> List[ForensicsFinding]:
        """Run all five specialists; return findings in canonical order."""
        agent_order: Tuple[str, ...] = (
            "signal", "execution", "sizing", "context", "process",
        )

        if not self.use_multiprocessing:
            return self._run_agents_inprocess(trade_id, agent_order)
        return self._run_agents_multiprocessing(trade_id, agent_order)

    def _run_agents_inprocess(
        self,
        trade_id: str,
        agent_order: Sequence[str],
    ) -> List[ForensicsFinding]:
        out: List[ForensicsFinding] = []
        for name in agent_order:
            factory = self.agent_factories.get(name)
            if factory is None:
                # No factory wired; emit unknown.
                out.append(
                    ForensicsFinding.unknown(
                        agent=name,  # type: ignore[arg-type]
                        error="no_agent_factory_wired",
                    )
                )
                continue
            try:
                agent = factory()
            except Exception as exc:  # noqa: BLE001
                out.append(
                    ForensicsFinding.unknown(
                        agent=name,  # type: ignore[arg-type]
                        error=f"factory_crash:{exc!r}",
                    )
                )
                continue
            try:
                finding = agent.safe_investigate(trade_id)
                if not isinstance(finding, ForensicsFinding):
                    finding = ForensicsFinding.unknown(
                        agent=name,  # type: ignore[arg-type]
                        error=f"non_finding_return:{type(finding).__name__}",
                    )
                out.append(finding)
            except Exception as exc:  # noqa: BLE001 - safe_investigate should never raise
                out.append(
                    ForensicsFinding.unknown(
                        agent=name,  # type: ignore[arg-type]
                        error=f"safe_investigate_crash:{exc!r}",
                    )
                )
        return out

    def _run_agents_multiprocessing(
        self,
        trade_id: str,
        agent_order: Sequence[str],
    ) -> List[ForensicsFinding]:
        config = dict(self.worker_config)
        payloads: List[Tuple[str, str, Dict[str, Any]]] = [
            (name, trade_id, config) for name in agent_order
        ]

        # Outer worker timeout: agent's own timeout + buffer.
        outer_timeout = self.per_agent_timeout_s + WORKER_TIMEOUT_BUFFER_S

        results_by_agent: Dict[str, ForensicsFinding] = {}
        try:
            # Use spawn so workers don't inherit ML model state from the parent.
            ctx = multiprocessing.get_context("spawn")
            pool_factory = self._pool_factory or ctx.Pool
            with pool_factory(processes=len(agent_order)) as pool:
                async_results = [
                    (name, pool.apply_async(self.worker_callable, ((name, trade_id, config),)))
                    for name in agent_order
                ]
                for name, ar in async_results:
                    try:
                        raw = ar.get(timeout=outer_timeout)
                    except multiprocessing.TimeoutError:
                        results_by_agent[name] = ForensicsFinding.unknown(
                            agent=name,  # type: ignore[arg-type]
                            error="timeout",
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001
                        results_by_agent[name] = ForensicsFinding.unknown(
                            agent=name,  # type: ignore[arg-type]
                            error=f"pool_crash:{exc!r}",
                        )
                        continue
                    try:
                        finding = _finding_from_dict(raw)
                    except Exception as exc:  # noqa: BLE001
                        finding = ForensicsFinding.unknown(
                            agent=name,  # type: ignore[arg-type]
                            error=f"finding_decode_failed:{exc!r}",
                        )
                    results_by_agent[name] = finding
        except Exception as exc:  # noqa: BLE001 - if the pool itself dies, fall back to all-unknown
            LOGGER.error("multiprocessing pool failed: %r", exc)
            return [
                ForensicsFinding.unknown(
                    agent=name,  # type: ignore[arg-type]
                    error=f"pool_setup_failed:{exc!r}",
                )
                for name in agent_order
            ]

        return [
            results_by_agent.get(
                name,
                ForensicsFinding.unknown(
                    agent=name,  # type: ignore[arg-type]
                    error="no_result",
                ),
            )
            for name in agent_order
        ]

    # ------------------------------------------------------------------
    # position metadata lookup
    # ------------------------------------------------------------------
    def _lookup_position_metadata(
        self, trade_id: str
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Return ``(symbol, loss_usd, loss_pct)`` if available."""
        symbol: Optional[str] = None
        loss_usd: Optional[float] = None
        loss_pct: Optional[float] = None

        # Try position_store first (more authoritative for closed trades).
        if self.position_store is not None:
            try:
                pos = self.position_store.get(trade_id)
            except Exception:  # noqa: BLE001
                pos = None
            if pos is not None:
                symbol = getattr(pos, "symbol", None) or symbol
                pnl = getattr(pos, "realized_pnl_usd", None)
                if pnl is not None:
                    try:
                        loss_usd = float(pnl)
                    except (TypeError, ValueError):
                        loss_usd = None
                entry_quote = getattr(pos, "entry_quote_usd", None)
                if loss_usd is not None and entry_quote:
                    try:
                        loss_pct = float(loss_usd) / float(entry_quote)
                    except (TypeError, ValueError, ZeroDivisionError):
                        loss_pct = None

        # Fall back to context_store snapshots for symbol if needed.
        if symbol is None and self.context_store is not None:
            for phase in ("close", "breaker", "fill", "signal"):
                try:
                    snap = self.context_store.get_snapshot(trade_id, phase)
                except Exception:  # noqa: BLE001
                    snap = None
                if snap is not None:
                    symbol = getattr(snap, "symbol", None) or symbol
                    if symbol:
                        break
        return symbol, loss_usd, loss_pct

    # ------------------------------------------------------------------
    # output writing
    # ------------------------------------------------------------------
    def _write_report(self, report: PostmortemReport) -> None:
        try:
            self.postmortem_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("could not create postmortem dir %s: %r", self.postmortem_dir, exc)
            return

        json_path = self.postmortem_dir / f"{report.trade_id}.json"
        md_path = self.postmortem_dir / f"{report.trade_id}.md"

        try:
            json_path.write_text(
                json.dumps(report.to_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("failed writing %s: %r", json_path, exc)

        try:
            md_path.write_text(_markdown_for_report(report), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("failed writing %s: %r", md_path, exc)

    # ------------------------------------------------------------------
    # feedback channels
    # ------------------------------------------------------------------
    def _apply_weight_delta(self, weight_delta: float) -> None:
        if abs(weight_delta) < 1e-12:
            return  # no-op: don't churn the audit log on zero deltas
        if self.outcome_weight_adjuster is None:
            LOGGER.debug(
                "synthesizer: no outcome_weight_adjuster wired; skipping delta=%s",
                weight_delta,
            )
            return
        try:
            self.outcome_weight_adjuster.apply_postmortem_delta(weight_delta)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("apply_postmortem_delta failed: %r", exc)

    def _recent_postmortems(
        self,
        *,
        within_seconds: int,
        root_cause_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Scan ``runs/postmortems/`` for matching postmortems.

        Filters by root_cause and symbol if provided. ``within_seconds``
        is measured against ``triggered_at_utc`` if parseable; postmortems
        with unparseable timestamps are EXCLUDED conservatively (they
        could be stale).
        """
        out: List[Dict[str, Any]] = []
        if not self.postmortem_dir.exists():
            return out
        now = _dt.datetime.now(_dt.timezone.utc)
        cutoff = now - _dt.timedelta(seconds=within_seconds)
        for path in sorted(self.postmortem_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            synth = data.get("synthesizer") or {}
            rc = synth.get("root_cause")
            if root_cause_filter and rc != root_cause_filter:
                continue
            if symbol_filter and data.get("symbol") != symbol_filter:
                continue
            ts = _parse_iso(data.get("triggered_at_utc") or "")
            if ts is None or ts < cutoff:
                continue
            out.append(data)
        return out

    def _maybe_append_retrain_queue(self, *, symbol: str) -> None:
        """If 3+ Signal-cause postmortems for ``symbol`` in 24h, append entry."""
        recent = self._recent_postmortems(
            within_seconds=SIGNAL_CLUSTER_WINDOW_S,
            root_cause_filter="Signal",
            symbol_filter=symbol,
        )
        if len(recent) < SIGNAL_CLUSTER_COUNT:
            return

        ids = [r.get("trade_id") for r in recent if r.get("trade_id")]
        entry = {
            "symbol": symbol,
            "reason": "signal_cause_cluster",
            "postmortem_ids": ids,
            "created_at_utc": _utcnow_iso(),
        }
        self._append_jsonl(self.retrain_queue_path, entry)

    def _maybe_append_risk_recommendation(self, *, symbol: str) -> None:
        """If 5+ Sizing-cause postmortems for ``symbol`` in 7d, append entry."""
        recent = self._recent_postmortems(
            within_seconds=SIZING_CLUSTER_WINDOW_S,
            root_cause_filter="Sizing",
            symbol_filter=symbol,
        )
        if len(recent) < SIZING_CLUSTER_COUNT:
            return

        ids = [r.get("trade_id") for r in recent if r.get("trade_id")]
        # The synthesizer doesn't set a specific param; this file is for
        # human review. We surface the symbol + evidence trail and let
        # the operator decide what knob to turn.
        entry = {
            "symbol": symbol,
            "param": "kelly_fraction",
            "current_value": None,
            "proposed_value": None,
            "evidence": ids,
            "created_at_utc": _utcnow_iso(),
        }
        self._append_jsonl(self.risk_recommendations_path, entry)

    def _append_jsonl(self, path: Path, entry: Mapping[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(entry), sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not append to %s: %r", path, exc)

    # ------------------------------------------------------------------
    # daily digest
    # ------------------------------------------------------------------
    def daily_digest(self, *, dispatch: bool = True) -> str:
        """Build a 24h digest string and (optionally) dispatch via notifier.

        Returns the formatted digest string regardless of whether the
        notifier is configured; the side-effect dispatch is best-effort.
        """
        recent = self._recent_postmortems(within_seconds=24 * 3600)
        # Top-3 losses by absolute USD.
        with_loss = [
            r for r in recent if isinstance(r.get("loss_usd"), (int, float))
        ]
        with_loss.sort(key=lambda r: float(r.get("loss_usd") or 0.0))  # most negative first
        top3 = with_loss[:3]

        # Root-cause distribution.
        rc_counts: Dict[str, int] = {}
        for r in recent:
            rc = (r.get("synthesizer") or {}).get("root_cause") or "Unknown"
            rc_counts[rc] = rc_counts.get(rc, 0) + 1

        # New retrain entries + risk recommendations from the past 24h.
        new_retrain = self._read_jsonl_recent(self.retrain_queue_path, hours=24)
        new_risk = self._read_jsonl_recent(self.risk_recommendations_path, hours=24)

        lines: List[str] = []
        lines.append("Loss Postmortem Daily Digest (last 24h)")
        lines.append(f"Investigated: {len(recent)} losses")
        if rc_counts:
            dist = ", ".join(
                f"{k}={v}" for k, v in sorted(rc_counts.items())
            )
            lines.append(f"Root-cause distribution: {dist}")
        else:
            lines.append("Root-cause distribution: (none)")
        if top3:
            lines.append("")
            lines.append("Top 3 losses by USD:")
            for r in top3:
                tid = r.get("trade_id", "?")
                sym = r.get("symbol") or "?"
                loss = r.get("loss_usd")
                rc = (r.get("synthesizer") or {}).get("root_cause") or "?"
                loss_str = f"${loss:,.2f}" if isinstance(loss, (int, float)) else "?"
                lines.append(f"  - {tid} ({sym}): {loss_str} — {rc}")
        if new_retrain:
            lines.append("")
            lines.append(f"New retrain queue entries: {len(new_retrain)}")
            for e in new_retrain:
                lines.append(
                    f"  - {e.get('symbol')}: {e.get('reason')} ({len(e.get('postmortem_ids') or [])} ids)"
                )
        if new_risk:
            lines.append("")
            lines.append(f"New risk recommendations (HUMAN REVIEW): {len(new_risk)}")
            for e in new_risk:
                lines.append(
                    f"  - {e.get('symbol')}: param={e.get('param')} ({len(e.get('evidence') or [])} ids)"
                )
        digest = "\n".join(lines)

        if dispatch:
            self._dispatch_digest(digest, rc_counts=rc_counts)
        return digest

    def _read_jsonl_recent(self, path: Path, *, hours: int) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=hours)
        out: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    ts = _parse_iso(entry.get("created_at_utc") or "")
                    if ts is None or ts < cutoff:
                        continue
                    out.append(entry)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not read %s: %r", path, exc)
        return out

    def _dispatch_digest(
        self,
        digest: str,
        *,
        rc_counts: Mapping[str, int],
    ) -> None:
        if self.notifier is None:
            LOGGER.debug("synthesizer.daily_digest: no notifier wired; logging only")
            LOGGER.info("postmortem_daily_digest:\n%s", digest)
            return
        try:
            # Prefer .alert (action-required visibility) — the digest is
            # operationally meaningful even on quiet days.
            alert_fn = getattr(self.notifier, "alert", None)
            if callable(alert_fn):
                alert_fn(digest)
                return
            info_fn = getattr(self.notifier, "info", None)
            if callable(info_fn):
                info_fn(digest)
                return
            LOGGER.warning(
                "synthesizer.daily_digest: notifier %r has no .alert or .info; logging only",
                self.notifier,
            )
            LOGGER.info("postmortem_daily_digest:\n%s", digest)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("notifier dispatch failed: %r", exc)


# ---------------------------------------------------------------------------
# helpers used by both the synthesizer and the workers
# ---------------------------------------------------------------------------


def _finding_from_dict(raw: Mapping[str, Any]) -> ForensicsFinding:
    """Reconstruct a :class:`ForensicsFinding` from a worker's dict return.

    The dict is the output of :meth:`ForensicsFinding.to_dict`. We use
    keyword construction so future fields land cleanly.
    """
    return ForensicsFinding(
        agent=raw["agent"],
        verdict=raw["verdict"],
        confidence=raw.get("confidence", 0.0),
        evidence=list(raw.get("evidence") or []),
        suggested_action=raw.get("suggested_action"),
        severity=raw.get("severity", 1),
        runtime_s=raw.get("runtime_s", 0.0),
        error=raw.get("error"),
    )


# Public alias (the brief allows either name).
Synthesizer = LossPostmortemSynthesizer


__all__ = [
    "LossPostmortemSynthesizer",
    "PostmortemReport",
    "Synthesizer",
    "WEIGHT_DELTA_MAX_ABS",
    "SIGNAL_CLUSTER_COUNT",
    "SIGNAL_CLUSTER_WINDOW_S",
    "SIZING_CLUSTER_COUNT",
    "SIZING_CLUSTER_WINDOW_S",
]
