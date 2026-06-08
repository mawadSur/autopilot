"""Base class + finding dataclass for the loss-postmortem forensics swarm.

This module defines the contract every specialist agent (Signal, Execution,
Sizing, Context, Process) must satisfy:

1. :class:`ForensicsFinding` is the structured verdict an agent returns. It
   serializes to JSON via :meth:`ForensicsFinding.to_dict` so the
   synthesizer can dump it into ``runs/postmortems/{trade_id}.json``.
2. :class:`BaseForensicsAgent` is the abstract base. Subclasses override
   :meth:`investigate`. The base class provides two safety wrappers —
   :meth:`_safe_run` (catches all exceptions, returns ``verdict="unknown"``)
   and :meth:`_with_timeout` (wall-clock limit, returns ``verdict="unknown"``
   on timeout). Both are deliberate: a buggy or slow agent must not crash
   the whole swarm.

The five specialist agents and the synthesizer glue ship in the next round.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Set, TypeVar

from state.trade_context_store import TradeContextStore

LOGGER = logging.getLogger(__name__)

# Agent identity: which forensic specialist produced the finding.
AgentName = Literal["signal", "execution", "sizing", "context", "process"]

# Finding verdict: one of the four canonical conclusions.
Verdict = Literal["innocent", "contributing", "primary_cause", "unknown"]

_VALID_AGENTS: Set[str] = {
    "signal", "execution", "sizing", "context", "process",
}
_VALID_VERDICTS: Set[str] = {
    "innocent", "contributing", "primary_cause", "unknown",
}

# Default per-agent investigation timeout. LLM-driven agents can run a few
# seconds; 60s is the bound the synthesizer enforces. Tunable per subclass
# if a specific agent has tighter or looser bounds.
DEFAULT_AGENT_TIMEOUT_S = 60.0

# Cap evidence string length to keep digests scannable.
EVIDENCE_MAX_LEN = 200

T = TypeVar("T")


# ---------------------------------------------------------------------------
# ForensicsFinding
# ---------------------------------------------------------------------------


@dataclass
class ForensicsFinding:
    """A single forensic agent's verdict on one losing trade.

    Fields
    ------
    agent
        Which specialist produced this finding.
    verdict
        Ranked conclusion. ``"innocent"`` = this agent found no contribution
        from its domain; ``"contributing"`` = its domain helped cause the
        loss but isn't the headline; ``"primary_cause"`` = its domain is
        the headline; ``"unknown"`` = the agent couldn't decide (used by
        the safety wrappers when an agent crashes or times out).
    confidence
        In [0, 1]. The synthesizer weights findings by confidence when
        breaking ties between two ``primary_cause`` verdicts from
        different domains.
    evidence
        Bullet-point reasoning, each ≤ 200 chars (longer strings are
        truncated by the dataclass __post_init__).
    suggested_action
        Optional structured action proposal. Examples:
            {"type": "raise_floor", "target": "supervisor.confidence_floor",
             "from": 0.6, "to": 0.65, "scope": "BTC/USD only"}
            {"type": "feature_request", "feature": "regime_distance"}
            {"type": "retrain", "symbol": "BTC/USD", "reason": "feature drift"}
    severity
        1-5. The synthesizer uses this for digest ordering.
    runtime_s
        Wall-clock seconds the agent took. Useful for spotting slow agents.
    error
        Set by safety wrappers when the agent crashed/timed out. Always
        paired with ``verdict="unknown"``.
    """

    agent: AgentName
    verdict: Verdict
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    suggested_action: Optional[Dict[str, Any]] = None
    severity: int = 1
    runtime_s: float = 0.0
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.agent not in _VALID_AGENTS:
            raise ValueError(
                f"agent must be one of {sorted(_VALID_AGENTS)!r}, got {self.agent!r}"
            )
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(_VALID_VERDICTS)!r}, "
                f"got {self.verdict!r}"
            )
        # Coerce + clamp confidence into [0, 1] but preserve NaN handling
        # safely: a NaN confidence is recorded as 0.0 (defensive — the
        # synthesizer treats 0.0 as "agent gave no signal").
        try:
            c = float(self.confidence)
        except (TypeError, ValueError):
            c = 0.0
        if c != c:  # NaN check
            c = 0.0
        self.confidence = max(0.0, min(1.0, c))

        if not isinstance(self.severity, int):
            try:
                self.severity = int(self.severity)
            except (TypeError, ValueError):
                self.severity = 1
        self.severity = max(1, min(5, self.severity))

        # Clamp runtime to non-negative; coerce to float.
        try:
            self.runtime_s = max(0.0, float(self.runtime_s))
        except (TypeError, ValueError):
            self.runtime_s = 0.0

        # Truncate evidence strings + drop non-strings.
        cleaned: List[str] = []
        for item in (self.evidence or []):
            if not isinstance(item, str):
                item = str(item)
            if len(item) > EVIDENCE_MAX_LEN:
                item = item[: EVIDENCE_MAX_LEN - 1] + "…"
            cleaned.append(item)
        self.evidence = cleaned

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict — all fields preserved."""
        return asdict(self)

    @classmethod
    def unknown(
        cls,
        *,
        agent: AgentName,
        error: str,
        runtime_s: float = 0.0,
    ) -> "ForensicsFinding":
        """Construct a verdict-unknown finding for the safety wrappers."""
        return cls(
            agent=agent,
            verdict="unknown",
            confidence=0.0,
            evidence=[],
            suggested_action=None,
            severity=1,
            runtime_s=runtime_s,
            error=error,
        )


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


class _TimeoutError(Exception):
    """Internal: raised when a forensic agent exceeds its wall-clock budget."""


def _run_with_timeout(
    fn: Callable[[], T],
    *,
    timeout_s: float,
) -> T:
    """Run ``fn`` in a daemon thread; raise :class:`_TimeoutError` if too slow.

    We use a thread (not signal.alarm) because:
    - signal-based timeouts only work on the main thread on Unix and break
      multi-process setups (the synthesizer eventually runs agents in
      child processes).
    - We need the agent's call site to be the only thing affected if it
      hangs — a thread is naturally scoped.

    The thread is daemon so a runaway agent doesn't block process exit.
    """
    result_box: Dict[str, Any] = {}
    exc_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surface to outer
            exc_box["error"] = exc

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout=max(0.001, float(timeout_s)))
    if th.is_alive():
        raise _TimeoutError(f"agent exceeded {timeout_s:.1f}s timeout")
    if "error" in exc_box:
        raise exc_box["error"]
    return result_box["value"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# BaseForensicsAgent
# ---------------------------------------------------------------------------


class BaseForensicsAgent:
    """Abstract base for forensics specialists.

    Subclasses set :attr:`agent_name` and override :meth:`investigate`.
    External callers (the synthesizer) should invoke :meth:`safe_investigate`
    rather than :meth:`investigate` directly so timeouts and exceptions
    convert to ``verdict="unknown"`` findings instead of propagating.

    Construction: every subclass receives the snapshot store so it can
    pull ``signal`` / ``fill`` / ``breaker`` snapshots for the trade under
    investigation. Subclasses are free to add extra constructor args.
    """

    # Subclasses override.
    agent_name: AgentName = "signal"  # placeholder; subclass MUST override

    def __init__(
        self,
        *,
        context_store: TradeContextStore,
        timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
    ) -> None:
        self.context_store = context_store
        self.timeout_s = float(timeout_s)

    # ------------------------------------------------------------------
    # contract
    # ------------------------------------------------------------------
    def investigate(self, trade_id: str) -> ForensicsFinding:
        """Inspect the snapshots for ``trade_id`` and return a finding.

        Default implementation: produce a placeholder ``"innocent"``
        finding. Subclasses MUST override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override .investigate()"
        )

    # ------------------------------------------------------------------
    # safety wrappers
    # ------------------------------------------------------------------
    def safe_investigate(self, trade_id: str) -> ForensicsFinding:
        """Run :meth:`investigate` with timeout + crash protection.

        On timeout: returns ``verdict="unknown"`` with ``error="timeout"``.
        On any other exception: returns ``verdict="unknown"`` with
        ``error=repr(exc)``.
        """
        import time

        start = time.monotonic()
        try:
            finding = _run_with_timeout(
                lambda: self.investigate(trade_id),
                timeout_s=self.timeout_s,
            )
        except _TimeoutError:
            elapsed = time.monotonic() - start
            LOGGER.warning(
                "%s timed out investigating trade_id=%s after %.2fs",
                type(self).__name__,
                trade_id,
                elapsed,
            )
            return ForensicsFinding.unknown(
                agent=self.agent_name,
                error="timeout",
                runtime_s=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - whole point of the wrapper
            elapsed = time.monotonic() - start
            LOGGER.warning(
                "%s crashed investigating trade_id=%s: %r",
                type(self).__name__,
                trade_id,
                exc,
            )
            return ForensicsFinding.unknown(
                agent=self.agent_name,
                error=repr(exc),
                runtime_s=elapsed,
            )

        # Subclass returned something unexpected; coerce into an unknown.
        if not isinstance(finding, ForensicsFinding):
            elapsed = time.monotonic() - start
            return ForensicsFinding.unknown(
                agent=self.agent_name,
                error=f"investigate returned non-ForensicsFinding: {type(finding).__name__}",
                runtime_s=elapsed,
            )
        # Stamp the runtime if subclass didn't.
        if finding.runtime_s <= 0.0:
            finding.runtime_s = time.monotonic() - start
        return finding


__all__ = [
    "AgentName",
    "BaseForensicsAgent",
    "DEFAULT_AGENT_TIMEOUT_S",
    "EVIDENCE_MAX_LEN",
    "ForensicsFinding",
    "Verdict",
]
