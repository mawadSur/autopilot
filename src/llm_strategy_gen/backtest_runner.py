"""BacktestRunner — third stage of the E3 loop (SKELETON).

Given a :class:`FeatureProposal`, runs a Sharpe-gate backtest and
returns a :class:`BacktestResult`. Intended interface is "real" but
the *current implementation* uses a deterministic stub for the
"sharpe_with_proposal" computation — a future PR replaces it with
real retraining on the dataset.

SECURITY WARNING — UNTRUSTED CODE EXECUTION
-------------------------------------------
The proposal's ``python_code`` is provided by an LLM. We currently
guard it with two layers ONLY:

1. ``ast.parse`` (already done in :class:`FeatureProposalGenerator`).
2. :func:`_validate_proposal_safety` — an AST-walk that rejects
   forbidden imports (``os``, ``subprocess``, ``shutil``, ``socket``,
   ``urllib``, ``requests``, ``pickle``, ``ctypes``, ``sys``, ``pathlib``,
   ``builtins``) and forbidden function calls (``__import__``,
   ``compile``, ``exec``, ``eval``, ``open``, ``getattr``, ``setattr``,
   ``delattr``, ``input``).

This is **not** a sandbox. It would not stop a determined adversary
who learns the rejection rules. Production deployment of this loop
requires real isolation — ideas in ``INTEGRATION.md``:

* Run ``exec`` inside an ephemeral Docker container with no network,
  read-only filesystem, dropped capabilities, and a strict CPU/memory
  budget.
* Or use a restricted-Python runtime (RestrictedPython) with explicit
  builtins allowlist.
* Until then, **never run this loop autonomously against an LLM API
  key with admin credentials**, and **never auto-merge** the PRs it
  opens.

The stub backtest
-----------------
Until real retraining is wired, ``run()`` does the following:

1. Validates the proposal's Python code passes
   :func:`_validate_proposal_safety`. If not, returns a
   ``BacktestResult`` with ``passed_gate=False`` and a note.
2. Computes a baseline Sharpe — for the skeleton this is read from
   ``baseline_meta_path`` (a JSON file with ``sharpe_baseline``); if
   absent, defaults to 0.0.
3. Computes a stub "Sharpe with proposal" by deterministically hashing
   the proposal's ``name + python_code`` to a number in
   ``[-0.1, +0.4]`` so the same proposal always produces the same
   number. We use a hash so tests and operators get a repeatable
   result without random flakiness.
4. Returns ``passed_gate = (sharpe_with - sharpe_baseline) >
   gate_threshold``.

Consumers should read the docstring carefully and treat
``BacktestResult`` as a **placeholder** — not a real Sharpe lift
estimate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from llm_strategy_gen.feature_proposal import FeatureProposal

LOGGER = logging.getLogger(__name__)


# Sharpe-lift threshold below which a proposal is rejected. Configurable
# via ``BacktestRunner(gate_threshold=...)`` and (in production) the
# ``AUTOPILOT_LLM_STRATEGY_GATE`` env var (consumed by WeeklyJob, not here).
DEFAULT_GATE_THRESHOLD = 0.2

# Forbidden modules / function calls. Conservative: anything that can
# touch the host filesystem, network, or subprocess gets rejected.
_FORBIDDEN_MODULES: Set[str] = frozenset({
    "os", "subprocess", "shutil", "socket", "urllib", "urllib2", "urllib3",
    "requests", "pickle", "ctypes", "sys", "pathlib", "builtins",
    "importlib", "marshal", "shelve", "tempfile", "io",
    "asyncio", "multiprocessing", "threading",
})
_FORBIDDEN_CALLS: Set[str] = frozenset({
    "__import__", "compile", "exec", "eval", "open", "input",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "breakpoint", "help",
})

# Stub backtest produces values in this range — wide enough that some
# proposals clear the default 0.2 gate, narrow enough that not every
# proposal does.
_STUB_LIFT_MIN = -0.1
_STUB_LIFT_MAX = 0.4


@dataclass(frozen=True)
class BacktestResult:
    """Result of running one proposal against the (stub) backtest.

    Attributes
    ----------
    proposal:
        The :class:`FeatureProposal` that was tested.
    sharpe_baseline:
        Sharpe of the baseline strategy, before adding the proposed
        feature. Currently read from a JSON file; future PR will compute
        it from a real backtest run.
    sharpe_with_proposal:
        Sharpe of the strategy with the proposed feature added. The
        SKELETON uses a deterministic stub keyed off proposal content;
        future PR replaces this with real retraining + backtest.
    sharpe_lift:
        ``sharpe_with_proposal - sharpe_baseline``.
    passed_gate:
        ``True`` iff ``sharpe_lift > gate_threshold`` AND the proposal
        passed the safety check. ``False`` for unsafe proposals
        regardless of stub Sharpe.
    gate_threshold:
        Threshold the runner used. Recorded so the downstream PR
        description can quote it.
    notes:
        Operational notes — surfaced in the PR description and the
        WeeklyJob log. Currently used for the safety-rejection reason
        and the explicit "stub backtest" warning.
    """

    proposal: FeatureProposal
    sharpe_baseline: float
    sharpe_with_proposal: float
    sharpe_lift: float
    passed_gate: bool
    gate_threshold: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "sharpe_baseline": float(self.sharpe_baseline),
            "sharpe_with_proposal": float(self.sharpe_with_proposal),
            "sharpe_lift": float(self.sharpe_lift),
            "passed_gate": bool(self.passed_gate),
            "gate_threshold": float(self.gate_threshold),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# safety check
# ---------------------------------------------------------------------------


def _walk_imports(tree: ast.AST) -> Iterable[str]:
    """Yield every imported module name (top-level component)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0]


def _walk_call_names(tree: ast.AST) -> Iterable[str]:
    """Yield function names from ``Call`` nodes whose func is a Name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node.func.id


def _walk_attribute_accesses(tree: ast.AST) -> Iterable[str]:
    """Yield ``foo.bar`` attribute accesses where ``foo`` is a top-level Name.

    We use this to catch ``os.system`` even when ``os`` was not imported
    explicitly (e.g., re-bound via ``__builtins__`` or accessed through
    a fallback import the LLM hid from us). Returns just the *root* name
    part (``os`` from ``os.system``).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                yield base.id


def _validate_proposal_safety(code: str) -> tuple[bool, str]:
    """Lightweight AST-only safety check on proposal code.

    Returns ``(ok, reason)``. ``ok=True`` means the code passed our
    rejection rules; ``ok=False`` means we found a forbidden construct
    and the caller should NOT execute the code.

    This is NOT a sandbox — see module docstring. Production deployment
    must add real isolation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error: {exc.msg}"

    for module in _walk_imports(tree):
        if module in _FORBIDDEN_MODULES:
            return False, f"forbidden_import: {module}"

    for call_name in _walk_call_names(tree):
        if call_name in _FORBIDDEN_CALLS:
            return False, f"forbidden_call: {call_name}"

    # Catch ``os.system`` / ``subprocess.run`` even without explicit imports.
    for root in _walk_attribute_accesses(tree):
        if root in _FORBIDDEN_MODULES:
            return False, f"forbidden_attribute_root: {root}"

    return True, ""


# ---------------------------------------------------------------------------
# stub backtest
# ---------------------------------------------------------------------------


def _stub_sharpe_with_proposal(
    *, baseline: float, proposal: FeatureProposal
) -> float:
    """Deterministic stub for "Sharpe with this proposal" — NOT REAL.

    Hashes ``proposal.name + python_code`` into a stable lift in
    ``[_STUB_LIFT_MIN, _STUB_LIFT_MAX]`` and returns
    ``baseline + lift``. Same proposal text → same number, every run.

    Replace this with real retraining + backtest in the follow-up PR.
    """
    digest = hashlib.sha256(
        (proposal.name + "\0" + proposal.python_code).encode("utf-8")
    ).digest()
    # 8 bytes -> uint64 -> [0, 1)
    n = int.from_bytes(digest[:8], "big") / 2**64
    span = _STUB_LIFT_MAX - _STUB_LIFT_MIN
    lift = _STUB_LIFT_MIN + n * span
    return float(baseline + lift)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


class BacktestRunner:
    """Run a Sharpe-gate stub backtest on a proposed feature.

    Parameters
    ----------
    dataset_path:
        Path to a parquet dataset. The skeleton does NOT read from it
        directly (the stub backtest doesn't need to); it's stored so
        the future real-retraining implementation has the path it
        needs without an interface change.
    gate_threshold:
        Sharpe lift threshold below which a proposal is rejected.
        Default 0.2 — tunable via the ``AUTOPILOT_LLM_STRATEGY_GATE``
        env var at the WeeklyJob level.
    """

    def __init__(
        self,
        *,
        dataset_path: Path,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.gate_threshold = float(gate_threshold)

    def run(
        self,
        proposal: FeatureProposal,
        baseline_meta_path: Optional[Path] = None,
    ) -> BacktestResult:
        """Return a :class:`BacktestResult` for ``proposal``.

        Order of operations:

        1. Safety-check the proposal code. If it fails, return a
           ``passed_gate=False`` result immediately — we do NOT run the
           stub backtest on unsafe code (defense in depth even though
           we never ``exec`` the code in the skeleton path).
        2. Read baseline Sharpe from ``baseline_meta_path`` JSON.
        3. Compute stub "Sharpe with proposal".
        4. Compare lift against gate threshold; build result.
        """
        ok, reason = _validate_proposal_safety(proposal.python_code)
        if not ok:
            LOGGER.warning(
                "BacktestRunner: rejecting %r on safety grounds: %s",
                proposal.name,
                reason,
            )
            return BacktestResult(
                proposal=proposal,
                sharpe_baseline=0.0,
                sharpe_with_proposal=0.0,
                sharpe_lift=0.0,
                passed_gate=False,
                gate_threshold=self.gate_threshold,
                notes=f"safety_reject: {reason}",
            )

        baseline = self._read_baseline_sharpe(baseline_meta_path)
        sharpe_with = _stub_sharpe_with_proposal(
            baseline=baseline, proposal=proposal
        )
        lift = sharpe_with - baseline
        passed = lift > self.gate_threshold

        return BacktestResult(
            proposal=proposal,
            sharpe_baseline=baseline,
            sharpe_with_proposal=sharpe_with,
            sharpe_lift=lift,
            passed_gate=passed,
            gate_threshold=self.gate_threshold,
            notes=(
                "STUB backtest — sharpe_with_proposal is a deterministic hash, "
                "not a real retraining result. Replace before production use."
            ),
        )

    def _read_baseline_sharpe(self, baseline_meta_path: Optional[Path]) -> float:
        if baseline_meta_path is None:
            return 0.0
        path = Path(baseline_meta_path)
        if not path.exists():
            LOGGER.info(
                "BacktestRunner: baseline meta %s missing; using 0.0", path
            )
            return 0.0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning(
                "BacktestRunner: could not read baseline meta %s (%r); using 0.0",
                path,
                exc,
            )
            return 0.0
        if not isinstance(data, dict):
            return 0.0
        try:
            return float(data.get("sharpe_baseline", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0


__all__ = [
    "BacktestResult",
    "BacktestRunner",
    "DEFAULT_GATE_THRESHOLD",
    "_validate_proposal_safety",  # exported for tests
]
