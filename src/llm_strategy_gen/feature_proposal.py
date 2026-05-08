"""FeatureProposalGenerator — second stage of the E3 loop (SKELETON).

Given a list of :class:`OutcomePattern` from :class:`OutcomeAnalyst`,
asks the LLM (Gemini, via an injected caller) to propose ONE Python
feature per pattern. Each proposal is a dict the caller turns into a
:class:`FeatureProposal`:

.. code-block:: json

    {
      "name": "high_vol_regime_atr_z",
      "description": "Z-score of ATR vs 240-bar baseline",
      "python_code": "def feature(df):\\n    ...",
      "expected_lift_sharpe": 0.18,
      "risk_notes": "May be unstable when ATR window has gaps"
    }

JSON parsing reuses :func:`utils.extract_json_object`, the same helper
the existing scanner / calibration agents use, so model output with
Markdown fences or stray prose is recovered correctly.

Each proposal's ``python_code`` is parsed with :mod:`ast` BEFORE we
return it. Anything that fails ``ast.parse`` is rejected. We do NOT
execute the code here — that's the responsibility of
:class:`BacktestRunner`, which adds the (still limited) safety check
on top.

Limits
------
* Hard cap on proposals returned (default 3, configurable).
* Hard cap on patterns processed (matches max_proposals — we don't
  send the LLM more clusters than we'll keep).
* Per-call timeout configurable; defaults to 30s.
* Graceful degradation on every failure mode (timeout, no API key,
  malformed JSON, syntax error) — we log and skip, never raise to
  the caller.
"""

from __future__ import annotations

import ast
import datetime as _dt
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from utils import extract_json_object

from llm_strategy_gen.outcome_analyst import OutcomePattern

LOGGER = logging.getLogger(__name__)


DEFAULT_MAX_PROPOSALS = 3
DEFAULT_LLM_TIMEOUT_S = 30
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class FeatureProposal:
    """One LLM-proposed feature ready for backtest gating.

    Attributes
    ----------
    name:
        Short snake_case identifier — used as the slug for the
        per-proposal artifact path.
    description:
        Plain-English description, one or two sentences.
    python_code:
        Raw Python source. Validated with ``ast.parse`` at construction
        time; downstream :class:`BacktestRunner` adds an additional
        unsafe-import / unsafe-call AST inspection before ``exec``.
    expected_lift_sharpe:
        LLM's predicted Sharpe lift over baseline. Treated as a hint;
        the real backtest is the source of truth.
    risk_notes:
        LLM's notes on potential regressions, edge cases, or training-
        data overfit risks. Logged into the PR description.
    proposed_at_utc:
        ISO-8601 timestamp recording when the proposal was generated.
    """

    name: str
    description: str
    python_code: str
    expected_lift_sharpe: float
    risk_notes: str
    proposed_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "python_code": self.python_code,
            "expected_lift_sharpe": float(self.expected_lift_sharpe),
            "risk_notes": self.risk_notes,
            "proposed_at_utc": self.proposed_at_utc,
        }


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _python_code_parses(code: str) -> bool:
    """Return True iff ``code`` is syntactically valid Python."""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class FeatureProposalGenerator:
    """Ask the LLM to propose Python features for the patterns we found.

    Parameters
    ----------
    llm_caller:
        Optional injectable callable
        ``(prompt: str, *, timeout_s: int) -> Dict[str, Any]``. The
        return value is fed directly through
        :func:`utils.extract_json_object`-compatible parsing if it's a
        string, or used as-is if it's already a dict. ``None`` means
        "no LLM available, return empty list" (with a warning).
    gemini_model:
        Hint passed through to the caller via prompt; we do NOT enforce
        it here.
    max_proposals:
        Maximum number of proposals returned per :meth:`propose` call.
        Also caps how many patterns we hand to the LLM.
    timeout_s:
        Per-call LLM timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        *,
        llm_caller: Optional[Callable[..., Any]] = None,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        timeout_s: int = DEFAULT_LLM_TIMEOUT_S,
    ) -> None:
        self.llm_caller = llm_caller
        self.gemini_model = str(gemini_model)
        self.max_proposals = max(1, int(max_proposals))
        self.timeout_s = int(timeout_s)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def propose(self, patterns: List[OutcomePattern]) -> List[FeatureProposal]:
        """Return up to ``max_proposals`` validated feature proposals.

        Returns an empty list (no exception) if:

        * ``patterns`` is empty,
        * no LLM caller was injected,
        * the API key is missing AND the caller doesn't opt in via
          ``_test_inject``,
        * every LLM call fails or returns invalid Python.
        """
        if not patterns:
            return []
        if self.llm_caller is None:
            LOGGER.warning("FeatureProposalGenerator: no llm_caller wired; returning []")
            return []

        if (
            os.getenv("GEMINI_API_KEY") is None
            and os.getenv("GOOGLE_API_KEY") is None
            and not getattr(self.llm_caller, "_test_inject", False)
        ):
            LOGGER.warning(
                "FeatureProposalGenerator: GEMINI_API_KEY missing; returning []"
            )
            return []

        out: List[FeatureProposal] = []
        for pattern in patterns[: self.max_proposals]:
            proposal = self._propose_one(pattern)
            if proposal is not None:
                out.append(proposal)
            if len(out) >= self.max_proposals:
                break
        return out

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _propose_one(self, pattern: OutcomePattern) -> Optional[FeatureProposal]:
        prompt = self._build_prompt(pattern)
        try:
            raw = self.llm_caller(prompt, timeout_s=self.timeout_s)
        except TypeError:
            try:
                raw = self.llm_caller(prompt)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                LOGGER.warning(
                    "FeatureProposalGenerator: LLM call failed for %r (%r)",
                    pattern.pattern_summary,
                    exc,
                )
                return None
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            LOGGER.warning(
                "FeatureProposalGenerator: LLM call failed for %r (%r)",
                pattern.pattern_summary,
                exc,
            )
            return None

        parsed: Dict[str, Any]
        if isinstance(raw, dict):
            parsed = raw
        elif isinstance(raw, str):
            try:
                parsed = extract_json_object(raw)
            except (ValueError, TypeError) as exc:
                LOGGER.warning(
                    "FeatureProposalGenerator: could not parse JSON from LLM (%r)",
                    exc,
                )
                return None
        else:
            LOGGER.warning(
                "FeatureProposalGenerator: unexpected LLM return type %r",
                type(raw).__name__,
            )
            return None

        name = str(parsed.get("name") or "").strip()
        description = str(parsed.get("description") or "").strip()
        python_code = str(parsed.get("python_code") or "")
        risk_notes = str(parsed.get("risk_notes") or "").strip()

        # Required: name + python_code must both be non-empty.
        if not name or not python_code:
            LOGGER.warning(
                "FeatureProposalGenerator: rejecting proposal with missing name/code "
                "(name=%r, has_code=%s)",
                name,
                bool(python_code),
            )
            return None

        # Validate the proposed Python parses; reject syntax errors.
        if not _python_code_parses(python_code):
            LOGGER.warning(
                "FeatureProposalGenerator: rejecting proposal %r — python_code "
                "failed ast.parse",
                name,
            )
            return None

        try:
            expected_lift = float(parsed.get("expected_lift_sharpe", 0.0) or 0.0)
        except (TypeError, ValueError):
            expected_lift = 0.0

        return FeatureProposal(
            name=name,
            description=description,
            python_code=python_code,
            expected_lift_sharpe=expected_lift,
            risk_notes=risk_notes,
            proposed_at_utc=_utcnow_iso(),
        )

    def _build_prompt(self, pattern: OutcomePattern) -> str:
        return (
            "You are a quant developer. Based on this loss pattern, propose ONE "
            "Python feature for the existing `compute_features()` pipeline that "
            "would have caught it.\n\n"
            f"Pattern: {pattern.pattern_summary}\n"
            f"Frequency: {pattern.frequency}\n"
            f"Signal-quality score: {pattern.signal_quality_score:.1f}\n"
            f"Evidence postmortem ids: {pattern.evidence_postmortem_ids[:10]}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  name           — snake_case identifier, <=40 chars\n"
            "  description    — 1-2 sentence description\n"
            "  python_code    — the feature implementation; must be valid Python\n"
            "  expected_lift_sharpe — float, your prediction of Sharpe lift\n"
            "  risk_notes     — notes on overfit / edge-case risks\n"
            "Do not include Markdown fences or commentary outside the JSON."
        )


__all__ = [
    "FeatureProposal",
    "FeatureProposalGenerator",
    "DEFAULT_MAX_PROPOSALS",
    "DEFAULT_LLM_TIMEOUT_S",
    "DEFAULT_GEMINI_MODEL",
]
