"""OutcomeAnalyst — first stage of the E3 LLM Strategy-Gen loop (SKELETON).

Reads recent loss postmortem JSONs (written by
:class:`loss_postmortem.synthesizer.LossPostmortemSynthesizer`) plus the
legacy ``performance_audit.json`` (written by the older
:class:`outcome_review_agent.logger.PerformanceTracker`). Buckets the
postmortems into root-cause + symbol clusters, then asks Gemini to label
each cluster with a one-line ``pattern_summary`` and a 0-100
``signal_quality_score``.

This stage is intentionally *bounded*:

* Max 3 Gemini calls per ``analyze()`` invocation. The bound is hard;
  we rank clusters by frequency and only send the top-N.
* Each call has a ``timeout_s`` budget. Default 30s.
* If the LLM call fails for ANY reason (no API key, transport error,
  unparseable JSON), we degrade gracefully: log a warning, skip the
  LLM-derived fields for that cluster, and keep the synthetic fallback
  values (``pattern_summary`` derived from postmortem evidence,
  ``signal_quality_score=50``).
* If the postmortem dir doesn't exist or is empty, ``analyze()``
  returns an empty list — no crash.

The output is a list of :class:`OutcomePattern` sorted by ``frequency``
descending. The downstream :class:`FeatureProposalGenerator` consumes
this list and asks the LLM for one feature per pattern.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# Hard ceiling on Gemini calls per analyze() invocation.
MAX_LLM_CALLS_PER_ANALYZE = 3
DEFAULT_LLM_TIMEOUT_S = 30
DEFAULT_WINDOW_DAYS = 7

# Default fallback score when the LLM is unavailable. Kept neutral on
# purpose so a degraded run still produces actionable patterns rather
# than silently zeroing out everything.
DEFAULT_SIGNAL_QUALITY_SCORE = 50.0


@dataclass(frozen=True)
class OutcomePattern:
    """One clustered pattern of recurring losses, ready for proposal-gen.

    Attributes
    ----------
    pattern_summary:
        One-sentence human-readable description of the cluster, e.g.
        ``"Signal-cause losses on BTC during high-volatility regime"``.
        Derived synthetically from cluster keys when the LLM is
        unavailable, refined by the LLM when it is.
    frequency:
        Number of postmortems / outcome reviews this pattern covers.
    signal_quality_score:
        LLM-assigned 0.0-100.0 score reflecting how confident the LLM is
        that the cluster represents a real, fixable signal-quality
        problem (vs. random noise). Defaults to 50.0 when the LLM call
        fails or is skipped.
    evidence_postmortem_ids:
        Postmortem trade-ids that contributed to this cluster, in the
        order encountered. Used downstream by the proposal generator to
        give the LLM concrete evidence rather than just a category label.
    """

    pattern_summary: str
    frequency: int
    signal_quality_score: float
    evidence_postmortem_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_summary": self.pattern_summary,
            "frequency": int(self.frequency),
            "signal_quality_score": float(self.signal_quality_score),
            "evidence_postmortem_ids": list(self.evidence_postmortem_ids),
        }


def _parse_iso(ts: Any) -> Optional[_dt.datetime]:
    if not isinstance(ts, str) or not ts:
        return None
    raw = ts.rstrip("Z")
    try:
        dt = _dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _cluster_key(root_cause: str, symbol: Optional[str]) -> Tuple[str, str]:
    return (str(root_cause or "Unknown"), str(symbol or "?"))


def _synthetic_summary(root_cause: str, symbol: Optional[str], frequency: int) -> str:
    sym = symbol or "unknown symbol"
    return (
        f"{root_cause}-cause losses on {sym} "
        f"({frequency} postmortem{'s' if frequency != 1 else ''})"
    )


class OutcomeAnalyst:
    """Read postmortems + audit log, return ranked outcome patterns.

    Parameters
    ----------
    postmortems_dir:
        Path to ``runs/postmortems/`` (or wherever the
        :class:`LossPostmortemSynthesizer` is configured to write).
        Missing dir is treated as zero postmortems, not an error.
    performance_audit_path:
        Optional path to the legacy ``performance_audit.json`` from
        :class:`outcome_review_agent.logger.PerformanceTracker`. If
        present, settled-trade reviews are folded into the clustering
        as additional ``Unknown``-root-cause evidence (the legacy log
        doesn't have a clean root_cause label, so we leave it broad).
    llm_caller:
        Optional injectable callable ``(prompt: str, timeout_s: int) ->
        Dict[str, Any]``. Production wires this to ``llm_judge``-style
        Gemini calls; tests inject a mock. ``None`` (default) means
        "skip LLM, use synthetic fallbacks for every cluster".
    timeout_s:
        Per-call LLM timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        *,
        postmortems_dir: Path,
        performance_audit_path: Optional[Path] = None,
        llm_caller: Optional[Any] = None,
        timeout_s: int = DEFAULT_LLM_TIMEOUT_S,
    ) -> None:
        self.postmortems_dir = Path(postmortems_dir)
        self.performance_audit_path = (
            Path(performance_audit_path) if performance_audit_path else None
        )
        self.llm_caller = llm_caller
        self.timeout_s = int(timeout_s)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def analyze(self, window_days: int = DEFAULT_WINDOW_DAYS) -> List[OutcomePattern]:
        """Return ranked outcome patterns over the past ``window_days``.

        Returns an empty list (no exception) when the postmortems
        directory is missing/empty AND there is no audit log.
        """
        if window_days <= 0:
            return []

        cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=window_days)
        clusters = self._read_postmortem_clusters(cutoff)
        self._fold_in_audit_log(clusters, cutoff)

        if not clusters:
            return []

        # Rank clusters by frequency descending. Ties broken by
        # (root_cause, symbol) for determinism.
        ranked = sorted(
            clusters.items(),
            key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]),
        )

        patterns: List[OutcomePattern] = []
        llm_call_budget = MAX_LLM_CALLS_PER_ANALYZE
        for (root_cause, symbol), evidence_ids in ranked:
            frequency = len(evidence_ids)
            synthetic = _synthetic_summary(root_cause, symbol, frequency)

            summary = synthetic
            score = DEFAULT_SIGNAL_QUALITY_SCORE

            if llm_call_budget > 0 and self.llm_caller is not None:
                refined = self._call_llm_for_cluster(
                    root_cause=root_cause,
                    symbol=symbol,
                    evidence_ids=evidence_ids,
                    fallback_summary=synthetic,
                )
                llm_call_budget -= 1
                if refined is not None:
                    summary, score = refined

            patterns.append(
                OutcomePattern(
                    pattern_summary=summary,
                    frequency=frequency,
                    signal_quality_score=score,
                    evidence_postmortem_ids=list(evidence_ids),
                )
            )
        return patterns

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _read_postmortem_clusters(
        self, cutoff: _dt.datetime
    ) -> Dict[Tuple[str, str], List[str]]:
        """Group postmortems within window by ``(root_cause, symbol)``."""
        clusters: Dict[Tuple[str, str], List[str]] = {}
        if not self.postmortems_dir.exists() or not self.postmortems_dir.is_dir():
            return clusters

        for path in sorted(self.postmortems_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                LOGGER.warning("OutcomeAnalyst: skipping unreadable %s (%r)", path, exc)
                continue
            if not isinstance(data, dict):
                continue

            ts = _parse_iso(data.get("triggered_at_utc"))
            # Postmortems with unparseable timestamps are EXCLUDED
            # conservatively. Matches LossPostmortemSynthesizer behavior.
            if ts is None or ts < cutoff:
                continue

            synth = data.get("synthesizer") or {}
            root_cause = synth.get("root_cause") or "Unknown"
            symbol = data.get("symbol") or "?"
            trade_id = data.get("trade_id")
            if not trade_id:
                continue

            key = _cluster_key(root_cause, symbol)
            clusters.setdefault(key, []).append(str(trade_id))
        return clusters

    def _fold_in_audit_log(
        self,
        clusters: Dict[Tuple[str, str], List[str]],
        cutoff: _dt.datetime,
    ) -> None:
        """Merge legacy performance_audit.json reviews into clusters.

        The legacy log doesn't carry root_cause; we file it under
        ``"Unknown"`` so it shows up in patterns but isn't mistaken for
        a Signal/Sizing-grade signal. ``symbol`` is taken from the
        review payload when present, else ``"?"``.
        """
        if self.performance_audit_path is None:
            return
        if not self.performance_audit_path.exists():
            return

        try:
            audit = json.loads(self.performance_audit_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning(
                "OutcomeAnalyst: could not read performance audit %s (%r)",
                self.performance_audit_path,
                exc,
            )
            return
        if not isinstance(audit, dict):
            return

        reviews = audit.get("reviews") or []
        if not isinstance(reviews, list):
            return

        for entry in reviews:
            if not isinstance(entry, dict):
                continue
            ts = _parse_iso(entry.get("settled_at") or entry.get("reviewed_at"))
            if ts is None or ts < cutoff:
                continue
            trade_id = entry.get("trade_id")
            if not trade_id:
                continue
            review = entry.get("outcome_review") or {}
            symbol = entry.get("symbol") or review.get("symbol") or "?"
            key = _cluster_key("Unknown", symbol)
            clusters.setdefault(key, []).append(str(trade_id))

    def _call_llm_for_cluster(
        self,
        *,
        root_cause: str,
        symbol: Optional[str],
        evidence_ids: List[str],
        fallback_summary: str,
    ) -> Optional[Tuple[str, float]]:
        """Invoke the injected LLM caller; degrade gracefully on failure.

        Returns ``(summary, score)`` on success, ``None`` on any failure
        (caller keeps the synthetic fallback).
        """
        if self.llm_caller is None:
            return None

        # Defensive: if there's no API key in env AND no caller is wired
        # to inject one, skip rather than raise.
        if (
            os.getenv("GEMINI_API_KEY") is None
            and os.getenv("GOOGLE_API_KEY") is None
            and not getattr(self.llm_caller, "_test_inject", False)
        ):
            LOGGER.warning(
                "OutcomeAnalyst: GEMINI_API_KEY missing; skipping LLM refinement for %s/%s",
                root_cause,
                symbol,
            )
            return None

        prompt = (
            "You are a quant analyst reviewing a cluster of trading "
            "losses for recurring patterns. Respond with a JSON object "
            "containing exactly two keys: 'pattern_summary' (a single "
            "sentence under 200 chars) and 'signal_quality_score' (a "
            "number 0-100; 100 = strong evidence of a real, fixable "
            f"signal-quality problem).\n\nRoot cause: {root_cause}\n"
            f"Symbol: {symbol}\nFrequency: {len(evidence_ids)}\n"
            f"Evidence postmortem ids: {evidence_ids[:10]}\n\n"
            f"Synthetic fallback summary (improve on this): "
            f"{fallback_summary}"
        )

        try:
            result = self.llm_caller(prompt, timeout_s=self.timeout_s)
        except TypeError:
            try:
                result = self.llm_caller(prompt)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                LOGGER.warning(
                    "OutcomeAnalyst: LLM caller raised %r; falling back",
                    exc,
                )
                return None
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            LOGGER.warning(
                "OutcomeAnalyst: LLM caller raised %r; falling back",
                exc,
            )
            return None

        if not isinstance(result, dict):
            LOGGER.warning(
                "OutcomeAnalyst: LLM caller returned %r (not dict); falling back",
                type(result).__name__,
            )
            return None

        summary = str(result.get("pattern_summary") or "").strip()
        if not summary:
            summary = fallback_summary

        raw_score = result.get("signal_quality_score", DEFAULT_SIGNAL_QUALITY_SCORE)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = DEFAULT_SIGNAL_QUALITY_SCORE
        score = max(0.0, min(100.0, score))

        return summary, score


__all__ = [
    "OutcomeAnalyst",
    "OutcomePattern",
    "MAX_LLM_CALLS_PER_ANALYZE",
    "DEFAULT_LLM_TIMEOUT_S",
    "DEFAULT_WINDOW_DAYS",
    "DEFAULT_SIGNAL_QUALITY_SCORE",
]
