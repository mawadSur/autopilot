"""Per-trade synchronous LLM weight adjuster (Lane C P1 #16).

This module implements decision D3 (per-trade synchronous outcome feedback,
NOT a daily batch): when a trade settles and the outcome-review agent
classifies it under the Process-vs-Outcome matrix, the adjuster
immediately updates the LLM weight used by the Bayesian fusion in
``calibration_agent.analyzer``.

Adjustments by classification:
    Deserved Success: +5%   (process and outcome both good -> trust LLM more)
    Good Failure:      0%   (good process, bad outcome -> noise, no signal)
    Dumb Luck:        -5%   (bad process, good outcome -> reduce LLM trust)
    Poetic Justice:  -10%   (bad process, bad outcome -> sharpest penalty)

The new weight is then EMA-smoothed against 1.0 with ``decay`` (default
0.95) so a single review can move the weight at most ``(1 - decay) * delta``
units in absolute terms. Final weight is clipped to ``[min_weight, max_weight]``.

State persistence uses ``fcntl.flock`` (mirroring Lane A's shakedown
JSON pattern) so multiple supervisor processes writing the same weight
file can't clobber each other. An audit JSONL file records every
adjustment for downstream analysis (loss postmortem swarm in Lane E).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# Mirrors the backoff schedule in live_supervisor's shakedown lock so the
# two pieces of per-process state behave consistently under contention.
_LOCK_BACKOFF_S: Tuple[float, ...] = (0.010, 0.020, 0.040, 0.080, 0.160)

# Adjustment table per Process-vs-Outcome classification.
_ADJUSTMENT_BY_CLASSIFICATION: Dict[str, float] = {
    "deserved success": 0.05,
    "good failure": 0.0,
    "dumb luck": -0.05,
    "poetic justice": -0.10,
}


def _normalize_classification(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    # Strip punctuation differences ("Deserved-Success", "deserved_success", etc.)
    text = text.replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text in _ADJUSTMENT_BY_CLASSIFICATION:
        return text
    # Tolerate longer phrases that contain the canonical label (LLMs sometimes
    # emit "Deserved Success: process was clean" etc.).
    for label in _ADJUSTMENT_BY_CLASSIFICATION:
        if text.startswith(label):
            return label
    return "unknown"


class _NullLock:
    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FileLock:
    """``fcntl.flock(LOCK_EX | LOCK_NB)`` with exponential-backoff retries.

    Mirrors ``live_supervisor._FileLock`` so the two state files (shakedown
    JSON, weight file) use the same locking discipline.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        try:
            import fcntl  # type: ignore[import-not-found]
        except ImportError:
            return self
        self._path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        self._fd = os.open(str(self._path), flags, 0o644)
        op = fcntl.LOCK_EX | fcntl.LOCK_NB
        for delay in _LOCK_BACKOFF_S:
            try:
                fcntl.flock(self._fd, op)
                return self
            except OSError:
                time.sleep(delay)
        try:
            fcntl.flock(self._fd, op)
        except OSError as exc:
            LOGGER.warning(
                "outcome-weight lock contended for %s after %d retries (%s); "
                "proceeding without lock",
                self._path,
                len(_LOCK_BACKOFF_S),
                exc,
            )
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is None:
            return
        try:
            import fcntl  # type: ignore[import-not-found]
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - best-effort unlock
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


def _acquire_file_lock(path: Path) -> Any:
    try:
        import fcntl  # noqa: F401  - test for availability only
    except ImportError:
        return _NullLock()
    return _FileLock(path)


class OutcomeWeightAdjuster:
    """Per-trade EMA adjuster for the LLM weight in Bayesian fusion."""

    def __init__(
        self,
        *,
        weight_file: Path,
        audit_file: Path,
        initial_weight: float = 1.0,
        max_weight: float = 2.0,
        min_weight: float = 0.5,
        decay: float = 0.95,
    ) -> None:
        if not 0.0 <= float(min_weight) < float(max_weight):
            raise ValueError("min_weight must be < max_weight and >= 0")
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("decay must be in (0, 1)")
        if not float(min_weight) <= float(initial_weight) <= float(max_weight):
            raise ValueError("initial_weight must lie within [min_weight, max_weight]")

        self.weight_file = Path(weight_file)
        self.audit_file = Path(audit_file)
        self.initial_weight = float(initial_weight)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.decay = float(decay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify_outcome(self, outcome_review: Mapping[str, Any] | Any) -> str:
        """Return the canonical classification string from a review payload.

        Accepts either a ``dict``-shaped review (the orchestrator's preferred
        format) or anything with a ``matrix_classification`` attribute (e.g.
        :class:`outcome_review_agent.models.OutcomeReview`).
        """
        if outcome_review is None:
            return "unknown"
        # dict-style
        if isinstance(outcome_review, Mapping):
            raw = outcome_review.get("matrix_classification") or outcome_review.get(
                "classification"
            )
        else:
            raw = getattr(outcome_review, "matrix_classification", None) or getattr(
                outcome_review, "classification", None
            )
        return _normalize_classification(raw)

    def compute_adjustment(self, outcome_class: str, current_weight: float) -> float:
        """Return the new (clipped) weight after applying the EMA adjustment.

        ``current_weight`` is the existing weight on disk. The EMA pulls the
        target (1 + delta) toward 1.0 with ``decay``, so even a string of
        Poetic Justice classifications can't dislocate the weight by more
        than the EMA budget per step.
        """
        normalized = _normalize_classification(outcome_class)
        delta = _ADJUSTMENT_BY_CLASSIFICATION.get(normalized, 0.0)
        target = 1.0 + delta
        new_weight = current_weight * (self.decay * target + (1.0 - self.decay) * 1.0)
        return self._clip(new_weight)

    def update_weight(self, trade_review: Mapping[str, Any] | Any) -> float:
        """Atomically read -> classify -> compute -> write the weight file.

        Returns the new weight. Appends a JSONL audit entry to
        ``self.audit_file`` describing the inputs and the resulting weight.
        """
        outcome_class = self.classify_outcome(trade_review)
        with _acquire_file_lock(self.weight_file):
            current = self._read_weight_locked()
            new_weight = self.compute_adjustment(outcome_class, current)
            self._write_weight_locked(new_weight)

        self._append_audit(
            outcome_class=outcome_class,
            previous_weight=current,
            new_weight=new_weight,
            trade_review=trade_review,
        )
        return new_weight

    def apply_postmortem_delta(self, weight_delta: float) -> float:
        """Hook reserved for the Loss Postmortem Swarm (Lane E).

        The postmortem will eventually emit a separate weight nudge that's
        independent of the per-trade outcome review. For now this method
        applies the delta directly through the same clip pipeline, but
        does NOT yet integrate with the swarm's full report shape.
        """
        delta = float(weight_delta)
        with _acquire_file_lock(self.weight_file):
            current = self._read_weight_locked()
            new_weight = self._clip(current + delta)
            self._write_weight_locked(new_weight)
        self._append_audit(
            outcome_class="postmortem_delta",
            previous_weight=current,
            new_weight=new_weight,
            trade_review={"postmortem_delta": delta},
        )
        return new_weight

    def current_weight(self) -> float:
        """Return the current (clipped) weight from disk."""
        with _acquire_file_lock(self.weight_file):
            return self._read_weight_locked()

    # ------------------------------------------------------------------
    # Internals (caller must hold the lock)
    # ------------------------------------------------------------------
    def _read_weight_locked(self) -> float:
        if not self.weight_file.exists():
            return self.initial_weight
        try:
            payload = json.loads(self.weight_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.initial_weight
        if not isinstance(payload, Mapping):
            return self.initial_weight
        try:
            return self._clip(float(payload.get("llm_weight", self.initial_weight)))
        except (TypeError, ValueError):
            return self.initial_weight

    def _write_weight_locked(self, weight: float) -> None:
        self.weight_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "llm_weight": float(weight),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Atomic-rename pattern (write to .tmp then rename) so partial writes
        # never leave a corrupt file even if the process is killed mid-write.
        tmp = self.weight_file.with_suffix(self.weight_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.weight_file)

    def _append_audit(
        self,
        *,
        outcome_class: str,
        previous_weight: float,
        new_weight: float,
        trade_review: Any,
    ) -> None:
        # Best-effort serialization: most reviews are dicts, but accept
        # OutcomeReview pydantic models too via model_dump.
        if hasattr(trade_review, "model_dump") and callable(trade_review.model_dump):
            review_payload: Any = trade_review.model_dump()
        elif isinstance(trade_review, Mapping):
            review_payload = dict(trade_review)
        else:
            review_payload = str(trade_review)

        entry = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "outcome_class": outcome_class,
            "previous_weight": float(previous_weight),
            "new_weight": float(new_weight),
            "delta": float(new_weight - previous_weight),
            "trade_review": review_payload,
        }
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def _clip(self, weight: float) -> float:
        return max(self.min_weight, min(self.max_weight, float(weight)))
