"""Auto-pause gate — combined daily-loss + confidence-distribution trip.

This is a different surface from the kill switch:
* The kill switch (``risk/circuit_breakers.CircuitBreakerSet``) is
  operator-set: a file at ``KILL_SWITCH_FILE`` is created manually (or by
  N consecutive errors — see Task 3). Once tripped, every symbol halts.
* The auto-pause gate is **bot-set**: it fires when *both* the daily loss
  exceeds a configurable percentage of bankroll AND the model's recent
  confidence distribution has shifted down enough to be statistically
  unusual (mean below baseline mean - 2σ). The "AND" matters: either
  signal alone is too noisy to halt on.

When the gate trips, the supervisor:
1. Writes a marker file at ``~/.autopilot_auto_paused`` (default).
2. Emits an ``autopilot_auto_pause_total`` counter with a ``reason`` label.
3. Sends a ``critical`` alert via the notifier.

The marker file is read by the kill-switch logic on the next tick (the
breaker context can mirror this seam if desired) so the trader halts on
the immediate next iteration.

Tunables
--------
``loss_threshold_pct`` — fraction of bankroll. Default 0.02 (2%).
``z_threshold``        — std-deviations below baseline mean. Default 2.0.
``recent_window``      — most recent confidences to average. Default 30.

Tests live in ``tests/prediction_market_scanner/test_auto_pause.py``.
"""
from __future__ import annotations

import logging
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


DEFAULT_LOSS_THRESHOLD_PCT = 0.02  # 2% of bankroll
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_RECENT_WINDOW = 30
DEFAULT_MARKER_PATH = Path.home() / ".autopilot_auto_paused"


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass
class AutoPauseDecision:
    """Result of a single :meth:`AutoPauseGate.evaluate` call."""

    should_pause: bool
    reason: str
    daily_pnl_usd: float
    loss_threshold_usd: float
    recent_mean: Optional[float]
    baseline_mean: Optional[float]
    baseline_std: Optional[float]
    z_threshold: float


class AutoPauseGate:
    """Combined daily-loss + confidence-shift auto-pause gate.

    The gate is stateless: every :meth:`evaluate` call reads the inputs
    and decides without persisting anything. The supervisor is
    responsible for writing the marker file + emitting metrics + sending
    alerts when the gate trips.
    """

    def __init__(
        self,
        *,
        loss_threshold_pct: float = DEFAULT_LOSS_THRESHOLD_PCT,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        recent_window: int = DEFAULT_RECENT_WINDOW,
        marker_path: Optional[Path] = None,
    ) -> None:
        if loss_threshold_pct < 0:
            raise ValueError("loss_threshold_pct must be non-negative")
        if z_threshold < 0:
            raise ValueError("z_threshold must be non-negative")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        self.loss_threshold_pct = float(loss_threshold_pct)
        self.z_threshold = float(z_threshold)
        self.recent_window = int(recent_window)
        self.marker_path = Path(marker_path) if marker_path else DEFAULT_MARKER_PATH

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        daily_pnl_usd: float,
        recent_confidences: Iterable[float],
        baseline_confidence_mean: float,
        baseline_confidence_std: float,
        bankroll_usd: float,
    ) -> Tuple[bool, str]:
        """Return (should_pause, reason).

        Pauses iff BOTH conditions hold:
        1. ``daily_pnl_usd < -(loss_threshold_pct * bankroll_usd)``.
        2. The most recent ``recent_window`` confidences average to
           below ``baseline_mean - z_threshold * baseline_std``.

        Either condition alone is permitted. NaN / inf inputs are treated
        defensively as "do not pause".
        """
        decision = self._decide(
            daily_pnl_usd=daily_pnl_usd,
            recent_confidences=recent_confidences,
            baseline_confidence_mean=baseline_confidence_mean,
            baseline_confidence_std=baseline_confidence_std,
            bankroll_usd=bankroll_usd,
        )
        return (decision.should_pause, decision.reason)

    def evaluate_detailed(
        self,
        *,
        daily_pnl_usd: float,
        recent_confidences: Iterable[float],
        baseline_confidence_mean: float,
        baseline_confidence_std: float,
        bankroll_usd: float,
    ) -> AutoPauseDecision:
        """Return the full :class:`AutoPauseDecision` for telemetry callers."""
        return self._decide(
            daily_pnl_usd=daily_pnl_usd,
            recent_confidences=recent_confidences,
            baseline_confidence_mean=baseline_confidence_mean,
            baseline_confidence_std=baseline_confidence_std,
            bankroll_usd=bankroll_usd,
        )

    def _decide(
        self,
        *,
        daily_pnl_usd: float,
        recent_confidences: Iterable[float],
        baseline_confidence_mean: float,
        baseline_confidence_std: float,
        bankroll_usd: float,
    ) -> AutoPauseDecision:
        loss_threshold_usd = self.loss_threshold_pct * float(
            max(bankroll_usd, 0.0)
        )

        # Defensive: NaN / inf in any monetary input -> don't pause.
        if not _is_finite(daily_pnl_usd) or not _is_finite(bankroll_usd):
            return AutoPauseDecision(
                should_pause=False,
                reason="non-finite inputs; gate inactive",
                daily_pnl_usd=float(daily_pnl_usd) if _is_finite(daily_pnl_usd) else 0.0,
                loss_threshold_usd=loss_threshold_usd,
                recent_mean=None,
                baseline_mean=None,
                baseline_std=None,
                z_threshold=self.z_threshold,
            )

        loss_condition = float(daily_pnl_usd) < -float(loss_threshold_usd)

        # Filter NaN/inf out of the confidence stream defensively, then
        # take the trailing N. If too few samples remain, the confidence
        # condition cannot be evaluated and the gate stays disarmed.
        clean_confidences: List[float] = [
            float(c) for c in recent_confidences if _is_finite(c)
        ]
        recent = clean_confidences[-self.recent_window:]

        if not recent:
            recent_mean: Optional[float] = None
        else:
            try:
                recent_mean = statistics.fmean(recent)
            except statistics.StatisticsError:
                recent_mean = None

        # Baseline must have non-zero std to make a z-score meaningful.
        baseline_ok = (
            _is_finite(baseline_confidence_mean)
            and _is_finite(baseline_confidence_std)
            and float(baseline_confidence_std) > 0.0
        )

        if recent_mean is None or not baseline_ok:
            confidence_condition = False
        else:
            cutoff = float(baseline_confidence_mean) - self.z_threshold * float(
                baseline_confidence_std
            )
            confidence_condition = recent_mean < cutoff

        should_pause = loss_condition and confidence_condition

        if should_pause:
            reason = (
                f"daily_pnl={daily_pnl_usd:.2f} below "
                f"-{loss_threshold_usd:.2f} AND recent_mean_conf="
                f"{recent_mean:.3f} below baseline_mean "
                f"{float(baseline_confidence_mean):.3f} - "
                f"{self.z_threshold}*std "
                f"{float(baseline_confidence_std):.3f}"
            )
        elif loss_condition and not confidence_condition:
            reason = "loss-only condition; confidence within baseline"
        elif confidence_condition and not loss_condition:
            reason = "confidence-shift only; pnl within tolerance"
        else:
            reason = "ok"

        return AutoPauseDecision(
            should_pause=should_pause,
            reason=reason,
            daily_pnl_usd=float(daily_pnl_usd),
            loss_threshold_usd=loss_threshold_usd,
            recent_mean=recent_mean,
            baseline_mean=float(baseline_confidence_mean) if _is_finite(baseline_confidence_mean) else None,
            baseline_std=float(baseline_confidence_std) if _is_finite(baseline_confidence_std) else None,
            z_threshold=self.z_threshold,
        )

    # ------------------------------------------------------------------
    # Side effects (write marker, etc.). Both are best-effort.
    # ------------------------------------------------------------------
    def write_marker(self, *, reason: str) -> bool:
        """Create the marker file. Returns True on success."""
        try:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            self.marker_path.write_text(
                f"auto-paused: {reason}\n",
                encoding="utf-8",
            )
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort
            LOGGER.warning(
                "AutoPauseGate.write_marker failed at %s: %s",
                self.marker_path,
                exc,
            )
            return False

    def is_marker_present(self) -> bool:
        """True iff the marker file currently exists on disk."""
        try:
            return self.marker_path.exists()
        except Exception:  # noqa: BLE001 - filesystem checks must not raise
            return False

    def clear_marker(self) -> bool:
        """Delete the marker file. Returns True iff it existed and was removed."""
        try:
            self.marker_path.unlink()
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001 - best-effort
            LOGGER.warning(
                "AutoPauseGate.clear_marker failed at %s: %s",
                self.marker_path,
                exc,
            )
            return False


__all__ = [
    "AutoPauseDecision",
    "AutoPauseGate",
    "DEFAULT_LOSS_THRESHOLD_PCT",
    "DEFAULT_MARKER_PATH",
    "DEFAULT_RECENT_WINDOW",
    "DEFAULT_Z_THRESHOLD",
]
