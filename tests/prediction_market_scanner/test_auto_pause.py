"""Tests for the auto-pause gate (``risk/auto_pause.py``).

Hermetic: marker-file paths are written into temp dirs and never reach
``~``. The integration scenario constructs a Supervisor with the existing
test stubs + auto_pause_gate to exercise the daily_close path.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from risk.auto_pause import AutoPauseDecision, AutoPauseGate
from state.confidence_history import ConfidenceHistory


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_gate(*, marker_dir: Path, **kwargs: Any) -> AutoPauseGate:
    return AutoPauseGate(
        marker_path=marker_dir / "auto_paused.marker",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# (a) Both conditions met -> pause
# ---------------------------------------------------------------------------


class TestAutoPauseBothConditionsMet(unittest.TestCase):
    def test_pauses_when_loss_breached_and_confidence_shifted_down(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, reason = gate.evaluate(
                # 5% loss on 10k bankroll = -500 USD; threshold is 2% = -200.
                daily_pnl_usd=-500.0,
                # Recent confidences far below baseline mean - 2σ = 0.6 - 0.2 = 0.4.
                recent_confidences=[0.30, 0.32, 0.28, 0.31] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertTrue(should)
        self.assertIn("daily_pnl", reason)
        self.assertIn("recent_mean_conf", reason)


# ---------------------------------------------------------------------------
# (b) Only loss condition met -> NO pause
# ---------------------------------------------------------------------------


class TestAutoPauseLossOnly(unittest.TestCase):
    def test_does_not_pause_when_only_loss_condition_holds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, reason = gate.evaluate(
                daily_pnl_usd=-500.0,
                # Recent confidences ~baseline mean -> not shifted.
                recent_confidences=[0.6, 0.62, 0.59, 0.61] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)
        self.assertIn("loss-only", reason)


# ---------------------------------------------------------------------------
# (c) Only confidence condition met -> NO pause
# ---------------------------------------------------------------------------


class TestAutoPauseConfidenceOnly(unittest.TestCase):
    def test_does_not_pause_when_only_confidence_condition_holds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, reason = gate.evaluate(
                # PnL within tolerance: 10 USD profit.
                daily_pnl_usd=10.0,
                recent_confidences=[0.30, 0.30, 0.30] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)
        self.assertIn("confidence-shift only", reason)


# ---------------------------------------------------------------------------
# (d) Neither condition -> NO pause
# ---------------------------------------------------------------------------


class TestAutoPauseNeitherCondition(unittest.TestCase):
    def test_does_not_pause_when_neither_condition_holds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, reason = gate.evaluate(
                daily_pnl_usd=10.0,
                recent_confidences=[0.6, 0.61, 0.59] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)
        self.assertEqual(reason, "ok")


# ---------------------------------------------------------------------------
# (e) Edge cases: empty list + NaN inputs
# ---------------------------------------------------------------------------


class TestAutoPauseEdgeCases(unittest.TestCase):
    def test_empty_confidence_list_disarms_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, reason = gate.evaluate(
                daily_pnl_usd=-500.0,  # would alone trip
                recent_confidences=[],
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)

    def test_nan_pnl_disarms_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, _reason = gate.evaluate(
                daily_pnl_usd=float("nan"),
                recent_confidences=[0.30] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)

    def test_inf_pnl_disarms_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            should, _reason = gate.evaluate(
                daily_pnl_usd=float("-inf"),
                recent_confidences=[0.30] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)

    def test_nan_confidences_filtered_out(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            # Mix of NaN + valid low confidences -> NaN filtered, low values
            # drive the recent mean below baseline cutoff. Combined with loss
            # condition -> pauses.
            confs = [float("nan"), 0.30, float("inf"), 0.28, 0.31] * 6
            should, _reason = gate.evaluate(
                daily_pnl_usd=-500.0,
                recent_confidences=confs,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertTrue(should)

    def test_zero_baseline_std_disarms_confidence_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            # std=0 means we can't compute z; gate must not trip.
            should, _reason = gate.evaluate(
                daily_pnl_usd=-500.0,
                recent_confidences=[0.30] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.0,
                bankroll_usd=10_000.0,
            )
        self.assertFalse(should)

    def test_rejects_negative_loss_threshold_pct(self) -> None:
        with self.assertRaises(ValueError):
            AutoPauseGate(loss_threshold_pct=-0.01)

    def test_rejects_negative_z_threshold(self) -> None:
        with self.assertRaises(ValueError):
            AutoPauseGate(z_threshold=-1.0)

    def test_rejects_zero_recent_window(self) -> None:
        with self.assertRaises(ValueError):
            AutoPauseGate(recent_window=0)


# ---------------------------------------------------------------------------
# Marker-file lifecycle
# ---------------------------------------------------------------------------


class TestAutoPauseMarkerFile(unittest.TestCase):
    def test_write_marker_creates_file_and_is_idempotent_via_clear(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker_path = Path(td) / "deep" / "marker"
            gate = AutoPauseGate(marker_path=marker_path)
            self.assertFalse(gate.is_marker_present())
            self.assertTrue(gate.write_marker(reason="test trip"))
            self.assertTrue(gate.is_marker_present())
            self.assertIn("test trip", marker_path.read_text())
            self.assertTrue(gate.clear_marker())
            self.assertFalse(gate.is_marker_present())
            # Clearing again returns False.
            self.assertFalse(gate.clear_marker())


# ---------------------------------------------------------------------------
# (f) Integration: Supervisor.daily_close() trips gate, writes marker, alerts
# ---------------------------------------------------------------------------


# Reuse the supervisor stubs; importing them lazily so this module stays
# self-contained on the unit-test layer.

class _StubPusher:
    def __init__(self) -> None:
        self.gauge_calls: List[Dict[str, Any]] = []
        self.counter_calls: List[Dict[str, Any]] = []
        self.histogram_calls: List[Dict[str, Any]] = []
        self.pushed = 0

    def is_enabled(self) -> bool:
        return True

    def gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.gauge_calls.append({"name": name, "value": float(value), "labels": labels or {}})

    def counter(
        self, name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.counter_calls.append(
            {"name": name, "increment": float(increment), "labels": labels or {}}
        )

    def histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        self.histogram_calls.append(
            {"name": name, "value": float(value), "labels": labels or {}}
        )

    def push(self) -> bool:
        self.pushed += 1
        return True


class TestSupervisorDailyCloseTriggersAutoPause(unittest.TestCase):
    """End-to-end: simulated daily_close trips the auto-pause gate, writes
    a marker file, emits ``autopilot_auto_pause_total`` and an alert."""

    def test_daily_close_with_auto_pause_writes_marker_and_emits_metric(
        self,
    ) -> None:
        from test_live_supervisor import (
            StubCircuitBreakers,
            StubExchange,
            StubNotifier,
            StubPositionStore,
            _build_supervisor,
        )

        with tempfile.TemporaryDirectory() as td:
            marker_path = Path(td) / ".autopilot_auto_paused"
            store = StubPositionStore(daily_pnl=-500.0)
            breakers = StubCircuitBreakers(daily_loss_limit_usd=1_000.0)
            notifier = StubNotifier()
            pusher = _StubPusher()
            history = ConfidenceHistory(redis_client=None)
            # Seed the history with low confidences so the gate's
            # confidence condition holds. Use 200 entries (default window).
            for _ in range(50):
                history.record("ETH/USDT", 0.3)
            # Plant baseline samples so baseline mean ~0.3, std ~0 -> gate
            # cannot trip on confidence alone (std=0). To make the trip
            # happen we instead inject a separate gate with a hand-picked
            # baseline.
            gate = AutoPauseGate(
                marker_path=marker_path,
                loss_threshold_pct=0.02,  # 2% on 10k bankroll = $200
            )

            sup, refs = _build_supervisor(
                position_store=store,
                circuit_breakers=breakers,
                notifier=notifier,
            )
            sup.metrics_pusher = pusher
            sup.auto_pause_gate = gate
            sup.confidence_history = history

            # Manually exercise the auto-pause check with hand-picked
            # baseline so the recent_mean (0.3) < baseline_mean (0.6) -
            # 2*std (0.1) = 0.4, AND daily_pnl (-500) < -200.
            # Drive through daily_close -- it pulls daily_pnl from the
            # store but the baseline read is via confidence_history.
            # We patch the baseline by manipulating the gate's evaluate
            # path through a wrapper that forwards a synthetic baseline.
            original_baseline = history.baseline

            def _forced_baseline(symbol: str) -> tuple:
                return (0.6, 0.1, 200)

            history.baseline = _forced_baseline  # type: ignore[assignment]
            try:
                sup.daily_close()
            finally:
                history.baseline = original_baseline  # type: ignore[assignment]

            self.assertTrue(
                marker_path.exists(),
                "auto-pause marker file must be written when gate trips",
            )
            counter_names = [c["name"] for c in pusher.counter_calls]
            self.assertIn("auto_pause_total", counter_names)
            # Alert fired with critical severity.
            critical_alerts = [
                c for c in notifier.alert_calls
                if c.get("severity") == "critical"
            ]
            self.assertTrue(
                critical_alerts,
                "auto-pause should send a critical-severity alert",
            )

    def test_daily_close_without_gate_is_no_op(self) -> None:
        """No auto_pause_gate => daily_close behaves exactly like before."""
        from test_live_supervisor import _build_supervisor

        sup, refs = _build_supervisor()
        # auto_pause_gate is None by default. Just make sure daily_close
        # doesn't crash and doesn't write the default marker path.
        sup.daily_close()
        # Default marker path should NOT exist after a clean close —
        # but we can't inspect it without polluting ~. Instead assert no
        # exception escaped (already handled by the call returning).


# ---------------------------------------------------------------------------
# Detailed-decision exposure (used by supervisor for reason fields)
# ---------------------------------------------------------------------------


class TestAutoPauseDetailedDecision(unittest.TestCase):
    def test_evaluate_detailed_returns_full_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = _make_gate(marker_dir=Path(td))
            decision = gate.evaluate_detailed(
                daily_pnl_usd=-500.0,
                recent_confidences=[0.30] * 10,
                baseline_confidence_mean=0.6,
                baseline_confidence_std=0.1,
                bankroll_usd=10_000.0,
            )
        self.assertIsInstance(decision, AutoPauseDecision)
        self.assertTrue(decision.should_pause)
        self.assertAlmostEqual(decision.daily_pnl_usd, -500.0)
        self.assertAlmostEqual(decision.loss_threshold_usd, 200.0)
        self.assertIsNotNone(decision.recent_mean)
        self.assertAlmostEqual(decision.recent_mean, 0.30, places=4)


if __name__ == "__main__":
    unittest.main()
