"""Tests for the ``live_supervisor.main()`` CLI entry point.

Targeted argument-handling tests that don't require a live Redis or live
exchange. The full Supervisor wiring is patched out — these tests assert
on parsing / validation behaviour around argv only.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from typing import Any, List
from unittest import mock

import live_supervisor


class _RecordingSupervisor:
    """Stand-in for ``Supervisor`` that records the symbols it was given."""

    instances: List["_RecordingSupervisor"] = []

    def __init__(self, *, config: Any, **_: Any) -> None:
        self.config = config
        _RecordingSupervisor.instances.append(self)

    def run_once(self) -> List[Any]:
        return []

    def run_loop(self, *, max_iterations: Any = None) -> dict:
        return {
            "iterations": 0,
            "total_ticks": 0,
            "action_counts": {},
            "interrupted": False,
        }


class MainArgvDedupeTests(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingSupervisor.instances = []

    def _patches(self):
        """Common stub set so ``main()`` can run without external services."""
        return [
            mock.patch.object(live_supervisor, "Supervisor", _RecordingSupervisor),
            mock.patch.object(
                live_supervisor, "CoinbaseExchange", lambda *a, **kw: object()
            ),
            mock.patch.object(
                live_supervisor, "PositionStore", lambda *a, **kw: object()
            ),
            mock.patch.object(
                live_supervisor, "CircuitBreakerSet", lambda *a, **kw: object()
            ),
            mock.patch.object(
                live_supervisor, "Notifier", lambda *a, **kw: object()
            ),
            # build_default_predict_fn is loaded inside main(); patch the
            # import path it uses (``from predictor import ...``).
            mock.patch(
                "predictor.build_default_predict_fn", return_value=None
            ),
        ]

    def test_duplicate_symbols_are_dropped(self) -> None:
        """``--symbols ETH/USD,ETH/USD,BTC/USD`` -> [ETH/USD, BTC/USD]."""
        argv = [
            "--symbols",
            "ETH/USD,ETH/USD,BTC/USD,BTC/USD",
            "--mode",
            "paper",
            "--once",
            "--shakedown-state-path",
            "/tmp/test-supervisor-dedupe.json",
        ]
        with self._patches()[0], self._patches()[1], self._patches()[2], \
                self._patches()[3], self._patches()[4], self._patches()[5]:
            rc = live_supervisor.main(argv)
        self.assertEqual(rc, 0)
        self.assertEqual(len(_RecordingSupervisor.instances), 1)
        self.assertEqual(
            _RecordingSupervisor.instances[0].config.symbols,
            ["ETH/USD", "BTC/USD"],
        )

    def test_only_duplicates_collapse_to_single_entry_not_empty(self) -> None:
        """``--symbols ETH/USD,ETH/USD`` keeps one entry, exits 0."""
        argv = [
            "--symbols",
            "ETH/USD,ETH/USD",
            "--mode",
            "paper",
            "--once",
            "--shakedown-state-path",
            "/tmp/test-supervisor-dedupe2.json",
        ]
        with self._patches()[0], self._patches()[1], self._patches()[2], \
                self._patches()[3], self._patches()[4], self._patches()[5]:
            rc = live_supervisor.main(argv)
        self.assertEqual(rc, 0)
        self.assertEqual(
            _RecordingSupervisor.instances[0].config.symbols, ["ETH/USD"]
        )

    def test_empty_symbols_returns_exit_code_2(self) -> None:
        """All-whitespace --symbols collapses to [] and exits 2."""
        argv = [
            "--symbols",
            ", , ,",
            "--mode",
            "paper",
            "--once",
        ]
        buf = io.StringIO()
        with self._patches()[0], self._patches()[1], self._patches()[2], \
                self._patches()[3], self._patches()[4], self._patches()[5], \
                redirect_stderr(buf):
            rc = live_supervisor.main(argv)
        self.assertEqual(rc, 2)
        self.assertIn("--symbols must contain at least one entry", buf.getvalue())


class _CapturingSupervisor:
    """Records the kwargs Supervisor was constructed with for assertion."""

    instances: List["_CapturingSupervisor"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = kwargs.get("config")
        self.auto_pause_gate = kwargs.get("auto_pause_gate")
        self.confidence_history = kwargs.get("confidence_history")
        _CapturingSupervisor.instances.append(self)

    def run_once(self) -> List[Any]:
        return []

    def run_loop(self, *, max_iterations: Any = None) -> dict:
        return {
            "iterations": 0,
            "total_ticks": 0,
            "action_counts": {},
            "interrupted": False,
        }


class AutoPauseCliWiringTests(unittest.TestCase):
    """Task 1: --auto-pause-* CLI flags wire AutoPauseGate + ConfidenceHistory."""

    def setUp(self) -> None:
        _CapturingSupervisor.instances = []

    def _patches(self):
        return [
            mock.patch.object(live_supervisor, "Supervisor", _CapturingSupervisor),
            mock.patch.object(
                live_supervisor, "CoinbaseExchange", lambda *a, **kw: object()
            ),
            mock.patch.object(
                live_supervisor, "PositionStore", lambda *a, **kw: object()
            ),
            mock.patch.object(
                live_supervisor, "CircuitBreakerSet", lambda *a, **kw: object()
            ),
            mock.patch.object(
                live_supervisor, "Notifier", lambda *a, **kw: object()
            ),
            mock.patch(
                "predictor.build_default_predict_fn", return_value=None
            ),
        ]

    def _run_main(self, argv: List[str]) -> int:
        ps = self._patches()
        with ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]:
            return live_supervisor.main(argv)

    def test_default_no_auto_pause_flag_means_gate_is_none(self) -> None:
        """Without --auto-pause-enabled, supervisor's gate must be None."""
        argv = [
            "--symbols",
            "ETH/USD",
            "--mode",
            "paper",
            "--once",
            "--shakedown-state-path",
            "/tmp/test-supervisor-no-autopause.json",
        ]
        rc = self._run_main(argv)
        self.assertEqual(rc, 0)
        self.assertEqual(len(_CapturingSupervisor.instances), 1)
        sup = _CapturingSupervisor.instances[0]
        # Gate stays None when the flag is absent (legacy default behavior).
        self.assertIsNone(sup.auto_pause_gate)
        # ConfidenceHistory is also None at the kwargs level — the
        # Supervisor's own __init__ defaults it to an in-process buffer.
        self.assertIsNone(sup.confidence_history)

    def test_auto_pause_enabled_constructs_gate_with_cli_tunables(self) -> None:
        """All four flags propagate into the AutoPauseGate + ConfidenceHistory."""
        from risk.auto_pause import AutoPauseGate
        from state.confidence_history import ConfidenceHistory

        argv = [
            "--symbols",
            "ETH/USD",
            "--mode",
            "paper",
            "--once",
            "--shakedown-state-path",
            "/tmp/test-supervisor-autopause-on.json",
            "--auto-pause-enabled",
            "--auto-pause-loss-pct",
            "0.03",
            "--auto-pause-confidence-window",
            "150",
            "--auto-pause-confidence-sigma",
            "2.5",
        ]
        rc = self._run_main(argv)
        self.assertEqual(rc, 0)
        self.assertEqual(len(_CapturingSupervisor.instances), 1)
        sup = _CapturingSupervisor.instances[0]
        self.assertIsInstance(sup.auto_pause_gate, AutoPauseGate)
        # Gate tunables pulled from argv.
        self.assertAlmostEqual(sup.auto_pause_gate.loss_threshold_pct, 0.03)
        self.assertAlmostEqual(sup.auto_pause_gate.z_threshold, 2.5)
        self.assertEqual(sup.auto_pause_gate.recent_window, 150)
        # Confidence history wired with the same window.
        self.assertIsInstance(sup.confidence_history, ConfidenceHistory)
        self.assertEqual(sup.confidence_history.window_size, 150)

    def test_auto_pause_default_loss_pct_when_only_flag_set(self) -> None:
        """Flag without explicit --auto-pause-loss-pct uses the 2% default."""
        argv = [
            "--symbols",
            "ETH/USD",
            "--mode",
            "paper",
            "--once",
            "--shakedown-state-path",
            "/tmp/test-supervisor-autopause-defaults.json",
            "--auto-pause-enabled",
        ]
        rc = self._run_main(argv)
        self.assertEqual(rc, 0)
        sup = _CapturingSupervisor.instances[0]
        self.assertIsNotNone(sup.auto_pause_gate)
        self.assertAlmostEqual(sup.auto_pause_gate.loss_threshold_pct, 0.02)
        self.assertAlmostEqual(sup.auto_pause_gate.z_threshold, 2.0)
        self.assertEqual(sup.auto_pause_gate.recent_window, 200)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
