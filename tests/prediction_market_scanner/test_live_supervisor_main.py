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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
