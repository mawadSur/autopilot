"""Unit tests for the hard-safety circuit breakers in ``src/risk/circuit_breakers.py``.

These breakers sit between the soft-penalty risk layer (Kelly + correlation
penalties) and the exchange ``place_order`` call. They are the last
opportunity to refuse a trade. The tests below exercise:

  * Schema validation for ``DecisionContext`` and ``CircuitBreakerVerdict``.
  * Each individual breaker's trip / pass behaviour at the boundary.
  * Aggregation when multiple breakers fire at once.
  * Kill-switch file lifecycle (presence, reset).
  * Env-var fallback and disabled-breaker INFO warning.

Tests are hermetic — every kill-switch path uses ``tempfile`` and every
env-var lookup is wrapped in ``patch.dict(os.environ, ...)``.
"""
from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from risk.circuit_breakers import (
    CircuitBreakerSet,
    CircuitBreakerVerdict,
    DecisionContext,
)


# A minimal env-clearing patch dict — every test that exercises env
# fallback constructs its own. The default-fixture tests use this to
# guarantee the breakers see a clean slate.
_CLEAN_ENV = {
    "DAILY_LOSS_LIMIT_USD": "",
    "MAX_DRAWDOWN_PCT": "",
    "MAX_TOTAL_NOTIONAL_USD": "",
    "MAX_PER_SYMBOL_NOTIONAL_USD": "",
    "KILL_SWITCH_FILE": "",
}


def _make_ctx(**overrides) -> DecisionContext:
    """Build a benign DecisionContext that no default breaker would trip."""
    base = dict(
        symbol="ETH-USD",
        side="buy",
        proposed_notional_usd=100.0,
        current_open_notional_usd=0.0,
        current_per_symbol_notional_usd=0.0,
        daily_realized_pnl_usd=0.0,
        equity_peak_usd=10_000.0,
        equity_current_usd=10_000.0,
        as_of_utc="2026-04-26T00:00:00Z",
    )
    base.update(overrides)
    return DecisionContext(**base)


class DecisionContextSchemaTests(unittest.TestCase):
    def test_valid_context_round_trips(self):
        ctx = _make_ctx()
        self.assertEqual(ctx.symbol, "ETH-USD")
        self.assertEqual(ctx.side, "buy")

    def test_extra_field_is_forbidden(self):
        with self.assertRaises(ValidationError):
            DecisionContext(
                symbol="ETH-USD",
                side="buy",
                proposed_notional_usd=100.0,
                current_open_notional_usd=0.0,
                current_per_symbol_notional_usd=0.0,
                daily_realized_pnl_usd=0.0,
                equity_peak_usd=10_000.0,
                equity_current_usd=10_000.0,
                as_of_utc="2026-04-26T00:00:00Z",
                stray_field="nope",
            )

    def test_negative_proposed_notional_rejected(self):
        with self.assertRaises(ValidationError):
            _make_ctx(proposed_notional_usd=-1.0)

    def test_invalid_side_rejected(self):
        with self.assertRaises(ValidationError):
            _make_ctx(side="hold")


class CircuitBreakerVerdictSchemaTests(unittest.TestCase):
    def test_valid_verdict_construction(self):
        v = CircuitBreakerVerdict(
            allow=True,
            tripped=[],
            reason="",
            recommended_action="allow",
            details={},
        )
        self.assertTrue(v.allow)
        self.assertEqual(v.recommended_action, "allow")

    def test_invalid_recommended_action_rejected(self):
        with self.assertRaises(ValidationError):
            CircuitBreakerVerdict(
                allow=False,
                tripped=["x"],
                reason="x",
                recommended_action="abort_universe",  # type: ignore[arg-type]
                details={},
            )


class NoBreakersConfiguredTests(unittest.TestCase):
    def test_no_breakers_configured_allows_with_warning(self):
        """All args None and env empty => allow=True; one INFO log per breaker."""
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet()

            with self.assertLogs("risk.circuit_breakers", level="INFO") as cap:
                verdict = breakers.check(_make_ctx())

            self.assertTrue(verdict.allow)
            self.assertEqual(verdict.tripped, [])
            self.assertEqual(verdict.recommended_action, "allow")

            # First call emits one INFO line per disabled breaker (5 total).
            joined = "\n".join(cap.output)
            for name in (
                "kill_switch",
                "daily_loss",
                "drawdown",
                "total_notional",
                "per_symbol_notional",
            ):
                self.assertIn(name, joined)

            # Second call must not re-emit; assertNoLogs would error if any
            # were emitted, so we use a manual capture.
            with self.assertLogs("risk.circuit_breakers", level="INFO") as cap2:
                # Emit a sentinel so assertLogs doesn't raise on empty.
                logging.getLogger("risk.circuit_breakers").info("sentinel")
                breakers.check(_make_ctx())
            # Only the sentinel line should be present.
            self.assertEqual(len(cap2.output), 1)
            self.assertIn("sentinel", cap2.output[0])


class KillSwitchTests(unittest.TestCase):
    def test_kill_switch_file_present_forces_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kill.flag"
            path.write_text("halt")
            breakers = CircuitBreakerSet(kill_switch_file=path)

            verdict = breakers.check(_make_ctx())

            self.assertFalse(verdict.allow)
            self.assertIn("kill_switch", verdict.tripped)
            self.assertEqual(verdict.recommended_action, "force_flat")
            self.assertTrue(breakers.is_kill_switch_tripped())

    def test_reset_kill_switch_removes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kill.flag"
            path.write_text("halt")
            breakers = CircuitBreakerSet(kill_switch_file=path)

            self.assertTrue(breakers.is_kill_switch_tripped())
            self.assertTrue(breakers.reset_kill_switch())
            self.assertFalse(path.exists())
            self.assertFalse(breakers.is_kill_switch_tripped())

    def test_reset_kill_switch_returns_false_when_not_tripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kill.flag"  # never created
            breakers = CircuitBreakerSet(kill_switch_file=path)
            self.assertFalse(breakers.reset_kill_switch())


class DailyLossLimitTests(unittest.TestCase):
    def test_daily_loss_limit_tripped(self):
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(daily_loss_limit_usd=1000.0)
            verdict = breakers.check(_make_ctx(daily_realized_pnl_usd=-2000.0))

            self.assertFalse(verdict.allow)
            self.assertIn("daily_loss", verdict.tripped)
            self.assertEqual(verdict.recommended_action, "halt_new_entries")
            self.assertTrue(verdict.details["daily_loss"]["tripped"])
            self.assertEqual(verdict.details["daily_loss"]["limit_usd"], 1000.0)
            self.assertEqual(verdict.details["daily_loss"]["actual_usd"], -2000.0)

    def test_daily_loss_limit_not_tripped_under_threshold(self):
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(daily_loss_limit_usd=1000.0)
            verdict = breakers.check(_make_ctx(daily_realized_pnl_usd=-500.0))

            self.assertTrue(verdict.allow)
            self.assertEqual(verdict.tripped, [])
            self.assertFalse(verdict.details["daily_loss"]["tripped"])


class DrawdownTests(unittest.TestCase):
    def test_drawdown_tripped(self):
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(max_drawdown_pct=5.0)
            verdict = breakers.check(
                _make_ctx(equity_peak_usd=10_000.0, equity_current_usd=9_400.0)
            )

            self.assertFalse(verdict.allow)
            self.assertIn("drawdown", verdict.tripped)
            self.assertEqual(verdict.recommended_action, "halt_new_entries")
            self.assertAlmostEqual(
                verdict.details["drawdown"]["actual_pct"], 6.0, places=4
            )

    def test_drawdown_not_tripped(self):
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(max_drawdown_pct=5.0)
            verdict = breakers.check(
                _make_ctx(equity_peak_usd=10_000.0, equity_current_usd=9_550.0)
            )

            self.assertTrue(verdict.allow)
            self.assertFalse(verdict.details["drawdown"]["tripped"])


class NotionalCapTests(unittest.TestCase):
    def test_total_notional_cap_tripped(self):
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(max_total_notional_usd=20_000.0)
            verdict = breakers.check(
                _make_ctx(
                    current_open_notional_usd=18_000.0,
                    proposed_notional_usd=3_000.0,
                )
            )

            self.assertFalse(verdict.allow)
            self.assertIn("total_notional", verdict.tripped)
            self.assertEqual(verdict.recommended_action, "halt_new_entries")
            self.assertEqual(
                verdict.details["total_notional"]["projected_usd"], 21_000.0
            )

    def test_per_symbol_cap_tripped(self):
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(max_per_symbol_notional_usd=5_000.0)
            verdict = breakers.check(
                _make_ctx(
                    symbol="BTC-USD",
                    current_per_symbol_notional_usd=4_500.0,
                    proposed_notional_usd=1_000.0,
                )
            )

            self.assertFalse(verdict.allow)
            self.assertIn("per_symbol_notional", verdict.tripped)
            self.assertEqual(verdict.recommended_action, "halt_new_entries")
            self.assertEqual(
                verdict.details["per_symbol_notional"]["symbol"], "BTC-USD"
            )

    def test_zero_proposed_notional_does_not_trip_caps(self):
        """A zero-size proposal cannot push exposure higher, so caps must pass.

        This guards against a regression where ``current + 0 > cap`` would
        be flagged as a breach when it merely reflects the existing book.
        """
        with patch.dict(os.environ, _CLEAN_ENV, clear=False):
            breakers = CircuitBreakerSet(
                max_total_notional_usd=20_000.0,
                max_per_symbol_notional_usd=5_000.0,
            )
            verdict = breakers.check(
                _make_ctx(
                    proposed_notional_usd=0.0,
                    current_open_notional_usd=25_000.0,
                    current_per_symbol_notional_usd=8_000.0,
                )
            )

            self.assertTrue(verdict.allow)
            self.assertEqual(verdict.tripped, [])
            self.assertFalse(verdict.details["total_notional"]["tripped"])
            self.assertFalse(verdict.details["per_symbol_notional"]["tripped"])


class MultipleBreakerTests(unittest.TestCase):
    def test_multiple_tripped_returns_most_severe_action(self):
        """Kill switch + daily loss => force_flat (most severe) wins."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kill.flag"
            path.write_text("halt")
            breakers = CircuitBreakerSet(
                kill_switch_file=path, daily_loss_limit_usd=1000.0
            )
            verdict = breakers.check(_make_ctx(daily_realized_pnl_usd=-5000.0))

            self.assertFalse(verdict.allow)
            self.assertIn("kill_switch", verdict.tripped)
            self.assertIn("daily_loss", verdict.tripped)
            self.assertEqual(verdict.recommended_action, "force_flat")


class EnvFallbackTests(unittest.TestCase):
    def test_env_vars_used_when_args_omitted(self):
        env = dict(_CLEAN_ENV)
        env.update(
            {
                "DAILY_LOSS_LIMIT_USD": "500",
                "MAX_DRAWDOWN_PCT": "3.0",
                "MAX_TOTAL_NOTIONAL_USD": "15000",
                "MAX_PER_SYMBOL_NOTIONAL_USD": "2500",
            }
        )
        with patch.dict(os.environ, env, clear=False):
            breakers = CircuitBreakerSet()

            self.assertEqual(breakers.daily_loss_limit_usd, 500.0)
            self.assertEqual(breakers.max_drawdown_pct, 3.0)
            self.assertEqual(breakers.max_total_notional_usd, 15_000.0)
            self.assertEqual(breakers.max_per_symbol_notional_usd, 2_500.0)

            # Sanity: a context that breaches the env-driven daily-loss
            # limit must trip even though we passed nothing to the ctor.
            verdict = breakers.check(_make_ctx(daily_realized_pnl_usd=-600.0))
            self.assertFalse(verdict.allow)
            self.assertIn("daily_loss", verdict.tripped)


if __name__ == "__main__":
    unittest.main()
