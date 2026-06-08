"""Characterization tests for the orchestrator kill-switch boundaries.

The kill-switch band was tightened from ``[0.01, 0.99]`` to ``[0.02, 0.98]``
in P1 #10. The actual band check now lives in
``RiskCalculator.passes_market_filters`` and the orchestrator delegates to
it. The boundary endpoints we lock in here are:

    p <= 0.02 -> kill switch (rejects 0.01 and 0.005)
    p >= 0.98 -> kill switch (rejects 0.99 and 0.995)

(0.02 and 0.98 themselves PASS the filter — the band is inclusive on the
*safe* side.) See P1 #10: Kelly explodes at p > 0.95, so the safety
margin needed to be wider.

Note on RiskMetrics.market_price field bounds: pydantic enforces
``gt=0.0, lt=1.0``. 0.005 and 0.995 are both valid (strictly between
0 and 1). The kill switch is a separate, tighter check.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from calibration_agent.models import CalibrationReport
from models import Market
from risk_management_agent.models import RiskAssessment
from risk_management_agent.risk_engine import RiskCalculator

from orchestrator import _build_risk_assessment


def _calibration(*, calibrated_true_prob: float, edge: float) -> CalibrationReport:
    return CalibrationReport(
        xgboost_prob=max(0.0, min(1.0, calibrated_true_prob - edge)),
        llm_adjustment_pct_points=edge * 100.0,
        calibrated_true_prob=calibrated_true_prob,
        confidence_score=80,
        key_drivers=["A driver"],
        key_uncertainties=["An uncertainty"],
        edge_vs_market=edge,
        action="paper-trade candidate",
        reasoning="Stub calibration for kill-switch boundary test.",
    )


def _market_at_price(implied_prob: float) -> Market:
    """Build a Market priced at ``implied_prob`` with bid/ask hugging the price.

    The bid/ask must remain strictly inside (0, 1) and produce a non-zero
    spread, which keeps RiskMetrics validation happy. We clamp the bid/ask
    by 0.001 from the implied price so spread > 0 but the price field is
    exactly the boundary value we care about.
    """
    eps = 0.0005
    bid = max(0.0001, implied_prob - eps)
    ask = min(0.9999, implied_prob + eps)
    return Market(
        market_id=f"mkt-kill-{implied_prob}",
        title="Kill-switch boundary market",
        category="Politics",
        implied_prob=implied_prob,
        bid_price=bid,
        ask_price=ask,
        volume_24h=25_000.0,
        price_history={"1h": 0.0, "6h": 0.0, "24h": 0.0},
        open_interest=40_000.0,
        resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
        rules_text="Resolves to YES if event occurs.",
    )


class KillSwitchBoundaryTests(unittest.TestCase):
    """Direct tests on the kill-switch logic in ``_build_risk_assessment``."""

    def setUp(self) -> None:
        self.calculator = RiskCalculator()

    def _assess(self, market: Market, calibrated_true_prob: float) -> RiskAssessment:
        edge = calibrated_true_prob - market.implied_prob
        calibration = _calibration(
            calibrated_true_prob=calibrated_true_prob, edge=edge,
        )
        risk_metrics = self.calculator.calculate_base_metrics(
            market=market,
            calibrated_true_prob=calibrated_true_prob,
            bankroll=10_000.0,
        )
        return _build_risk_assessment(
            risk_metrics=risk_metrics, calibration=calibration,
        )

    def test_below_lower_boundary_005_triggers_kill_switch(self):
        """0.005 < 0.01, so kill_switch must trigger."""
        market = _market_at_price(0.005)
        # Use a clearly higher calibration so the only reason to reject is the kill switch.
        assessment = self._assess(market, calibrated_true_prob=0.55)

        self.assertTrue(
            assessment.kill_switch_triggered,
            msg="Expected kill_switch to trigger at market_price=0.005 (<= 0.01)",
        )
        self.assertFalse(assessment.allow_trade)
        self.assertEqual(assessment.final_recommendation, "reject")

    def test_at_lower_boundary_010_triggers_kill_switch_inclusive(self):
        """0.01 <= 0.01, so kill_switch is inclusive on the lower bound."""
        market = _market_at_price(0.01)
        assessment = self._assess(market, calibrated_true_prob=0.55)

        self.assertTrue(
            assessment.kill_switch_triggered,
            msg=(
                "Kill switch uses <= 0.01 (inclusive). 0.01 should reject. "
                "If this assertion ever flips, audit src/orchestrator.py:270."
            ),
        )
        self.assertFalse(assessment.allow_trade)
        self.assertEqual(assessment.final_recommendation, "reject")

    def test_at_upper_boundary_099_triggers_kill_switch_inclusive(self):
        """0.99 >= 0.99, so kill_switch is inclusive on the upper bound."""
        market = _market_at_price(0.99)
        # Calibration must stay in [0,1]; pick a sub-market value so edge is non-positive
        # and the only "kill" signal is the price boundary itself.
        assessment = self._assess(market, calibrated_true_prob=0.5)

        self.assertTrue(
            assessment.kill_switch_triggered,
            msg=(
                "Kill switch uses >= 0.99 (inclusive). 0.99 should reject. "
                "If this assertion ever flips, audit src/orchestrator.py:271."
            ),
        )
        self.assertFalse(assessment.allow_trade)
        self.assertEqual(assessment.final_recommendation, "reject")

    def test_above_upper_boundary_0995_triggers_kill_switch(self):
        """0.995 > 0.99, so kill_switch must trigger."""
        market = _market_at_price(0.995)
        assessment = self._assess(market, calibrated_true_prob=0.5)

        self.assertTrue(
            assessment.kill_switch_triggered,
            msg="Expected kill_switch to trigger at market_price=0.995 (>= 0.99)",
        )
        self.assertFalse(assessment.allow_trade)
        self.assertEqual(assessment.final_recommendation, "reject")

    def test_safe_interior_price_does_not_trigger_kill_switch(self):
        """Sanity: a safely-priced market with positive edge does NOT kill-switch."""
        market = _market_at_price(0.42)
        assessment = self._assess(market, calibrated_true_prob=0.55)

        self.assertFalse(
            assessment.kill_switch_triggered,
            msg="Kill switch must NOT trigger on a safe interior price with positive edge.",
        )


if __name__ == "__main__":
    unittest.main()
