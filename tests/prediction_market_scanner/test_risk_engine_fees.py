"""Tests for the Polymarket fee deduction inside ``RiskCalculator`` (P1 #9)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from models import Market
from risk_management_agent.risk_engine import (
    RiskCalculator,
    apply_polymarket_fees,
)


def _make_market(*, implied_prob: float = 0.50) -> Market:
    return Market(
        market_id="test-market",
        title="Will the test pass?",
        category="Politics",
        implied_prob=implied_prob,
        bid_price=max(0.01, implied_prob - 0.01),
        ask_price=min(0.99, implied_prob + 0.01),
        volume_24h=50_000.0,
        price_history={"24h_ago": implied_prob, "1h_ago": implied_prob},
        open_interest=10_000.0,
        resolution_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        rules_text="Resolves YES iff the test passes.",
    )


class ApplyPolymarketFeesTests(unittest.TestCase):
    def test_zero_fee_is_passthrough(self) -> None:
        adjusted = apply_polymarket_fees(0.6, 0.5, polymarket_fee_bps=0)
        self.assertAlmostEqual(adjusted, 0.6)

    def test_two_percent_fee_shrinks_edge(self) -> None:
        # gross edge 0.10 -> after 2% fee -> 0.10 * 0.98 = 0.098 -> p_adj = 0.598
        adjusted = apply_polymarket_fees(0.6, 0.5, polymarket_fee_bps=200)
        self.assertAlmostEqual(adjusted, 0.598, places=6)

    def test_negative_edge_passes_through(self) -> None:
        # When the trader has no edge, fees should not amplify the loss.
        adjusted = apply_polymarket_fees(0.4, 0.5, polymarket_fee_bps=200)
        self.assertAlmostEqual(adjusted, 0.4)


class RiskCalculatorFeesTests(unittest.TestCase):
    def test_kelly_with_fees_strictly_smaller_than_no_fees(self) -> None:
        market = _make_market(implied_prob=0.50)
        no_fees = RiskCalculator(polymarket_fee_bps=0)
        with_fees = RiskCalculator(polymarket_fee_bps=200)

        m_no = no_fees.calculate_base_metrics(
            market=market,
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
        )
        m_yes = with_fees.calculate_base_metrics(
            market=market,
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
        )

        self.assertGreater(m_no.raw_kelly_size_pct, 0.0)
        self.assertGreater(m_yes.raw_kelly_size_pct, 0.0)
        self.assertLess(m_yes.raw_kelly_size_pct, m_no.raw_kelly_size_pct)
        self.assertLess(m_yes.adjusted_position_size_pct, m_no.adjusted_position_size_pct)

    def test_ev_shrinks_by_approximately_fee_amount(self) -> None:
        market = _make_market(implied_prob=0.50)
        no_fees = RiskCalculator(polymarket_fee_bps=0)
        with_fees = RiskCalculator(polymarket_fee_bps=200)

        m_no = no_fees.calculate_base_metrics(
            market=market,
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
        )
        m_yes = with_fees.calculate_base_metrics(
            market=market,
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
        )

        # The position sizing is also smaller with fees (Kelly was reduced),
        # so EV should be strictly lower. We additionally check the relative
        # shrinkage is in the right ballpark: the per-dollar EV (EV / notional)
        # should drop by ~2% (the fee).
        self.assertGreater(m_no.expected_value_estimate, 0.0)
        self.assertGreater(m_yes.expected_value_estimate, 0.0)
        self.assertLess(m_yes.expected_value_estimate, m_no.expected_value_estimate)

        notional_no = m_no.bankroll * m_no.adjusted_position_size_pct / 100.0
        notional_yes = m_yes.bankroll * m_yes.adjusted_position_size_pct / 100.0
        ev_per_notional_no = m_no.expected_value_estimate / notional_no
        ev_per_notional_yes = m_yes.expected_value_estimate / notional_yes
        # Per-notional EV should shrink by ~2% (the fee), since the gross edge
        # (0.10 / 0.50 = 20% per dollar) becomes (0.098 / 0.50 = 19.6%).
        ratio = ev_per_notional_yes / ev_per_notional_no
        self.assertAlmostEqual(ratio, 0.98, places=2)

    def test_zero_edge_remains_zero_with_fees(self) -> None:
        market = _make_market(implied_prob=0.50)
        with_fees = RiskCalculator(polymarket_fee_bps=200)
        metrics = with_fees.calculate_base_metrics(
            market=market,
            calibrated_true_prob=0.50,
            bankroll=10_000.0,
        )
        self.assertEqual(metrics.raw_kelly_size_pct, 0.0)
        self.assertEqual(metrics.expected_value_estimate, 0.0)


if __name__ == "__main__":
    unittest.main()
