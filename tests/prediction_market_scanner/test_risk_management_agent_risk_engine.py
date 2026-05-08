import unittest
from datetime import datetime, timezone

from models import Market
from risk_management_agent.models import RiskMetrics
from risk_management_agent.risk_engine import RiskCalculator


class RiskCalculatorTests(unittest.TestCase):
    def _market(self, *, implied_prob=0.55, bid_price=0.53, ask_price=0.57, volume_24h=20000.0, category="Politics") -> Market:
        return Market(
            market_id="mkt-1",
            title="Test Market",
            category=category,
            implied_prob=implied_prob,
            bid_price=bid_price,
            ask_price=ask_price,
            volume_24h=volume_24h,
            price_history={"1h": 0.01, "6h": 0.02, "24h": 0.03},
            open_interest=40000.0,
            resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if the event occurs.",
        )

    def test_calculates_raw_and_fractional_kelly_without_penalties(self):
        # Polymarket fees are now deducted by default (P1 #9). To exercise the
        # pre-fee math this test always cared about, opt out with fee_bps=0.
        calculator = RiskCalculator(polymarket_fee_bps=0)

        metrics = calculator.calculate_base_metrics(
            market=self._market(),
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
            market_price=0.55,
            existing_open_positions=[],
        )

        self.assertIsInstance(metrics, RiskMetrics)
        self.assertAlmostEqual(metrics.raw_kelly_size_pct, (0.60 - 0.55) / (1.0 - 0.55) * 100.0, places=6)
        self.assertAlmostEqual(metrics.fractional_kelly_size_pct, metrics.raw_kelly_size_pct * 0.25, places=6)
        self.assertAlmostEqual(metrics.adjusted_position_size_pct, metrics.fractional_kelly_size_pct, places=6)
        self.assertFalse(metrics.liquidity_penalty_applied)
        self.assertFalse(metrics.correlation_penalty_applied)
        self.assertAlmostEqual(metrics.max_loss_if_wrong, 10_000.0 * metrics.adjusted_position_size_pct / 100.0, places=6)

    def test_applies_liquidity_penalty_for_wide_spread(self):
        calculator = RiskCalculator()

        metrics = calculator.calculate_base_metrics(
            market=self._market(bid_price=0.50, ask_price=0.56),
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
            market_price=0.55,
        )

        self.assertTrue(metrics.liquidity_penalty_applied)
        self.assertEqual(metrics.liquidity_penalty_multiplier, 0.5)
        self.assertAlmostEqual(
            metrics.adjusted_position_size_pct,
            metrics.fractional_kelly_size_pct * 0.5,
            places=6,
        )

    def test_applies_liquidity_penalty_for_low_volume(self):
        calculator = RiskCalculator()

        metrics = calculator.calculate_base_metrics(
            market=self._market(volume_24h=5_000.0),
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
            market_price=0.55,
        )

        self.assertTrue(metrics.liquidity_penalty_applied)
        self.assertEqual(metrics.liquidity_penalty_multiplier, 0.5)

    def test_applies_linear_correlation_penalty_for_same_category_positions(self):
        calculator = RiskCalculator()
        existing_positions = [
            self._market(category="Politics"),
            {"category": "politics"},
            self._market(category="Crypto"),
        ]

        metrics = calculator.calculate_base_metrics(
            market=self._market(category="Politics"),
            calibrated_true_prob=0.60,
            bankroll=10_000.0,
            market_price=0.55,
            existing_open_positions=existing_positions,
        )

        self.assertTrue(metrics.correlation_penalty_applied)
        self.assertEqual(metrics.same_category_open_positions, 2)
        self.assertAlmostEqual(metrics.correlation_penalty_multiplier, 0.4, places=6)
        self.assertAlmostEqual(
            metrics.adjusted_position_size_pct,
            metrics.fractional_kelly_size_pct * 0.4,
            places=6,
        )

    def test_zero_or_negative_edge_results_in_zero_size(self):
        calculator = RiskCalculator()

        metrics = calculator.calculate_base_metrics(
            market=self._market(implied_prob=0.60),
            calibrated_true_prob=0.55,
            bankroll=10_000.0,
            market_price=0.60,
        )

        self.assertEqual(metrics.raw_kelly_size_pct, 0.0)
        self.assertEqual(metrics.fractional_kelly_size_pct, 0.0)
        self.assertEqual(metrics.adjusted_position_size_pct, 0.0)
        self.assertEqual(metrics.max_loss_if_wrong, 0.0)
        self.assertEqual(metrics.expected_value_estimate, 0.0)


if __name__ == "__main__":
    unittest.main()
