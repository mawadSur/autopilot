import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from calibration_agent.ml_service import get_xgboost_probability
from models import Market


class CalibrationBaselineTests(unittest.TestCase):
    def _market(self, *, implied_prob=0.52) -> Market:
        return Market(
            market_id="mkt-1",
            title="Test Market",
            category="Politics",
            implied_prob=implied_prob,
            bid_price=max(0.0, implied_prob - 0.01),
            ask_price=min(1.0, implied_prob + 0.01),
            volume_24h=15000.0,
            price_history={"1h": 0.01, "6h": 0.02, "24h": 0.03},
            open_interest=40000.0,
            resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if the event occurs.",
        )

    def test_returns_jittered_probability_from_market(self):
        with patch("calibration_agent.ml_service.random.uniform", return_value=0.02):
            probability = get_xgboost_probability(self._market(implied_prob=0.52))

        self.assertAlmostEqual(probability, 0.54)

    def test_clamps_probability_to_zero_and_one(self):
        with patch("calibration_agent.ml_service.random.uniform", return_value=0.02):
            self.assertEqual(get_xgboost_probability(self._market(implied_prob=0.99)), 1.0)

        with patch("calibration_agent.ml_service.random.uniform", return_value=-0.02):
            self.assertEqual(get_xgboost_probability(self._market(implied_prob=0.01)), 0.0)

    def test_rejects_non_market_inputs(self):
        with self.assertRaisesRegex(TypeError, "Market"):
            get_xgboost_probability({"implied_prob": 0.5})


if __name__ == "__main__":
    unittest.main()
