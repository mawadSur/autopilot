import unittest
from datetime import datetime, timezone

from models import Market
from ranker import calculate_priority


class RankerTests(unittest.TestCase):
    def _market(self, **overrides):
        base = {
            "market_id": "mkt-1",
            "title": "Will event happen?",
            "category": "Politics",
            "implied_prob": 0.55,
            "bid_price": 0.50,
            "ask_price": 0.52,
            "volume_24h": 20000.0,
            "price_history": {"1h": 0.0, "6h": 0.0, "24h": 0.0},
            "open_interest": 25000.0,
            "resolution_date": datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
            "rules_text": "Resolves on official records.",
        }
        base.update(overrides)
        return Market(**base)

    def test_calculate_priority_rewards_liquidity_sweet_spot_and_positive_anomalies(self):
        market = self._market(
            bid_price=0.48,
            ask_price=0.50,
            volume_24h=50000.0,
            resolution_date=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
        )

        assessment = calculate_priority(
            market,
            anomaly_flags=["VOL_SPIKE", "DECOUPLED"],
            clarity_score=90.0,
            now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        )

        self.assertGreaterEqual(assessment.research_priority, 85)
        self.assertGreaterEqual(assessment.component_scores["liquidity"], 80.0)
        self.assertEqual(assessment.component_scores["time_to_resolution"], 100.0)
        self.assertIn("volume spike", assessment.reason.lower())
        self.assertIn("clear resolution rules", assessment.reason.lower())

    def test_calculate_priority_penalizes_weak_liquidity_and_unclear_rules(self):
        market = self._market(
            bid_price=0.40,
            ask_price=0.58,
            volume_24h=800.0,
            resolution_date=datetime(2026, 4, 20, 18, 0, tzinfo=timezone.utc),
        )

        assessment = calculate_priority(
            market,
            anomaly_flags=["AMBIGUOUS", "WIDE_SPREAD"],
            clarity_score=35.0,
            now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        )

        self.assertLessEqual(assessment.research_priority, 20)
        self.assertEqual(assessment.component_scores["anomalies"], 0.0)
        self.assertIn("wide spread", assessment.reason.lower())
        self.assertTrue(
            "unclear resolution rules" in assessment.reason.lower()
            or "very near resolution" in assessment.reason.lower()
        )

    def test_calculate_priority_returns_weighted_component_scores(self):
        market = self._market(
            bid_price=0.49,
            ask_price=0.53,
            volume_24h=10000.0,
            resolution_date=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
        )

        assessment = calculate_priority(
            market,
            anomaly_flags=["INFO_EDGE"],
            clarity_score=80.0,
            now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        )

        expected = round(
            (0.40 * assessment.component_scores["liquidity"])
            + (0.20 * assessment.component_scores["time_to_resolution"])
            + (0.20 * assessment.component_scores["anomalies"])
            + (0.20 * assessment.component_scores["clarity"])
        )
        self.assertEqual(assessment.research_priority, expected)
        self.assertIn("3-10 day", assessment.reason.lower())


if __name__ == "__main__":
    unittest.main()
