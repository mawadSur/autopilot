import unittest
from datetime import datetime, timezone

from analyzer import (
    attach_category_spread_averages,
    evaluate_market,
    get_filter_reasons,
)
from models import Market


class AnalyzerTests(unittest.TestCase):
    def _market(self, **overrides):
        base = {
            "market_id": "mkt-1",
            "title": "Will event happen?",
            "category": "Politics",
            "implied_prob": 0.52,
            "bid_price": 0.50,
            "ask_price": 0.54,
            "volume_24h": 6000.0,
            "price_history": {"1h": 0.0, "6h": 0.0, "24h": 0.0},
            "open_interest": 20000.0,
            "resolution_date": datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
            "rules_text": "Resolves on the official source.",
            "avg_volume_7d": 1500.0,
            "volume_change_1h": 0.2,
        }
        base.update(overrides)
        return Market(**base)

    def test_get_filter_reasons_catches_spread_volume_and_resolution_filters(self):
        market = self._market(
            ask_price=0.80,
            volume_24h=500.0,
            resolution_date=datetime(2026, 4, 20, 18, 0, tzinfo=timezone.utc),
        )

        reasons = get_filter_reasons(
            market,
            now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(reasons, ["SPREAD_TOO_WIDE", "LOW_VOLUME", "NEAR_RESOLUTION"])
        self.assertEqual(evaluate_market(market, now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)), [])

    def test_evaluate_market_flags_volume_spike_decoupled_wide_spread_and_info_edge(self):
        quiet_a = self._market(market_id="quiet-a", bid_price=0.50, ask_price=0.51)
        quiet_b = self._market(market_id="quiet-b", bid_price=0.48, ask_price=0.49)
        quiet_c = self._market(market_id="quiet-c", bid_price=0.47, ask_price=0.48)
        target = self._market(
            market_id="target",
            bid_price=0.40,
            ask_price=0.44,
            price_history={"1h": 0.08, "6h": 0.01, "24h": 0.02},
            volume_24h=9000.0,
            avg_volume_7d=1500.0,
            volume_change_1h=0.05,
            resolution_date=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
        )
        markets = [quiet_a, quiet_b, quiet_c, target]

        attach_category_spread_averages(
            markets,
            now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        )
        flags = evaluate_market(target, now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc))

        self.assertEqual(flags, ["VOL_SPIKE", "DECOUPLED", "WIDE_SPREAD", "INFO_EDGE"])

    def test_attach_category_spread_averages_skips_filtered_markets_in_baseline(self):
        liquid = self._market(market_id="liquid", bid_price=0.50, ask_price=0.52)
        stale = self._market(
            market_id="stale",
            bid_price=0.20,
            ask_price=0.50,
            volume_24h=400.0,
        )
        attach_category_spread_averages(
            [liquid, stale],
            now=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        )

        self.assertAlmostEqual(liquid.category_avg_spread, 0.02)
        self.assertAlmostEqual(stale.category_avg_spread, 0.02)


if __name__ == "__main__":
    unittest.main()
