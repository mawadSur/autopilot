import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from llm_judge import LLMJudgeResult
from main import build_scan_results, export_scan_results, render_cli_table
from models import Market


class MainScannerTests(unittest.TestCase):
    def _market(self, **overrides):
        base = {
            "market_id": "mkt-1",
            "title": "Will event happen?",
            "category": "Politics",
            "implied_prob": 0.55,
            "bid_price": 0.50,
            "ask_price": 0.52,
            "volume_24h": 20000.0,
            "price_history": {"1h": 0.03, "6h": 0.04, "24h": 0.08},
            "open_interest": 25000.0,
            "resolution_date": datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
            "rules_text": "Resolves on official records.",
            "avg_volume_7d": 2500.0,
            "volume_change_1h": 0.05,
        }
        base.update(overrides)
        return Market(**base)

    def test_build_scan_results_sorts_descending_and_exports_exact_schema(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        markets = [
            self._market(
                market_id="alpha",
                title="Will Alpha happen?",
                bid_price=0.49,
                ask_price=0.51,
                volume_24h=60000.0,
                resolution_date=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
                avg_volume_7d=5000.0,
            ),
            self._market(
                market_id="beta",
                title="Will Beta happen?",
                bid_price=0.47,
                ask_price=0.49,
                volume_24h=15000.0,
                resolution_date=datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc),
                avg_volume_7d=10000.0,
            ),
            self._market(
                market_id="filtered",
                title="Will Filtered happen?",
                bid_price=0.40,
                ask_price=0.61,
                volume_24h=900.0,
                resolution_date=datetime(2026, 4, 20, 18, 0, tzinfo=timezone.utc),
                avg_volume_7d=200.0,
            ),
        ]

        def fake_fetch_markets(**kwargs):
            self.assertEqual(kwargs["min_volume_24h"], 5000.0)
            self.assertEqual(kwargs["page_size"], 100)
            self.assertIsNone(kwargs["max_pages"])
            return markets

        def fake_judge_market(title, rules_text):
            if "Alpha" in title:
                return LLMJudgeResult(clarity_score=92, narrative_momentum=70, anomaly_flags=[])
            return LLMJudgeResult(clarity_score=61, narrative_momentum=45, anomaly_flags=["AMBIGUOUS"])

        rows = build_scan_results(
            now=now,
            fetch_markets_fn=fake_fetch_markets,
            judge_market_fn=fake_judge_market,
        )

        self.assertEqual([row["market_id"] for row in rows], ["alpha", "beta"])
        self.assertGreater(rows[0]["research_priority"], rows[1]["research_priority"])
        self.assertEqual(
            list(rows[0].keys()),
            [
                "market_id",
                "title",
                "category",
                "implied_prob",
                "spread",
                "volume_24h",
                "move_24h",
                "days_to_resolution",
                "clarity_score",
                "anomaly_flags",
                "research_priority",
            ],
        )
        self.assertIn("VOL_SPIKE", rows[0]["anomaly_flags"])
        self.assertIn("INFO_EDGE", rows[0]["anomaly_flags"])
        self.assertIn("AMBIGUOUS", rows[1]["anomaly_flags"])

    def test_render_cli_table_limits_output_rows(self):
        rows = []
        for index in range(1, 26):
            rows.append(
                {
                    "market_id": f"mkt-{index}",
                    "title": f"Market {index:02d}",
                    "category": "Politics",
                    "implied_prob": 0.50,
                    "spread": 0.02,
                    "volume_24h": 10000.0 + index,
                    "move_24h": 0.01,
                    "days_to_resolution": 5.0,
                    "clarity_score": 80,
                    "anomaly_flags": ["INFO_EDGE"],
                    "research_priority": 100 - index,
                }
            )

        table = render_cli_table(rows, limit=20)

        self.assertIn("Market 01", table)
        self.assertIn("Market 20", table)
        self.assertNotIn("Market 21", table)
        self.assertIn("Priority", table)

    def test_export_scan_results_writes_timestamped_json(self):
        now = datetime(2026, 4, 20, 12, 30, tzinfo=timezone.utc)
        rows = [
            {
                "market_id": "alpha",
                "title": "Will Alpha happen?",
                "category": "Politics",
                "implied_prob": 0.51,
                "spread": 0.02,
                "volume_24h": 50000.0,
                "move_24h": 0.08,
                "days_to_resolution": 4.0,
                "clarity_score": 92,
                "anomaly_flags": ["VOL_SPIKE", "INFO_EDGE"],
                "research_priority": 91,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = export_scan_results(rows, output_dir=tmp_dir, now=now)
            self.assertEqual(output_path.name, "scan_20260420T123000Z.json")
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(payload, rows)


if __name__ == "__main__":
    unittest.main()
