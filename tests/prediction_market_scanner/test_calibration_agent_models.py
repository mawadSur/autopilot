import unittest

from pydantic import ValidationError

from calibration_agent.models import CalibrationReport


class CalibrationReportTests(unittest.TestCase):
    def test_valid_report_passes(self):
        report = CalibrationReport(
            xgboost_prob=0.58,
            llm_adjustment_pct_points=2.5,
            calibrated_true_prob=0.605,
            confidence_score=72,
            key_drivers=["Fresh poll data improved the base rate.", "Resolution language is objective."],
            key_uncertainties=["Turnout assumptions remain noisy.", "Late macro headlines could shift sentiment."],
            edge_vs_market=0.055,
            action="monitor",
            reasoning="The baseline moved modestly higher because the qualitative evidence is directionally supportive but not decisive.",
        )

        self.assertEqual(report.action, "monitor")
        self.assertAlmostEqual(report.calibrated_true_prob, 0.605)

    def test_invalid_probability_range_fails(self):
        with self.assertRaises(ValidationError):
            CalibrationReport(
                xgboost_prob=1.2,
                llm_adjustment_pct_points=0.0,
                calibrated_true_prob=0.6,
                confidence_score=50,
                key_drivers=["Driver"],
                key_uncertainties=["Uncertainty"],
                edge_vs_market=0.1,
                action="pass",
                reasoning="Invalid baseline probability.",
            )

    def test_invalid_action_fails(self):
        with self.assertRaises(ValidationError):
            CalibrationReport(
                xgboost_prob=0.55,
                llm_adjustment_pct_points=-1.0,
                calibrated_true_prob=0.54,
                confidence_score=60,
                key_drivers=["Driver"],
                key_uncertainties=["Uncertainty"],
                edge_vs_market=-0.02,
                action="buy",
                reasoning="Unsupported action.",
            )

    def test_blank_list_entry_fails(self):
        with self.assertRaises(ValidationError):
            CalibrationReport(
                xgboost_prob=0.55,
                llm_adjustment_pct_points=-1.0,
                calibrated_true_prob=0.54,
                confidence_score=60,
                key_drivers=["Driver", "   "],
                key_uncertainties=["Uncertainty"],
                edge_vs_market=-0.02,
                action="pass",
                reasoning="Blank supporting evidence should not validate.",
            )

    def test_json_schema_includes_expected_fields(self):
        schema = CalibrationReport.model_json_schema()
        properties = schema["properties"]

        self.assertEqual(
            set(properties),
            {
                "xgboost_prob",
                "llm_adjustment_pct_points",
                "calibrated_true_prob",
                "confidence_score",
                "key_drivers",
                "key_uncertainties",
                "edge_vs_market",
                "action",
                "reasoning",
            },
        )
        self.assertEqual(properties["confidence_score"]["minimum"], 0)
        self.assertEqual(properties["confidence_score"]["maximum"], 100)
        self.assertEqual(
            properties["action"]["enum"],
            ["pass", "monitor", "paper-trade candidate"],
        )
        self.assertEqual(
            properties["xgboost_prob"]["description"],
            "The baseline probability from the ML model.",
        )


if __name__ == "__main__":
    unittest.main()
