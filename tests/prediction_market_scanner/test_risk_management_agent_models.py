import unittest

from pydantic import ValidationError

from risk_management_agent.models import RiskAssessment


class RiskAssessmentTests(unittest.TestCase):
    def test_valid_assessment_passes(self):
        assessment = RiskAssessment(
            allow_trade=True,
            simulated_position_size_pct=2.5,
            max_loss_if_wrong=125.0,
            expected_value_estimate=18.75,
            top_risk_reasons=[
                "Market liquidity is thinner than the scanner threshold suggests.",
                "News confirmation is still sparse.",
            ],
            kill_switch_triggered=False,
            final_recommendation="small",
            risk_logic_summary="Position size was capped at 2.5% because liquidity and evidence quality both impose conservative penalties.",
        )

        self.assertTrue(assessment.allow_trade)
        self.assertEqual(assessment.final_recommendation, "small")
        self.assertAlmostEqual(assessment.simulated_position_size_pct, 2.5)

    def test_invalid_position_size_range_fails(self):
        with self.assertRaises(ValidationError):
            RiskAssessment(
                allow_trade=False,
                simulated_position_size_pct=125.0,
                max_loss_if_wrong=0.0,
                expected_value_estimate=-5.0,
                top_risk_reasons=["Oversized risk."],
                kill_switch_triggered=True,
                final_recommendation="reject",
                risk_logic_summary="Position size cannot exceed bankroll.",
            )

    def test_invalid_recommendation_fails(self):
        with self.assertRaises(ValidationError):
            RiskAssessment(
                allow_trade=False,
                simulated_position_size_pct=0.0,
                max_loss_if_wrong=0.0,
                expected_value_estimate=-12.0,
                top_risk_reasons=["No edge after fees."],
                kill_switch_triggered=False,
                final_recommendation="buy",
                risk_logic_summary="Unsupported recommendation.",
            )

    def test_blank_risk_reason_fails(self):
        with self.assertRaises(ValidationError):
            RiskAssessment(
                allow_trade=False,
                simulated_position_size_pct=0.0,
                max_loss_if_wrong=0.0,
                expected_value_estimate=-1.0,
                top_risk_reasons=["   "],
                kill_switch_triggered=True,
                final_recommendation="reject",
                risk_logic_summary="Blank reasons should not validate.",
            )

    def test_schema_includes_expected_fields(self):
        schema = RiskAssessment.model_json_schema()
        properties = schema["properties"]

        self.assertEqual(
            set(properties),
            {
                "allow_trade",
                "simulated_position_size_pct",
                "max_loss_if_wrong",
                "expected_value_estimate",
                "top_risk_reasons",
                "kill_switch_triggered",
                "final_recommendation",
                "risk_logic_summary",
            },
        )
        self.assertEqual(properties["simulated_position_size_pct"]["minimum"], 0.0)
        self.assertEqual(properties["simulated_position_size_pct"]["maximum"], 100.0)
        self.assertEqual(
            properties["final_recommendation"]["enum"],
            ["reject", "small", "medium", "high-conviction paper trade"],
        )
        self.assertEqual(
            properties["allow_trade"]["description"],
            "True if the trade meets all safety and risk criteria.",
        )


if __name__ == "__main__":
    unittest.main()
