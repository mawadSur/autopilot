import unittest

from pydantic import ValidationError

from synthesis_agent.models import SynthesisReport


class SynthesisReportTests(unittest.TestCase):
    def test_report_accepts_valid_payload(self):
        report = SynthesisReport(
            implied_probability=0.42,
            narrative_direction="mixed",
            has_unique_evidence=True,
            reasons_market_is_right=[
                "The deadline is close and uncertainty is already priced in.",
                "The market has digested the same public reporting.",
                "Contradictory evidence remains unresolved.",
            ],
            reasons_market_is_wrong=[
                "A niche source surfaced new evidence after the last price move.",
                "The crowd is overweighting stale consensus.",
                "The market may be underestimating one key catalyst.",
            ],
            verdict="possible edge",
            explanation="There may be some informational edge, but it is not yet decisive.",
        )

        self.assertEqual(report.verdict, "possible edge")
        self.assertEqual(len(report.reasons_market_is_right), 3)
        self.assertTrue(report.has_unique_evidence)

    def test_report_rejects_reason_lists_with_wrong_length(self):
        with self.assertRaises(ValidationError):
            SynthesisReport(
                implied_probability=0.55,
                narrative_direction="bullish",
                has_unique_evidence=False,
                reasons_market_is_right=["One", "Two"],
                reasons_market_is_wrong=["One", "Two", "Three"],
                verdict="no edge",
                explanation="Explanation",
            )

    def test_report_rejects_invalid_verdict(self):
        with self.assertRaises(ValidationError):
            SynthesisReport(
                implied_probability=0.55,
                narrative_direction="bearish",
                has_unique_evidence=False,
                reasons_market_is_right=["One", "Two", "Three"],
                reasons_market_is_wrong=["One", "Two", "Three"],
                verdict="weak edge",
                explanation="Explanation",
            )

    def test_schema_includes_descriptions_and_exact_reason_lengths(self):
        schema = SynthesisReport.model_json_schema()
        properties = schema["properties"]

        self.assertEqual(
            properties["implied_probability"]["description"],
            "Current market-implied probability (e.g. 0.45 for 45%)",
        )
        self.assertEqual(
            properties["verdict"]["description"],
            "Final trading verdict. Must be 'no edge' if evidence is mixed or weak.",
        )
        self.assertEqual(properties["reasons_market_is_right"]["minItems"], 3)
        self.assertEqual(properties["reasons_market_is_right"]["maxItems"], 3)
        self.assertEqual(properties["reasons_market_is_wrong"]["minItems"], 3)
        self.assertEqual(properties["reasons_market_is_wrong"]["maxItems"], 3)


if __name__ == "__main__":
    unittest.main()
