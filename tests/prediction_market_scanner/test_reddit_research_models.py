import unittest

from pydantic import ValidationError

from reddit_research_agent.models import RedditResearchReport


class RedditResearchReportTests(unittest.TestCase):
    def test_report_accepts_valid_payload(self):
        report = RedditResearchReport(
            pro_argument="Recent filings support a higher probability than the market implies.",
            anti_argument="The thesis still depends on a catalyst that has not been confirmed.",
            key_evidence=["Official filing timestamp", "Linked primary-source announcement draft"],
            key_assumptions=["The filing is authentic", "No adverse update arrives before resolution"],
            conviction_score=8,
            evidence_quality_score=7,
            pricing_assessment="underpriced",
            assessment_reasoning="The community is citing fresher evidence than the market price appears to reflect.",
        )

        self.assertEqual(report.pricing_assessment, "underpriced")
        self.assertEqual(report.conviction_score, 8)
        self.assertEqual(len(report.key_evidence), 2)

    def test_report_rejects_out_of_range_scores(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(
                pro_argument="Pro",
                anti_argument="Anti",
                key_evidence=["Source"],
                key_assumptions=["Assumption"],
                conviction_score=11,
                evidence_quality_score=7,
                pricing_assessment="fairly priced",
                assessment_reasoning="Reasoning",
            )

    def test_report_rejects_invalid_pricing_assessment(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(
                pro_argument="Pro",
                anti_argument="Anti",
                key_evidence=["Source"],
                key_assumptions=["Assumption"],
                conviction_score=5,
                evidence_quality_score=6,
                pricing_assessment="cheap",
                assessment_reasoning="Reasoning",
            )


if __name__ == "__main__":
    unittest.main()
