import unittest

from pydantic import ValidationError

from reddit_research_agent.models import RedditResearchReport


def _valid_payload(**overrides):
    payload = dict(
        bullish_thesis="Recent filings support a higher probability than the market implies.",
        bearish_thesis="The thesis still depends on a catalyst that has not been confirmed.",
        key_evidence=["Official filing timestamp", "Linked primary-source announcement draft"],
        key_assumptions=["The filing is authentic", "No adverse update arrives before resolution"],
        conviction_score=8,
        evidence_quality_score=72,
        misinformation_risk_score=18,
        sentiment_score=35,
        key_sources=["https://example.com/post", "u/sourcehound"],
        summary="Multiple high-quality threads cite the same primary source and converge bullishly.",
        pricing_assessment="underpriced",
        assessment_reasoning="The community is citing fresher evidence than the market price appears to reflect.",
    )
    payload.update(overrides)
    return payload


class RedditResearchReportTests(unittest.TestCase):
    def test_report_accepts_valid_payload(self):
        report = RedditResearchReport(**_valid_payload())

        self.assertEqual(report.pricing_assessment, "underpriced")
        self.assertEqual(report.conviction_score, 8)
        self.assertEqual(len(report.key_evidence), 2)
        self.assertEqual(report.bullish_thesis.startswith("Recent filings"), True)
        self.assertEqual(report.bearish_thesis.startswith("The thesis"), True)
        self.assertEqual(report.evidence_quality_score, 72)
        self.assertEqual(report.misinformation_risk_score, 18)
        self.assertEqual(report.sentiment_score, 35)
        self.assertEqual(len(report.key_sources), 2)
        self.assertTrue(report.summary)

    def test_report_rejects_out_of_range_scores(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(conviction_score=11))

    def test_report_rejects_invalid_pricing_assessment(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(pricing_assessment="cheap"))

    def test_report_rejects_misinformation_risk_above_100(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(misinformation_risk_score=101))

    def test_report_rejects_sentiment_outside_signed_range(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(sentiment_score=-150))
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(sentiment_score=150))

    def test_report_rejects_evidence_quality_above_100(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(evidence_quality_score=101))

    def test_report_rejects_more_than_ten_key_sources(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(
                **_valid_payload(key_sources=[f"https://example.com/{i}" for i in range(11)])
            )

    def test_report_rejects_unknown_extra_field(self):
        with self.assertRaises(ValidationError):
            RedditResearchReport(**_valid_payload(pro_argument="legacy field name"))


if __name__ == "__main__":
    unittest.main()
