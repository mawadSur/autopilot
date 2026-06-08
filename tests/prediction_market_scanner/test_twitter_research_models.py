import unittest

from pydantic import ValidationError

from twitter_research_agent.models import TwitterResearchReport


def _valid_payload(**overrides):
    payload = dict(
        bullish_thesis="Credentialed analysts on X cite a fresh primary source supporting YES.",
        bearish_thesis="Counter-thread points out the catalyst is already largely priced in.",
        evidence_quality_score=64,
        misinformation_risk_score=22,
        sentiment_score=18,
        key_sources=["https://x.com/quant_takes/status/1", "@analyst_one"],
        summary="Twitter discourse leans modestly bullish with one credible counter-thread.",
        tweet_count=42,
    )
    payload.update(overrides)
    return payload


class TwitterResearchReportTests(unittest.TestCase):
    def test_report_accepts_valid_payload(self):
        report = TwitterResearchReport(**_valid_payload())

        self.assertEqual(report.evidence_quality_score, 64)
        self.assertEqual(report.misinformation_risk_score, 22)
        self.assertEqual(report.sentiment_score, 18)
        self.assertEqual(report.tweet_count, 42)
        self.assertEqual(len(report.key_sources), 2)
        self.assertTrue(report.bullish_thesis.startswith("Credentialed"))
        self.assertTrue(report.bearish_thesis.startswith("Counter-thread"))
        self.assertTrue(report.summary)

    def test_report_rejects_evidence_quality_above_100(self):
        with self.assertRaises(ValidationError):
            TwitterResearchReport(**_valid_payload(evidence_quality_score=101))

    def test_report_rejects_misinformation_risk_above_100(self):
        with self.assertRaises(ValidationError):
            TwitterResearchReport(**_valid_payload(misinformation_risk_score=150))

    def test_report_rejects_sentiment_outside_signed_range(self):
        with self.assertRaises(ValidationError):
            TwitterResearchReport(**_valid_payload(sentiment_score=-150))
        with self.assertRaises(ValidationError):
            TwitterResearchReport(**_valid_payload(sentiment_score=150))

    def test_report_rejects_negative_tweet_count(self):
        with self.assertRaises(ValidationError):
            TwitterResearchReport(**_valid_payload(tweet_count=-1))

    def test_report_rejects_more_than_ten_key_sources(self):
        with self.assertRaises(ValidationError):
            TwitterResearchReport(
                **_valid_payload(key_sources=[f"@user_{i}" for i in range(11)])
            )

    def test_report_rejects_unknown_extra_field(self):
        with self.assertRaises(ValidationError):
            TwitterResearchReport(**_valid_payload(pricing_assessment="underpriced"))


if __name__ == "__main__":
    unittest.main()
