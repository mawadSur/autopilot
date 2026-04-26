import unittest

from twitter_research_agent.analyzer import (
    SYSTEM_PROMPT_TEMPLATE,
    TwitterAgent,
    _build_system_prompt,
)
from twitter_research_agent.models import TwitterResearchReport


class FakeHttpOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


class FakeTypesModule:
    def HttpOptions(self, **kwargs):
        return FakeHttpOptions(**kwargs)

    def GenerateContentConfig(self, **kwargs):
        return FakeGenerationConfig(**kwargs)


class TwitterAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_discussion_returns_twitter_research_report(self):
        response_payload = TwitterResearchReport(
            bullish_thesis="Multiple credentialed analysts cite a fresh primary source supporting YES.",
            bearish_thesis="Counter-thread argues the catalyst was already priced after last week's news.",
            evidence_quality_score=71,
            misinformation_risk_score=15,
            sentiment_score=28,
            key_sources=["https://x.com/quant_takes/status/1", "@analyst_one"],
            summary="Twitter discourse leans modestly bullish with one credible dissenting thread.",
            tweet_count=37,
        )
        client = FakeClient(FakeResponse(parsed=response_payload))
        fake_types = FakeTypesModule()
        agent = TwitterAgent(api_key="test-key", client=client, types_module=fake_types)

        report = await agent.analyze_discussion(
            market_title="Will the Fed cut rates in March?",
            implied_prob=0.42,
            twitter_context=(
                "TWEET 1 | @quant_takes | likes=205 | retweets=66\n"
                '"Primary-source link supports a YES resolution."'
            ),
        )

        self.assertEqual(report.evidence_quality_score, 71)
        self.assertEqual(report.misinformation_risk_score, 15)
        self.assertEqual(report.sentiment_score, 28)
        self.assertEqual(report.tweet_count, 37)
        self.assertTrue(report.bullish_thesis.startswith("Multiple credentialed"))
        self.assertTrue(report.bearish_thesis.startswith("Counter-thread"))
        self.assertIn("@analyst_one", report.key_sources)
        self.assertEqual(len(client.models.calls), 1)

        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-2.5-pro")
        self.assertIn("Twitter Discussion Context:", call["contents"])
        self.assertEqual(
            call["config"].kwargs["system_instruction"],
            _build_system_prompt("Will the Fed cut rates in March?", 0.42),
        )
        self.assertEqual(call["config"].kwargs["response_mime_type"], "application/json")
        self.assertIs(call["config"].kwargs["response_schema"], TwitterResearchReport)
        self.assertEqual(call["config"].kwargs["temperature"], 0.2)
        self.assertEqual(call["config"].kwargs["http_options"].kwargs["timeout"], 30)

    def test_system_prompt_template_documents_spec_fields(self):
        for field in (
            "bullish_thesis",
            "bearish_thesis",
            "evidence_quality_score",
            "misinformation_risk_score",
            "sentiment_score",
            "key_sources",
            "summary",
            "tweet_count",
        ):
            self.assertIn(field, SYSTEM_PROMPT_TEMPLATE)
        self.assertIn("Twitter/X research analyst", SYSTEM_PROMPT_TEMPLATE)

    async def test_analyze_discussion_validates_inputs(self):
        agent = TwitterAgent(
            api_key="test-key",
            client=FakeClient(FakeResponse(parsed={})),
            types_module=FakeTypesModule(),
        )

        with self.assertRaises(ValueError):
            await agent.analyze_discussion("", 0.5, "context")

        with self.assertRaises(ValueError):
            await agent.analyze_discussion("Market", 1.2, "context")

        with self.assertRaises(ValueError):
            await agent.analyze_discussion("Market", 0.5, "")


if __name__ == "__main__":
    unittest.main()
