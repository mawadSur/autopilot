import unittest

from reddit_research_agent.analyzer import RedditAgent, _build_system_prompt
from reddit_research_agent.models import RedditResearchReport


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


class FakeClientFactoryModule:
    def __init__(self):
        self.calls = []

    def Client(self, *, api_key):
        self.calls.append(api_key)
        return FakeClient(FakeResponse(parsed={}))


class FakeTypesModule:
    def HttpOptions(self, **kwargs):
        return FakeHttpOptions(**kwargs)

    def GenerateContentConfig(self, **kwargs):
        return FakeGenerationConfig(**kwargs)


class RedditAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_discussion_returns_reddit_research_report(self):
        response_payload = RedditResearchReport(
            pro_argument="Primary-source reporting indicates the catalyst is more likely than the market implies.",
            anti_argument="The thread still relies on one timing assumption that could break late.",
            key_evidence=["Regulatory filing timestamp", "Direct link to executive statement"],
            key_assumptions=["No contradictory filing appears", "The reported date is authentic"],
            conviction_score=7,
            evidence_quality_score=8,
            pricing_assessment="underpriced",
            assessment_reasoning="Reddit surfaced fresher primary-source evidence than the current market price appears to reflect.",
        )
        client = FakeClient(FakeResponse(parsed=response_payload))
        fake_types = FakeTypesModule()
        agent = RedditAgent(api_key="test-key", client=client, types_module=fake_types)

        report = await agent.analyze_discussion(
            market_title="Will GPT-5 launch by June?",
            implied_prob=0.42,
            reddit_context="THREAD 1 | r/LocalLLaMA | score=100 | comments=20 | author=u/sourcehound",
        )

        self.assertEqual(report.pricing_assessment, "underpriced")
        self.assertEqual(report.evidence_quality_score, 8)
        self.assertEqual(len(client.models.calls), 1)

        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-2.5-pro")
        self.assertIn("Reddit Discussion Context:", call["contents"])
        self.assertEqual(
            call["config"].kwargs["system_instruction"],
            _build_system_prompt("Will GPT-5 launch by June?", 0.42),
        )
        self.assertEqual(call["config"].kwargs["response_mime_type"], "application/json")
        self.assertIs(call["config"].kwargs["response_schema"], RedditResearchReport)
        self.assertEqual(call["config"].kwargs["temperature"], 0.2)
        self.assertEqual(call["config"].kwargs["http_options"].kwargs["timeout"], 30)

    async def test_analyze_discussion_validates_inputs(self):
        agent = RedditAgent(api_key="test-key", client=FakeClient(FakeResponse(parsed={})), types_module=FakeTypesModule())

        with self.assertRaises(ValueError):
            await agent.analyze_discussion("", 0.5, "context")

        with self.assertRaises(ValueError):
            await agent.analyze_discussion("Market", 1.2, "context")

        with self.assertRaises(ValueError):
            await agent.analyze_discussion("Market", 0.5, "")

    def test_constructor_uses_new_client_factory(self):
        fake_genai = FakeClientFactoryModule()
        agent = RedditAgent(
            api_key="test-key",
            genai_module=fake_genai,
            types_module=FakeTypesModule(),
        )

        self.assertEqual(fake_genai.calls, ["test-key"])
        self.assertIsNotNone(agent.client)


if __name__ == "__main__":
    unittest.main()
