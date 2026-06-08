import unittest

from synthesis_agent.analyzer import SynthesisAgent, SYSTEM_PROMPT
from synthesis_agent.models import SynthesisReport


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


class SynthesisAgentTests(unittest.TestCase):
    def _market_data(self) -> dict:
        return {
            "market_id": "mkt-1",
            "title": "Will candidate X win?",
            "category": "Politics",
            "implied_probability": 0.42,
            "market_implied_probability": 0.42,
            "volume_24h": 12000.0,
            "spread": 0.02,
        }

    def _social_context(self) -> str:
        return (
            "REDDIT DISCUSSION CONTEXT\n"
            "THREAD 1 | r/politics | score=120 | comments=40 | author=u/example\n"
            "TITLE: Will candidate X win?\n"
            "TOP COMMENTS:\n"
            "- u/analyst | score=80: Poll cross-tabs suggest upside."
        )

    def _news_context(self) -> str:
        return (
            "NEWS COVERAGE CONTEXT\n"
            "ARTICLE 1 | 2026-04-01T10:00:00+00:00 | Debate announced\n"
            "Summary: Coverage is factual and moderately supportive of a tighter race.\n"
            "Link: https://example.com/story"
        )

    def test_synthesize_edge_returns_synthesis_report(self):
        response_payload = SynthesisReport(
            implied_probability=0.42,
            narrative_direction="mixed",
            has_unique_evidence=True,
            reasons_market_is_right=[
                "The market already prices in the visible polling trend.",
                "Resolution criteria are straightforward.",
                "There is still meaningful headline risk.",
            ],
            reasons_market_is_wrong=[
                "Reddit surfaced one piece of evidence the market may be underweighting.",
                "Recent news shifted the timeline faster than price responded.",
                "The crowd found a niche but credible source.",
            ],
            verdict="stale",
            explanation="The market has not yet absorbed the latest narrative.",
        )
        client = FakeClient(FakeResponse(parsed=response_payload))
        agent = SynthesisAgent("test-key", client=client, types_module=FakeTypesModule())

        report = agent.synthesize_edge(
            self._market_data(),
            self._social_context(),
            self._news_context(),
        )

        self.assertEqual(report.verdict, "stale")
        self.assertEqual(len(client.models.calls), 1)
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-2.5-pro")
        self.assertEqual(call["config"].kwargs["system_instruction"], SYSTEM_PROMPT)
        self.assertEqual(call["config"].kwargs["response_mime_type"], "application/json")
        self.assertIs(call["config"].kwargs["response_schema"], SynthesisReport)
        self.assertEqual(call["config"].kwargs["temperature"], 0.1)
        self.assertEqual(call["config"].kwargs["http_options"].kwargs["timeout"], 30)
        self.assertIn("--- MARKET DATA ---", call["contents"])
        self.assertIn("--- SOCIAL & REDDIT NARRATIVE ---", call["contents"])
        self.assertIn("--- NEWS & RSS COVERAGE ---", call["contents"])
        self.assertIn('"title": "Will candidate X win?"', call["contents"])
        self.assertIn("Poll cross-tabs suggest upside.", call["contents"])
        self.assertIn("Coverage is factual and moderately supportive of a tighter race.", call["contents"])

    def test_constructor_uses_new_client_factory(self):
        fake_genai = FakeClientFactoryModule()
        agent = SynthesisAgent(
            "test-key",
            genai_module=fake_genai,
            types_module=FakeTypesModule(),
        )

        self.assertEqual(fake_genai.calls, ["test-key"])
        self.assertIsNotNone(agent.client)

    def test_system_prompt_describes_all_four_verdict_categories(self):
        for value in ("stale", "efficient", "overreactive", "unclear"):
            self.assertIn(value, SYSTEM_PROMPT)
        for legacy_value in ("no edge", "possible edge", "strong research edge"):
            self.assertNotIn(legacy_value, SYSTEM_PROMPT)

    def test_invalid_inputs_raise(self):
        agent = SynthesisAgent("test-key", client=FakeClient(FakeResponse(parsed={})), types_module=FakeTypesModule())

        with self.assertRaisesRegex(ValueError, "market_data"):
            agent.synthesize_edge({}, self._social_context(), self._news_context())
        with self.assertRaisesRegex(ValueError, "social_context"):
            agent.synthesize_edge(self._market_data(), "", self._news_context())
        with self.assertRaisesRegex(ValueError, "news_context"):
            agent.synthesize_edge(self._market_data(), self._social_context(), "")


if __name__ == "__main__":
    unittest.main()
