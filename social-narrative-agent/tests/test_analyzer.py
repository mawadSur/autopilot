import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyzer import NarrativeAnalyzer, SYSTEM_PROMPT
from models import NarrativeAnalysis


class FakeResponsesAPI:
    def __init__(self, parsed_result):
        self.parsed_result = parsed_result
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(output_parsed=self.parsed_result)


class FakeChatCompletionsAPI:
    def __init__(self, parsed_result):
        self.parsed_result = parsed_result
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(parsed=self.parsed_result)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class NarrativeAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    def _analysis_payload(self):
        return {
            "bullish_thesis": "A credible primary-source leak is moving sentiment faster than price.",
            "bearish_thesis": "The discussion could still be circular rumor amplification with no confirmation.",
            "unresolved_questions": [
                "Is there a direct primary-source confirmation?",
                "Are high-engagement accounts independently sourced?",
            ],
            "signal_quality_score": 7,
            "crowd_overconfidence_score": 8,
            "misinformation_risk": 5,
            "crowd_beliefs": [
                "A catalyst is imminent.",
                "Insiders are signaling early.",
                "The market is still underreacting.",
                "Skeptics are being ignored.",
                "Recent chatter is more confident than confirmed.",
            ],
            "market_alignment": "ahead",
            "reasoning": "Social conviction is stronger than the current odds and appears to be reacting to newer claims.",
        }

    async def test_analyze_narrative_returns_pydantic_model_via_responses_api(self):
        responses_api = FakeResponsesAPI(self._analysis_payload())
        client = types.SimpleNamespace(responses=responses_api)
        analyzer = NarrativeAnalyzer(client=client, model="gpt-4o-mini")

        result = await analyzer.analyze_narrative(
            social_text="TOPIC: GPT-5 release date\nT1 HOT | post | 100 | alpha | Launch chatter is accelerating.",
            current_market_odds=0.41,
        )

        self.assertIsInstance(result, NarrativeAnalysis)
        self.assertEqual(result.market_alignment, "ahead")
        self.assertEqual(result.signal_quality_score, 7)
        self.assertEqual(
            responses_api.calls[0]["input"][0]["content"],
            SYSTEM_PROMPT.format(current_market_odds=0.41),
        )
        self.assertEqual(responses_api.calls[0]["text_format"], NarrativeAnalysis)

    async def test_analyze_narrative_falls_back_to_chat_completions_parse(self):
        completions_api = FakeChatCompletionsAPI(NarrativeAnalysis.model_validate(self._analysis_payload()))
        chat = types.SimpleNamespace(completions=completions_api)
        client = types.SimpleNamespace(chat=chat)
        analyzer = NarrativeAnalyzer(client=client)

        result = await analyzer.analyze_narrative(
            social_text="TOPIC: Election winner\nT1 HOT | post | 88 | alpha | Narrative is diverging from market pricing.",
            current_market_odds=0.63,
        )

        self.assertIsInstance(result, NarrativeAnalysis)
        self.assertEqual(result.crowd_overconfidence_score, 8)
        self.assertEqual(completions_api.calls[0]["response_format"], NarrativeAnalysis)
        self.assertEqual(
            completions_api.calls[0]["messages"][0]["content"],
            SYSTEM_PROMPT.format(current_market_odds=0.63),
        )

    async def test_analyze_narrative_validates_inputs(self):
        analyzer = NarrativeAnalyzer(client=types.SimpleNamespace(responses=FakeResponsesAPI(self._analysis_payload())))

        with self.assertRaises(ValueError):
            await analyzer.analyze_narrative(social_text="   ", current_market_odds=0.5)
        with self.assertRaises(ValueError):
            await analyzer.analyze_narrative(social_text="x", current_market_odds=1.5)


if __name__ == "__main__":
    unittest.main()
