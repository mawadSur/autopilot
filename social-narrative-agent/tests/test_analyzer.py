import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyzer import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    DEFAULT_CLAIM_EXTRACTION_MODEL,
    ClaimExtractionResult,
    ExtractedClaim,
    NarrativeAnalyzer,
    SYSTEM_PROMPT,
)
from models import NarrativeAnalysis


class FakeResponsesAPI:
    def __init__(self, parsed_results):
        self.parsed_results = list(parsed_results)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.parsed_results:
            raise AssertionError("No queued parsed results left for responses.parse")
        return types.SimpleNamespace(output_parsed=self.parsed_results.pop(0))


class FakeChatCompletionsAPI:
    def __init__(self, parsed_results):
        self.parsed_results = list(parsed_results)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.parsed_results:
            raise AssertionError("No queued parsed results left for chat.completions.parse")
        message = types.SimpleNamespace(parsed=self.parsed_results.pop(0))
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

    def _claim_payload(self):
        return {
            "claims": [
                {
                    "claim": "Launch timing is being pulled forward into this quarter.",
                    "source_context": "Top-level high-engagement Reddit post citing recent chatter.",
                },
                {
                    "claim": "No official date has been confirmed yet.",
                    "source_context": "Reply quoting the absence of primary-source confirmation.",
                },
            ]
        }

    async def test_extract_claims_returns_structured_claim_list(self):
        responses_api = FakeResponsesAPI([self._claim_payload()])
        client = types.SimpleNamespace(responses=responses_api)
        analyzer = NarrativeAnalyzer(client=client)

        result = await analyzer.extract_claims(
            "TOPIC: GPT-5 release date\nT1 HOT | post | 100 | alpha | Launch chatter is accelerating.",
        )

        self.assertIsInstance(result, ClaimExtractionResult)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].claim, "Launch timing is being pulled forward into this quarter.")
        self.assertEqual(responses_api.calls[0]["text_format"], ClaimExtractionResult)
        self.assertEqual(responses_api.calls[0]["model"], DEFAULT_CLAIM_EXTRACTION_MODEL)
        self.assertEqual(responses_api.calls[0]["input"][0]["content"], CLAIM_EXTRACTION_SYSTEM_PROMPT)

    async def test_analyze_narrative_preprocesses_claims_before_main_pass(self):
        responses_api = FakeResponsesAPI([self._claim_payload(), self._analysis_payload()])
        client = types.SimpleNamespace(responses=responses_api)
        analyzer = NarrativeAnalyzer(client=client, model="gpt-4o-mini")

        result = await analyzer.analyze_narrative(
            social_text="TOPIC: GPT-5 release date\nT1 HOT | post | 100 | alpha | Launch chatter is accelerating.",
            current_market_odds=0.41,
        )

        self.assertIsInstance(result, NarrativeAnalysis)
        self.assertEqual(result.market_alignment, "ahead")
        self.assertEqual(result.signal_quality_score, 7)
        self.assertEqual(len(responses_api.calls), 2)
        self.assertEqual(responses_api.calls[1]["input"][0]["content"], SYSTEM_PROMPT.format(current_market_odds=0.41))
        self.assertEqual(responses_api.calls[1]["text_format"], NarrativeAnalysis)
        self.assertIn("CLAIM: Launch timing is being pulled forward into this quarter.", responses_api.calls[1]["input"][1]["content"])
        self.assertNotIn("T1 HOT | post | 100 | alpha", responses_api.calls[1]["input"][1]["content"])

    async def test_analyze_narrative_falls_back_to_chat_completions_parse(self):
        completions_api = FakeChatCompletionsAPI([
            ClaimExtractionResult.model_validate(self._claim_payload()),
            NarrativeAnalysis.model_validate(self._analysis_payload()),
        ])
        chat = types.SimpleNamespace(completions=completions_api)
        client = types.SimpleNamespace(chat=chat)
        analyzer = NarrativeAnalyzer(client=client)

        result = await analyzer.analyze_narrative(
            social_text="TOPIC: Election winner\nT1 HOT | post | 88 | alpha | Narrative is diverging from market pricing.",
            current_market_odds=0.63,
        )

        self.assertIsInstance(result, NarrativeAnalysis)
        self.assertEqual(result.crowd_overconfidence_score, 8)
        self.assertEqual(completions_api.calls[0]["response_format"], ClaimExtractionResult)
        self.assertEqual(completions_api.calls[1]["response_format"], NarrativeAnalysis)
        self.assertEqual(completions_api.calls[1]["messages"][0]["content"], SYSTEM_PROMPT.format(current_market_odds=0.63))

    async def test_analyze_narrative_validates_inputs(self):
        analyzer = NarrativeAnalyzer(client=types.SimpleNamespace(responses=FakeResponsesAPI([self._claim_payload(), self._analysis_payload()])))

        with self.assertRaises(ValueError):
            await analyzer.analyze_narrative(social_text="   ", current_market_odds=0.5)
        with self.assertRaises(ValueError):
            await analyzer.analyze_narrative(social_text="x", current_market_odds=1.5)
        with self.assertRaises(ValueError):
            await analyzer.extract_claims("   ")

    def test_format_claims_for_analysis_handles_empty_claim_list(self):
        analyzer = NarrativeAnalyzer(client=types.SimpleNamespace())
        formatted = analyzer._format_claims_for_analysis(ClaimExtractionResult(claims=[]))
        self.assertIn("No distinct claims extracted", formatted)

    def test_claim_extraction_result_dedupes_duplicate_claims(self):
        deduped = ClaimExtractionResult(
            claims=[
                ExtractedClaim(claim="Same claim", source_context="Thread A"),
                ExtractedClaim(claim="same   claim", source_context="Thread B"),
            ]
        )
        self.assertEqual(len(deduped.claims), 1)


if __name__ == "__main__":
    unittest.main()
