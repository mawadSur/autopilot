import unittest
from unittest.mock import patch

import requests

import llm_judge
from llm_judge import LLMJudgeResult, judge_market


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "json": json,
            "timeout": timeout,
        })
        return self.response

    def close(self):
        self.closed = True


class LLMJudgeTests(unittest.TestCase):
    def test_judge_market_parses_scores_and_grounding_metadata(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"clarity_score": 84, "narrative_momentum": 67, "ambiguous": false, "reasoning": "Clear trigger and rising coverage."}'
                                }
                            ]
                        },
                        "groundingMetadata": {
                            "webSearchQueries": ["market topic latest news"],
                            "groundingChunks": [
                                {"web": {"title": "Reuters headline", "uri": "https://example.com/reuters"}},
                                {"web": {"title": "AP headline", "uri": "https://example.com/ap"}},
                            ],
                        },
                    }
                ]
            }
        )
        session = FakeSession(response)

        result = judge_market(
            "Will X happen?",
            "This market resolves to Yes if the official filing is published by the deadline.",
            api_key="test-key",
            session=session,
        )

        self.assertIsInstance(result, LLMJudgeResult)
        self.assertEqual(result.clarity_score, 84)
        self.assertEqual(result.narrative_momentum, 67)
        self.assertEqual(result.anomaly_flags, [])
        self.assertEqual(result.search_queries, ["market topic latest news"])
        self.assertEqual(result.source_titles, ["Reuters headline", "AP headline"])
        self.assertEqual(session.calls[0]["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(session.calls[0]["timeout"], 30)
        self.assertIn("tools", session.calls[0]["json"])
        self.assertEqual(session.calls[0]["json"]["tools"], [{"google_search": {}}])
        self.assertFalse(session.closed)

    def test_judge_market_flags_ambiguous_from_subjective_language(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '```json\n{"clarity_score": 48, "narrative_momentum": 40, "ambiguous": false, "reasoning": "Model did not mark ambiguity."}\n```'
                                }
                            ]
                        }
                    }
                ]
            }
        )
        session = FakeSession(response)

        result = judge_market(
            "Will the ruling stand?",
            "This market resolves according to official sources unless substantial evidence suggests otherwise.",
            api_key="test-key",
            session=session,
            use_search_grounding=False,
        )

        self.assertEqual(result.clarity_score, 48)
        self.assertEqual(result.narrative_momentum, 40)
        self.assertEqual(result.anomaly_flags, ["AMBIGUOUS"])
        self.assertIn("substantial evidence", result.reasoning.lower())
        self.assertNotIn("tools", session.calls[0]["json"])

    def test_judge_market_salvages_truncated_json_like_output(self):
        response = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"clarity_score": 91, "narrative_momentum": 63, "ambiguous": false, "reasoning": "Clear terms with active coverage"'
                                }
                            ]
                        }
                    }
                ]
            }
        )
        session = FakeSession(response)

        result = judge_market(
            "Will the acquisition close?",
            "This market resolves to Yes if the merger is formally completed before the listed deadline.",
            api_key="test-key",
            session=session,
            use_search_grounding=False,
        )

        self.assertEqual(result.clarity_score, 91)
        self.assertEqual(result.narrative_momentum, 63)
        self.assertEqual(result.anomaly_flags, [])
        self.assertIn("clear terms", result.reasoning.lower())

    @patch.dict("os.environ", {}, clear=True)
    def test_judge_market_requires_api_key(self):
        with patch.object(llm_judge.cfg, "gemini_api_key", None):
            with self.assertRaises(RuntimeError):
                judge_market(
                    "Will the bill pass?",
                    "This market resolves to Yes if the bill is signed into law before June 1.",
                    session=FakeSession(FakeResponse({"candidates": []})),
                )


if __name__ == "__main__":
    unittest.main()
