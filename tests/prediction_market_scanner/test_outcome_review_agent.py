import json
import tempfile
import unittest
from pathlib import Path

from outcome_review_agent.analyzer import OutcomeReviewAgent
from outcome_review_agent.logger import PerformanceTracker
from outcome_review_agent.models import OutcomeReview


def _sample_review_dict() -> dict:
    return {
        "matrix_classification": "Deserved Success",
        "thesis_held": True,
        "unknown_at_entry": False,
        "calibration_reasonable": True,
        "resulting_detected": False,
        "research_module_flaw": False,
        "risk_module_flaw": False,
        "key_takeaways": ["Stay the course on high-conviction setups."],
        "reasoning": "Strong process and outcome aligned.",
    }


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


def _build_agent(parsed_payload=None) -> OutcomeReviewAgent:
    response = FakeResponse(parsed=parsed_payload or _sample_review_dict())
    return OutcomeReviewAgent(
        api_key="test-key",
        client=FakeClient(response),
        types_module=FakeTypesModule(),
    )


class ReviewTradeTests(unittest.TestCase):
    def test_returns_plain_dict_with_full_payload(self):
        agent = _build_agent()

        result = agent.review_trade(
            {
                "trade_id": "abc",
                "trade_log": {"market_id": "abc", "side": "yes"},
                "final_outcome": True,
                "post_settlement_news": "Resolution confirmed by official source.",
            }
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["matrix_classification"], "Deserved Success")
        self.assertTrue(result["thesis_held"])

    def test_defaults_trade_log_to_whole_payload_and_news_to_sentinel(self):
        agent = _build_agent()

        agent.review_trade({"market_id": "xyz", "final_outcome": False})

        prompt_arg = agent.client.models.calls[0]["contents"]
        self.assertIn('"market_id": "xyz"', prompt_arg)
        self.assertIn("No post-settlement news provided.", prompt_arg)

    def test_raises_when_final_outcome_missing(self):
        agent = _build_agent()
        with self.assertRaisesRegex(ValueError, "final_outcome"):
            agent.review_trade({"trade_id": "abc"})

    def test_raises_when_final_outcome_not_bool(self):
        agent = _build_agent()
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            agent.review_trade({"final_outcome": "win"})

    def test_raises_when_payload_not_mapping(self):
        agent = _build_agent()
        with self.assertRaisesRegex(TypeError, "must be a Mapping"):
            agent.review_trade(["not", "a", "mapping"])  # type: ignore[arg-type]


class PerformanceTrackerWiringTests(unittest.TestCase):
    """End-to-end wiring: PerformanceTracker discovers settled trade files, calls
    OutcomeReviewAgent via the new ``review_trade`` adapter, writes audit."""

    def test_processes_settled_trade_via_review_trade_adapter(self):
        agent = _build_agent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "trade_execution_mkt-1.json").write_text(
                json.dumps(
                    {
                        "trade_id": "mkt-1",
                        "status": "settled",
                        "final_outcome": True,
                        "post_settlement_news": "Resolution confirmed.",
                        "settled_at": "2026-04-25T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )

            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=agent,
            )
            result = tracker.process_settled_trades()

            self.assertEqual(result["new_reviews"], 1)
            self.assertEqual(result["total_reviews"], 1)

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(audit["reviews"]), 1)
            review_payload = audit["reviews"][0]["outcome_review"]
            self.assertEqual(review_payload["matrix_classification"], "Deserved Success")
            self.assertTrue(review_payload["thesis_held"])

    def test_skips_already_reviewed_trades_on_second_run(self):
        agent = _build_agent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "trade_execution_mkt-2.json").write_text(
                json.dumps(
                    {
                        "trade_id": "mkt-2",
                        "status": "settled",
                        "final_outcome": False,
                        "post_settlement_news": "Counter-evidence emerged.",
                    }
                ),
                encoding="utf-8",
            )
            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=agent,
            )
            tracker.process_settled_trades()
            second = tracker.process_settled_trades()

            self.assertEqual(second["new_reviews"], 0)
            self.assertEqual(second["total_reviews"], 1)
            self.assertEqual(len(agent.client.models.calls), 1)

    def test_ignores_unsettled_trades(self):
        agent = _build_agent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "trade_execution_mkt-3.json").write_text(
                json.dumps(
                    {
                        "trade_id": "mkt-3",
                        "status": "open",
                        "final_outcome": False,
                    }
                ),
                encoding="utf-8",
            )
            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=agent,
            )
            result = tracker.process_settled_trades()

            self.assertEqual(result["new_reviews"], 0)
            self.assertEqual(len(agent.client.models.calls), 0)


class OutcomeReviewSchemaTests(unittest.TestCase):
    def test_schema_is_strict_about_unknown_fields(self):
        with self.assertRaises(Exception):
            OutcomeReview.model_validate({**_sample_review_dict(), "unknown_field": True})

    def test_schema_round_trips(self):
        review = OutcomeReview.model_validate(_sample_review_dict())
        self.assertEqual(review.model_dump()["matrix_classification"], "Deserved Success")


if __name__ == "__main__":
    unittest.main()
