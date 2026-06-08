import json
import tempfile
import unittest
from pathlib import Path

from iterative_improver.analyzer import IterativeImproverAgent, SYSTEM_PROMPT
from iterative_improver.models import (
    FAILURE_DIAGNOSES,
    FeatureRecommendation,
    RetrainingRecommendation,
)
from outcome_review_agent.logger import PerformanceTracker


def _three_features() -> list:
    return [
        {"name": "rolling_macro_vol_30d", "description": "30d realized vol of macro index", "rationale": "captures regime shift"},
        {"name": "news_freshness_seconds", "description": "age of latest news at entry", "rationale": "guards against stale narrative"},
        {"name": "calibration_temp_z", "description": "rolling z of softmax temperature", "rationale": "detects calibration drift"},
    ]


def _sample_recommendation_dict(diagnosis: str = "regime_shift", priority: int = 7) -> dict:
    return {
        "failure_diagnosis": diagnosis,
        "retraining_priority": priority,
        "new_features": _three_features(),
        "reasoning": "Volatility regime shifted post-debate; model missed the dispersion.",
    }


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
        return type("HttpOptions", (), {"kwargs": kwargs})()

    def GenerateContentConfig(self, **kwargs):
        return type("GenerateContentConfig", (), {"kwargs": kwargs})()


def _build_agent(parsed_payload=None) -> IterativeImproverAgent:
    response = FakeResponse(parsed=parsed_payload or _sample_recommendation_dict())
    return IterativeImproverAgent(
        api_key="test-key",
        client=FakeClient(response),
        types_module=FakeTypesModule(),
    )


class RetrainingRecommendationSchemaTests(unittest.TestCase):
    def test_valid_recommendation_round_trips(self):
        rec = RetrainingRecommendation.model_validate(_sample_recommendation_dict())
        self.assertEqual(rec.failure_diagnosis, "regime_shift")
        self.assertEqual(rec.retraining_priority, 7)
        self.assertEqual(len(rec.new_features), 3)

    def test_must_have_exactly_three_features(self):
        too_few = _sample_recommendation_dict()
        too_few["new_features"] = too_few["new_features"][:2]
        with self.assertRaises(Exception):
            RetrainingRecommendation.model_validate(too_few)

        too_many = _sample_recommendation_dict()
        too_many["new_features"] = _three_features() + [
            {"name": "extra", "description": "...", "rationale": "..."}
        ]
        with self.assertRaises(Exception):
            RetrainingRecommendation.model_validate(too_many)

    def test_priority_bounds(self):
        for bad_priority in (-1, 11, 99):
            payload = _sample_recommendation_dict()
            payload["retraining_priority"] = bad_priority
            with self.assertRaises(Exception):
                RetrainingRecommendation.model_validate(payload)

    def test_failure_diagnosis_must_be_in_allowed_set(self):
        payload = _sample_recommendation_dict()
        payload["failure_diagnosis"] = "vibes_off"
        with self.assertRaises(Exception):
            RetrainingRecommendation.model_validate(payload)

    def test_extra_fields_rejected(self):
        payload = {**_sample_recommendation_dict(), "extra": True}
        with self.assertRaises(Exception):
            RetrainingRecommendation.model_validate(payload)

    def test_feature_recommendation_requires_all_fields(self):
        with self.assertRaises(Exception):
            FeatureRecommendation.model_validate({"name": "x", "description": "y"})

    def test_failure_diagnoses_constant_matches_schema(self):
        for diagnosis in FAILURE_DIAGNOSES:
            payload = _sample_recommendation_dict(diagnosis=diagnosis)
            self.assertEqual(
                RetrainingRecommendation.model_validate(payload).failure_diagnosis,
                diagnosis,
            )


class ImproveAfterTradeTests(unittest.TestCase):
    def test_returns_validated_recommendation(self):
        agent = _build_agent()
        rec = agent.improve_after_trade(
            prediction=0.62,
            outcome=False,
            model_meta={"model_version": "v3.2", "trained_on": "2026-01-15"},
            trade_log={"market_id": "abc", "side": "yes"},
        )
        self.assertIsInstance(rec, RetrainingRecommendation)
        self.assertEqual(rec.failure_diagnosis, "regime_shift")

    def test_prompt_includes_prediction_pct_and_outcome(self):
        agent = _build_agent()
        agent.improve_after_trade(
            prediction=0.62,  # → 62.0%
            outcome=False,
            model_meta={"v": "1"},
            trade_log={"a": 1},
        )
        prompt = agent.client.models.calls[0]["contents"]
        self.assertIn("Model Prediction: 62.0%", prompt)
        self.assertIn("Actual Outcome: loss", prompt)

    def test_prediction_above_one_treated_as_percent(self):
        agent = _build_agent()
        agent.improve_after_trade(
            prediction=42.5,  # already a percent — keep as-is
            outcome=True,
            model_meta={"v": "1"},
            trade_log={"a": 1},
        )
        prompt = agent.client.models.calls[0]["contents"]
        self.assertIn("Model Prediction: 42.5%", prompt)
        self.assertIn("Actual Outcome: win", prompt)

    def test_rejects_invalid_inputs(self):
        agent = _build_agent()
        with self.assertRaisesRegex(ValueError, "prediction is required"):
            agent.improve_after_trade(prediction=None, outcome=True, model_meta={"v": "1"}, trade_log={"a": 1})  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "outcome must be a bool"):
            agent.improve_after_trade(prediction=0.5, outcome="win", model_meta={"v": "1"}, trade_log={"a": 1})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "model_meta"):
            agent.improve_after_trade(prediction=0.5, outcome=True, model_meta={}, trade_log={"a": 1})
        with self.assertRaisesRegex(ValueError, "trade_log"):
            agent.improve_after_trade(prediction=0.5, outcome=True, model_meta={"v": "1"}, trade_log=None)


class ReviewTradeAdapterTests(unittest.TestCase):
    def test_extracts_prediction_from_calibration_calibrated_true_prob(self):
        agent = _build_agent()
        result = agent.review_trade(
            {
                "trade_id": "abc",
                "final_outcome": False,
                "calibration": {"calibrated_true_prob": 0.55},
                "model_meta": {"v": "3.2"},
            }
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["failure_diagnosis"], "regime_shift")
        prompt = agent.client.models.calls[0]["contents"]
        self.assertIn("Model Prediction: 55.0%", prompt)

    def test_explicit_prediction_wins_over_calibration(self):
        agent = _build_agent()
        agent.review_trade(
            {
                "trade_id": "abc",
                "final_outcome": True,
                "prediction": 0.81,
                "calibration": {"calibrated_true_prob": 0.55},
                "model_meta": {"v": "x"},
            }
        )
        prompt = agent.client.models.calls[0]["contents"]
        self.assertIn("Model Prediction: 81.0%", prompt)

    def test_falls_back_to_sentinel_for_missing_model_meta(self):
        agent = _build_agent()
        agent.review_trade(
            {
                "trade_id": "abc",
                "final_outcome": False,
                "calibration": {"calibrated_true_prob": 0.4},
            }
        )
        prompt = agent.client.models.calls[0]["contents"]
        self.assertIn("No model metadata available.", prompt)

    def test_raises_when_final_outcome_missing(self):
        agent = _build_agent()
        with self.assertRaisesRegex(ValueError, "final_outcome"):
            agent.review_trade({"trade_id": "abc", "calibration": {"calibrated_true_prob": 0.5}})

    def test_raises_when_no_prediction_source_found(self):
        agent = _build_agent()
        with self.assertRaisesRegex(ValueError, "prediction"):
            agent.review_trade({"trade_id": "abc", "final_outcome": True, "model_meta": {"v": "x"}})

    def test_raises_when_payload_not_mapping(self):
        agent = _build_agent()
        with self.assertRaisesRegex(TypeError, "must be a Mapping"):
            agent.review_trade(["nope"])  # type: ignore[arg-type]

    def test_extracts_prediction_from_calibration_pydantic_object(self):
        class _CalibrationStub:
            calibrated_true_prob = 0.33

        agent = _build_agent()
        agent.review_trade(
            {
                "trade_id": "abc",
                "final_outcome": False,
                "calibration": _CalibrationStub(),
                "model_meta": {"v": "y"},
            }
        )
        prompt = agent.client.models.calls[0]["contents"]
        self.assertIn("Model Prediction: 33.0%", prompt)


class PerformanceTrackerConditionalIntegrationTests(unittest.TestCase):
    """Conditional agents only run when their predicate matches."""

    class _StubOutcomeReviewAgent:
        def __init__(self, outcome_review):
            self._review = outcome_review
            self.calls = []

        def review_trade(self, trade_payload):
            self.calls.append(trade_payload)
            return dict(self._review)

    @staticmethod
    def _is_good_failure(_trade_payload, outcome_review):
        return outcome_review.get("matrix_classification") == "Good Failure"

    def _write_settled_trade(self, store: Path, *, market_id: str, prediction: float = 0.62):
        (store / f"trade_execution_{market_id}.json").write_text(
            json.dumps(
                {
                    "trade_id": market_id,
                    "status": "settled",
                    "final_outcome": False,
                    "calibration": {"calibrated_true_prob": prediction},
                    "model_meta": {"version": "v3"},
                }
            ),
            encoding="utf-8",
        )

    def test_invokes_agent_only_for_good_failure(self):
        outcome_agent_good_failure = self._StubOutcomeReviewAgent(
            {"matrix_classification": "Good Failure"}
        )
        improver = _build_agent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            self._write_settled_trade(store, market_id="m1")

            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=outcome_agent_good_failure,
                conditional_review_agents={
                    "iterative_improver": (self._is_good_failure, improver),
                },
            )
            tracker.process_settled_trades()

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))
            entry = audit["reviews"][0]
            self.assertIn("iterative_improver_review", entry)
            self.assertEqual(entry["iterative_improver_review"]["failure_diagnosis"], "regime_shift")
            self.assertEqual(len(improver.client.models.calls), 1)

    def test_skips_agent_when_predicate_returns_false(self):
        outcome_agent_dumb_luck = self._StubOutcomeReviewAgent(
            {"matrix_classification": "Dumb Luck"}
        )
        improver = _build_agent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            self._write_settled_trade(store, market_id="m2")

            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=outcome_agent_dumb_luck,
                conditional_review_agents={
                    "iterative_improver": (self._is_good_failure, improver),
                },
            )
            tracker.process_settled_trades()

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))
            entry = audit["reviews"][0]
            self.assertNotIn("iterative_improver_review", entry)
            self.assertEqual(len(improver.client.models.calls), 0)

    def test_works_alongside_additional_review_agents(self):
        """Backwards compat: conditional + unconditional agents coexist."""
        outcome_agent_good_failure = self._StubOutcomeReviewAgent(
            {"matrix_classification": "Good Failure"}
        )
        improver = _build_agent()

        class _UnconditionalAgent:
            def __init__(self):
                self.calls = []

            def review_trade(self, trade_payload):
                self.calls.append(trade_payload)
                return {"always": True}

        unconditional = _UnconditionalAgent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            self._write_settled_trade(store, market_id="m3")

            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=outcome_agent_good_failure,
                additional_review_agents={"data_quality": unconditional},
                conditional_review_agents={
                    "iterative_improver": (self._is_good_failure, improver),
                },
            )
            tracker.process_settled_trades()

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))
            entry = audit["reviews"][0]
            self.assertIn("data_quality_review", entry)
            self.assertIn("iterative_improver_review", entry)


class SystemPromptSpecComplianceTests(unittest.TestCase):
    """The system prompt must enumerate the failure modes and the constraint contract."""

    def test_prompt_lists_three_diagnoses(self):
        for diagnosis in FAILURE_DIAGNOSES:
            self.assertIn(diagnosis, SYSTEM_PROMPT)

    def test_prompt_specifies_exactly_three_features(self):
        self.assertIn("EXACTLY three", SYSTEM_PROMPT)

    def test_prompt_specifies_priority_range(self):
        self.assertIn("0-10", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
