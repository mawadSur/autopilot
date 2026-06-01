import json
import tempfile
import unittest
from pathlib import Path

from data_quality_auditor.analyzer import (
    DataQualityAuditor,
    INTEGRITY_SYSTEM_PROMPT,
    INTERPRETATION_SYSTEM_PROMPT,
    PIPELINE_SYSTEM_PROMPT,
)
from data_quality_auditor.models import (
    DataQualityAudit,
    FAILURE_MODE_NAMES,
    FocusedAuditFinding,
)
from outcome_review_agent.logger import PerformanceTracker


def _clean_audit_dict() -> dict:
    finding = {"detected": False, "evidence": "No anomalies observed."}
    return {
        "stale_data": finding,
        "duplicate_sources": finding,
        "missing_primary_sources": finding,
        "misleading_sentiment": finding,
        "scraping_gaps": finding,
        "timestamp_mismatches": finding,
        "incorrect_market_metadata": finding,
        "data_failure": False,
        "failure_mode": None,
        "severity": None,
        "recommended_fix": "No action needed.",
    }


def _failure_audit_dict() -> dict:
    payload = _clean_audit_dict()
    payload["stale_data"] = {
        "detected": True,
        "evidence": "Market price snapshot was 14m old at entry, exceeded 5m staleness budget.",
    }
    payload["data_failure"] = True
    payload["failure_mode"] = "stale_data"
    payload["severity"] = 4
    payload["recommended_fix"] = "Tighten freshness gate to 60s and re-fetch on stale snapshots."
    return payload


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


def _build_auditor(parsed_payload=None) -> DataQualityAuditor:
    response = FakeResponse(parsed=parsed_payload or _clean_audit_dict())
    return DataQualityAuditor(
        api_key="test-key",
        client=FakeClient(response),
        types_module=FakeTypesModule(),
    )


class DataQualityAuditSchemaTests(unittest.TestCase):
    def test_clean_audit_validates(self):
        audit = DataQualityAudit.model_validate(_clean_audit_dict())
        self.assertFalse(audit.data_failure)
        self.assertIsNone(audit.failure_mode)
        self.assertIsNone(audit.severity)

    def test_failure_requires_failure_mode_and_severity(self):
        bad = _failure_audit_dict()
        bad["failure_mode"] = None
        with self.assertRaisesRegex(ValueError, "failure_mode is required"):
            DataQualityAudit.model_validate(bad)

        bad = _failure_audit_dict()
        bad["severity"] = None
        with self.assertRaisesRegex(ValueError, "severity is required"):
            DataQualityAudit.model_validate(bad)

    def test_clean_audit_must_not_carry_failure_mode_or_severity(self):
        bad = _clean_audit_dict()
        bad["failure_mode"] = "stale_data"
        with self.assertRaisesRegex(ValueError, "failure_mode must be null"):
            DataQualityAudit.model_validate(bad)

        bad = _clean_audit_dict()
        bad["severity"] = 1
        with self.assertRaisesRegex(ValueError, "severity must be null"):
            DataQualityAudit.model_validate(bad)

    def test_severity_bounds(self):
        bad = _failure_audit_dict()
        bad["severity"] = 6
        with self.assertRaises(Exception):
            DataQualityAudit.model_validate(bad)

    def test_unknown_failure_mode_rejected(self):
        bad = _failure_audit_dict()
        bad["failure_mode"] = "vibes_too_off"
        with self.assertRaises(Exception):
            DataQualityAudit.model_validate(bad)

    def test_extra_fields_rejected(self):
        bad = {**_clean_audit_dict(), "unknown_field": True}
        with self.assertRaises(Exception):
            DataQualityAudit.model_validate(bad)

    def test_failure_mode_names_constant_matches_schema_fields(self):
        for name in FAILURE_MODE_NAMES:
            self.assertIn(name, DataQualityAudit.model_fields)


class AuditTradeTests(unittest.TestCase):
    def test_returns_validated_audit(self):
        auditor = _build_auditor()
        audit = auditor.audit_trade(
            trade_payload={"market_id": "abc"},
            news_context="Coverage neutral.",
            features_window={"price_t-1": 0.42, "price_t-2": 0.41},
        )
        self.assertIsInstance(audit, DataQualityAudit)
        self.assertFalse(audit.data_failure)

    def test_rejects_empty_inputs(self):
        auditor = _build_auditor()
        with self.assertRaisesRegex(ValueError, "trade_payload"):
            auditor.audit_trade(trade_payload=None, news_context="x", features_window="y")
        with self.assertRaisesRegex(ValueError, "news_context"):
            auditor.audit_trade(trade_payload={"a": 1}, news_context="", features_window="y")
        with self.assertRaisesRegex(ValueError, "features_window"):
            auditor.audit_trade(trade_payload={"a": 1}, news_context="x", features_window=None)


class ReviewTradeAdapterTests(unittest.TestCase):
    def test_pulls_news_from_research_news_report_when_top_level_absent(self):
        auditor = _build_auditor()
        result = auditor.review_trade(
            {
                "trade_id": "abc",
                "research": {"news_report": "AP coverage of debate."},
                "scanner": {"price": 0.42, "volume_24h": 12000},
            }
        )
        self.assertIsInstance(result, dict)
        self.assertFalse(result["data_failure"])

        prompt = auditor.client.models.calls[0]["contents"]
        self.assertIn("AP coverage of debate.", prompt)

    def test_falls_back_to_scanner_for_features_window_when_absent(self):
        auditor = _build_auditor()
        auditor.review_trade(
            {
                "trade_id": "abc",
                "scanner": {"price": 0.42},
                "news_context": "Quiet news cycle.",
            }
        )
        prompt = auditor.client.models.calls[0]["contents"]
        self.assertIn('"price": 0.42', prompt)
        self.assertIn("Quiet news cycle.", prompt)

    def test_uses_sentinels_when_news_and_features_both_missing(self):
        auditor = _build_auditor()
        auditor.review_trade({"trade_id": "abc"})
        prompt = auditor.client.models.calls[0]["contents"]
        self.assertIn("No news context available.", prompt)
        self.assertIn("No feature snapshot available.", prompt)

    def test_explicit_features_window_slot_wins_over_scanner(self):
        auditor = _build_auditor()
        auditor.review_trade(
            {
                "trade_id": "abc",
                "scanner": {"price": 0.42},
                "features_window": [{"close": 0.40, "ts": "2026-04-25T00:00:00Z"}],
                "news_context": "x",
            }
        )
        prompt = auditor.client.models.calls[0]["contents"]
        self.assertIn('"close": 0.4', prompt)

    def test_raises_when_payload_not_mapping(self):
        auditor = _build_auditor()
        with self.assertRaisesRegex(TypeError, "must be a Mapping"):
            auditor.review_trade(["not", "a", "mapping"])  # type: ignore[arg-type]


class PerformanceTrackerMultiAgentTests(unittest.TestCase):
    """Tracker runs both the primary outcome-review agent AND any additional agents."""

    class _StubOutcomeReviewAgent:
        def __init__(self):
            self.calls = []

        def review_trade(self, trade_payload):
            self.calls.append(trade_payload)
            return {"matrix_classification": "Deserved Success", "thesis_held": True}

    def test_writes_both_outcome_and_data_quality_into_audit(self):
        outcome_agent = self._StubOutcomeReviewAgent()
        data_quality_auditor = _build_auditor(parsed_payload=_failure_audit_dict())

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "trade_execution_mkt-9.json").write_text(
                json.dumps(
                    {
                        "trade_id": "mkt-9",
                        "status": "settled",
                        "final_outcome": False,
                        "post_settlement_news": "Counter-evidence.",
                        "scanner": {"price": 0.51},
                        "research": {"news_report": "Wire reports neutral."},
                    }
                ),
                encoding="utf-8",
            )

            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=outcome_agent,
                additional_review_agents={"data_quality": data_quality_auditor},
            )
            result = tracker.process_settled_trades()

            self.assertEqual(result["new_reviews"], 1)

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))
            entry = audit["reviews"][0]
            self.assertIn("outcome_review", entry)
            self.assertIn("data_quality_review", entry)
            self.assertEqual(entry["data_quality_review"]["failure_mode"], "stale_data")
            self.assertEqual(entry["data_quality_review"]["severity"], 4)
            self.assertEqual(entry["outcome_review"]["matrix_classification"], "Deserved Success")
            # Both agents got called exactly once.
            self.assertEqual(len(outcome_agent.calls), 1)
            self.assertEqual(len(data_quality_auditor.client.models.calls), 1)

    def test_default_no_additional_agents_keeps_existing_behavior(self):
        outcome_agent = self._StubOutcomeReviewAgent()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "trade_execution_mkt-10.json").write_text(
                json.dumps(
                    {
                        "trade_id": "mkt-10",
                        "status": "settled",
                        "final_outcome": True,
                    }
                ),
                encoding="utf-8",
            )

            tracker = PerformanceTracker(
                trade_store_dir=store,
                outcome_review_agent=outcome_agent,
            )
            tracker.process_settled_trades()

            audit = json.loads((store / "performance_audit.json").read_text(encoding="utf-8"))
            entry = audit["reviews"][0]
            self.assertIn("outcome_review", entry)
            self.assertNotIn("data_quality_review", entry)


def _clean_focused_dict() -> dict:
    return {
        "data_failure": False,
        "failure_modes": [],
        "primary_failure_mode": None,
        "severity": 1,
        "audit_trail": "All in-scope checks passed.",
        "recommended_fix": "No action needed.",
    }


def _failure_focused_dict(*, modes, primary, severity=4) -> dict:
    return {
        "data_failure": True,
        "failure_modes": list(modes),
        "primary_failure_mode": primary,
        "severity": severity,
        "audit_trail": "Detected stale market snapshot at trade entry.",
        "recommended_fix": "Tighten freshness gate.",
    }


def _build_focused_auditor(parsed_payload=None) -> DataQualityAuditor:
    response = FakeResponse(parsed=parsed_payload or _clean_focused_dict())
    return DataQualityAuditor(
        api_key="test-key",
        client=FakeClient(response),
        types_module=FakeTypesModule(),
    )


class FocusedAuditFindingSchemaTests(unittest.TestCase):
    def test_clean_finding_validates(self):
        f = FocusedAuditFinding.model_validate(_clean_focused_dict())
        self.assertFalse(f.data_failure)
        self.assertEqual(f.failure_modes, [])
        self.assertIsNone(f.primary_failure_mode)
        self.assertEqual(f.severity, 1)

    def test_failure_requires_modes_and_primary(self):
        bad = _failure_focused_dict(modes=[], primary="stale_data")
        with self.assertRaisesRegex(ValueError, "failure_modes must be non-empty"):
            FocusedAuditFinding.model_validate(bad)

        bad = _failure_focused_dict(modes=["stale_data"], primary=None)
        with self.assertRaisesRegex(ValueError, "primary_failure_mode is required"):
            FocusedAuditFinding.model_validate(bad)

    def test_clean_must_not_carry_modes_or_primary(self):
        bad = _clean_focused_dict()
        bad["failure_modes"] = ["stale_data"]
        with self.assertRaisesRegex(ValueError, "failure_modes must be empty"):
            FocusedAuditFinding.model_validate(bad)

        bad = _clean_focused_dict()
        bad["primary_failure_mode"] = "stale_data"
        with self.assertRaisesRegex(ValueError, "primary_failure_mode must be null"):
            FocusedAuditFinding.model_validate(bad)

    def test_primary_failure_mode_must_appear_in_failure_modes(self):
        bad = _failure_focused_dict(modes=["stale_data"], primary="duplicate_sources")
        with self.assertRaisesRegex(ValueError, "primary_failure_mode must appear"):
            FocusedAuditFinding.model_validate(bad)

    def test_unknown_mode_rejected_in_list(self):
        with self.assertRaises(Exception):
            FocusedAuditFinding.model_validate(
                {**_clean_focused_dict(), "failure_modes": ["vibes_too_off"]}
            )

    def test_severity_required_even_on_clean(self):
        bad = _clean_focused_dict()
        del bad["severity"]
        with self.assertRaises(Exception):
            FocusedAuditFinding.model_validate(bad)

    def test_supports_multiple_failure_modes(self):
        f = FocusedAuditFinding.model_validate(
            _failure_focused_dict(
                modes=["stale_data", "missing_primary_sources"],
                primary="missing_primary_sources",
                severity=3,
            )
        )
        self.assertEqual(f.failure_modes, ["stale_data", "missing_primary_sources"])
        self.assertEqual(f.primary_failure_mode, "missing_primary_sources")


class FocusedAuditMethodTests(unittest.TestCase):
    _INPUTS = {
        "trade_payload": {"market_id": "abc"},
        "news_context": "AP coverage of debate.",
        "features_window": [{"close": 0.42, "ts": "2026-04-25T00:00:00Z"}],
    }

    def _assert_focused_call_used_prompt(self, auditor, expected_prompt):
        call = auditor.client.models.calls[0]
        self.assertIs(call["config"].kwargs["response_schema"], FocusedAuditFinding)
        self.assertEqual(call["config"].kwargs["system_instruction"], expected_prompt)

    def test_audit_integrity_uses_integrity_prompt(self):
        auditor = _build_focused_auditor()
        result = auditor.audit_integrity(**self._INPUTS)
        self.assertIsInstance(result, FocusedAuditFinding)
        self._assert_focused_call_used_prompt(auditor, INTEGRITY_SYSTEM_PROMPT)

    def test_audit_interpretation_uses_interpretation_prompt(self):
        auditor = _build_focused_auditor()
        auditor.audit_interpretation(**self._INPUTS)
        self._assert_focused_call_used_prompt(auditor, INTERPRETATION_SYSTEM_PROMPT)

    def test_audit_pipeline_uses_pipeline_prompt(self):
        auditor = _build_focused_auditor()
        auditor.audit_pipeline(**self._INPUTS)
        self._assert_focused_call_used_prompt(auditor, PIPELINE_SYSTEM_PROMPT)

    def test_returns_typed_finding_with_failure_payload(self):
        auditor = _build_focused_auditor(
            parsed_payload=_failure_focused_dict(
                modes=["stale_data"], primary="stale_data", severity=4
            )
        )
        result = auditor.audit_integrity(**self._INPUTS)
        self.assertTrue(result.data_failure)
        self.assertEqual(result.primary_failure_mode, "stale_data")
        self.assertEqual(result.severity, 4)

    def test_unified_audit_trade_unaffected(self):
        """Augment guarantee: audit_trade still returns the rich DataQualityAudit shape."""
        auditor = _build_auditor()  # unified path uses _clean_audit_dict
        result = auditor.audit_trade(
            trade_payload={"market_id": "abc"},
            news_context="Coverage neutral.",
            features_window={"price_t-1": 0.42},
        )
        self.assertIsInstance(result, DataQualityAudit)
        self.assertFalse(result.data_failure)

    def test_focused_methods_reject_empty_inputs(self):
        auditor = _build_focused_auditor()
        with self.assertRaisesRegex(ValueError, "trade_payload"):
            auditor.audit_integrity(trade_payload=None, news_context="x", features_window="y")
        with self.assertRaisesRegex(ValueError, "news_context"):
            auditor.audit_pipeline(trade_payload={"a": 1}, news_context="", features_window="y")
        with self.assertRaisesRegex(ValueError, "features_window"):
            auditor.audit_interpretation(trade_payload={"a": 1}, news_context="x", features_window=None)


if __name__ == "__main__":
    unittest.main()
