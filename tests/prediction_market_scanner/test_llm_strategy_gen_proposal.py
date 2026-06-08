"""Tests for :mod:`llm_strategy_gen.feature_proposal`."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from llm_strategy_gen.feature_proposal import (
    DEFAULT_MAX_PROPOSALS,
    FeatureProposal,
    FeatureProposalGenerator,
)
from llm_strategy_gen.outcome_analyst import OutcomePattern


class _FakeLLM:
    """Test-injected LLM caller — bypasses env-var check."""

    _test_inject = True

    def __init__(self, *, responses=None, raises=None):
        # responses: list of dicts/strings returned in sequence; final
        # response is reused if the caller exhausts the list.
        self.responses = list(responses or [])
        self.raises = raises
        self.calls = 0

    def __call__(self, prompt, timeout_s=30):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        if not self.responses:
            return {}
        idx = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[idx]


def _good_proposal(name="atr_z_score") -> dict:
    return {
        "name": name,
        "description": "Z-scored ATR vs 240-bar baseline.",
        "python_code": (
            "def feature(df):\n"
            "    return (df['atr'] - df['atr'].rolling(240).mean()) / "
            "df['atr'].rolling(240).std()"
        ),
        "expected_lift_sharpe": 0.18,
        "risk_notes": "Sensitive to ATR window gaps.",
    }


def _patterns(n: int = 3) -> list[OutcomePattern]:
    return [
        OutcomePattern(
            pattern_summary=f"pattern {i}",
            frequency=10 - i,
            signal_quality_score=70.0,
            evidence_postmortem_ids=[f"t{i}-{j}" for j in range(3)],
        )
        for i in range(n)
    ]


class FeatureProposalGeneratorBasicTests(unittest.TestCase):
    def test_empty_patterns_returns_empty(self):
        gen = FeatureProposalGenerator(llm_caller=_FakeLLM(responses=[_good_proposal()]))
        self.assertEqual(gen.propose([]), [])

    def test_no_llm_caller_returns_empty(self):
        gen = FeatureProposalGenerator(llm_caller=None)
        self.assertEqual(gen.propose(_patterns(3)), [])

    def test_three_valid_proposals(self):
        responses = [
            _good_proposal("a_one"),
            _good_proposal("a_two"),
            _good_proposal("a_three"),
        ]
        fake = _FakeLLM(responses=responses)
        gen = FeatureProposalGenerator(llm_caller=fake)
        proposals = gen.propose(_patterns(3))

        self.assertEqual(len(proposals), 3)
        self.assertEqual(fake.calls, 3)
        for p in proposals:
            self.assertIsInstance(p, FeatureProposal)
            self.assertTrue(p.name)
            self.assertIn("def feature", p.python_code)
            self.assertEqual(p.expected_lift_sharpe, 0.18)

    def test_max_proposals_respected(self):
        # 5 patterns, but max_proposals=2 caps it at 2 LLM calls.
        fake = _FakeLLM(responses=[_good_proposal()])
        gen = FeatureProposalGenerator(llm_caller=fake, max_proposals=2)
        proposals = gen.propose(_patterns(5))
        self.assertEqual(len(proposals), 2)
        self.assertEqual(fake.calls, 2)

    def test_default_max_proposals_is_three(self):
        fake = _FakeLLM(responses=[_good_proposal()])
        gen = FeatureProposalGenerator(llm_caller=fake)
        self.assertEqual(gen.max_proposals, DEFAULT_MAX_PROPOSALS)
        self.assertEqual(DEFAULT_MAX_PROPOSALS, 3)


class FeatureProposalGeneratorRejectionTests(unittest.TestCase):
    def test_syntax_error_in_python_code_rejected(self):
        bad = _good_proposal()
        bad["python_code"] = "def feature(df:\n    return df  # missing paren"
        good = _good_proposal("good_one")
        fake = _FakeLLM(responses=[bad, good, good])
        gen = FeatureProposalGenerator(llm_caller=fake)

        proposals = gen.propose(_patterns(3))
        # First rejected (syntax error), next two accepted.
        self.assertEqual(len(proposals), 2)
        self.assertTrue(all(p.name == "good_one" for p in proposals))
        self.assertEqual(fake.calls, 3)

    def test_missing_name_rejected(self):
        bad = _good_proposal()
        bad["name"] = ""
        fake = _FakeLLM(responses=[bad])
        gen = FeatureProposalGenerator(llm_caller=fake)
        self.assertEqual(gen.propose(_patterns(1)), [])

    def test_missing_python_code_rejected(self):
        bad = _good_proposal()
        bad["python_code"] = ""
        fake = _FakeLLM(responses=[bad])
        gen = FeatureProposalGenerator(llm_caller=fake)
        self.assertEqual(gen.propose(_patterns(1)), [])

    def test_unparseable_llm_string_rejected(self):
        # extract_json_object should fail on this; caller returns string
        # instead of dict so we go through the JSON parse path.
        fake = _FakeLLM(responses=["not even close to JSON"])
        gen = FeatureProposalGenerator(llm_caller=fake)
        self.assertEqual(gen.propose(_patterns(1)), [])

    def test_json_string_with_fence_recovered(self):
        # Markdown-fenced JSON should be unwrapped via extract_json_object.
        wrapped = "```json\n" + json.dumps(_good_proposal("fenced")) + "\n```"
        fake = _FakeLLM(responses=[wrapped])
        gen = FeatureProposalGenerator(llm_caller=fake)
        proposals = gen.propose(_patterns(1))
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].name, "fenced")

    def test_unexpected_return_type_rejected(self):
        fake = _FakeLLM(responses=[12345])
        gen = FeatureProposalGenerator(llm_caller=fake)
        self.assertEqual(gen.propose(_patterns(1)), [])


class FeatureProposalGeneratorTimeoutTests(unittest.TestCase):
    def test_timeout_returns_empty(self):
        fake = _FakeLLM(raises=TimeoutError("simulated"))
        gen = FeatureProposalGenerator(llm_caller=fake)
        self.assertEqual(gen.propose(_patterns(2)), [])

    def test_partial_failure_keeps_good_ones(self):
        class _IntermittentLLM:
            _test_inject = True

            def __init__(self):
                self.calls = 0

            def __call__(self, prompt, timeout_s=30):
                self.calls += 1
                if self.calls == 2:
                    raise TimeoutError("flaked on call 2")
                return _good_proposal(f"feat_{self.calls}")

        fake = _IntermittentLLM()
        gen = FeatureProposalGenerator(llm_caller=fake)
        proposals = gen.propose(_patterns(3))
        self.assertEqual(len(proposals), 2)
        self.assertEqual([p.name for p in proposals], ["feat_1", "feat_3"])

    def test_missing_api_key_no_inject(self):
        class _ProdLLM:
            _test_inject = False
            calls = 0

            def __call__(self, prompt, timeout_s=30):
                self.calls += 1
                return _good_proposal()

        prod = _ProdLLM()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            gen = FeatureProposalGenerator(llm_caller=prod)
            proposals = gen.propose(_patterns(2))
        self.assertEqual(prod.calls, 0)
        self.assertEqual(proposals, [])


class FeatureProposalGeneratorTimestampTests(unittest.TestCase):
    def test_proposed_at_utc_is_iso(self):
        fake = _FakeLLM(responses=[_good_proposal()])
        gen = FeatureProposalGenerator(llm_caller=fake)
        proposals = gen.propose(_patterns(1))
        self.assertEqual(len(proposals), 1)
        ts = proposals[0].proposed_at_utc
        self.assertTrue(ts.endswith("Z"))
        # Best-effort parse:
        import datetime as _dt
        parsed = _dt.datetime.fromisoformat(ts.rstrip("Z"))
        self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
