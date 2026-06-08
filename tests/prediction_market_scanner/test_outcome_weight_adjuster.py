"""Tests for ``OutcomeWeightAdjuster`` (Lane C P1 #16, D3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from calibration_agent.analyzer import bayesian_fusion
from calibration_agent.outcome_weight_adjuster import OutcomeWeightAdjuster


class _AdjusterTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        root = Path(self._tmpdir.name)
        self.weight_file = root / "outcome_llm_weight.json"
        self.audit_file = root / "outcome_llm_weight_audit.jsonl"

    def _adjuster(self, **kwargs) -> OutcomeWeightAdjuster:
        defaults = dict(
            weight_file=self.weight_file,
            audit_file=self.audit_file,
            initial_weight=1.0,
            max_weight=2.0,
            min_weight=0.5,
            decay=0.95,
        )
        defaults.update(kwargs)
        return OutcomeWeightAdjuster(**defaults)


class BoundedMagnitudeTests(_AdjusterTestBase):
    def test_ten_dumb_luck_never_below_min(self) -> None:
        adjuster = self._adjuster()
        for _ in range(10):
            new_weight = adjuster.update_weight({"matrix_classification": "Dumb Luck"})
            self.assertGreaterEqual(new_weight, 0.5)
            self.assertLessEqual(new_weight, 2.0)
        self.assertGreaterEqual(adjuster.current_weight(), 0.5)

    def test_ten_deserved_success_never_above_max(self) -> None:
        adjuster = self._adjuster()
        for _ in range(10):
            new_weight = adjuster.update_weight(
                {"matrix_classification": "Deserved Success"}
            )
            self.assertLessEqual(new_weight, 2.0)
            self.assertGreaterEqual(new_weight, 0.5)
        self.assertLessEqual(adjuster.current_weight(), 2.0)

    def test_a_hundred_poetic_justice_never_below_min(self) -> None:
        # Sanity for the hardest penalty.
        adjuster = self._adjuster()
        for _ in range(100):
            adjuster.update_weight({"matrix_classification": "Poetic Justice"})
        self.assertGreaterEqual(adjuster.current_weight(), 0.5)


class ClassificationParseTests(_AdjusterTestBase):
    def test_unknown_classification_yields_zero_delta(self) -> None:
        adjuster = self._adjuster()
        before = adjuster.current_weight()
        after = adjuster.update_weight({"matrix_classification": "Mystery"})
        # Decay applies even on zero-delta? Yes: target=1.0 so decay(1.0) preserves.
        self.assertAlmostEqual(after, before, places=6)

    def test_outcome_review_pydantic_object_accepted(self) -> None:
        # Anything with `matrix_classification` attribute should work.
        class _Review:
            matrix_classification = "Deserved Success"

        adjuster = self._adjuster()
        before = adjuster.current_weight()
        after = adjuster.update_weight(_Review())
        self.assertGreater(after, before)

    def test_classify_outcome_normalizes(self) -> None:
        adjuster = self._adjuster()
        self.assertEqual(
            adjuster.classify_outcome({"matrix_classification": "deserved_success"}),
            "deserved success",
        )
        self.assertEqual(
            adjuster.classify_outcome({"matrix_classification": "Poetic-Justice"}),
            "poetic justice",
        )
        self.assertEqual(
            adjuster.classify_outcome({"matrix_classification": "garbage"}),
            "unknown",
        )


class AuditLogTests(_AdjusterTestBase):
    def test_audit_records_every_adjustment(self) -> None:
        adjuster = self._adjuster()
        adjuster.update_weight({"matrix_classification": "Deserved Success", "trade_id": "a"})
        adjuster.update_weight({"matrix_classification": "Dumb Luck", "trade_id": "b"})
        adjuster.update_weight({"matrix_classification": "Good Failure", "trade_id": "c"})

        self.assertTrue(self.audit_file.exists())
        lines = self.audit_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        entries = [json.loads(line) for line in lines]
        self.assertEqual(entries[0]["outcome_class"], "deserved success")
        self.assertEqual(entries[1]["outcome_class"], "dumb luck")
        self.assertEqual(entries[2]["outcome_class"], "good failure")
        for entry in entries:
            self.assertIn("ts_utc", entry)
            self.assertIn("previous_weight", entry)
            self.assertIn("new_weight", entry)
            self.assertIn("delta", entry)
            self.assertIn("trade_review", entry)


class WeightFileFormatTests(_AdjusterTestBase):
    def test_weight_file_is_json_with_llm_weight_key(self) -> None:
        adjuster = self._adjuster()
        adjuster.update_weight({"matrix_classification": "Deserved Success"})
        payload = json.loads(self.weight_file.read_text(encoding="utf-8"))
        self.assertIn("llm_weight", payload)
        self.assertIn("updated_at", payload)
        self.assertGreater(float(payload["llm_weight"]), 1.0)

    def test_initial_weight_used_when_file_missing(self) -> None:
        adjuster = self._adjuster(initial_weight=1.5)
        self.assertAlmostEqual(adjuster.current_weight(), 1.5)


class PostmortemHookTests(_AdjusterTestBase):
    def test_apply_postmortem_delta_is_clipped(self) -> None:
        adjuster = self._adjuster()
        # Massive delta should clip to max.
        new_weight = adjuster.apply_postmortem_delta(weight_delta=10.0)
        self.assertEqual(new_weight, 2.0)
        # Massive negative delta should clip to min.
        new_weight = adjuster.apply_postmortem_delta(weight_delta=-10.0)
        self.assertEqual(new_weight, 0.5)


class FusionLlmWeightCouplingTests(unittest.TestCase):
    """Test the Bayesian fusion respects llm_weight."""

    def test_llm_weight_half_versus_one_changes_posterior(self) -> None:
        # XGB and LLM disagree; with llm_weight=1.0 the LLM pulls posterior.
        post_full, _ = bayesian_fusion(
            xgb_prob=0.30,
            xgb_confidence=0.8,
            llm_prob=0.80,
            llm_confidence=1.0,
            llm_weight=1.0,
        )
        # With llm_weight=0.5, the LLM's effective pseudo-count halves, so
        # the posterior should be pulled less far from XGB.
        post_half, _ = bayesian_fusion(
            xgb_prob=0.30,
            xgb_confidence=0.8,
            llm_prob=0.80,
            llm_confidence=1.0,
            llm_weight=0.5,
        )
        # Both posteriors are between 0.30 and 0.80; the half-weighted one
        # is closer to the XGB baseline.
        self.assertGreater(post_full, post_half)
        self.assertGreater(post_full, 0.30)
        self.assertGreater(post_half, 0.30)
        self.assertLess(post_full, 0.80)
        self.assertLess(post_half, 0.80)
        # Closer to XGB:
        self.assertLess(abs(post_half - 0.30), abs(post_full - 0.30))


if __name__ == "__main__":
    unittest.main()
