"""Tests for ``calibration_agent.analyzer.bayesian_fusion`` (P1 #14)."""

from __future__ import annotations

import math
import unittest

from calibration_agent.analyzer import bayesian_fusion


class BayesianFusionTests(unittest.TestCase):
    def test_unanimous_high_confidence_inputs_concentrate(self) -> None:
        # Both estimators agree at 0.75 with high confidence.
        posterior, confidence = bayesian_fusion(
            xgb_prob=0.75,
            xgb_confidence=1.0,
            llm_prob=0.75,
            llm_confidence=1.0,
        )
        self.assertAlmostEqual(posterior, 0.75, places=2)
        self.assertGreater(confidence, 0.95)

    def test_conflicting_inputs_weighted_toward_higher_confidence(self) -> None:
        # XGB says 0.40 with c=0.8, LLM says 0.70 with c=0.6.
        # Posterior should land between them, weighted toward XGB.
        posterior, _ = bayesian_fusion(
            xgb_prob=0.40,
            xgb_confidence=0.8,
            llm_prob=0.70,
            llm_confidence=0.6,
        )
        self.assertGreater(posterior, 0.40)
        self.assertLess(posterior, 0.70)
        # Weighted toward XGB -> closer to 0.40 than to 0.70.
        self.assertLess(abs(posterior - 0.40), abs(posterior - 0.70))

    def test_weak_xgb_falls_back_to_llm(self) -> None:
        # XGB confidence = 0 means its pseudo-count is 0; posterior should
        # mostly reflect the LLM.
        posterior, _ = bayesian_fusion(
            xgb_prob=0.10,
            xgb_confidence=0.0,
            llm_prob=0.80,
            llm_confidence=1.0,
        )
        self.assertGreater(posterior, 0.70)

    def test_weak_llm_falls_back_to_xgb(self) -> None:
        posterior, _ = bayesian_fusion(
            xgb_prob=0.85,
            xgb_confidence=1.0,
            llm_prob=0.20,
            llm_confidence=0.0,
        )
        self.assertGreater(posterior, 0.75)

    def test_edge_case_p_zero_does_not_nan(self) -> None:
        posterior, confidence = bayesian_fusion(
            xgb_prob=0.0,
            xgb_confidence=1.0,
            llm_prob=0.0,
            llm_confidence=1.0,
        )
        self.assertFalse(math.isnan(posterior))
        self.assertFalse(math.isnan(confidence))
        self.assertGreaterEqual(posterior, 0.0)
        self.assertLess(posterior, 0.05)

    def test_edge_case_p_one_does_not_nan(self) -> None:
        posterior, confidence = bayesian_fusion(
            xgb_prob=1.0,
            xgb_confidence=1.0,
            llm_prob=1.0,
            llm_confidence=1.0,
        )
        self.assertFalse(math.isnan(posterior))
        self.assertFalse(math.isnan(confidence))
        self.assertLessEqual(posterior, 1.0)
        self.assertGreater(posterior, 0.95)

    def test_zero_confidence_both_falls_to_uniform_midpoint(self) -> None:
        posterior, confidence = bayesian_fusion(
            xgb_prob=0.20,
            xgb_confidence=0.0,
            llm_prob=0.80,
            llm_confidence=0.0,
        )
        # Posterior is the average of the priors after shrinkage; should
        # land near 0.5 because both pseudo-counts are zero.
        self.assertAlmostEqual(posterior, 0.5, places=2)
        # Confidence is small because the posterior variance is large.
        self.assertLess(confidence, 0.95)

    def test_confidence_increases_with_evidence(self) -> None:
        _, low_evidence = bayesian_fusion(0.6, 0.1, 0.6, 0.1)
        _, high_evidence = bayesian_fusion(0.6, 1.0, 0.6, 1.0)
        self.assertLess(low_evidence, high_evidence)


if __name__ == "__main__":
    unittest.main()
