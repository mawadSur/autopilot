"""NaN/finite confidence guard tests for ``src/predictor.py``.

Phase 0 safety hardening (P0 #6): the supervisor's confidence floor is a ``<``
comparison, which silently passes NaN. Both predictors (LegacyTransformer and
XGBoost) must therefore validate their confidence floats and return the
neutral result on any non-finite or out-of-range value.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import numpy as np

from predictor import (
    LegacyTransformerPredictor,
    XGBoostPredictor,
    _NEUTRAL_RESULT,
    _is_valid_confidence,
    _validated_decision,
)


# ---------------------------------------------------------------------------
# _is_valid_confidence + _validated_decision
# ---------------------------------------------------------------------------


class IsValidConfidenceTests(unittest.TestCase):
    def test_finite_in_range_is_valid(self) -> None:
        self.assertTrue(_is_valid_confidence(0.0))
        self.assertTrue(_is_valid_confidence(0.5))
        self.assertTrue(_is_valid_confidence(1.0))

    def test_nan_is_invalid(self) -> None:
        self.assertFalse(_is_valid_confidence(float("nan")))

    def test_pos_inf_is_invalid(self) -> None:
        self.assertFalse(_is_valid_confidence(float("inf")))

    def test_neg_inf_is_invalid(self) -> None:
        self.assertFalse(_is_valid_confidence(float("-inf")))

    def test_negative_is_invalid(self) -> None:
        self.assertFalse(_is_valid_confidence(-0.5))

    def test_above_one_is_invalid(self) -> None:
        self.assertFalse(_is_valid_confidence(1.5))


# ---------------------------------------------------------------------------
# LegacyTransformerPredictor._probs_to_decision
# ---------------------------------------------------------------------------


def _make_legacy_predictor(
    *, thr_long: float = 0.55, thr_short: float = 0.60, num_classes: int = 3
) -> LegacyTransformerPredictor:
    """Bypass torch / load_model_bundle for pure decision-mapping tests."""
    p = LegacyTransformerPredictor.__new__(LegacyTransformerPredictor)
    p.thr_long = thr_long
    p.thr_short = thr_short
    p.num_classes = num_classes
    p.margin = 0.0
    return p


class LegacyTransformerNanGuardTests(unittest.TestCase):
    def test_nan_proba_routes_to_neutral(self) -> None:
        p = _make_legacy_predictor()
        side, conf = p._probs_to_decision(
            np.array([0.05, 0.10, float("nan")])
        )
        self.assertEqual((side, conf), _NEUTRAL_RESULT)

    def test_pos_inf_proba_routes_to_neutral(self) -> None:
        p = _make_legacy_predictor()
        # +inf will pick the long branch (>= thr_long) but must be rejected.
        side, conf = p._probs_to_decision(
            np.array([0.05, 0.10, float("inf")])
        )
        self.assertEqual((side, conf), _NEUTRAL_RESULT)

    def test_negative_proba_routes_to_neutral(self) -> None:
        # Construct a probs vector where the dominance check would pick
        # short (-0.5 selected because |p_short| > p_long won't be reached
        # via the threshold gate — but we want to specifically exercise the
        # validator). Use a binary head so the helper is invoked directly.
        p = _make_legacy_predictor(num_classes=1, thr_long=-1.0, thr_short=-1.0)
        # _probs_to_decision treats single value as P(long); set p_long=-0.5
        # which would pass the thr_long gate (>= -1.0) but is invalid.
        side, conf = p._probs_to_decision(np.array([-0.5]))
        self.assertEqual((side, conf), _NEUTRAL_RESULT)

    def test_validated_decision_passes_legitimate_value(self) -> None:
        side, conf = _validated_decision("buy", 0.7)
        self.assertEqual(side, "buy")
        self.assertAlmostEqual(conf, 0.7, places=6)


# ---------------------------------------------------------------------------
# XGBoostPredictor._predict NaN routing
# ---------------------------------------------------------------------------


class _FakeXgbModel:
    """Stub ``predict_proba`` that returns whatever value the test injects."""

    def __init__(self, proba: float) -> None:
        self._proba = proba

    def predict_proba(self, X: Any) -> np.ndarray:
        # XGBoost's predict_proba shape: (n_samples, 2) — column 1 is P(long).
        return np.array([[1.0 - self._proba, self._proba]], dtype=np.float64)


class _FakeBuffer:
    """Mimics enough of the per-symbol pandas buffer for ``_predict``."""

    def __init__(self, n: int = 300) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def copy(self) -> "_FakeBuffer":  # pragma: no cover - trivial
        return self


def _build_xgb_predictor(proba: float) -> XGBoostPredictor:
    p = XGBoostPredictor.__new__(XGBoostPredictor)
    p.model = _FakeXgbModel(proba)
    p.feature_cols = ["f1", "f2"]
    p.thr_long = 0.5
    p._buffers = {"ETH/USD": _FakeBuffer()}
    return p


class XgbPredictNanGuardTests(unittest.TestCase):
    """Patch ``compute_features`` so we exercise the proba branch only."""

    def _patched_call(self, predictor: XGBoostPredictor, symbol: str = "ETH/USD"):
        import pandas as pd

        feats = pd.DataFrame(
            {
                "f1": np.linspace(0.1, 0.5, 10),
                "f2": np.linspace(0.2, 0.6, 10),
            }
        )
        with mock.patch("predictor.compute_features", return_value=feats) if False \
                else mock.patch("utils.compute_features", return_value=feats):
            return predictor._predict(symbol)

    def test_nan_proba_returns_none(self) -> None:
        p = _build_xgb_predictor(float("nan"))
        result = self._patched_call(p)
        self.assertIsNone(result)

    def test_pos_inf_proba_returns_none(self) -> None:
        p = _build_xgb_predictor(float("inf"))
        result = self._patched_call(p)
        self.assertIsNone(result)

    def test_negative_proba_returns_none(self) -> None:
        # XGBoost normally returns [0, 1], but a corrupted booster could
        # yield negatives. Validator must reject.
        p = _build_xgb_predictor(-0.5)
        result = self._patched_call(p)
        self.assertIsNone(result)

    def test_valid_proba_returns_float(self) -> None:
        p = _build_xgb_predictor(0.75)
        result = self._patched_call(p)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result, 0.75, places=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
