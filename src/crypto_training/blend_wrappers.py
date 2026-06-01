"""Persistable wrapper classes for blended-proba XGBoost models.

These are picklable from any context (no closure capture) so the canonical
``scripts/validate_xgboost_winrate.py`` validator can load them via
``joblib.load`` without needing the training script on the path.

The wrappers expose the standard sklearn ``predict_proba(X) -> (n, 2)`` API,
so the validator's ``_get_probas`` works unchanged.
"""

from __future__ import annotations

import numpy as np


class DetBlend:
    """Single-booster raw + calibrated blend.

    Stores a fitted ``xgb.XGBClassifier`` (``booster``) and its
    ``CalibratedClassifierCV`` (``cal``). ``predict_proba`` returns
    ``alpha * P_raw + (1 - alpha) * P_calibrated`` stacked as
    ``[P(class 0), P(class 1)]``.

    The blend is "raw=high-recall, calibrated=high-precision-on-val": a
    high alpha (~0.6-0.9) lets the raw booster's confident tails through
    where sigmoid calibration would compress them, lifting precision at
    high thresholds without retraining.
    """

    def __init__(self, booster, cal, alpha: float):
        self.booster = booster
        self.cal = cal
        self.alpha = float(alpha)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p_raw = self.booster.predict_proba(X)[:, 1]
        p_cal = self.cal.predict_proba(X)[:, 1]
        p = self.alpha * p_raw + (1 - self.alpha) * p_cal
        return np.column_stack([1 - p, p])


class StochasticEnsemble:
    """K-booster bagging-style ensemble with per-member sigmoid calibration."""

    def __init__(self, calibrated_members, raw_members):
        self.calibrated_members = calibrated_members
        self.raw_members = raw_members
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.calibrated_members], axis=0)

    def predict_raw_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.raw_members], axis=0)


class BlendedEnsemble:
    """Stochastic ensemble + linear raw/calibrated blend."""

    def __init__(self, ensemble: StochasticEnsemble, alpha: float):
        self.ensemble = ensemble
        self.alpha = float(alpha)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p_cal = self.ensemble.predict_proba(X)[:, 1]
        p_raw = self.ensemble.predict_raw_proba(X)[:, 1]
        p = self.alpha * p_raw + (1 - self.alpha) * p_cal
        return np.column_stack([1 - p, p])
