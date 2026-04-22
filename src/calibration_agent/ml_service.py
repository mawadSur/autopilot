from __future__ import annotations

import random

from models import Market


DEFAULT_JITTER_RANGE = 0.02


def get_xgboost_probability(market: Market) -> float:
    if not isinstance(market, Market):
        raise TypeError("market must be a Market instance")

    market.refresh_derived_fields()
    _ = market.volume_24h
    _ = market.spread
    _ = market.days_to_resolution

    baseline = float(market.implied_prob)
    # TODO: Replace this mock with a call to the trained XGBoost inference endpoint.
    jitter = random.uniform(-DEFAULT_JITTER_RANGE, DEFAULT_JITTER_RANGE)
    return max(0.0, min(1.0, baseline + jitter))
