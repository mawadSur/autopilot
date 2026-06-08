from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from models import Market


WEIGHT_LIQUIDITY = 0.40
WEIGHT_TIME_TO_RESOLUTION = 0.20
WEIGHT_ANOMALIES = 0.20
WEIGHT_CLARITY = 0.20
MAX_FILTER_SPREAD = 0.15
MAX_LIQUIDITY_VOLUME = 100_000.0
MIN_LIQUIDITY_VOLUME = 1_000.0
SWEET_SPOT_MIN_DAYS = 3.0
SWEET_SPOT_MAX_DAYS = 10.0
TIME_SCORE_DECAY_MAX_DAYS = 30.0
ANOMALY_SCORES = {
    "VOL_SPIKE": 45.0,
    "DECOUPLED": 35.0,
    "WIDE_SPREAD": 20.0,
    "INFO_EDGE": 10.0,
    "AMBIGUOUS": -35.0,
}


@dataclass
class PriorityAssessment:
    research_priority: int
    reason: str
    component_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "research_priority": self.research_priority,
            "reason": self.reason,
            "component_scores": dict(self.component_scores),
        }


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _normalize_log(value: float, *, lower: float, upper: float) -> float:
    safe_value = max(float(value), lower)
    lower_log = math.log10(lower)
    upper_log = math.log10(upper)
    if upper_log <= lower_log:
        return 0.0
    normalized = (math.log10(safe_value) - lower_log) / (upper_log - lower_log)
    return 100.0 * _clamp(normalized, 0.0, 1.0)


def _score_liquidity(market: Market) -> float:
    volume_score = _normalize_log(
        market.volume_24h,
        lower=MIN_LIQUIDITY_VOLUME,
        upper=MAX_LIQUIDITY_VOLUME,
    )
    spread_score = 100.0 * (1.0 - _clamp(market.spread / MAX_FILTER_SPREAD, 0.0, 1.0))
    return _clamp((0.65 * volume_score) + (0.35 * spread_score))


def _score_time_to_resolution(days_to_resolution: float) -> float:
    days = max(float(days_to_resolution), 0.0)
    if SWEET_SPOT_MIN_DAYS <= days <= SWEET_SPOT_MAX_DAYS:
        return 100.0
    if days < SWEET_SPOT_MIN_DAYS:
        return _clamp(100.0 * (days / SWEET_SPOT_MIN_DAYS))
    if days >= TIME_SCORE_DECAY_MAX_DAYS:
        return 0.0
    decay_window = TIME_SCORE_DECAY_MAX_DAYS - SWEET_SPOT_MAX_DAYS
    score = 100.0 * (1.0 - ((days - SWEET_SPOT_MAX_DAYS) / decay_window))
    return _clamp(score)


def _score_anomalies(anomaly_flags: Sequence[str]) -> float:
    score = 0.0
    seen = set()
    for flag in anomaly_flags:
        normalized = str(flag or "").strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        score += ANOMALY_SCORES.get(normalized, 12.0)
    return _clamp(score)


def _reason_from_components(
    *,
    market: Market,
    anomaly_flags: Sequence[str],
    clarity_score: float,
    liquidity_score: float,
    time_score: float,
) -> str:
    normalized_flags = {str(flag or "").strip().upper() for flag in anomaly_flags if str(flag or "").strip()}
    positives: List[str] = []
    cautions: List[str] = []

    if "VOL_SPIKE" in normalized_flags:
        positives.append("high volume spike")
    if "DECOUPLED" in normalized_flags:
        positives.append("price-volume decoupling")
    if "WIDE_SPREAD" in normalized_flags:
        cautions.append("wide spread")

    if clarity_score >= 80.0:
        positives.append("clear resolution rules")
    elif clarity_score < 50.0 or "AMBIGUOUS" in normalized_flags:
        cautions.append("unclear resolution rules")

    if SWEET_SPOT_MIN_DAYS <= market.days_to_resolution <= SWEET_SPOT_MAX_DAYS:
        positives.append("inside the 3-10 day window")
    elif market.days_to_resolution < 1.0:
        cautions.append("very near resolution")
    elif market.days_to_resolution < SWEET_SPOT_MIN_DAYS:
        cautions.append("near deadline")
    elif time_score < 40.0:
        cautions.append("far from resolution")

    if liquidity_score >= 75.0:
        positives.append("strong liquidity")
    elif market.volume_24h < MIN_LIQUIDITY_VOLUME * 3.0:
        cautions.append("thin liquidity")
    elif market.spread >= 0.08:
        cautions.append("wide spread")

    parts = positives[:3] if positives else cautions[:3]
    if not parts:
        parts = ["balanced market setup"]
    if len(parts) == 1:
        return parts[0].capitalize()
    if len(parts) == 2:
        return f"{parts[0].capitalize()} with {parts[1]}"
    return f"{parts[0].capitalize()} with {parts[1]} and {parts[2]}"


def calculate_priority(
    market: Market,
    *,
    anomaly_flags: Optional[Sequence[str]] = None,
    clarity_score: float = 50.0,
    now: Optional[datetime] = None,
) -> PriorityAssessment:
    market.refresh_derived_fields(now=now)
    liquidity_score = _score_liquidity(market)
    time_score = _score_time_to_resolution(market.days_to_resolution)
    anomaly_score = _score_anomalies(anomaly_flags or ())
    normalized_clarity_score = _clamp(clarity_score)

    weighted_score = (
        (WEIGHT_LIQUIDITY * liquidity_score)
        + (WEIGHT_TIME_TO_RESOLUTION * time_score)
        + (WEIGHT_ANOMALIES * anomaly_score)
        + (WEIGHT_CLARITY * normalized_clarity_score)
    )
    research_priority = int(round(_clamp(weighted_score)))
    reason = _reason_from_components(
        market=market,
        anomaly_flags=anomaly_flags or (),
        clarity_score=normalized_clarity_score,
        liquidity_score=liquidity_score,
        time_score=time_score,
    )
    return PriorityAssessment(
        research_priority=research_priority,
        reason=reason,
        component_scores={
            "liquidity": round(liquidity_score, 2),
            "time_to_resolution": round(time_score, 2),
            "anomalies": round(anomaly_score, 2),
            "clarity": round(normalized_clarity_score, 2),
        },
    )
