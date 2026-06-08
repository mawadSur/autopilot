from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


FAILURE_DIAGNOSES = ("regime_shift", "calibration_error", "narrative_overfit")

FailureDiagnosis = Literal["regime_shift", "calibration_error", "narrative_overfit"]


class FeatureRecommendation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Snake_case feature identifier (e.g. 'rolling_news_sentiment_zscore_30d').",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Concrete description of what the feature computes from available inputs.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this feature addresses the diagnosed blind spot.",
    )


class RetrainingRecommendation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    failure_diagnosis: FailureDiagnosis = Field(
        ...,
        description="One of: regime_shift, calibration_error, narrative_overfit.",
    )
    retraining_priority: int = Field(
        ...,
        ge=0,
        le=10,
        description="0=defer indefinitely, 10=block trading until retrained.",
    )
    new_features: List[FeatureRecommendation] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly three new feature recommendations targeting the diagnosed blind spot.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Concise explanation tying the diagnosis to the trade evidence.",
    )
