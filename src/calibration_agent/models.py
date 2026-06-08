from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CalibrationReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    xgboost_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The baseline probability from the ML model.",
    )
    llm_adjustment_pct_points: float = Field(
        ...,
        description="The adjustment made by the LLM in percentage points, such as +2.5 or -1.0.",
    )
    calibrated_true_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The final estimated true probability after calibration.",
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="How confident the agent is in this calibration.",
    )
    key_drivers: List[str] = Field(
        ...,
        description="Specific evidence supporting the calibration.",
    )
    key_uncertainties: List[str] = Field(
        ...,
        description="Explicitly quantified unknowns that could change the calibration.",
    )
    edge_vs_market: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="The calibrated true probability minus the market implied probability.",
    )
    action: Literal["pass", "monitor", "paper-trade candidate"] = Field(
        ...,
        description="Recommended action based on the estimated edge and confidence.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Explanation of why the baseline was adjusted or kept, and why the action was chosen.",
    )

    @field_validator("key_drivers", "key_uncertainties")
    @classmethod
    def _validate_string_lists(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
