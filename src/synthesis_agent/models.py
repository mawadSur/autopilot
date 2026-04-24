from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SynthesisReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    implied_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current market-implied probability (e.g. 0.45 for 45%)",
    )
    narrative_direction: Literal["bullish", "bearish", "mixed"] = Field(
        ...,
        description="Narrative-implied direction based on news and social sentiment",
    )
    has_unique_evidence: bool = Field(
        ...,
        description="True if the narrative contains unique, verifiable evidence not yet reflected in price",
    )
    reasons_market_is_right: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Top 3 distinct reasons the market's current odds could be perfectly correct",
    )
    reasons_market_is_wrong: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Top 3 distinct reasons the market's current odds could be mispriced",
    )
    verdict: Literal["no edge", "possible edge", "strong research edge"] = Field(
        ...,
        description="Final trading verdict. Must be 'no edge' if evidence is mixed or weak.",
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Concise explanation in plain English justifying the verdict",
    )

    @field_validator("reasons_market_is_right", "reasons_market_is_wrong")
    @classmethod
    def _validate_reason_lists(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if len(cleaned) != 3:
            raise ValueError("Reason lists must contain exactly 3 items")
        if any(not value for value in cleaned):
            raise ValueError("Reason list entries must be non-empty strings")
        return cleaned
