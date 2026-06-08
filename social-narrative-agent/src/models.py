from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SocialPost(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    platform: str = Field(..., min_length=1)
    author_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    is_reply: bool
    is_quote: bool
    linked_urls: List[HttpUrl] = Field(default_factory=list)
    engagement_score: int = Field(..., ge=0)


class NarrativeAnalysis(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    bullish_thesis: str = Field(..., min_length=1)
    bearish_thesis: str = Field(..., min_length=1)
    unresolved_questions: List[str] = Field(default_factory=list)
    signal_quality_score: int = Field(..., ge=0, le=10)
    crowd_overconfidence_score: int = Field(..., ge=0, le=10)
    misinformation_risk: int = Field(..., ge=0, le=10)
    crowd_beliefs: List[str] = Field(..., min_length=5, max_length=5)
    market_alignment: Literal["ahead", "behind", "aligned"]
    reasoning: str = Field(..., min_length=1)

    @field_validator("unresolved_questions", "crowd_beliefs")
    @classmethod
    def _validate_non_empty_entries(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
