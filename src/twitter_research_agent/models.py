from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Mirrors the canonical research-agent shape (see news_research_agent.models and
# reddit_research_agent.models): bullish/bearish theses + 0..100 quality scores +
# signed sentiment. tweet_count is Twitter-specific (analogous to the listing length
# implied by other agents' fetcher payloads) and is exposed so calibration/synthesis
# can downweight reports backed by very few tweets.
class TwitterResearchReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    bullish_thesis: str = Field(..., min_length=1)
    bearish_thesis: str = Field(..., min_length=1)
    evidence_quality_score: int = Field(..., ge=0, le=100)
    misinformation_risk_score: int = Field(..., ge=0, le=100)
    sentiment_score: int = Field(..., ge=-100, le=100)
    key_sources: List[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(..., min_length=1)
    tweet_count: int = Field(..., ge=0)

    @field_validator("key_sources")
    @classmethod
    def _validate_string_lists(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
