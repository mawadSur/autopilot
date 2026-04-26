from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Renamed pro_argument/anti_argument -> bullish_thesis/bearish_thesis to align with the
# cross-agent research spec; production callers (calibration_agent, orchestrator) consume
# the report opaquely so the rename only touches reddit-agent tests + orchestrator fixture.
class RedditResearchReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    bullish_thesis: str = Field(..., min_length=1)
    bearish_thesis: str = Field(..., min_length=1)
    key_evidence: List[str] = Field(default_factory=list)
    key_assumptions: List[str] = Field(default_factory=list)
    conviction_score: int = Field(..., ge=0, le=10)
    evidence_quality_score: int = Field(..., ge=0, le=100)
    misinformation_risk_score: int = Field(..., ge=0, le=100)
    sentiment_score: int = Field(..., ge=-100, le=100)
    key_sources: List[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(..., min_length=1)
    pricing_assessment: Literal["underpriced", "overpriced", "fairly priced", "unclear"]
    assessment_reasoning: str = Field(..., min_length=1)

    @field_validator("key_evidence", "key_assumptions", "key_sources")
    @classmethod
    def _validate_non_empty_entries(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
