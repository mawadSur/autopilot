from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Kept legacy source_quality_score (0-10) for back-compat with calibration_agent default
# placeholder + downstream callers; added evidence_quality_score (0-100) per spec alongside.
class NewsResearchReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    timeline: List[str] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    source_quality_score: int = Field(..., ge=0, le=10)
    bullish_thesis: str = Field(..., min_length=1)
    bearish_thesis: str = Field(..., min_length=1)
    evidence_quality_score: int = Field(..., ge=0, le=100)
    misinformation_risk_score: int = Field(..., ge=0, le=100)
    sentiment_score: int = Field(..., ge=-100, le=100)
    key_sources: List[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(..., min_length=1)

    @field_validator("timeline", "key_facts", "key_sources")
    @classmethod
    def _validate_string_lists(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
