from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RedditResearchReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    pro_argument: str = Field(..., min_length=1)
    anti_argument: str = Field(..., min_length=1)
    key_evidence: List[str] = Field(default_factory=list)
    key_assumptions: List[str] = Field(default_factory=list)
    conviction_score: int = Field(..., ge=0, le=10)
    evidence_quality_score: int = Field(..., ge=0, le=10)
    pricing_assessment: Literal["underpriced", "overpriced", "fairly priced", "unclear"]
    assessment_reasoning: str = Field(..., min_length=1)

    @field_validator("key_evidence", "key_assumptions")
    @classmethod
    def _validate_non_empty_entries(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
