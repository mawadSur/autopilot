from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


FAILURE_MODE_NAMES = (
    "stale_data",
    "duplicate_sources",
    "missing_primary_sources",
    "misleading_sentiment",
    "scraping_gaps",
    "timestamp_mismatches",
    "incorrect_market_metadata",
)

FailureModeName = Literal[
    "stale_data",
    "duplicate_sources",
    "missing_primary_sources",
    "misleading_sentiment",
    "scraping_gaps",
    "timestamp_mismatches",
    "incorrect_market_metadata",
]


class FailureModeFinding(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    detected: bool = Field(
        ...,
        description="True if this failure mode was observed in the trade context.",
    )
    evidence: str = Field(
        ...,
        min_length=1,
        description="Concise rigorous-analysis text supporting the detection verdict.",
    )


class DataQualityAudit(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    stale_data: FailureModeFinding
    duplicate_sources: FailureModeFinding
    missing_primary_sources: FailureModeFinding
    misleading_sentiment: FailureModeFinding
    scraping_gaps: FailureModeFinding
    timestamp_mismatches: FailureModeFinding
    incorrect_market_metadata: FailureModeFinding

    data_failure: bool = Field(
        ...,
        description="Overall verdict: did a data-integrity failure compromise the trade?",
    )
    failure_mode: Optional[FailureModeName] = Field(
        None,
        description="Specific failure mode name when data_failure is True; otherwise null.",
    )
    severity: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Severity 1-5 when data_failure is True; otherwise null.",
    )
    recommended_fix: str = Field(
        ...,
        min_length=1,
        description="Technical recommendation to prevent recurrence.",
    )

    @model_validator(mode="after")
    def _validate_failure_consistency(self) -> "DataQualityAudit":
        if self.data_failure:
            if self.failure_mode is None:
                raise ValueError("failure_mode is required when data_failure is True")
            if self.severity is None:
                raise ValueError("severity is required when data_failure is True")
        else:
            if self.failure_mode is not None:
                raise ValueError("failure_mode must be null when data_failure is False")
            if self.severity is not None:
                raise ValueError("severity must be null when data_failure is False")
        return self


class FocusedAuditFinding(BaseModel):
    """Result of a focused (modular) audit covering a subset of failure modes.

    Returned by ``audit_integrity()`` (modes 1-3), ``audit_interpretation()``
    (modes 4 + 7), and ``audit_pipeline()`` (modes 5 + 6). Severity is always
    set (1=negligible, even on clean audits) per the modular design contract.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    data_failure: bool = Field(
        ...,
        description="True if any in-scope failure mode was detected.",
    )
    failure_modes: List[FailureModeName] = Field(
        default_factory=list,
        description="Names of detected failure modes (subset of FAILURE_MODE_NAMES).",
    )
    primary_failure_mode: Optional[FailureModeName] = Field(
        None,
        description="The single most important failure mode when data_failure is True; null otherwise.",
    )
    severity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Severity 1-5 (1=negligible even on clean audits, 5=critical).",
    )
    audit_trail: str = Field(
        ...,
        min_length=1,
        description="Detailed reasoning behind the verdict.",
    )
    recommended_fix: str = Field(
        ...,
        min_length=1,
        description="Technical recommendation to prevent recurrence.",
    )

    @model_validator(mode="after")
    def _validate_failure_consistency(self) -> "FocusedAuditFinding":
        if self.data_failure:
            if not self.failure_modes:
                raise ValueError("failure_modes must be non-empty when data_failure is True")
            if self.primary_failure_mode is None:
                raise ValueError("primary_failure_mode is required when data_failure is True")
            if self.primary_failure_mode not in self.failure_modes:
                raise ValueError("primary_failure_mode must appear in failure_modes")
        else:
            if self.failure_modes:
                raise ValueError("failure_modes must be empty when data_failure is False")
            if self.primary_failure_mode is not None:
                raise ValueError("primary_failure_mode must be null when data_failure is False")
        return self
