from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OutcomeReview(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    matrix_classification: str = Field(
        ...,
        description=(
            "One of: Deserved Success, Good Failure, Dumb Luck, or Poetic Justice."
        ),
    )
    thesis_held: bool = Field(
        ...,
        description="Whether the original thesis held based on available evidence.",
    )
    unknown_at_entry: bool = Field(
        ...,
        description="Whether post-settlement information was impossible to know at trade entry.",
    )
    calibration_reasonable: bool = Field(
        ...,
        description="Whether the original calibration was reasonable given entry-time information.",
    )
    resulting_detected: bool = Field(
        ...,
        description="True when outcome appears driven by luck/noise more than process quality.",
    )
    research_module_flaw: bool = Field(
        ...,
        description="True if there is a detectable flaw in research quality or evidence handling.",
    )
    risk_module_flaw: bool = Field(
        ...,
        description="True if there is a detectable flaw in risk sizing, constraints, or execution discipline.",
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Actionable post-mortem takeaways for model/process improvement.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Concise explanation for the matrix classification and flaw/resulting diagnosis.",
    )
