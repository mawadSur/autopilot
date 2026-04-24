from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutcomeReview(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    trade_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the settled trade.",
    )
    classification: Literal[
        "deserved success",
        "good failure",
        "dumb luck",
        "poetic justice",
    ] = Field(
        ...,
        description="Outcome quadrant classification based on process quality and result quality.",
    )
    process_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="How well the agents followed the strategy from 0 to 10.",
    )
    outcome_score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Outcome quality based on simulated profit/loss from 0 to 10.",
    )
    new_info_impact: str = Field(
        ...,
        min_length=1,
        description="What emerged after entry that the model missed.",
    )
    confidence_in_classification: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in this classification from 0 to 100.",
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Concise explanation of why this outcome fits the selected quadrant.",
    )
    strategy_adjustment_needed: bool = Field(
        ...,
        description="True if this indicates a systemic flaw, False if this was statistical noise.",
    )
