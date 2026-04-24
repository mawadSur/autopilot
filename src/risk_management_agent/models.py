from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskMetrics(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    market_price: float = Field(..., gt=0.0, lt=1.0)
    calibrated_true_prob: float = Field(..., ge=0.0, le=1.0)
    bankroll: float = Field(..., ge=0.0)
    raw_kelly_size_pct: float = Field(..., ge=0.0, le=100.0)
    fractional_kelly_size_pct: float = Field(..., ge=0.0, le=100.0)
    liquidity_penalty_multiplier: float = Field(..., ge=0.0, le=1.0)
    correlation_penalty_multiplier: float = Field(..., ge=0.0, le=1.0)
    same_category_open_positions: int = Field(..., ge=0)
    liquidity_penalty_applied: bool
    correlation_penalty_applied: bool
    adjusted_position_size_pct: float = Field(..., ge=0.0, le=100.0)
    max_loss_if_wrong: float = Field(..., ge=0.0)
    expected_value_estimate: float


class RiskAssessment(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    allow_trade: bool = Field(
        ...,
        description="True if the trade meets all safety and risk criteria.",
    )
    simulated_position_size_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="The recommended size as a percentage of bankroll, such as 2.5.",
    )
    max_loss_if_wrong: float = Field(
        ...,
        ge=0.0,
        description="The dollar amount at risk based on the current bankroll.",
    )
    expected_value_estimate: float = Field(
        ...,
        description="The mathematical expected value estimate of the trade.",
    )
    top_risk_reasons: List[str] = Field(
        ...,
        description="The primary factors that limited the size or blocked the trade.",
    )
    kill_switch_triggered: bool = Field(
        ...,
        description="True if extreme risks like systemic correlation or zero liquidity were found.",
    )
    final_recommendation: Literal["reject", "small", "medium", "high-conviction paper trade"] = Field(
        ...,
        description="Final risk recommendation after sizing and safety checks.",
    )
    risk_logic_summary: str = Field(
        ...,
        min_length=1,
        description="Concise explanation of the sizing math and risk penalties applied.",
    )

    @field_validator("top_risk_reasons")
    @classmethod
    def _validate_reason_list(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must be non-empty strings")
        return cleaned
