"""Pydantic request/response models for the prediction-market FastAPI.

Response shapes for downstream agents (``CalibrationReport``,
``RiskAssessment``, ``RiskMetrics``, ``RedditResearchReport``,
``NewsResearchReport``) are re-exported from their owning agents so the API
contract stays in lockstep with the agent schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Re-export agent response models so API consumers can import directly from
# ``src.api.models`` without reaching into the agent packages.
from calibration_agent.models import CalibrationReport  # noqa: F401
from news_research_agent.models import NewsResearchReport  # noqa: F401
from reddit_research_agent.models import RedditResearchReport  # noqa: F401
from risk_management_agent.models import RiskAssessment, RiskMetrics  # noqa: F401


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    service: str = "autopilot-prediction-market"
    version: str = "0.1.0"


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n: int = Field(default=20, ge=0, le=200)
    category: Optional[str] = Field(default=None, description="Optional category filter (case-insensitive).")
    min_volume_24h: float = Field(default=5_000.0, ge=0.0)
    page_size: int = Field(default=100, ge=1, le=500)
    max_pages: Optional[int] = Field(default=None, ge=1)


class ScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan_id: str
    count: int
    results: List[Dict[str, Any]]


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    top_n: int = Field(default=20, ge=1, le=200)
    category: Optional[str] = None
    subreddits: Optional[List[str]] = None


class ResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str
    reddit_query: str
    news_query: str
    reddit_report: Dict[str, Any]
    news_report: Dict[str, Any]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    top_n: int = Field(default=20, ge=1, le=200)
    category: Optional[str] = None


class RiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    calibration: CalibrationReport
    bankroll: float = Field(default=10_000.0, ge=0.0)
    top_n: int = Field(default=20, ge=1, le=200)
    category: Optional[str] = None


class RiskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str
    risk_metrics: RiskMetrics
    risk_assessment: RiskAssessment


class PaperTradeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=50)
    category: Optional[str] = None
    bankroll: float = Field(default=10_000.0, ge=0.0)
    subreddits: Optional[List[str]] = None


class PaperTradeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str
    trade_log_path: str
    event_payload: Dict[str, Any]


class SettleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_id: str = Field(..., min_length=1)
    outcome: str = Field(..., description="'win' or 'loss' (also accepts true/false/yes/no/1/0).")
    market_outcome: Optional[str] = Field(
        default=None,
        description="Did the market resolve YES? Defaults to outcome (always-long-YES convention).",
    )
    exit_price: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    realized_pnl_usd: Optional[float] = None
    settled_at: Optional[str] = None
    news: Optional[str] = Field(default=None, description="Post-settlement news context for OutcomeReviewAgent.")


class SettleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_log_path: str
    payload: Dict[str, Any]


class TradesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    trades: List[Dict[str, Any]]


class PostmortemsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    reviews: List[Dict[str, Any]]
    aggregates: Dict[str, Any] = Field(default_factory=dict)
