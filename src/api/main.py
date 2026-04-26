"""FastAPI app for the prediction-market scanner + agent pipeline.

This is the prediction-market HTTP control plane — distinct from the legacy
crypto-trading API at ``src/main.py``. Both apps coexist on different ports.

Run locally::

    ./.venv/bin/uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001

Endpoints (see ``src/api/models.py`` for request/response shapes):

    GET  /health        — liveness probe
    POST /scan          — run the Polymarket scanner
    POST /research      — Reddit + News research for one market
    POST /predict       — calibration agent (XGBoost + Gemini) for one market
    POST /risk          — risk engine + assessment for one market
    POST /paper-trade   — end-to-end pipeline → trade_execution_<id>.json
    POST /settle        — mark a trade log settled (in-place mutation)
    GET  /trades        — list trade logs filtered by status/source
    GET  /postmortems   — list reviews from performance_audit.json

Environment variables read by this app:

    AUTOPILOT_TRADE_STORE   directory for trade_execution_<id>.json (default: repo root)
    APP_VERSION             API version reported by /health (default: 0.1.0)
    RESEARCH_MOCK           when truthy, research agents return deterministic mock data
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# sys.path shim mirrors orchestrator.py / main.py so flat ``from main import ...``
# imports resolve to the *root* main.py (the Polymarket scanner CLI). The src/
# entry must follow the repo root so we don't shadow the root main.py with
# src/main.py (the legacy crypto FastAPI).
_API_DIR = Path(__file__).resolve().parent
_SRC_DIR = _API_DIR.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(1, str(_SRC_DIR))

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from calibration_agent.analyzer import CalibrationAgent
from calibration_agent.ml_service import (
    extract_market_features,
    extract_research_features,
    get_xgboost_probability,
)
from calibration_agent.models import CalibrationReport
from fetcher import DEFAULT_MIN_VOLUME_24H, DEFAULT_PAGE_SIZE, fetch_active_markets
from main import build_scan_results  # type: ignore[import-not-found]
from mark_trade_settled import mark_settled
from models import Market
from news_research_agent.analyzer import NewsAgent
from news_research_agent.fetcher import GoogleNewsRSSFetcher
from orchestrator import (
    build_news_search_query,
    build_reddit_search_query,
    run_final_risk_gate,
)
from reddit_research_agent.analyzer import RedditAgent
from reddit_research_agent.fetcher import RedditDeepDiver
from risk_management_agent.risk_engine import RiskCalculator

from src.api.dependencies import (
    REPO_ROOT,
    TRADE_STORE_ENV_VAR,
    get_audit_file_path,
    get_performance_tracker,
    get_risk_calculator,
    get_sqlite_store,
    get_trade_store_dir,
)
from src.api.models import (
    HealthResponse,
    PaperTradeRequest,
    PaperTradeResponse,
    PostmortemsResponse,
    PredictRequest,
    ResearchRequest,
    ResearchResponse,
    RiskRequest,
    RiskResponse,
    ScanRequest,
    ScanResponse,
    SettleRequest,
    SettleResponse,
    TradesResponse,
)


LOGGER = logging.getLogger(__name__)
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
SERVICE_NAME = "autopilot-prediction-market"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _parse_bool_token(value: str) -> bool:
    """Accept 'win|loss|true|false|yes|no|1|0' (case-insensitive)."""

    normalized = (value or "").strip().lower()
    if normalized in {"win", "won", "true", "yes", "y", "1"}:
        return True
    if normalized in {"loss", "lost", "false", "no", "n", "0"}:
        return False
    raise HTTPException(
        status_code=422,
        detail=f"Invalid outcome token {value!r}; expected one of win|loss|yes|no|true|false|1|0.",
    )


def _to_serializable(value: Any) -> Any:
    """Mirror orchestrator._to_serializable so payloads round-trip through JSON."""

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_serializable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it's awaitable, otherwise return it as-is."""

    if inspect.isawaitable(value):
        return await value
    return value


def _scan_id() -> str:
    return datetime.now(timezone.utc).strftime("scan-%Y%m%dT%H%M%S%fZ")


def _resolve_market(
    market_id: str,
    *,
    min_volume_24h: float,
    page_size: int,
    max_pages: Optional[int],
    fetch_markets_fn: Any = None,
) -> Market:
    """Find a ``Market`` instance by id via the Polymarket fetcher.

    Raises 404 when the market is not present in the active set.

    ``fetch_markets_fn`` is resolved at *call time* (defaulting to the module
    attribute ``fetch_active_markets``) so unittests can monkey-patch
    ``api_main.fetch_active_markets`` without having to re-import this helper.
    Capturing the default at definition time would freeze the original symbol
    and silently leak real Polymarket HTTP requests during the test suite.
    """

    target = (market_id or "").strip()
    if not target:
        raise HTTPException(status_code=422, detail="market_id must be a non-empty string.")
    if fetch_markets_fn is None:
        # Lazy lookup so ``patch.object(api_main, "fetch_active_markets", ...)`` works.
        fetch_markets_fn = globals()["fetch_active_markets"]
    for market in fetch_markets_fn(
        min_volume_24h=min_volume_24h,
        page_size=page_size,
        max_pages=max_pages,
    ):
        if str(market.market_id) == target:
            return market
    raise HTTPException(status_code=404, detail=f"market_id={target!r} not found in active scan.")


def _trade_log_path(store_dir: Path, market_id: str) -> Path:
    return store_dir / f"trade_execution_{market_id}.json"


def _matches_status(payload: Dict[str, Any], status_filter: Optional[str]) -> bool:
    if status_filter is None:
        return True
    actual = str(payload.get("status") or "").strip().lower()
    return actual == status_filter.strip().lower()


def _matches_source(payload: Dict[str, Any], source_filter: Optional[str]) -> bool:
    if source_filter is None:
        return True
    actual = str(payload.get("source") or "").strip().lower()
    return actual == source_filter.strip().lower()


def create_app() -> FastAPI:
    """FastAPI app factory.

    Title: ``autopilot-prediction-market``
    Version: ``APP_VERSION`` env (default ``0.1.0``)

    CORS is permissive (``allow_origins=["*"]``) for local development. Tighten
    these settings before exposing the app outside localhost.
    """

    app = FastAPI(title=SERVICE_NAME, version=APP_VERSION)

    # NOTE: CORS is permissive on purpose for local-dev convenience. Lock down
    # ``allow_origins`` / ``allow_methods`` before any production deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------- /health -----------------------
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=SERVICE_NAME, version=APP_VERSION)

    # ----------------------- /scan -----------------------
    @app.post("/scan", response_model=ScanResponse)
    async def scan(request: ScanRequest) -> ScanResponse:
        try:
            results = await asyncio.to_thread(
                build_scan_results,
                min_volume_24h=request.min_volume_24h,
                page_size=request.page_size,
                max_pages=request.max_pages,
                category=request.category,
            )
        except Exception as exc:  # pragma: no cover - exercised via integration
            LOGGER.exception("Scan failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"scan failed: {exc}") from exc

        top_n = max(0, int(request.top_n))
        truncated = list(results)[:top_n] if top_n else list(results)
        return ScanResponse(scan_id=_scan_id(), count=len(truncated), results=truncated)

    # ----------------------- helpers shared by /research /predict /risk /paper-trade
    async def _run_research(market: Market, *, subreddits: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        reddit_query = build_reddit_search_query(market)
        news_query = build_news_search_query(market)

        # Fetchers and analyzers are sync; wrap in to_thread so the event loop
        # stays free for concurrent requests.
        reddit_diver = RedditDeepDiver(reddit_query, subreddits=list(subreddits or []) or None)
        news_aggregator = GoogleNewsRSSFetcher(news_query)
        reddit_agent = RedditAgent()
        news_agent = NewsAgent()

        reddit_context, news_context = await asyncio.gather(
            asyncio.to_thread(reddit_diver.fetch_threads),
            asyncio.to_thread(news_aggregator.fetch_news_context),
        )

        reddit_call = reddit_agent.analyze_discussion(
            market_title=market.title,
            implied_prob=market.implied_prob,
            reddit_context=reddit_context,
            search_query=reddit_query,
        )
        news_call = news_agent.analyze_news(
            market_title=market.title,
            implied_prob=market.implied_prob,
            news_context=news_context,
            search_query=news_query,
        )
        reddit_report = await _maybe_await(reddit_call)
        news_report = await _maybe_await(news_call)

        return {
            "reddit_query": reddit_query,
            "news_query": news_query,
            "reddit_report": reddit_report,
            "news_report": news_report,
        }

    async def _run_calibration(market: Market, research: Dict[str, Any]) -> CalibrationReport:
        xgboost_baseline = await asyncio.to_thread(get_xgboost_probability, market)
        agent = CalibrationAgent()

        def _calibrate() -> CalibrationReport:
            method = getattr(agent, "calibrate", None) or getattr(agent, "calibrate_probability", None)
            if method is None:
                raise HTTPException(status_code=500, detail="CalibrationAgent missing calibrate method.")
            try:
                return method(
                    market=market,
                    reddit_report=research["reddit_report"],
                    news_report=research["news_report"],
                    xgboost_prob=xgboost_baseline,
                )
            except TypeError:
                return method(
                    market=market,
                    news_report=research["news_report"],
                    xgboost_prob=xgboost_baseline,
                    reddit_report=research["reddit_report"],
                )

        return await asyncio.to_thread(_calibrate)

    # ----------------------- /research -----------------------
    @app.post("/research", response_model=ResearchResponse)
    async def research(request: ResearchRequest) -> ResearchResponse:
        market = await asyncio.to_thread(
            _resolve_market,
            request.market_id,
            min_volume_24h=DEFAULT_MIN_VOLUME_24H,
            page_size=DEFAULT_PAGE_SIZE,
            max_pages=None,
        )
        try:
            payload = await _run_research(market, subreddits=request.subreddits)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - real LLM fail path
            LOGGER.exception("/research failed for %s: %s", request.market_id, exc)
            raise HTTPException(status_code=500, detail=f"research failed: {exc}") from exc

        return ResearchResponse(
            market_id=market.market_id,
            reddit_query=payload["reddit_query"],
            news_query=payload["news_query"],
            reddit_report=_to_serializable(payload["reddit_report"]),
            news_report=_to_serializable(payload["news_report"]),
        )

    # ----------------------- /predict -----------------------
    @app.post("/predict", response_model=CalibrationReport)
    async def predict(request: PredictRequest) -> CalibrationReport:
        market = await asyncio.to_thread(
            _resolve_market,
            request.market_id,
            min_volume_24h=DEFAULT_MIN_VOLUME_24H,
            page_size=DEFAULT_PAGE_SIZE,
            max_pages=None,
        )
        try:
            research_payload = await _run_research(market)
            calibration = await _run_calibration(market, research_payload)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - real LLM fail path
            LOGGER.exception("/predict failed for %s: %s", request.market_id, exc)
            raise HTTPException(status_code=500, detail=f"predict failed: {exc}") from exc
        return calibration

    # ----------------------- /risk -----------------------
    @app.post("/risk", response_model=RiskResponse)
    async def risk(
        request: RiskRequest,
        risk_calculator: RiskCalculator = Depends(get_risk_calculator),
    ) -> RiskResponse:
        market = await asyncio.to_thread(
            _resolve_market,
            request.market_id,
            min_volume_24h=DEFAULT_MIN_VOLUME_24H,
            page_size=DEFAULT_PAGE_SIZE,
            max_pages=None,
        )
        try:
            risk_metrics = await asyncio.to_thread(
                risk_calculator.calculate_base_metrics,
                market=market,
                calibrated_true_prob=request.calibration.calibrated_true_prob,
                bankroll=request.bankroll,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Build a self-contained RiskAssessment without re-running the agent.
        from orchestrator import _build_risk_assessment  # type: ignore[attr-defined]

        risk_assessment = _build_risk_assessment(
            risk_metrics=risk_metrics,
            calibration=request.calibration,
        )
        return RiskResponse(
            market_id=market.market_id,
            risk_metrics=risk_metrics,
            risk_assessment=risk_assessment,
        )

    # ----------------------- /paper-trade -----------------------
    @app.post("/paper-trade", response_model=PaperTradeResponse)
    async def paper_trade(
        request: PaperTradeRequest,
        risk_calculator: RiskCalculator = Depends(get_risk_calculator),
    ) -> PaperTradeResponse:
        store_dir = get_trade_store_dir()
        store_dir.mkdir(parents=True, exist_ok=True)

        try:
            scan_rows = await asyncio.to_thread(
                build_scan_results,
                min_volume_24h=DEFAULT_MIN_VOLUME_24H,
                page_size=DEFAULT_PAGE_SIZE,
                max_pages=None,
                category=request.category,
            )
        except Exception as exc:  # pragma: no cover - integration
            LOGGER.exception("paper-trade scan failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"scan failed: {exc}") from exc

        scanner_row: Optional[Dict[str, Any]] = None
        for row in scan_rows[: max(1, int(request.top_n))]:
            if str(row.get("market_id") or "") == request.market_id:
                scanner_row = dict(row)
                break
        if scanner_row is None:
            for row in scan_rows:
                if str(row.get("market_id") or "") == request.market_id:
                    scanner_row = dict(row)
                    break
        if scanner_row is None:
            scanner_row = {"market_id": request.market_id}

        market = await asyncio.to_thread(
            _resolve_market,
            request.market_id,
            min_volume_24h=DEFAULT_MIN_VOLUME_24H,
            page_size=DEFAULT_PAGE_SIZE,
            max_pages=None,
        )
        try:
            research_payload = await _run_research(market, subreddits=request.subreddits)
            calibration = await _run_calibration(market, research_payload)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - real LLM fail path
            LOGGER.exception("/paper-trade pipeline failed for %s: %s", request.market_id, exc)
            raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc

        execution = await asyncio.to_thread(
            run_final_risk_gate,
            calibration=calibration,
            market=market,
            scanner_row=scanner_row,
            reddit_report=research_payload["reddit_report"],
            news_report=research_payload["news_report"],
            bankroll=request.bankroll,
            risk_calculator=risk_calculator,
        )

        # ``run_final_risk_gate`` always writes to the orchestrator's REPO_ROOT.
        # When AUTOPILOT_TRADE_STORE points elsewhere, mirror the file into the
        # configured store dir so /trades and /settle find it.
        source_path: Path = execution["log_path"]
        target_path = _trade_log_path(store_dir, market.market_id)
        if source_path.resolve() != target_path.resolve():
            payload_text = source_path.read_text(encoding="utf-8")
            target_path.write_text(payload_text, encoding="utf-8")
            try:
                source_path.unlink()
            except OSError:
                LOGGER.warning("Could not remove orchestrator log %s", source_path)
            execution["log_path"] = target_path

        return PaperTradeResponse(
            market_id=market.market_id,
            trade_log_path=str(execution["log_path"]),
            event_payload=_to_serializable(execution["event_payload"]),
        )

    # ----------------------- /settle -----------------------
    @app.post("/settle", response_model=SettleResponse)
    async def settle(request: SettleRequest) -> SettleResponse:
        store_dir = get_trade_store_dir()
        log_path = _trade_log_path(store_dir, request.market_id)
        if not log_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"trade log not found: {log_path}",
            )

        final_outcome = _parse_bool_token(request.outcome)
        market_outcome = (
            _parse_bool_token(request.market_outcome) if request.market_outcome is not None else final_outcome
        )

        try:
            payload = await asyncio.to_thread(
                mark_settled,
                log_path,
                final_outcome=final_outcome,
                post_settlement_news=request.news,
                settled_at=request.settled_at,
                market_outcome=market_outcome,
                exit_price=request.exit_price,
                realized_pnl_usd=request.realized_pnl_usd,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return SettleResponse(trade_log_path=str(log_path), payload=_to_serializable(payload))

    # ----------------------- /trades -----------------------
    @app.get("/trades", response_model=TradesResponse)
    async def trades(
        status: Optional[str] = Query(default=None, description="Filter by status (e.g. 'open', 'settled')."),
        source: Optional[str] = Query(default=None, description="Filter by source (e.g. 'orchestrator', 'shadow', 'backfill')."),
        limit: int = Query(default=100, ge=1, le=10_000),
    ) -> TradesResponse:
        # Prefer the SQLite mirror when AUTOPILOT_SQLITE_PATH is set — falls
        # back to the canonical JSON walk otherwise so behavior is unchanged
        # for callers who haven't opted in.
        sqlite_store = get_sqlite_store()
        if sqlite_store is not None:
            try:
                rows = await asyncio.to_thread(
                    sqlite_store.list_trades,
                    status=status,
                    source=source,
                    limit=int(limit),
                )
                return TradesResponse(count=len(rows), trades=rows)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("SQLite /trades read failed; falling back to JSON: %s", exc)

        store_dir = get_trade_store_dir()
        if not store_dir.is_dir():
            return TradesResponse(count=0, trades=[])

        rows: List[Dict[str, Any]] = []
        files = sorted(store_dir.glob("trade_execution_*.json"))
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Skipping unreadable trade log %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                continue
            if not _matches_status(payload, status):
                continue
            if not _matches_source(payload, source):
                continue
            payload.setdefault("_trade_log_path", str(path))
            rows.append(payload)
            if len(rows) >= int(limit):
                break
        return TradesResponse(count=len(rows), trades=rows)

    # ----------------------- /postmortems -----------------------
    @app.get("/postmortems", response_model=PostmortemsResponse)
    async def postmortems(
        limit: int = Query(default=100, ge=1, le=10_000),
    ) -> PostmortemsResponse:
        # Prefer the SQLite mirror when enabled. Aggregates still come from
        # the JSON file because they're computed by PerformanceTracker; an
        # empty dict is returned when SQLite is the source and no audit JSON
        # exists yet.
        sqlite_store = get_sqlite_store()
        if sqlite_store is not None:
            try:
                rows = await asyncio.to_thread(
                    sqlite_store.list_reviews,
                    limit=int(limit),
                )
                aggregates: Dict[str, Any] = {}
                audit_path = get_audit_file_path()
                if audit_path.is_file():
                    try:
                        audit = json.loads(audit_path.read_text(encoding="utf-8"))
                        aggregates = dict(audit.get("aggregates") or {})
                    except (OSError, json.JSONDecodeError) as exc:
                        LOGGER.warning("Could not load aggregates from %s: %s", audit_path, exc)
                return PostmortemsResponse(count=len(rows), reviews=rows, aggregates=aggregates)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("SQLite /postmortems read failed; falling back to JSON: %s", exc)

        audit_path = get_audit_file_path()
        if not audit_path.is_file():
            return PostmortemsResponse(count=0, reviews=[], aggregates={})

        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"failed to read audit file: {exc}") from exc

        reviews = list(audit.get("reviews") or [])
        truncated = reviews[: int(limit)]
        aggregates = dict(audit.get("aggregates") or {})
        return PostmortemsResponse(count=len(truncated), reviews=truncated, aggregates=aggregates)

    return app


app = create_app()


__all__ = [
    "APP_VERSION",
    "SERVICE_NAME",
    "TRADE_STORE_ENV_VAR",
    "app",
    "create_app",
    "get_audit_file_path",
    "get_performance_tracker",
    "get_risk_calculator",
    "get_trade_store_dir",
    "REPO_ROOT",
]
