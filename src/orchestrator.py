from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(1, str(SRC_DIR))

from calibration_agent.analyzer import CalibrationAgent
from calibration_agent.ml_service import get_xgboost_probability
from calibration_agent.models import CalibrationReport
from fetcher import DEFAULT_MIN_VOLUME_24H, DEFAULT_PAGE_SIZE, fetch_active_markets
from main import build_scan_results
from models import Market
from reddit_research_agent.analyzer import RedditAgent
from reddit_research_agent.fetcher import RedditDeepDiver


LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_N = 5
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"

ScanResultsFn = Callable[..., Sequence[Dict[str, Any]]]
FetchMarketsFn = Callable[..., Sequence[Market]]
BaselineFn = Callable[[Market], float]
_CALIBRATION_CONTEXT: Dict[int, Dict[str, Any]] = {}
_NEWS_IMPORT_CANDIDATES = (
    ("news_research_agent.fetcher", "GoogleNewsRSSFetcher", "news_research_agent.analyzer", "NewsAgent"),
    ("news_agent.fetcher", "NewsAggregator", "news_agent.analyzer", "NewsAgent"),
)


def _extract_title_category(source: Market | Dict[str, Any]) -> tuple[str, str]:
    if isinstance(source, Market):
        return source.title.strip(), source.category.strip()
    title = str(source.get("title") or "").strip()
    category = str(source.get("category") or "").strip()
    return title, category


def build_research_query(source: Market | Dict[str, Any], *, suffix: str) -> str:
    title, category = _extract_title_category(source)
    if not title:
        raise ValueError("source must include a non-empty title")

    parts = [title]
    if category and category.lower() not in title.lower():
        parts.append(category)
    cleaned_suffix = str(suffix or "").strip()
    if cleaned_suffix:
        parts.append(cleaned_suffix)
    return " ".join(part for part in parts if part)


def build_reddit_search_query(source: Market | Dict[str, Any]) -> str:
    return build_research_query(source, suffix="reddit discussion")


def build_news_search_query(source: Market | Dict[str, Any]) -> str:
    return build_research_query(source, suffix="news")


def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return func(**kwargs)

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in parameters and parameters[key].kind != inspect.Parameter.POSITIONAL_ONLY
    }
    return func(**filtered_kwargs)


def _resolve_method(obj: Any, method_names: Sequence[str]) -> Callable[..., Any]:
    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if callable(method):
            return method
    raise RuntimeError(
        f"{type(obj).__name__} does not implement any of the required methods: {', '.join(method_names)}"
    )


async def _invoke_method(obj: Any, method_names: Sequence[str], **kwargs: Any) -> Any:
    method = _resolve_method(obj, method_names)
    if inspect.iscoroutinefunction(method):
        result = _call_with_supported_kwargs(method, **kwargs)
        return await result
    return await asyncio.to_thread(lambda: _call_with_supported_kwargs(method, **kwargs))


def _instantiate_reddit_diver(
    reddit_diver_cls: Any,
    *,
    search_query: str,
    subreddits: Optional[Sequence[str]],
) -> Any:
    attempts = (
        lambda: reddit_diver_cls(search_query, subreddits=subreddits),
        lambda: reddit_diver_cls(search_query=search_query, subreddits=subreddits),
        lambda: reddit_diver_cls(search_query),
    )
    for attempt in attempts:
        try:
            return attempt()
        except TypeError:
            continue
    raise RuntimeError(f"Unable to construct Reddit diver from {reddit_diver_cls!r}")


def _instantiate_news_aggregator(news_aggregator_cls: Any, *, search_query: str) -> Any:
    attempts = (
        lambda: news_aggregator_cls(search_query),
        lambda: news_aggregator_cls(search_query=search_query),
        lambda: news_aggregator_cls(topic_query=search_query),
        lambda: news_aggregator_cls(topic=search_query),
    )
    for attempt in attempts:
        try:
            return attempt()
        except TypeError:
            continue
    raise RuntimeError(f"Unable to construct NewsAggregator from {news_aggregator_cls!r}")


def _import_symbol(module_name: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _resolve_news_dependencies(
    *,
    news_aggregator_cls: Any | None,
    news_agent: Any | None,
) -> tuple[Any, Any]:
    resolved_aggregator_cls = news_aggregator_cls
    resolved_news_agent = news_agent

    if resolved_aggregator_cls is not None and resolved_news_agent is not None:
        return resolved_aggregator_cls, resolved_news_agent

    for fetcher_module, fetcher_symbol, analyzer_module, analyzer_symbol in _NEWS_IMPORT_CANDIDATES:
        try:
            if resolved_aggregator_cls is None:
                resolved_aggregator_cls = _import_symbol(fetcher_module, fetcher_symbol)
            if resolved_news_agent is None:
                news_agent_cls = _import_symbol(analyzer_module, analyzer_symbol)
                resolved_news_agent = news_agent_cls()
            break
        except Exception:
            continue

    if resolved_aggregator_cls is None or resolved_news_agent is None:
        raise RuntimeError(
            "NewsAggregator and NewsAgent are not implemented in this repo yet. "
            "Pass them into analyze_top_markets(...) or add a news agent package."
        )
    return resolved_aggregator_cls, resolved_news_agent


def _build_market_features(market: Market) -> dict:
    market.refresh_derived_fields()
    return {
        "market_id": market.market_id,
        "title": market.title,
        "category": market.category,
        "implied_prob": market.implied_prob,
        "market_implied_prob": market.implied_prob,
        "volume_24h": market.volume_24h,
        "spread": market.spread,
        "days_to_resolution": market.days_to_resolution,
        "open_interest": market.open_interest,
        "bid_price": market.bid_price,
        "ask_price": market.ask_price,
        "price_history": dict(market.price_history),
        "rules_text": market.rules_text,
    }


def _store_calibration_context(
    calibration: CalibrationReport,
    *,
    market: Market,
    scanner_priority: int,
) -> None:
    _CALIBRATION_CONTEXT[id(calibration)] = {
        "market_id": market.market_id,
        "market_title": market.title,
        "market_implied_prob": market.implied_prob,
        "scanner_priority": int(scanner_priority),
    }


def _get_calibration_context(calibration: CalibrationReport) -> Dict[str, Any]:
    return dict(_CALIBRATION_CONTEXT.get(id(calibration), {}))


def _run_calibration_agent(
    calibration_agent: Any,
    *,
    market: Market,
    reddit_report: Any,
    news_report: Any,
    xgboost_baseline: float,
) -> CalibrationReport:
    method = _resolve_method(calibration_agent, ("calibrate", "calibrate_probability"))
    return _call_with_supported_kwargs(
        method,
        market=market,
        reddit_report=reddit_report,
        news_report=news_report,
        research_summaries={
            "reddit": reddit_report,
            "news": news_report,
        },
        xgboost_prob=xgboost_baseline,
        xgboost_baseline=xgboost_baseline,
        market_features=_build_market_features(market),
        market_implied_prob=market.implied_prob,
    )


async def _analyze_market(
    *,
    market: Market,
    scan_row: Dict[str, Any],
    reddit_diver_cls: Any,
    reddit_agent: Any,
    news_aggregator_cls: Any,
    news_agent: Any,
    calibration_agent: Any,
    xgboost_probability_fn: BaselineFn,
    subreddits: Optional[Sequence[str]],
) -> CalibrationReport:
    reddit_query = build_reddit_search_query(market)
    news_query = build_news_search_query(market)
    reddit_diver = _instantiate_reddit_diver(
        reddit_diver_cls,
        search_query=reddit_query,
        subreddits=subreddits,
    )
    news_aggregator = _instantiate_news_aggregator(
        news_aggregator_cls,
        search_query=news_query,
    )

    reddit_context, news_context = await asyncio.gather(
        _invoke_method(reddit_diver, ("fetch_threads", "fetch_discussion_context")),
        _invoke_method(news_aggregator, ("fetch_news_context", "fetch_context", "fetch_news")),
    )
    reddit_report, news_report = await asyncio.gather(
        _invoke_method(
            reddit_agent,
            ("analyze_discussion",),
            market_title=market.title,
            implied_prob=market.implied_prob,
            reddit_context=reddit_context,
            search_query=reddit_query,
        ),
        _invoke_method(
            news_agent,
            ("analyze_news", "analyze_coverage", "analyze_context", "analyze_discussion"),
            market_title=market.title,
            implied_prob=market.implied_prob,
            current_market_odds=market.implied_prob,
            news_context=news_context,
            context=news_context,
            search_query=news_query,
        ),
    )

    xgboost_baseline = await asyncio.to_thread(xgboost_probability_fn, market)
    calibration = await asyncio.to_thread(
        lambda: _run_calibration_agent(
            calibration_agent,
            market=market,
            reddit_report=reddit_report,
            news_report=news_report,
            xgboost_baseline=xgboost_baseline,
        )
    )
    _store_calibration_context(
        calibration,
        market=market,
        scanner_priority=int(scan_row.get("research_priority") or 0),
    )
    return calibration


async def analyze_top_markets(
    top_n: int = DEFAULT_TOP_N,
    *,
    min_volume_24h: float = DEFAULT_MIN_VOLUME_24H,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: Optional[int] = None,
    category: Optional[str] = None,
    subreddits: Optional[Sequence[str]] = None,
    build_scan_results_fn: ScanResultsFn = build_scan_results,
    fetch_markets_fn: FetchMarketsFn = fetch_active_markets,
    reddit_diver_cls: Any = RedditDeepDiver,
    reddit_agent: Any | None = None,
    news_aggregator_cls: Any | None = None,
    news_agent: Any | None = None,
    calibration_agent: Any | None = None,
    xgboost_probability_fn: BaselineFn = get_xgboost_probability,
    logger: Optional[logging.Logger] = None,
) -> List[CalibrationReport]:
    logger = logger or LOGGER
    top_limit = max(0, int(top_n))
    if top_limit == 0:
        return []

    resolved_news_aggregator_cls, resolved_news_agent = _resolve_news_dependencies(
        news_aggregator_cls=news_aggregator_cls,
        news_agent=news_agent,
    )
    resolved_reddit_agent = reddit_agent if reddit_agent is not None else RedditAgent()
    resolved_calibration_agent = calibration_agent if calibration_agent is not None else CalibrationAgent()

    captured_markets: Dict[str, Market] = {}

    def _capturing_fetch_markets(**kwargs: Any) -> Sequence[Market]:
        markets = list(fetch_markets_fn(**kwargs))
        for market in markets:
            captured_markets[market.market_id] = market
        return markets

    scan_rows = list(
        _call_with_supported_kwargs(
            build_scan_results_fn,
            min_volume_24h=min_volume_24h,
            page_size=page_size,
            max_pages=max_pages,
            category=category,
            fetch_markets_fn=_capturing_fetch_markets,
        )
    )[:top_limit]
    if not scan_rows:
        return []

    if not captured_markets:
        for market in fetch_markets_fn(
            min_volume_24h=min_volume_24h,
            page_size=page_size,
            max_pages=max_pages,
        ):
            captured_markets[market.market_id] = market

    selected_markets: List[tuple[Dict[str, Any], Market]] = []
    for scan_row in scan_rows:
        market_id = str(scan_row.get("market_id") or "").strip()
        market = captured_markets.get(market_id)
        if market is None:
            logger.warning(
                "Skipping ranked market without a full Market payload: %s",
                scan_row.get("title") or market_id or "<unknown>",
            )
            continue
        selected_markets.append((scan_row, market))

    tasks = [
        _analyze_market(
            market=market,
            scan_row=scan_row,
            reddit_diver_cls=reddit_diver_cls,
            reddit_agent=resolved_reddit_agent,
            news_aggregator_cls=resolved_news_aggregator_cls,
            news_agent=resolved_news_agent,
            calibration_agent=resolved_calibration_agent,
            xgboost_probability_fn=xgboost_probability_fn,
            subreddits=subreddits,
        )
        for scan_row, market in selected_markets
    ]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    completed: List[CalibrationReport] = []
    for (_, market), result in zip(selected_markets, task_results):
        if isinstance(result, Exception):
            logger.warning("Multi-agent analysis failed for %s: %s", market.title, result)
            continue
        completed.append(result)
    return completed


def _format_probability(probability: float) -> str:
    return f"{float(probability) * 100.0:.1f}%"


def _format_percentage_points(value: float) -> str:
    sign = "+" if float(value) > 0.0 else ""
    return f"{sign}{float(value):.2f} pts"


def _action_text(action: str) -> str:
    normalized = str(action or "").strip().lower()
    if normalized == "paper-trade candidate":
        return f"{ANSI_BOLD}{ANSI_GREEN}{action}{ANSI_RESET}"
    if normalized == "monitor":
        return f"{ANSI_BOLD}{ANSI_YELLOW}{action}{ANSI_RESET}"
    return f"{ANSI_RED}{action}{ANSI_RESET}"


def render_alpha_assessment_table(calibrations: Sequence[CalibrationReport]) -> str:
    title = f"{ANSI_BOLD}Alpha Assessment{ANSI_RESET}"
    if not calibrations:
        return f"{title}\n(no calibrated opportunities)"

    headers = ["Market", "XGBoost Prob", "LLM Adj", "Calibrated Prob", "Edge vs Market", "Action"]
    rows = []
    for calibration in calibrations:
        context = _get_calibration_context(calibration)
        rows.append([
            str(context.get("market_title") or "Unknown Market"),
            _format_probability(calibration.xgboost_prob),
            _format_percentage_points(calibration.llm_adjustment_pct_points),
            _format_probability(calibration.calibrated_true_prob),
            _format_percentage_points(calibration.edge_vs_market * 100.0),
            str(calibration.action),
        ])

    widths = [max(len(header), max(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    body_rows = []
    for row in rows:
        rendered_cells = []
        for idx, cell in enumerate(row):
            padded = cell.ljust(widths[idx])
            if idx == len(row) - 1:
                padded = _action_text(padded)
            rendered_cells.append(padded)
        body_rows.append("| " + " | ".join(rendered_cells) + " |")

    return "\n".join([title, separator, header_row, separator, *body_rows, separator])


def print_alpha_assessment_table(calibrations: Sequence[CalibrationReport]) -> str:
    rendered = render_alpha_assessment_table(calibrations)
    print(rendered)
    return rendered


def print_final_report(calibration: CalibrationReport) -> str:
    context = _get_calibration_context(calibration)
    market_title = str(context.get("market_title") or "Unknown Market")
    market_implied_prob = float(context.get("market_implied_prob") or 0.0)
    scanner_priority = context.get("scanner_priority")
    lines = [
        f"{ANSI_BOLD}Market:{ANSI_RESET} {market_title}",
        f"{ANSI_BOLD}Scanner Priority:{ANSI_RESET} {scanner_priority if scanner_priority is not None else 'n/a'}",
        f"{ANSI_BOLD}Market Implied Probability:{ANSI_RESET} {_format_probability(market_implied_prob)}",
        f"{ANSI_BOLD}XGBoost Baseline:{ANSI_RESET} {_format_probability(calibration.xgboost_prob)}",
        f"{ANSI_BOLD}LLM Adjustment:{ANSI_RESET} {_format_percentage_points(calibration.llm_adjustment_pct_points)}",
        f"{ANSI_BOLD}Calibrated True Probability:{ANSI_RESET} {_format_probability(calibration.calibrated_true_prob)}",
        f"{ANSI_BOLD}Edge vs Market:{ANSI_RESET} {_format_percentage_points(calibration.edge_vs_market * 100.0)}",
        f"{ANSI_BOLD}Confidence:{ANSI_RESET} {int(calibration.confidence_score)}/100",
        f"{ANSI_BOLD}Action:{ANSI_RESET} {_action_text(calibration.action)}",
        f"{ANSI_BOLD}Reasoning:{ANSI_RESET} {calibration.reasoning}",
    ]

    if calibration.key_drivers:
        lines.append(f"{ANSI_BOLD}Key Drivers:{ANSI_RESET}")
        lines.extend(f"- {driver}" for driver in calibration.key_drivers)
    if calibration.key_uncertainties:
        lines.append(f"{ANSI_BOLD}Key Uncertainties:{ANSI_RESET}")
        lines.extend(f"- {uncertainty}" for uncertainty in calibration.key_uncertainties)

    rendered = "\n".join(lines)
    print(rendered)
    return rendered


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent prediction-market orchestrator.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-volume-24h", type=float, default=DEFAULT_MIN_VOLUME_24H)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--subreddit", action="append", dest="subreddits", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    calibrations = asyncio.run(
        analyze_top_markets(
            top_n=args.top,
            min_volume_24h=args.min_volume_24h,
            page_size=args.page_size,
            max_pages=args.max_pages,
            category=args.category,
            subreddits=args.subreddits,
        )
    )
    print_alpha_assessment_table(calibrations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
