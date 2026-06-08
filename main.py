from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyzer import attach_category_spread_averages, evaluate_market, passes_market_filters
from fetcher import DEFAULT_MIN_VOLUME_24H, DEFAULT_PAGE_SIZE, fetch_active_markets
from llm_judge import LLMJudgeResult, judge_market
from models import Market
from ranker import calculate_priority


LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_N = 20
EXPORT_FIELDS = (
    "market_id",
    "title",
    "category",
    "implied_prob",
    "spread",
    "volume_24h",
    "move_24h",
    "days_to_resolution",
    "clarity_score",
    "anomaly_flags",
    "research_priority",
)


FetchMarketsFn = Callable[..., Sequence[Market]]
AttachSpreadsFn = Callable[..., Dict[str, float]]
PassesFiltersFn = Callable[..., bool]
EvaluateMarketFn = Callable[..., List[str]]
JudgeMarketFn = Callable[..., LLMJudgeResult]
CalculatePriorityFn = Callable[..., Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _combine_flags(*flag_groups: Sequence[str]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for group in flag_groups:
        for flag in group:
            normalized = str(flag or "").strip().upper()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            combined.append(normalized)
    return combined


def _fallback_judge_result(exc: Exception) -> LLMJudgeResult:
    return LLMJudgeResult(
        clarity_score=50,
        narrative_momentum=0,
        anomaly_flags=[],
        reasoning=f"LLM judge unavailable: {exc}",
    )


def _serialize_market_result(
    market: Market,
    *,
    clarity_score: int,
    anomaly_flags: Sequence[str],
    research_priority: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    market.refresh_derived_fields(now=now)
    return {
        "market_id": market.market_id,
        "title": market.title,
        "category": market.category,
        "implied_prob": float(market.implied_prob),
        "spread": float(market.spread),
        "volume_24h": float(market.volume_24h),
        "move_24h": float(market.price_history.get("24h", 0.0)),
        "days_to_resolution": float(market.days_to_resolution),
        "clarity_score": int(clarity_score),
        "anomaly_flags": list(anomaly_flags),
        "research_priority": int(research_priority),
    }


def build_scan_results(
    *,
    now: Optional[datetime] = None,
    min_volume_24h: float = DEFAULT_MIN_VOLUME_24H,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: Optional[int] = None,
    category: Optional[str] = None,
    fetch_markets_fn: FetchMarketsFn = fetch_active_markets,
    attach_spreads_fn: AttachSpreadsFn = attach_category_spread_averages,
    passes_filters_fn: PassesFiltersFn = passes_market_filters,
    evaluate_market_fn: EvaluateMarketFn = evaluate_market,
    judge_market_fn: JudgeMarketFn = judge_market,
    calculate_priority_fn: CalculatePriorityFn = calculate_priority,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    logger = logger or LOGGER
    current_time = now or _utc_now()
    normalized_category = str(category or "").strip().lower()
    markets = list(
        fetch_markets_fn(
            min_volume_24h=min_volume_24h,
            page_size=page_size,
            max_pages=max_pages,
        )
    )
    attach_spreads_fn(markets, now=current_time)

    results: List[Dict[str, Any]] = []
    for market in markets:
        if normalized_category and str(market.category or "").strip().lower() != normalized_category:
            continue
        market.refresh_derived_fields(now=current_time)
        if not passes_filters_fn(market, now=current_time):
            continue

        anomaly_flags = list(evaluate_market_fn(market, now=current_time))
        try:
            judge_result = judge_market_fn(market.title, market.rules_text)
        except Exception as exc:  # pragma: no cover - exercised via warning path tests if needed
            logger.warning("LLM judge failed for market %s: %s", market.market_id, exc)
            judge_result = _fallback_judge_result(exc)

        combined_flags = _combine_flags(anomaly_flags, judge_result.anomaly_flags)
        priority = calculate_priority_fn(
            market,
            anomaly_flags=combined_flags,
            clarity_score=judge_result.clarity_score,
            now=current_time,
        )
        results.append(
            _serialize_market_result(
                market,
                clarity_score=judge_result.clarity_score,
                anomaly_flags=combined_flags,
                research_priority=priority.research_priority,
                now=current_time,
            )
        )

    results.sort(
        key=lambda row: (
            int(row["research_priority"]),
            int(row["clarity_score"]),
            float(row["volume_24h"]),
        ),
        reverse=True,
    )
    return results


def _truncate(text: str, width: int) -> str:
    value = str(text or "")
    if width < 4 or len(value) <= width:
        return value[:width]
    return value[: width - 3] + "..."


def _format_pct(value: Any, *, signed: bool = False) -> str:
    numeric = float(value)
    prefix = "+" if signed and numeric > 0.0 else ""
    return f"{prefix}{numeric * 100.0:.1f}%"


def _format_volume(value: Any) -> str:
    numeric = float(value)
    if numeric >= 1_000_000:
        return f"${numeric / 1_000_000:.1f}M"
    if numeric >= 1_000:
        return f"${numeric / 1_000:.1f}K"
    return f"${numeric:.0f}"


def _table_row_view(index: int, row: Dict[str, Any]) -> Dict[str, str]:
    flags = ", ".join(row.get("anomaly_flags", [])) or "-"
    return {
        "#": str(index),
        "Priority": str(int(row["research_priority"])),
        "Clarity": str(int(row["clarity_score"])),
        "Spread": _format_pct(row["spread"]),
        "Vol24h": _format_volume(row["volume_24h"]),
        "Move24h": _format_pct(row["move_24h"], signed=True),
        "Days": f"{float(row['days_to_resolution']):.1f}",
        "Category": _truncate(str(row["category"]), 14),
        "Flags": _truncate(flags, 28),
        "Title": _truncate(str(row["title"]), 52),
    }


def render_cli_table(rows: Sequence[Dict[str, Any]], *, limit: int = DEFAULT_TOP_N) -> str:
    top_rows = list(rows[: max(0, int(limit))])
    if not top_rows:
        return "No eligible markets found."

    display_rows = [_table_row_view(index, row) for index, row in enumerate(top_rows, start=1)]
    columns = ["#", "Priority", "Clarity", "Spread", "Vol24h", "Move24h", "Days", "Category", "Flags", "Title"]
    widths = {
        column: max(len(column), *(len(display_row[column]) for display_row in display_rows))
        for column in columns
    }
    numeric_columns = {"#", "Priority", "Clarity", "Spread", "Vol24h", "Move24h", "Days"}

    def format_row(values: Dict[str, str]) -> str:
        parts = []
        for column in columns:
            value = values[column]
            if column in numeric_columns:
                parts.append(value.rjust(widths[column]))
            else:
                parts.append(value.ljust(widths[column]))
        return " | ".join(parts)

    header = format_row({column: column for column in columns})
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(format_row(display_row) for display_row in display_rows)
    return f"{header}\n{separator}\n{body}"


def export_scan_results(
    rows: Sequence[Dict[str, Any]],
    *,
    output_dir: Path | str = REPO_ROOT / "output",
    now: Optional[datetime] = None,
) -> Path:
    current_time = now or _utc_now()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / f"scan_{current_time.strftime('%Y%m%dT%H%M%SZ')}.json"
    export_rows = [
        {field: row[field] for field in EXPORT_FIELDS}
        for row in rows
    ]
    output_path.write_text(json.dumps(export_rows, indent=2), encoding="utf-8")
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Polymarket scanner end-to-end.")
    parser.add_argument("--min-volume-24h", type=float, default=DEFAULT_MIN_VOLUME_24H)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (e.g., Politics, Crypto)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    current_time = _utc_now()
    results = build_scan_results(
        now=current_time,
        min_volume_24h=args.min_volume_24h,
        page_size=args.page_size,
        max_pages=args.max_pages,
        category=args.category,
    )
    print(render_cli_table(results, limit=args.top))
    output_path = export_scan_results(results, output_dir=args.output_dir, now=current_time)
    print(f"\nExported {len(results)} results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
