from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import fmean
from typing import Dict, List, Optional, Sequence

from models import Market


MAX_SPREAD_FILTER = 0.15
MIN_VOLUME_FILTER = 1000.0
MIN_DAYS_TO_RESOLUTION = 1.0
VOL_SPIKE_MULTIPLE = 5.0
DECOUPLED_PRICE_MOVE_1H = 0.05
DECOUPLED_VOLUME_CHANGE_1H = 0.10
WIDE_SPREAD_MULTIPLE = 2.0
INFO_EDGE_MIN_DAYS = 3.0
INFO_EDGE_MAX_DAYS = 7.0


def _normalize_category(category: str) -> str:
    return str(category or "uncategorized").strip().lower() or "uncategorized"


def get_filter_reasons(market: Market, *, now: Optional[datetime] = None) -> List[str]:
    market.refresh_derived_fields(now=now)
    reasons: List[str] = []
    if market.spread > MAX_SPREAD_FILTER:
        reasons.append("SPREAD_TOO_WIDE")
    if market.volume_24h < MIN_VOLUME_FILTER:
        reasons.append("LOW_VOLUME")
    if market.days_to_resolution < MIN_DAYS_TO_RESOLUTION:
        reasons.append("NEAR_RESOLUTION")
    return reasons


def passes_market_filters(market: Market, *, now: Optional[datetime] = None) -> bool:
    return not get_filter_reasons(market, now=now)


def compute_category_spread_averages(
    markets: Sequence[Market],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, float]:
    grouped_spreads: Dict[str, List[float]] = defaultdict(list)
    for market in markets:
        if not passes_market_filters(market, now=now):
            continue
        grouped_spreads[_normalize_category(market.category)].append(market.spread)
    return {
        category: fmean(spreads)
        for category, spreads in grouped_spreads.items()
        if spreads
    }


def attach_category_spread_averages(
    markets: Sequence[Market],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, float]:
    averages = compute_category_spread_averages(markets, now=now)
    for market in markets:
        market.category_avg_spread = averages.get(_normalize_category(market.category))
    return averages


def evaluate_market(market: Market, *, now: Optional[datetime] = None) -> List[str]:
    if get_filter_reasons(market, now=now):
        return []

    flags: List[str] = []
    avg_volume_7d = market.avg_volume_7d
    if avg_volume_7d is not None and avg_volume_7d > 0.0 and market.volume_24h > VOL_SPIKE_MULTIPLE * avg_volume_7d:
        flags.append("VOL_SPIKE")

    volume_change_1h = market.volume_change_1h
    price_move_1h = abs(market.price_history.get("1h", 0.0))
    if (
        volume_change_1h is not None
        and price_move_1h > DECOUPLED_PRICE_MOVE_1H
        and abs(volume_change_1h) < DECOUPLED_VOLUME_CHANGE_1H
    ):
        flags.append("DECOUPLED")

    category_avg_spread = market.category_avg_spread
    if category_avg_spread is not None and category_avg_spread > 0.0 and market.spread > WIDE_SPREAD_MULTIPLE * category_avg_spread:
        flags.append("WIDE_SPREAD")

    if INFO_EDGE_MIN_DAYS <= market.days_to_resolution <= INFO_EDGE_MAX_DAYS:
        flags.append("INFO_EDGE")

    return flags
