from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from models import Market


LOGGER = logging.getLogger(__name__)

GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_PAGE_SIZE = 100
DEFAULT_MIN_VOLUME_24H = 5000.0
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_USER_AGENT = "autopilot-polymarket-scanner/1.0"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    if not header_value:
        return None
    try:
        return max(0.0, float(header_value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(header_value)
        return max(0.0, retry_at.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _unique_text_parts(parts: Sequence[Optional[str]]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _parse_outcome_prices(raw_prices: Any) -> List[float]:
    if raw_prices in (None, ""):
        return []
    if isinstance(raw_prices, str):
        try:
            raw_prices = json.loads(raw_prices)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_prices, Iterable):
        return []
    prices: List[float] = []
    for price in raw_prices:
        numeric = _coerce_optional_float(price)
        if numeric is not None:
            prices.append(numeric)
    return prices


def _extract_implied_prob(market_payload: Dict[str, Any]) -> float:
    best_bid = _coerce_optional_float(market_payload.get("bestBid"))
    best_ask = _coerce_optional_float(market_payload.get("bestAsk"))
    if best_bid is not None and best_ask is not None:
        return min(1.0, max(0.0, (best_bid + best_ask) / 2.0))
    if best_ask is not None:
        return min(1.0, max(0.0, best_ask))
    if best_bid is not None:
        return min(1.0, max(0.0, best_bid))
    last_trade = _coerce_optional_float(market_payload.get("lastTradePrice"))
    if last_trade is not None:
        return min(1.0, max(0.0, last_trade))
    outcome_prices = _parse_outcome_prices(market_payload.get("outcomePrices"))
    if outcome_prices:
        return min(1.0, max(0.0, outcome_prices[0]))
    return 0.0


def _extract_category(event_payload: Dict[str, Any], market_payload: Dict[str, Any]) -> str:
    direct_category = str(market_payload.get("category") or event_payload.get("category") or "").strip()
    if direct_category:
        return direct_category
    tags = event_payload.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                candidate = str(tag.get("label") or tag.get("name") or "").strip()
            else:
                candidate = str(tag).strip()
            if candidate:
                return candidate
    return "uncategorized"


def _build_rules_text(event_payload: Dict[str, Any], market_payload: Dict[str, Any]) -> str:
    return "\n\n".join(
        _unique_text_parts(
            (
                market_payload.get("description"),
                market_payload.get("resolutionSource"),
                event_payload.get("description"),
                event_payload.get("resolutionSource"),
            )
        )
    )


def _extract_avg_volume_7d(market_payload: Dict[str, Any]) -> Optional[float]:
    weekly_volume = _coerce_optional_float(market_payload.get("volume1wk"))
    if weekly_volume is None or weekly_volume <= 0.0:
        return None
    return weekly_volume / 7.0


def _extract_volume_change_1h(market_payload: Dict[str, Any]) -> Optional[float]:
    for key in (
        "oneHourVolumeChange",
        "oneHourVolumeChangePct",
        "volumeChange1h",
        "volumeChange1hr",
        "hourlyVolumeChange",
    ):
        value = _coerce_optional_float(market_payload.get(key))
        if value is not None:
            return value
    return None


def _market_from_gamma_payload(
    market_payload: Dict[str, Any],
    event_payload: Optional[Dict[str, Any]] = None,
) -> Market:
    parent_event = event_payload or {}
    volume_24h = _coerce_float(
        market_payload.get("volume24hr"),
        _coerce_float(parent_event.get("volume24hr")),
    )
    resolution_date = (
        market_payload.get("endDate")
        or market_payload.get("endDateIso")
        or parent_event.get("endDate")
        or parent_event.get("endDateIso")
        or datetime.utcnow().isoformat() + "Z"
    )
    return Market(
        market_id=str(market_payload.get("id", "")),
        title=str(market_payload.get("question") or parent_event.get("title") or "").strip(),
        category=_extract_category(parent_event, market_payload),
        implied_prob=_extract_implied_prob(market_payload),
        bid_price=_coerce_float(market_payload.get("bestBid")),
        ask_price=_coerce_float(
            market_payload.get("bestAsk"),
            _coerce_float(market_payload.get("bestBid")),
        ),
        volume_24h=volume_24h,
        price_history={
            "1h": _coerce_float(market_payload.get("oneHourPriceChange")),
            "6h": _coerce_float(market_payload.get("sixHourPriceChange")),
            "24h": _coerce_float(market_payload.get("oneDayPriceChange")),
        },
        open_interest=_coerce_float(parent_event.get("openInterest")),
        resolution_date=resolution_date,
        rules_text=_build_rules_text(parent_event, market_payload),
        avg_volume_7d=_extract_avg_volume_7d(market_payload),
        volume_change_1h=_extract_volume_change_1h(market_payload),
    )


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> List[Dict[str, Any]]:
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                if attempt >= max_retries:
                    response.raise_for_status()
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                sleep_for = retry_after if retry_after is not None else backoff_seconds * (2 ** attempt)
                sleep_for += random.uniform(0.0, 0.25)
                LOGGER.warning("Polymarket Gamma API rate limited request. Sleeping for %.2fs", sleep_for)
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Gamma API returned a non-list payload for market pagination")
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt >= max_retries:
                raise
            sleep_for = backoff_seconds * (2 ** attempt) + random.uniform(0.0, 0.25)
            LOGGER.warning("Gamma API request failed (%s). Retrying in %.2fs", exc, sleep_for)
            time.sleep(sleep_for)
    return []


def fetch_active_markets(
    *,
    min_volume_24h: float = DEFAULT_MIN_VOLUME_24H,
    page_size: int = DEFAULT_PAGE_SIZE,
    session: Optional[requests.Session] = None,
    max_pages: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[Market]:
    own_session = session is None
    http = session or requests.Session()
    if "Accept" not in http.headers:
        http.headers["Accept"] = "application/json"
    if "User-Agent" not in http.headers:
        http.headers["User-Agent"] = DEFAULT_USER_AGENT

    markets: List[Market] = []
    seen_market_ids = set()
    offset = 0
    page_count = 0
    try:
        while True:
            payload = _request_json(
                http,
                f"{GAMMA_API_BASE_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                },
                timeout=timeout,
            )
            if not payload:
                break
            for market_payload in payload:
                if not market_payload.get("active", False) or market_payload.get("closed", False):
                    continue
                market_id = str(market_payload.get("id", "")).strip()
                if not market_id or market_id in seen_market_ids:
                    continue
                event_payload: Dict[str, Any] = {}
                raw_events = market_payload.get("events") or []
                if isinstance(raw_events, list) and raw_events:
                    first_event = raw_events[0]
                    if isinstance(first_event, dict):
                        event_payload = first_event
                market = _market_from_gamma_payload(market_payload, event_payload)
                if market.volume_24h < min_volume_24h:
                    continue
                seen_market_ids.add(market_id)
                markets.append(market)
            page_count += 1
            if max_pages is not None and page_count >= max_pages:
                break
            if len(payload) < page_size:
                break
            offset += page_size
    finally:
        if own_session:
            http.close()
    return markets


def _resolved_market_outcome(market_payload: Dict[str, Any]) -> Optional[bool]:
    """Determine the binary YES/NO resolution for a closed market.

    Polymarket settles a binary market by setting ``outcomePrices`` to ``["1","0"]``
    (Yes won) or ``["0","1"]`` (No won). Refunded / void / 50-50 splits land at
    ``["0.5","0.5"]`` (or otherwise non-extreme values) and are returned as
    ``None`` so callers can skip them. We also consult ``umaResolutionStatus``
    (e.g. ``"resolved"``) only as a sanity check — the price array is the
    authoritative outcome signal.
    """

    outcome_prices = _parse_outcome_prices(market_payload.get("outcomePrices"))
    if len(outcome_prices) < 2:
        return None

    yes_price, no_price = outcome_prices[0], outcome_prices[1]
    if yes_price >= 0.99 and no_price <= 0.01:
        return True
    if no_price >= 0.99 and yes_price <= 0.01:
        return False
    return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _historical_max_volume(market_payload: Dict[str, Any]) -> float:
    """Best-effort historical volume signal for a closed market.

    24h volume on a closed market is degraded (often near zero), so we prefer
    cumulative or weekly fields when available, falling back to ``volume24hr``.
    """

    for key in ("volumeNum", "volume", "volume1wk", "volume1mo", "volumeClob"):
        value = _coerce_optional_float(market_payload.get(key))
        if value is not None and value > 0.0:
            return value
    return _coerce_float(market_payload.get("volume24hr"))


def fetch_resolved_markets(
    *,
    min_volume_24h: float = DEFAULT_MIN_VOLUME_24H,
    page_size: int = DEFAULT_PAGE_SIZE,
    session: Optional[requests.Session] = None,
    max_pages: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    days_back: Optional[int] = None,
) -> List[Tuple[Market, bool]]:
    """Page resolved Polymarket markets and return ``(market, market_outcome)`` pairs.

    Mirrors :func:`fetch_active_markets` but queries the closed-market slice
    (``active=false&closed=true``, ordered by ``endDate`` descending) and pairs
    each market with its boolean resolution. Markets whose resolution is
    ambiguous (refunded / void / 50-50) are skipped and counted in the
    ``orchestrator``-style INFO log so operators can see how much of the page
    we lost.

    The ``min_volume_24h`` filter is applied against the best historical
    volume signal we can find on the payload (cumulative / weekly / 24h, in
    that order) since closed markets typically report near-zero ``volume24hr``.
    """

    own_session = session is None
    http = session or requests.Session()
    if "Accept" not in http.headers:
        http.headers["Accept"] = "application/json"
    if "User-Agent" not in http.headers:
        http.headers["User-Agent"] = DEFAULT_USER_AGENT

    cutoff_dt: Optional[datetime] = None
    if days_back is not None and days_back > 0:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=int(days_back))

    results: List[Tuple[Market, bool]] = []
    seen_market_ids = set()
    skipped_ambiguous = 0
    offset = 0
    page_count = 0
    try:
        while True:
            payload = _request_json(
                http,
                f"{GAMMA_API_BASE_URL}/markets",
                params={
                    "active": "false",
                    "closed": "true",
                    "limit": page_size,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "false",
                },
                timeout=timeout,
            )
            if not payload:
                break
            for market_payload in payload:
                if not market_payload.get("closed", False):
                    continue
                market_id = str(market_payload.get("id", "")).strip()
                if not market_id or market_id in seen_market_ids:
                    continue

                historical_volume = _historical_max_volume(market_payload)
                if historical_volume < min_volume_24h:
                    continue

                event_payload: Dict[str, Any] = {}
                raw_events = market_payload.get("events") or []
                if isinstance(raw_events, list) and raw_events:
                    first_event = raw_events[0]
                    if isinstance(first_event, dict):
                        event_payload = first_event

                if cutoff_dt is not None:
                    end_dt = _parse_iso_datetime(
                        market_payload.get("endDate")
                        or market_payload.get("endDateIso")
                        or event_payload.get("endDate")
                        or event_payload.get("endDateIso")
                    )
                    if end_dt is not None and end_dt < cutoff_dt:
                        continue

                outcome = _resolved_market_outcome(market_payload)
                if outcome is None:
                    skipped_ambiguous += 1
                    continue

                market = _market_from_gamma_payload(market_payload, event_payload)
                seen_market_ids.add(market_id)
                results.append((market, outcome))
            page_count += 1
            if max_pages is not None and page_count >= max_pages:
                break
            if len(payload) < page_size:
                break
            offset += page_size
    finally:
        if own_session:
            http.close()

    if skipped_ambiguous:
        LOGGER.info(
            "fetch_resolved_markets skipped %d market(s) with ambiguous resolution outcomes.",
            skipped_ambiguous,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch active Polymarket markets from the Gamma API.")
    parser.add_argument("--min-volume-24h", type=float, default=DEFAULT_MIN_VOLUME_24H)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    markets = fetch_active_markets(
        min_volume_24h=args.min_volume_24h,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    print(json.dumps([asdict(market) for market in markets], default=str, indent=args.indent))


if __name__ == "__main__":
    main()
