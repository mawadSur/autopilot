"""Agent A4 — :class:`ContextForensicsAgent` of the loss-postmortem swarm.

This is the only agent in the swarm that uses Gemini, and it does so under
strict bounds: at most ONE Gemini call per :meth:`investigate` invocation,
30s timeout on that call, and graceful degradation when the API key is
unset OR the call fails. The deterministic checks (news headline count,
Polymarket macro price shift) always run regardless of LLM availability —
the LLM call is purely for SUMMARIZATION of the fetched context, never
for fetching or judging it.

What this agent investigates
----------------------------
"What did the model NOT see that mattered?" Pulls a 1h-window context
around the trade timestamp and emits red flags for:

1. **News in the 1h window before the trade** — uses the existing
   :mod:`news_research_agent.fetcher` (Google News RSS) so we don't
   reinvent fetching. Headlines are filtered post-hoc by
   ``published_iso`` to the 1h window. Deterministic threshold: > 5
   headlines in the hour ⇒ "elevated news density" red flag. The
   optional Gemini call summarises the headline pack into a one-line
   take so the digest is human-scannable.
2. **Polymarket macro shift** — pulls a small slice of high-volume macro
   markets (Politics, Macro, Federal Reserve, Crypto categories) and
   compares their current implied probability against the 1h price
   change reported by Gamma (``price_history["1h"]``). Any market that
   moved > 5pp in the hour ⇒ "macro shift" red flag.
3. **Vol spike on correlated symbols** — best-effort placeholder. We
   read ``ticker_buffer`` from the signal snapshot if present; if the
   absolute price change in the 30 min before signal exceeds 2x the
   buffer's median absolute change, flag it. If no ticker buffer,
   emit an informational bullet only (not a red flag).
4. **Sentiment divergence** — optional. We don't run an X/Twitter
   firehose here; the bullet "sentiment check unavailable — would
   require X/Twitter integration" is emitted only when no other
   signal fires (so a quiet market still surfaces the gap).

Verdict
-------
- **0 red flags + clean checks** → ``"innocent"``.
- **1-2 red flags** → ``"contributing"`` with confidence 0.5.
- **3+ red flags** → ``"primary_cause"`` with confidence 0.75.

The Gemini call's success or failure does NOT push the verdict toward
primary_cause on its own — a missing LLM is an observability gap, not
evidence about the trade. If the LLM is unavailable we still emit
findings from the deterministic checks and add the evidence bullet
``"LLM summarization unavailable — context check uses raw fetches only"``.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence

from loss_postmortem.base import (
    BaseForensicsAgent,
    DEFAULT_AGENT_TIMEOUT_S,
    ForensicsFinding,
)
from state.trade_context_store import TradeContextStore

LOGGER = logging.getLogger(__name__)

# Tighter cap than the per-agent 60s default — the LLM call is the only
# blocking I/O of any consequence and 30s is the conservative bound.
GEMINI_CALL_TIMEOUT_S = 30.0

# Window for "did anything happen recently?" checks, applied to both news
# headlines and Polymarket macro markets.
NEWS_WINDOW_SECONDS = 3600
MACRO_WINDOW_SECONDS = 3600

# Macro categories worth scanning for cross-market shifts. Matched
# case-insensitively against ``Market.category``.
MACRO_CATEGORIES: tuple[str, ...] = (
    "politics",
    "macro",
    "federal reserve",
    "crypto",
)

# Deterministic red-flag thresholds.
HEADLINE_DENSITY_RED_FLAG = 5  # > 5 headlines in window => red flag
MACRO_SHIFT_RED_FLAG_PP = 0.05  # > 5pp move on any macro market => red flag
VOL_SPIKE_MULTIPLIER = 2.0  # > 2x baseline median abs delta => red flag

# Very-high news cluster tier. W1A's integration test agent flagged that
# A4 capped at "contributing" even when 7 in-window headlines fired —
# at this level the news cluster is the headline finding. ``>=`` tier:
# at or above 10 headlines in the 1h window we promote A4 directly to
# ``primary_cause`` regardless of how many other red flags fired.
_VERY_HIGH_NEWS_CLUSTER = 10

# Verdict thresholds (number of red flags).
PRIMARY_CAUSE_RED_FLAGS = 3
CONTRIBUTING_RED_FLAGS = 1

# BTC dominance cache settings. CoinGecko's free /global endpoint reports
# the BTC market-cap dominance percentage (0-100). We cache the value in
# Redis under a single fixed key with an hour TTL — the 1h TTL doubles as
# a "fetched at most 1h ago" guarantee for the shift check below. A 1pp
# delta versus the prior cached value (rolled into ``btc_dominance:prev``
# on each live refresh) qualifies as a contributing cross-asset signal:
# never enough on its own to promote A4 to primary, but pairs with news
# / macro / vol-spike red flags to add weight.
BTC_DOMINANCE_CACHE_KEY = "btc_dominance:cached"
BTC_DOMINANCE_PREV_KEY = "btc_dominance:prev"
BTC_DOMINANCE_CACHE_TTL_S = 3600  # 1 hour
BTC_DOMINANCE_HTTP_TIMEOUT_S = 4.0
BTC_DOMINANCE_ENDPOINT = "https://api.coingecko.com/api/v3/global"
BTC_DOMINANCE_SHIFT_PP = 1.0  # 1pp shift in last 1h => contributing flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; return ``None`` on any failure.

    Accepts a few wire variants (trailing ``Z``, no tz). Always returns a
    UTC-aware ``datetime`` on success.
    """

    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _filter_headlines_to_window(
    headlines: Sequence[Dict[str, Any]],
    *,
    window_end: datetime,
    window_seconds: int,
) -> List[Dict[str, Any]]:
    """Keep only headlines with ``published_iso`` inside ``(end - window, end]``."""

    window_start = window_end - timedelta(seconds=window_seconds)
    out: List[Dict[str, Any]] = []
    for entry in headlines or []:
        if not isinstance(entry, dict):
            continue
        published = _parse_iso_timestamp(entry.get("published_iso"))
        if published is None:
            continue
        if window_start <= published <= window_end:
            out.append(entry)
    return out


def _macro_market_shifted(market: Any) -> Optional[float]:
    """Return the absolute 1h price change in pp if it exceeds the threshold.

    ``Market.price_history`` always carries a ``"1h"`` key (post_init forces
    it) so this is safe to read directly. Returns ``None`` when the market
    didn't move enough to qualify, OR when its category isn't one of the
    macro categories we care about.
    """

    category = str(getattr(market, "category", "") or "").strip().lower()
    if category not in MACRO_CATEGORIES:
        return None
    price_history = getattr(market, "price_history", None) or {}
    try:
        change_1h = float(price_history.get("1h", 0.0))
    except (TypeError, ValueError):
        return None
    abs_change = abs(change_1h)
    if abs_change > MACRO_SHIFT_RED_FLAG_PP:
        return abs_change
    return None


def _ticker_buffer_vol_spike(
    ticker_buffer: Sequence[Dict[str, float]],
    *,
    window_seconds: int = 1800,
) -> Optional[float]:
    """Detect a > 2x baseline-median absolute-price-delta spike.

    Returns the spike multiplier if a spike is present; ``None`` otherwise
    (including when the buffer is too short to compute a baseline).
    """

    if not ticker_buffer or len(ticker_buffer) < 4:
        return None
    prices: List[float] = []
    for tick in ticker_buffer:
        if not isinstance(tick, dict):
            continue
        try:
            prices.append(float(tick.get("price", 0.0)))
        except (TypeError, ValueError):
            continue
    if len(prices) < 4:
        return None
    deltas = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    if not deltas:
        return None
    baseline = median(deltas)
    if baseline <= 0.0:
        return None
    # The "recent" portion: last quarter of the buffer.
    tail_n = max(1, len(deltas) // 4)
    recent_max = max(deltas[-tail_n:])
    multiplier = recent_max / baseline
    if multiplier > VOL_SPIKE_MULTIPLIER:
        return multiplier
    return None


def _run_with_inline_timeout(
    fn: Callable[[], Any],
    *,
    timeout_s: float,
) -> Any:
    """Run ``fn`` in a daemon thread with a wall-clock cap.

    This is a local helper rather than reusing :func:`base._run_with_timeout`
    because we don't want a Gemini timeout to bubble up as the agent's
    overall timeout — the agent must continue and emit deterministic
    findings even when the LLM stalls.

    Raises :class:`TimeoutError` (the stdlib variant) on overrun so the
    caller can distinguish from other exceptions.
    """

    result_box: Dict[str, Any] = {}
    exc_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surface to outer
            exc_box["error"] = exc

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout=max(0.001, float(timeout_s)))
    if th.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout_s:.1f}s")
    if "error" in exc_box:
        raise exc_box["error"]
    return result_box.get("value")


# ---------------------------------------------------------------------------
# BTC dominance — cached cross-asset signal (CoinGecko free /global)
# ---------------------------------------------------------------------------


def _fetch_btc_dominance_cached(redis_client: Optional[Any]) -> Optional[float]:
    """Return BTC market-cap dominance as a percentage (0-100), or ``None``.

    Reads from Redis key ``btc_dominance:cached`` (TTL 1h). On miss, fetches
    from CoinGecko's free ``/global`` endpoint (no API key required) and
    repopulates the cache. Always returns ``None`` rather than raising —
    this matches the ``_safe_run`` discipline on the rest of A4 so a
    transient outage cannot crash the agent.

    Args:
        redis_client: optional Redis-like object with ``get(key)`` and
            ``set(key, value, ex=ttl)`` semantics. When ``None`` (or any
            cache op raises), the function falls through to the HTTP
            fetch and skips caching.

    Returns:
        Float dominance percentage in [0, 100], or ``None`` on failure.
    """

    # ---- 1) cache hit? ------------------------------------------------
    if redis_client is not None:
        try:
            raw = redis_client.get(BTC_DOMINANCE_CACHE_KEY)
        except Exception:  # noqa: BLE001 - degrade gracefully
            raw = None
        if raw is not None:
            try:
                # Redis client may return bytes or str depending on
                # decode_responses setting — normalize both.
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                value = float(raw)
                if 0.0 <= value <= 100.0:
                    return value
            except (TypeError, ValueError):
                # Bad cache entry; fall through to the live fetch.
                pass

    # ---- 2) live fetch (best-effort, never raises out) ---------------
    try:
        import requests  # local import: keeps cold path cheap
    except Exception:  # noqa: BLE001
        return None

    try:
        response = requests.get(
            BTC_DOMINANCE_ENDPOINT,
            timeout=BTC_DOMINANCE_HTTP_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - includes Timeout, ConnectionError
        LOGGER.warning("context: BTC dominance fetch failed: %r", exc)
        return None

    try:
        if getattr(response, "status_code", 200) >= 400:
            return None
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - malformed JSON
        LOGGER.warning("context: BTC dominance response not JSON: %r", exc)
        return None

    try:
        # CoinGecko shape: {"data": {"market_cap_percentage": {"btc": 51.2, ...}}}
        market_cap_pct = (
            payload.get("data", {}).get("market_cap_percentage", {})
            if isinstance(payload, dict)
            else {}
        )
        btc_pct = float(market_cap_pct.get("btc"))
    except (TypeError, ValueError, AttributeError):
        return None
    if not (0.0 <= btc_pct <= 100.0):
        return None

    # ---- 3) repopulate cache (best-effort) ---------------------------
    # On a cache miss we have a fresh value to store. Before overwriting
    # the canonical key we copy whatever's there into the "prev" slot so
    # the shift check has a recent comparison point. The prev key
    # piggy-backs on the same TTL — if both expire, the delta check
    # naturally degrades to "no comparison available".
    if redis_client is not None:
        try:
            existing = redis_client.get(BTC_DOMINANCE_CACHE_KEY)
        except Exception:  # noqa: BLE001
            existing = None
        if existing is not None:
            try:
                redis_client.set(
                    BTC_DOMINANCE_PREV_KEY,
                    existing,
                    ex=BTC_DOMINANCE_CACHE_TTL_S,
                )
            except Exception:  # noqa: BLE001
                pass
        try:
            redis_client.set(
                BTC_DOMINANCE_CACHE_KEY,
                f"{btc_pct:.6f}",
                ex=BTC_DOMINANCE_CACHE_TTL_S,
            )
        except Exception:  # noqa: BLE001
            pass
    return btc_pct


def _fetch_btc_dominance_prev(redis_client: Optional[Any]) -> Optional[float]:
    """Read the previously cached BTC dominance, if any.

    Used to compute the 1h shift after a live refresh. Returns ``None`` when
    the prev key is absent or holds a malformed value.
    """

    if redis_client is None:
        return None
    try:
        raw = redis_client.get(BTC_DOMINANCE_PREV_KEY)
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 100.0:
        return value
    return None


# ---------------------------------------------------------------------------
# X / Twitter sentiment — bearer-auth fetch + word-count sentiment + 1h cache
# ---------------------------------------------------------------------------

# X API endpoint + auth + bounds. Endpoint is the v2 recent-search surface;
# 50 results is a comfortable default — enough signal to average over without
# burning API quota. 4s HTTP timeout matches the BTC dominance fetcher
# discipline upstream.
X_API_RECENT_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
X_API_HTTP_TIMEOUT_S = 4.0
X_API_MAX_RESULTS = 50

# Cache key prefix + TTL. Keyed by (symbol, ISO date_hour) so the result
# rolls over hourly without explicit invalidation.
X_SENTIMENT_CACHE_KEY_PREFIX = "x_sentiment"
X_SENTIMENT_CACHE_TTL_S = 3600  # 1 hour

# Sentiment evidence trigger threshold. ``|score| > 0.3`` flips the bullet
# in :meth:`ContextForensicsAgent.investigate` to a contributing red flag.
X_SENTIMENT_RED_FLAG_ABS = 0.3

# Skeleton word lists. Production should swap to a real sentiment model
# (e.g. cardiffnlp/twitter-roberta-base-sentiment) and proper tokenization
# — these tiny lists exist purely so this PR can ship a runnable end-to-end
# integration without dragging in a heavyweight dependency. See the
# docstring for the upgrade path.
_X_POSITIVE_WORDS: frozenset[str] = frozenset(
    [
        "bull", "bullish", "rally", "surge", "moon", "pump", "buy",
        "long", "breakout", "winning", "win", "gain", "gains", "rich",
        "strong", "strength", "rocket", "up", "uptrend", "ath", "good",
        "great", "love", "best",
    ]
)

_X_NEGATIVE_WORDS: frozenset[str] = frozenset(
    [
        "bear", "bearish", "crash", "dump", "sell", "short", "tank",
        "loss", "losses", "down", "downtrend", "weak", "weakness",
        "rekt", "rug", "scam", "fud", "panic", "fear", "drop", "dump",
        "bad", "hate", "worst", "bleeding",
    ]
)


def _x_score_text(text: str) -> int:
    """Return a simple +1/-1/0 score for a single tweet's text.

    Word-count heuristic: positive_count - negative_count, sign-clipped to
    {-1, 0, 1} so ties or quiet text don't dominate the aggregate. Splits
    on whitespace and lowercases — no proper tokenizer here. Documented as
    a temporary skeleton (see module-level note).
    """

    if not text or not isinstance(text, str):
        return 0
    pos = 0
    neg = 0
    for token in text.lower().split():
        # Strip very common punctuation cling-ons; cheap pre-filter.
        token = token.strip(".,!?;:'\"()[]{}#@-")
        if not token:
            continue
        if token in _X_POSITIVE_WORDS:
            pos += 1
        elif token in _X_NEGATIVE_WORDS:
            neg += 1
    if pos == neg:
        return 0
    return 1 if pos > neg else -1


def _fetch_x_sentiment(
    symbol: str,
    *,
    redis_client: Optional[Any] = None,
) -> Optional[float]:
    """Pull recent X (Twitter) chatter for ``symbol`` + return a compound score.

    Wiring contract:

    * Auth: ``Authorization: Bearer <X_API_KEY>`` from env.
    * Endpoint: GET ``https://api.twitter.com/2/tweets/search/recent``
      with ``query=$symbol``, ``max_results=50``,
      ``tweet.fields=public_metrics``.
    * 4s HTTP timeout. On timeout / 4xx-5xx / non-JSON, return ``None``
      silently — never raise out of A4 (matches the rest of the agent's
      ``_safe_run`` discipline).
    * Cache: Redis key ``x_sentiment:{symbol}:{YYYY-MM-DDTHH}`` with TTL=1h
      so we don't burn API quota on every investigate() call inside the
      same hour.
    * Sentiment scoring: simple positive/negative word-count heuristic
      averaged across returned tweets, clipped to ``[-1, 1]``. This is a
      skeleton — production should swap to a real sentiment model
      (cardiffnlp/twitter-roberta-base-sentiment is the cheap reference)
      with proper tokenization. The signature stays the same.

    Args:
        symbol: trading symbol the sentiment was requested for (e.g.
            ``"BTC/USD"``). Used both as the search query and as the
            cache key component.
        redis_client: optional Redis-like client (``get`` / ``set(ex=)``).
            When ``None``, the cache is skipped (live fetch every call).

    Returns:
        Float sentiment in ``[-1.0, 1.0]`` on success; ``None`` when
        ``X_API_KEY`` is unset or any failure path fires.
    """

    api_key = os.getenv("X_API_KEY")
    if not api_key:
        # No key configured — degrade silently. A4's sentiment evidence
        # bullet already records the gap.
        return None

    # ---- 1) cache hit? ---------------------------------------------
    cache_key = (
        f"{X_SENTIMENT_CACHE_KEY_PREFIX}:{symbol}:"
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H')}"
    )
    if redis_client is not None:
        try:
            raw = redis_client.get(cache_key)
        except Exception:  # noqa: BLE001 - degrade gracefully
            raw = None
        if raw is not None:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                value = float(raw)
                if -1.0 <= value <= 1.0:
                    return value
            except (TypeError, ValueError):
                # Bad cache entry; fall through to the live fetch.
                pass

    # ---- 2) live fetch (best-effort, never raises out) -------------
    try:
        import requests  # local import keeps cold path cheap
    except Exception:  # noqa: BLE001
        return None

    try:
        response = requests.get(
            X_API_RECENT_SEARCH_URL,
            params={
                "query": symbol,
                "max_results": X_API_MAX_RESULTS,
                "tweet.fields": "public_metrics",
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=X_API_HTTP_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - includes Timeout, ConnectionError
        LOGGER.warning("context: X sentiment fetch failed: %r", exc)
        return None

    try:
        if getattr(response, "status_code", 200) >= 400:
            return None
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - malformed JSON
        LOGGER.warning("context: X sentiment response not JSON: %r", exc)
        return None

    tweets = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            tweets = data
    if not tweets:
        # Empty response → neutral sentiment (0.0). Distinct from "API
        # failed" (None) so the caller can tell them apart.
        score = 0.0
    else:
        scores = [_x_score_text(str(t.get("text", "") or "")) for t in tweets]
        # Mean of per-tweet scores in {-1, 0, 1} → already in [-1, 1].
        score = sum(scores) / max(1, len(scores))
        # Defensive clip for safety.
        score = max(-1.0, min(1.0, score))

    # ---- 3) cache (best-effort) ------------------------------------
    if redis_client is not None:
        try:
            redis_client.set(
                cache_key,
                f"{score:.6f}",
                ex=X_SENTIMENT_CACHE_TTL_S,
            )
        except Exception:  # noqa: BLE001
            pass

    return score


# ---------------------------------------------------------------------------
# ContextForensicsAgent
# ---------------------------------------------------------------------------


class ContextForensicsAgent(BaseForensicsAgent):
    """A4: pulls 1h-window context the model didn't have at signal time.

    Constructor knobs (all optional, but injected by tests for determinism):

    - ``news_fetcher_factory(query: str) -> fetcher`` — defaults to
      :class:`news_research_agent.fetcher.GoogleNewsRSSFetcher`. The
      returned object must have ``fetch_news() -> list[dict]`` returning
      entries with at least ``"title"`` and ``"published_iso"`` keys
      (the existing fetcher already yields this shape).
    - ``markets_fetcher() -> list[Market]`` — defaults to
      :func:`fetcher.fetch_active_markets`. Tests pass a stub that
      returns a fixed market list.
    - ``gemini_caller(prompt: str) -> str`` — defaults to a thin wrapper
      around :func:`llm_judge._request_gemini_json`. Tests pass a stub
      that returns a canned summary, raises to simulate failure, or
      sleeps to simulate a hang.
    """

    agent_name = "context"

    def __init__(
        self,
        *,
        context_store: TradeContextStore,
        timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        news_fetcher_factory: Optional[Callable[[str], Any]] = None,
        markets_fetcher: Optional[Callable[[], Sequence[Any]]] = None,
        gemini_caller: Optional[Callable[[str], str]] = None,
        gemini_timeout_s: float = GEMINI_CALL_TIMEOUT_S,
        redis_client: Optional[Any] = None,
        btc_dominance_fetcher: Optional[Callable[[Optional[Any]], Optional[float]]] = None,
        x_sentiment_fetcher: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        super().__init__(context_store=context_store, timeout_s=timeout_s)
        self._news_fetcher_factory = news_fetcher_factory
        self._markets_fetcher = markets_fetcher
        self._gemini_caller = gemini_caller
        self._gemini_timeout_s = float(gemini_timeout_s)
        # Optional Redis client used for the BTC-dominance + X-sentiment
        # caches. When absent we still try to fetch live but skip caching.
        self._redis_client = redis_client
        # Tests can inject a fake fetcher to avoid hitting CoinGecko.
        # Default delegates to ``_fetch_btc_dominance_cached``.
        self._btc_dominance_fetcher = btc_dominance_fetcher
        # Tests can inject a fake X sentiment fetcher to avoid hitting the
        # X API. Default delegates to ``_fetch_x_sentiment`` (module-level)
        # with the agent's Redis client for caching.
        self._x_sentiment_fetcher = x_sentiment_fetcher

    # ------------------------------------------------------------------
    # contract
    # ------------------------------------------------------------------
    def investigate(self, trade_id: str) -> ForensicsFinding:
        evidence: List[str] = []
        red_flags = 0
        suggested_actions: List[Dict[str, Any]] = []
        extreme_news_cluster = False

        snap = self.context_store.get_signal_snapshot(trade_id)
        if snap is None:
            # Without a signal snapshot we have no symbol/timestamp anchor.
            # Emit an informational unknown-style finding rather than
            # hallucinating context.
            return ForensicsFinding(
                agent="context",
                verdict="unknown",
                confidence=0.0,
                evidence=[
                    f"no signal-phase snapshot for trade_id={trade_id}",
                ],
                severity=1,
                error="missing_signal_snapshot",
            )

        trade_time = _parse_iso_timestamp(snap.captured_at_utc) or datetime.now(timezone.utc)
        symbol = str(snap.symbol or "").strip() or "unknown"

        # ---- 1) News headline density --------------------------------
        try:
            headlines_in_window = self._fetch_news_in_window(
                symbol=symbol, window_end=trade_time
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            LOGGER.warning("context: news fetch failed: %r", exc)
            headlines_in_window = []
            evidence.append("news fetch unavailable — skipping news density check")

        if headlines_in_window:
            n = len(headlines_in_window)
            top_title = str(headlines_in_window[0].get("title", "")).strip()
            evidence.append(
                f"news: {n} headline(s) in 1h window before trade"
                + (f" — top: {top_title}" if top_title else "")
            )
            if n > HEADLINE_DENSITY_RED_FLAG:
                red_flags += 1
                evidence.append(
                    f"red flag: > {HEADLINE_DENSITY_RED_FLAG} headlines in 1h "
                    "(elevated news density)"
                )
                suggested_actions.append(
                    {"type": "add_news_feature", "source": "google_news_rss"}
                )
            if n >= _VERY_HIGH_NEWS_CLUSTER:
                # Extreme cluster — A4 promotes directly to primary_cause.
                extreme_news_cluster = True
                evidence.append(
                    f"extreme news cluster ({n} headlines in 1h window) — "
                    f">= {_VERY_HIGH_NEWS_CLUSTER}; primary contextual driver"
                )
        else:
            evidence.append("news: no headlines in 1h window before trade")

        # ---- 2) Polymarket macro shift -------------------------------
        try:
            shifted_markets = self._fetch_macro_shifts()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("context: polymarket macro fetch failed: %r", exc)
            shifted_markets = []
            evidence.append(
                "polymarket macro fetch unavailable — skipping macro shift check"
            )

        if shifted_markets:
            red_flags += 1
            top = shifted_markets[0]
            evidence.append(
                f"red flag: macro market '{top['title'][:80]}' moved "
                f"{top['shift_pp'] * 100:.1f}pp in 1h"
            )
            suggested_actions.append(
                {"type": "add_macro_signal_bridge", "source": "polymarket"}
            )
        else:
            evidence.append("polymarket macro: no > 5pp shifts in 1h window")

        # ---- 3) Vol spike on correlated symbols ----------------------
        spike = _ticker_buffer_vol_spike(snap.ticker_buffer or [])
        if spike is not None:
            red_flags += 1
            evidence.append(
                f"red flag: ticker buffer shows {spike:.1f}x baseline "
                "abs-delta in last 30 min"
            )
            suggested_actions.append(
                {"type": "add_btc_dominance_feature"}
            )
        elif snap.ticker_buffer:
            evidence.append("vol spike check: no > 2x baseline anomaly in ticker buffer")
        else:
            evidence.append(
                "vol spike check: no ticker buffer captured — feature gap"
            )

        # ---- 3b) BTC dominance shift (cross-asset, contributing-only) -
        # Best-effort. Never raises. Never promotes to primary on its
        # own — caps at one contributing red flag.
        # The check is OFF unless the agent has either a redis_client
        # (cache backing) or an explicitly-injected fetcher. This keeps
        # the default A4 wiring (no Redis) from accidentally hitting
        # CoinGecko on every investigate() call in test / dev.
        if self._redis_client is not None or self._btc_dominance_fetcher is not None:
            try:
                btc_pct = self._fetch_btc_dominance()
            except Exception as exc:  # noqa: BLE001 - belt + braces
                LOGGER.warning("context: BTC dominance check failed: %r", exc)
                btc_pct = None
        else:
            btc_pct = None
        if btc_pct is not None:
            prev_btc_pct = _fetch_btc_dominance_prev(self._redis_client)
            if prev_btc_pct is not None:
                shift = btc_pct - prev_btc_pct
                if abs(shift) > BTC_DOMINANCE_SHIFT_PP:
                    red_flags += 1
                    evidence.append(
                        f"red flag: BTC dominance shifted "
                        f"{shift:+.2f}pp in last 1h "
                        f"(now {btc_pct:.2f}%)"
                    )
                    suggested_actions.append(
                        {"type": "add_btc_dominance_feature"}
                    )
                else:
                    evidence.append(
                        f"BTC dominance: {btc_pct:.2f}% "
                        f"(1h shift {shift:+.2f}pp, under "
                        f"{BTC_DOMINANCE_SHIFT_PP:.1f}pp threshold)"
                    )
            else:
                evidence.append(
                    f"BTC dominance: {btc_pct:.2f}% "
                    "(no prior cached sample for 1h delta)"
                )

        # ---- 4) Sentiment (X / Twitter) -------------------------------
        # Best-effort. Returns None when X_API_KEY is unset (no key, no
        # noise) and surfaces a contributing red flag when ``|score|``
        # exceeds the documented threshold. Wrap in try/except so the
        # check NEVER propagates upward — sentiment is decoration, not
        # the swarm's source of truth.
        try:
            x_score = self._fetch_x_sentiment(symbol)
        except Exception as exc:  # noqa: BLE001 - paranoid belt + braces
            LOGGER.warning("context: X sentiment helper raised: %r", exc)
            x_score = None
        if x_score is None:
            evidence.append(
                "sentiment check unavailable — X_API_KEY unset or fetch failed"
            )
        else:
            evidence.append(f"X sentiment: {x_score:+.2f} (over recent 50 tweets)")
            if abs(x_score) > X_SENTIMENT_RED_FLAG_ABS:
                red_flags += 1
                direction = "bullish" if x_score > 0 else "bearish"
                evidence.append(
                    f"red flag: x_sentiment_shift — {direction} "
                    f"({x_score:+.2f}, > {X_SENTIMENT_RED_FLAG_ABS:.2f})"
                )
                suggested_actions.append(
                    {"type": "add_x_sentiment_feature", "source": "x_recent_search"}
                )

        # ---- 5) Optional LLM summarization ---------------------------
        # ONE bounded call. Failure => still emit deterministic findings.
        if headlines_in_window or shifted_markets:
            llm_summary = self._maybe_summarize_with_gemini(
                trade_id=trade_id,
                symbol=symbol,
                headlines=headlines_in_window,
                shifted_markets=shifted_markets,
            )
            if llm_summary:
                evidence.append(f"LLM summary: {llm_summary}")
            # When LLM is unavailable, _maybe_summarize_with_gemini already
            # appended an explanatory bullet directly to ``evidence`` via
            # the return contract below — so nothing more to do here.

        # ---- Verdict -------------------------------------------------
        verdict, confidence, severity = self._classify(
            red_flags, extreme_news_cluster=extreme_news_cluster
        )

        return ForensicsFinding(
            agent="context",
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            suggested_action=suggested_actions[0] if suggested_actions else None,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # check helpers
    # ------------------------------------------------------------------
    def _fetch_news_in_window(
        self, *, symbol: str, window_end: datetime
    ) -> List[Dict[str, Any]]:
        """Pull headlines for ``symbol`` and filter to the 1h pre-trade window."""

        factory = self._news_fetcher_factory or self._default_news_fetcher_factory
        fetcher = factory(symbol)
        raw_headlines = fetcher.fetch_news()
        if not isinstance(raw_headlines, list):
            return []
        return _filter_headlines_to_window(
            raw_headlines,
            window_end=window_end,
            window_seconds=NEWS_WINDOW_SECONDS,
        )

    def _fetch_btc_dominance(self) -> Optional[float]:
        """Return the cached/live BTC dominance as a percentage in [0, 100].

        Delegates to the constructor-injected fetcher when supplied;
        otherwise calls :func:`_fetch_btc_dominance_cached` with this
        agent's redis client. Returns ``None`` on any failure.
        """

        fetcher = self._btc_dominance_fetcher
        if fetcher is None:
            return _fetch_btc_dominance_cached(self._redis_client)
        return fetcher(self._redis_client)

    def _fetch_x_sentiment(self, symbol: str) -> Optional[float]:
        """Return the cached/live X sentiment score in [-1, 1] or ``None``.

        Delegates to the constructor-injected fetcher when supplied;
        otherwise calls :func:`_fetch_x_sentiment` (module-level) with this
        agent's redis client for caching. Returns ``None`` on any failure
        OR when X_API_KEY is unset.
        """

        fetcher = self._x_sentiment_fetcher
        if fetcher is not None:
            return fetcher(symbol)
        return _fetch_x_sentiment(symbol, redis_client=self._redis_client)

    def _fetch_macro_shifts(self) -> List[Dict[str, Any]]:
        """Return macro markets that moved > 5pp in the 1h window, sorted desc."""

        fetcher = self._markets_fetcher or self._default_markets_fetcher
        markets = list(fetcher() or [])
        shifted: List[Dict[str, Any]] = []
        for market in markets:
            shift = _macro_market_shifted(market)
            if shift is None:
                continue
            shifted.append(
                {
                    "title": str(getattr(market, "title", "") or ""),
                    "category": str(getattr(market, "category", "") or ""),
                    "shift_pp": float(shift),
                }
            )
        shifted.sort(key=lambda m: m["shift_pp"], reverse=True)
        return shifted

    # ------------------------------------------------------------------
    # LLM summarization (the ONE Gemini call)
    # ------------------------------------------------------------------
    def _maybe_summarize_with_gemini(
        self,
        *,
        trade_id: str,
        symbol: str,
        headlines: Sequence[Dict[str, Any]],
        shifted_markets: Sequence[Dict[str, Any]],
    ) -> Optional[str]:
        """Best-effort one-line summary; never raises.

        Returns the summary string on success, ``None`` if no API key is
        configured and no caller-supplied stub was injected, or ``None``
        on any failure (including timeout). Mutates the agent's evidence
        log indirectly via the return value: callers append the summary
        bullet themselves; we only emit the "unavailable" bullet here.
        """

        caller = self._gemini_caller
        if caller is None:
            api_key = (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )
            if not api_key:
                # API key absent — degrade silently. Leave the evidence
                # log untouched here; callers see the absence as a
                # missing summary bullet, which is benign.
                return None
            caller = self._default_gemini_caller

        prompt = self._build_summary_prompt(
            trade_id=trade_id,
            symbol=symbol,
            headlines=headlines,
            shifted_markets=shifted_markets,
        )
        try:
            summary = _run_with_inline_timeout(
                lambda: caller(prompt),
                timeout_s=self._gemini_timeout_s,
            )
        except TimeoutError:
            LOGGER.warning(
                "context: Gemini summarization timed out after %.1fs",
                self._gemini_timeout_s,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - graceful degrade
            LOGGER.warning("context: Gemini summarization failed: %r", exc)
            return None

        text = str(summary or "").strip()
        if not text:
            return None
        # Cap to a single line; the ForensicsFinding evidence cap will
        # truncate further if needed.
        return text.splitlines()[0].strip()

    # ------------------------------------------------------------------
    # default factories (lazy imports keep import cost off the cold path)
    # ------------------------------------------------------------------
    @staticmethod
    def _default_news_fetcher_factory(query: str) -> Any:
        from news_research_agent.fetcher import GoogleNewsRSSFetcher

        return GoogleNewsRSSFetcher(query)

    @staticmethod
    def _default_markets_fetcher() -> Sequence[Any]:
        from fetcher import fetch_active_markets

        return fetch_active_markets()

    @staticmethod
    def _default_gemini_caller(prompt: str) -> str:
        """Make ONE bounded Gemini ``generateContent`` call.

        Reuses :func:`llm_judge._request_gemini_json` for retry + timeout
        plumbing. Returns the model's first text part.
        """

        import requests

        from llm_judge import (
            DEFAULT_GEMINI_MODEL,
            _extract_response_text,
            _request_gemini_json,
            _resolve_api_key,
            _resolve_model,
        )

        api_key = _resolve_api_key(None)
        if not api_key:
            raise RuntimeError("no Gemini API key available")
        model = _resolve_model(None) or DEFAULT_GEMINI_MODEL
        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "You summarize the recent context around a losing "
                            "trade in one short sentence under 160 characters. "
                            "Be neutral and specific. No markdown."
                        )
                    }
                ]
            },
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 200,
            },
        }
        session = requests.Session()
        try:
            response_json = _request_gemini_json(
                session,
                api_key=api_key,
                model=model,
                payload=payload,
                timeout_s=int(GEMINI_CALL_TIMEOUT_S),
            )
        finally:
            session.close()
        return _extract_response_text(response_json)

    # ------------------------------------------------------------------
    # prompt + verdict
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary_prompt(
        *,
        trade_id: str,
        symbol: str,
        headlines: Sequence[Dict[str, Any]],
        shifted_markets: Sequence[Dict[str, Any]],
    ) -> str:
        lines = [
            f"Trade {trade_id} on {symbol} just lost. "
            "Summarize the contextual signal in one short sentence.",
            "",
            "Headlines in the 1h window before the trade:",
        ]
        if headlines:
            for entry in headlines[:8]:
                title = str(entry.get("title", "")).strip()
                published = str(entry.get("published", "")).strip()
                if title:
                    lines.append(f"- [{published}] {title}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("Polymarket macro markets that moved > 5pp in 1h:")
        if shifted_markets:
            for m in shifted_markets[:8]:
                lines.append(
                    f"- {m['category']}: {m['title']} "
                    f"({m['shift_pp'] * 100:.1f}pp)"
                )
        else:
            lines.append("- (none)")
        return "\n".join(lines)

    @staticmethod
    def _classify(
        red_flags: int, *, extreme_news_cluster: bool = False
    ) -> tuple[str, float, int]:
        """Map red-flag count + tier flags to (verdict, confidence, severity).

        ``extreme_news_cluster`` is a hard promotion: a >= 10 headline 1h
        cluster jumps directly to primary_cause regardless of how many
        other red flags fired (W1A tightening — the cluster IS the
        story).
        """

        if extreme_news_cluster or red_flags >= PRIMARY_CAUSE_RED_FLAGS:
            return ("primary_cause", 0.75, 4)
        if red_flags >= CONTRIBUTING_RED_FLAGS:
            return ("contributing", 0.5, 2)
        return ("innocent", 0.3, 1)


__all__ = [
    "BTC_DOMINANCE_CACHE_KEY",
    "BTC_DOMINANCE_CACHE_TTL_S",
    "BTC_DOMINANCE_ENDPOINT",
    "BTC_DOMINANCE_PREV_KEY",
    "BTC_DOMINANCE_SHIFT_PP",
    "ContextForensicsAgent",
    "GEMINI_CALL_TIMEOUT_S",
    "HEADLINE_DENSITY_RED_FLAG",
    "MACRO_CATEGORIES",
    "MACRO_SHIFT_RED_FLAG_PP",
    "X_API_HTTP_TIMEOUT_S",
    "X_API_MAX_RESULTS",
    "X_API_RECENT_SEARCH_URL",
    "X_SENTIMENT_CACHE_KEY_PREFIX",
    "X_SENTIMENT_CACHE_TTL_S",
    "X_SENTIMENT_RED_FLAG_ABS",
    "_VERY_HIGH_NEWS_CLUSTER",
    "_fetch_btc_dominance_cached",
    "_fetch_btc_dominance_prev",
    "_fetch_x_sentiment",
]
