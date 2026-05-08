"""Tests for :class:`loss_postmortem.context_forensics.ContextForensicsAgent`.

Five scenarios (per the brief in autopilot_lane_e_swarm_briefs_2026_05_08.md):

  (a) High-impact news in window → contributing
  (b) Polymarket macro shift > 5pp → contributing
  (c) Quiet market (no news, no macro shifts) → innocent
  (d) LLM timeout → deterministic checks still emit, verdict driven by them
  (e) Mocked clean state with stub Gemini caller → innocent

Plus several supporting tests around:

  - missing GEMINI_API_KEY → graceful degrade (no LLM bullet, no crash)
  - missing signal snapshot → verdict="unknown" with explanatory error
  - news fetch raising → "news fetch unavailable" bullet, no crash
  - Polymarket fetch raising → "polymarket macro fetch unavailable" bullet
  - 3+ red flags → verdict=primary_cause

All external I/O (news fetch, Polymarket fetch, Gemini) is stubbed via
constructor injection — no network calls in this suite.
"""

from __future__ import annotations

import os
import time
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence
from unittest.mock import patch

import fakeredis

from loss_postmortem.context_forensics import (
    BTC_DOMINANCE_CACHE_KEY,
    BTC_DOMINANCE_CACHE_TTL_S,
    BTC_DOMINANCE_PREV_KEY,
    HEADLINE_DENSITY_RED_FLAG,
    ContextForensicsAgent,
    _VERY_HIGH_NEWS_CLUSTER,
    _fetch_btc_dominance_cached,
    _fetch_x_sentiment,
)
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


TRADE_TIME = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)


def _store() -> TradeContextStore:
    return TradeContextStore(
        redis_client=fakeredis.FakeRedis(decode_responses=True),
        namespace="test",
    )


def _signal_snapshot(
    *,
    trade_id: str = "trade-A4",
    symbol: str = "BTC/USD",
    captured_at: datetime = TRADE_TIME,
    ticker_buffer: List[Dict[str, float]] | None = None,
) -> TradeContextSnapshot:
    return TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.isoformat(),
        phase="signal",
        feature_buffer={"feat_a": 1.0},
        model_probs={"long": 0.62},
        model_confidence=0.62,
        ticker_buffer=ticker_buffer or [],
    )


def _headline(
    *,
    title: str,
    published_at: datetime,
) -> Dict[str, Any]:
    return {
        "title": title,
        "link": "https://example.com/article",
        "published": published_at.strftime("%a, %d %b %Y %H:%M:%S +0000"),
        "published_iso": published_at.isoformat(),
        "summary": "",
    }


class _StubNewsFetcher:
    def __init__(self, headlines: Sequence[Dict[str, Any]]) -> None:
        self._headlines = list(headlines)

    def fetch_news(self) -> List[Dict[str, Any]]:
        return list(self._headlines)


def _news_factory(
    headlines: Sequence[Dict[str, Any]],
):
    def factory(_query: str) -> _StubNewsFetcher:
        return _StubNewsFetcher(headlines)

    return factory


def _raising_news_factory(_query: str) -> _StubNewsFetcher:
    raise RuntimeError("simulated news outage")


class _StubMarket:
    """Light stand-in for src.models.Market — the agent only reads three attrs."""

    def __init__(self, *, title: str, category: str, change_1h: float) -> None:
        self.title = title
        self.category = category
        self.price_history = {"1h": float(change_1h), "6h": 0.0, "24h": 0.0}


def _markets_fetcher(markets: Sequence[Any]):
    def fetcher() -> Sequence[Any]:
        return list(markets)

    return fetcher


def _raising_markets_fetcher() -> Sequence[Any]:
    raise RuntimeError("simulated polymarket outage")


def _stub_gemini(text: str = "Quiet news cycle around BTC/USD."):
    def caller(_prompt: str) -> str:
        return text

    return caller


def _hanging_gemini(seconds: float = 5.0):
    def caller(_prompt: str) -> str:
        time.sleep(seconds)
        return "should never arrive"

    return caller


def _raising_gemini(_prompt: str) -> str:
    raise RuntimeError("simulated Gemini failure")


def _seed_signal(store: TradeContextStore, snap: TradeContextSnapshot) -> None:
    store.record_snapshot(snap)


# ---------------------------------------------------------------------------
# Five core scenarios
# ---------------------------------------------------------------------------


class ContextForensicsCoreScenarios(unittest.TestCase):
    """The five scenarios called out in the brief."""

    def test_high_impact_news_in_window_yields_contributing(self) -> None:
        """(a) > 5 headlines in the 1h window before the trade → contributing."""

        store = _store()
        snap = _signal_snapshot()
        _seed_signal(store, snap)

        # 7 headlines spread across the 1h pre-trade window (> threshold).
        headlines = [
            _headline(
                title=f"Macro headline {i}",
                published_at=TRADE_TIME - timedelta(minutes=10 + i * 5),
            )
            for i in range(7)
        ]
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory(headlines),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini("Heavy news flow on BTC."),
        )
        finding = agent.investigate("trade-A4")
        self.assertEqual(finding.agent, "context")
        self.assertEqual(finding.verdict, "contributing")
        self.assertGreaterEqual(finding.confidence, 0.4)
        # Density red flag explicitly mentioned.
        joined = " | ".join(finding.evidence)
        self.assertIn(f"> {HEADLINE_DENSITY_RED_FLAG}", joined)
        # Suggested action surfaces a feature request.
        self.assertEqual(
            (finding.suggested_action or {}).get("type"), "add_news_feature"
        )

    def test_polymarket_macro_shift_yields_contributing(self) -> None:
        """(b) Macro market moved > 5pp in 1h → contributing."""

        store = _store()
        _seed_signal(store, _signal_snapshot())

        markets = [
            _StubMarket(
                title="Will the Fed cut rates in May?",
                category="Federal Reserve",
                change_1h=0.08,  # 8pp move
            ),
            _StubMarket(  # category not macro → ignored
                title="Will the Lakers win game 7?",
                category="Sports",
                change_1h=0.20,
            ),
        ]
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher(markets),
            gemini_caller=_stub_gemini(),
        )
        finding = agent.investigate("trade-A4")
        self.assertEqual(finding.verdict, "contributing")
        joined = " | ".join(finding.evidence)
        self.assertIn("Will the Fed cut rates", joined)
        self.assertIn("8.0pp", joined)
        # Sports market should NOT have been counted.
        self.assertNotIn("Lakers", joined)

    def test_quiet_market_yields_innocent(self) -> None:
        """(c) No news, no macro shifts, no vol spike → innocent."""

        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([
                _StubMarket(
                    title="Will SPX close above 5800?",
                    category="Macro",
                    change_1h=0.001,  # < 5pp → not flagged
                )
            ]),
            gemini_caller=_stub_gemini(),
        )
        finding = agent.investigate("trade-A4")
        self.assertEqual(finding.verdict, "innocent")
        self.assertIsNone(finding.suggested_action)
        joined = " | ".join(finding.evidence)
        self.assertIn("no headlines", joined)
        self.assertIn("no > 5pp shifts", joined)

    def test_llm_timeout_does_not_lean_verdict_toward_primary(self) -> None:
        """(d) Hanging Gemini call → deterministic checks still drive verdict.

        The brief: "don't lean primary_cause just because LLM is unreachable".
        We seed a single news headline (1 red flag worth) and let Gemini
        hang. The verdict must follow the deterministic count (contributing
        from the > 5 threshold not being hit means innocent here), and
        crucially must NOT escalate to primary_cause due to the timeout.
        """

        store = _store()
        _seed_signal(store, _signal_snapshot())

        # One headline — under the > 5 threshold → 0 red flags from news.
        headlines = [
            _headline(
                title="A single relevant headline",
                published_at=TRADE_TIME - timedelta(minutes=20),
            )
        ]
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory(headlines),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_hanging_gemini(seconds=5.0),
            gemini_timeout_s=0.05,  # tighten for fast test
        )
        finding = agent.investigate("trade-A4")
        # 0 red flags → innocent (deterministic).
        self.assertEqual(finding.verdict, "innocent")
        self.assertIsNone(finding.error)
        # No "LLM summary:" bullet should be present (timed out).
        joined = " | ".join(finding.evidence)
        self.assertNotIn("LLM summary:", joined)

    def test_mocked_clean_state_with_stub_gemini_yields_innocent(self) -> None:
        """(e) All inputs clean + working Gemini stub → innocent."""

        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini("All quiet."),
        )
        finding = agent.investigate("trade-A4")
        self.assertEqual(finding.verdict, "innocent")
        self.assertIsNone(finding.error)


# ---------------------------------------------------------------------------
# Supporting tests around graceful degradation + edge cases
# ---------------------------------------------------------------------------


class ContextForensicsDegradationTests(unittest.TestCase):
    def test_missing_signal_snapshot_yields_unknown(self) -> None:
        store = _store()  # nothing seeded
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
        )
        finding = agent.investigate("does-not-exist")
        self.assertEqual(finding.verdict, "unknown")
        self.assertEqual(finding.error, "missing_signal_snapshot")

    def test_no_gemini_api_key_degrades_gracefully(self) -> None:
        """When no API key + no injected caller, agent must not crash."""

        store = _store()
        _seed_signal(store, _signal_snapshot())

        # Headlines fetched but well below the density red flag.
        headlines = [
            _headline(
                title="One mild headline",
                published_at=TRADE_TIME - timedelta(minutes=15),
            )
        ]
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory(headlines),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=None,  # use default which checks env
        )
        # Strip both possible env vars so the default caller bails out.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            finding = agent.investigate("trade-A4")
        self.assertEqual(finding.verdict, "innocent")
        self.assertIsNone(finding.error)
        joined = " | ".join(finding.evidence)
        # No LLM summary bullet should appear when the API key is missing.
        self.assertNotIn("LLM summary:", joined)

    def test_gemini_failure_does_not_escalate_verdict(self) -> None:
        """Gemini raising mid-call must not push verdict above the deterministic floor."""

        store = _store()
        _seed_signal(store, _signal_snapshot())

        headlines = [
            _headline(
                title="One single headline",
                published_at=TRADE_TIME - timedelta(minutes=10),
            )
        ]
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory(headlines),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_raising_gemini,
        )
        finding = agent.investigate("trade-A4")
        # 1 headline only → 0 red flags → innocent.
        self.assertEqual(finding.verdict, "innocent")
        joined = " | ".join(finding.evidence)
        self.assertNotIn("LLM summary:", joined)

    def test_news_fetch_failure_emits_unavailable_bullet(self) -> None:
        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_raising_news_factory,
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
        )
        finding = agent.investigate("trade-A4")
        joined = " | ".join(finding.evidence)
        self.assertIn("news fetch unavailable", joined)
        # Other checks should still complete.
        self.assertIn("no > 5pp shifts", joined)

    def test_polymarket_fetch_failure_emits_unavailable_bullet(self) -> None:
        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_raising_markets_fetcher,
            gemini_caller=_stub_gemini(),
        )
        finding = agent.investigate("trade-A4")
        joined = " | ".join(finding.evidence)
        self.assertIn("polymarket macro fetch unavailable", joined)


class ContextForensicsRedFlagAggregationTests(unittest.TestCase):
    def test_three_red_flags_escalate_to_primary_cause(self) -> None:
        """News density + macro shift + vol spike = 3 red flags → primary_cause."""

        store = _store()
        # Build a ticker buffer with a clear trailing spike: the median
        # absolute delta is small (~1) and the last quarter contains a
        # delta > 2x baseline (>= 5).
        ticker_buffer: List[Dict[str, float]] = (
            [{"price": 100.0 + (i % 2)} for i in range(12)]
            + [{"price": 110.0}, {"price": 100.0}, {"price": 110.0}]
        )
        snap = _signal_snapshot(ticker_buffer=ticker_buffer)
        _seed_signal(store, snap)

        # 7 headlines → red flag #1
        headlines = [
            _headline(
                title=f"Storm headline {i}",
                published_at=TRADE_TIME - timedelta(minutes=5 + i * 4),
            )
            for i in range(7)
        ]
        # Macro market shifted 10pp → red flag #2
        markets = [
            _StubMarket(
                title="Will inflation print > 3.5%?",
                category="Macro",
                change_1h=0.10,
            )
        ]
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory(headlines),
            markets_fetcher=_markets_fetcher(markets),
            gemini_caller=_stub_gemini("Volatile macro context."),
        )
        finding = agent.investigate("trade-A4")
        self.assertEqual(finding.verdict, "primary_cause")
        self.assertGreaterEqual(finding.confidence, 0.7)
        # All three red flag bullets should be present.
        joined = " | ".join(finding.evidence)
        self.assertIn(f"> {HEADLINE_DENSITY_RED_FLAG}", joined)
        self.assertIn("inflation", joined)
        self.assertIn("baseline abs-delta", joined)


# ---------------------------------------------------------------------------
# News-cluster verdict ladder (W1A tightening)
# ---------------------------------------------------------------------------


class ContextForensicsNewsClusterLadderTests(unittest.TestCase):
    """Verify the >= 10 headline tier promotes A4 directly to primary_cause."""

    def _run_with_n_headlines(self, n: int):
        store = _store()
        _seed_signal(store, _signal_snapshot())
        # Place N headlines packed inside the 1h pre-trade window. Even
        # for n=50 every entry must land in (TRADE_TIME-60min, TRADE_TIME].
        # Use seconds spacing so we can fit a large cluster in the window.
        headlines = []
        if n > 0:
            # Spread evenly between 1s and 59min before trade time. Even
            # at n=50 the spacing is ~71s — comfortably inside the window.
            window_seconds = 59 * 60  # 59 min, leave 1 min buffer at edge
            for i in range(n):
                offset_s = 1 + int((window_seconds - 1) * i / max(1, n - 1))
                published = TRADE_TIME - timedelta(seconds=offset_s)
                headlines.append(
                    _headline(
                        title=f"Cluster headline {i}",
                        published_at=published,
                    )
                )
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory(headlines),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini("Cluster summary."),
        )
        return agent.investigate("trade-A4")

    def test_one_headline_stays_innocent(self) -> None:
        """0 headlines -> innocent. With n=1 the > 5 threshold is not hit."""
        # n=0 case: explicit check the no-headlines path is innocent
        # (preserved behaviour).
        finding = self._run_with_n_headlines(0)
        self.assertEqual(finding.verdict, "innocent")
        joined = " | ".join(finding.evidence)
        self.assertIn("no headlines", joined)
        self.assertNotIn("extreme news cluster", joined)

    def test_one_headline_below_density_threshold(self) -> None:
        """1 headline -> innocent (under HEADLINE_DENSITY_RED_FLAG=5)."""
        finding = self._run_with_n_headlines(1)
        # Single headline is below the > 5 density threshold AND below
        # the >= 10 very-high cluster threshold — innocent.
        self.assertEqual(finding.verdict, "innocent")
        joined = " | ".join(finding.evidence)
        self.assertNotIn("extreme news cluster", joined)

    def test_seven_headlines_yields_contributing(self) -> None:
        """7 headlines -> contributing (existing tier preserved)."""
        finding = self._run_with_n_headlines(7)
        self.assertEqual(finding.verdict, "contributing")
        joined = " | ".join(finding.evidence)
        # Density red flag fires but the very-high tier does not.
        self.assertIn(f"> {HEADLINE_DENSITY_RED_FLAG}", joined)
        self.assertNotIn("extreme news cluster", joined)

    def test_ten_headlines_promotes_to_primary(self) -> None:
        """n=10 (== _VERY_HIGH_NEWS_CLUSTER) -> primary_cause with extreme bullet."""
        finding = self._run_with_n_headlines(_VERY_HIGH_NEWS_CLUSTER)
        self.assertEqual(finding.verdict, "primary_cause")
        joined = " | ".join(finding.evidence)
        self.assertIn("extreme news cluster", joined)
        self.assertIn(f"{_VERY_HIGH_NEWS_CLUSTER} headlines in 1h window", joined)

    def test_fifty_headlines_still_primary(self) -> None:
        """n=50 -> primary_cause (no upper bound issue)."""
        finding = self._run_with_n_headlines(50)
        self.assertEqual(finding.verdict, "primary_cause")
        joined = " | ".join(finding.evidence)
        self.assertIn("extreme news cluster", joined)
        self.assertIn("50 headlines in 1h window", joined)


# ---------------------------------------------------------------------------
# BTC dominance cached fetcher + X sentiment scaffold
# ---------------------------------------------------------------------------


class _StubResponse:
    """Minimal stand-in for ``requests.Response`` (only attrs A4 reads)."""

    def __init__(
        self,
        *,
        json_data: Any = None,
        status_code: int = 200,
        raises_on_get: bool = False,
        raises_on_json: bool = False,
    ) -> None:
        self._json = json_data
        self.status_code = status_code
        self._raises_on_get = raises_on_get
        self._raises_on_json = raises_on_json

    def json(self) -> Any:
        if self._raises_on_json:
            raise ValueError("malformed JSON")
        return self._json


def _coingecko_payload(btc_pct: float) -> Dict[str, Any]:
    return {
        "data": {
            "market_cap_percentage": {
                "btc": btc_pct,
                "eth": 18.0,
                "usdt": 5.0,
            }
        }
    }


class BtcDominanceCachedFetcherTests(unittest.TestCase):
    """Cover the four documented branches of ``_fetch_btc_dominance_cached``."""

    def test_cache_hit_returns_immediately_without_http_call(self) -> None:
        """A valid Redis entry must short-circuit the live fetch entirely."""
        client = fakeredis.FakeRedis(decode_responses=True)
        client.set(BTC_DOMINANCE_CACHE_KEY, "52.345", ex=BTC_DOMINANCE_CACHE_TTL_S)

        # Spy on requests.get globally — module-level patch survives the
        # local import inside _fetch_btc_dominance_cached.
        with patch("requests.get") as spy:
            value = _fetch_btc_dominance_cached(client)

        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, 52.345, places=3)
        # Cache hit path — no HTTP call should have been made.
        self.assertEqual(
            spy.call_count,
            0,
            msg="requests.get must NOT be called on cache hit",
        )

    def test_cache_miss_fetches_live_and_repopulates(self) -> None:
        """No cached entry → HTTP fetch → cache populated for next call."""
        client = fakeredis.FakeRedis(decode_responses=True)

        with patch(
            "requests.get",
            return_value=_StubResponse(json_data=_coingecko_payload(51.234)),
        ) as spy:
            value = _fetch_btc_dominance_cached(client)

        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, 51.234, places=3)
        self.assertEqual(spy.call_count, 1)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)

        # Cache must now hold the fetched value.
        cached = client.get(BTC_DOMINANCE_CACHE_KEY)
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(float(cached), 51.234, places=3)

    def test_cache_miss_rolls_existing_value_into_prev(self) -> None:
        """An existing canonical value moves to the prev key on refresh."""

        # Forge an "expired" state by writing the canonical key directly
        # but leaving prev empty. Then bypass the cache hit by deleting
        # the canonical key right before the fetch — emulating TTL
        # expiry while preserving the value somewhere fakeredis can read.
        # Easier: write to BOTH the canonical key AND something else,
        # then call the live fetch path by passing a client where the
        # canonical key returns the stale value, then the fetch logic
        # will read it as "existing" and roll into prev.
        # Simplest realisation: pre-seed canonical, monkey-patch the
        # cache-hit branch to skip via delete-after-read.
        client = fakeredis.FakeRedis(decode_responses=True)
        client.set(BTC_DOMINANCE_CACHE_KEY, "50.000")
        # Wipe just before the fetch path so the cache-hit path bails.
        # We do this by patching _fetch helper: easier approach is to
        # let cache-hit fire, get back the stale value (50.0), and then
        # simulate the TTL-expired refresh by manually calling fetch
        # with the canonical key cleared.
        # Simulate expiry: clear canonical, leave nothing.
        # Actually, the function's contract: cache hit returns the value.
        # To exercise the prev-roll branch, we need a state where
        # canonical_key.get() in cache-hit returns None (so fetch live
        # runs) but the same key.get() in the repopulate phase returns
        # the OLD value. fakeredis can't toggle that — so instead we
        # use a proxy that returns None on the first .get and the old
        # value on the second .get.
        class _ProxyClient:
            def __init__(self, real):
                self.real = real
                self._gets = 0

            def get(self, key):
                self._gets += 1
                # First .get is the cache-hit probe — pretend miss.
                if self._gets == 1 and key == BTC_DOMINANCE_CACHE_KEY:
                    return None
                return self.real.get(key)

            def set(self, key, value, ex=None):
                return self.real.set(key, value, ex=ex)

        proxy = _ProxyClient(client)
        with patch(
            "requests.get",
            return_value=_StubResponse(json_data=_coingecko_payload(53.5)),
        ):
            value = _fetch_btc_dominance_cached(proxy)
        self.assertAlmostEqual(value, 53.5, places=3)
        # Prev should now hold the prior canonical value (50.000).
        self.assertEqual(client.get(BTC_DOMINANCE_PREV_KEY), "50.000")
        # Canonical should hold the new value.
        new_canonical = client.get(BTC_DOMINANCE_CACHE_KEY)
        self.assertIsNotNone(new_canonical)
        self.assertAlmostEqual(float(new_canonical), 53.5, places=3)

    def test_http_timeout_returns_none(self) -> None:
        """requests.get raising Timeout → None, no crash."""
        client = fakeredis.FakeRedis(decode_responses=True)

        class _SimulatedTimeout(Exception):
            pass

        with patch("requests.get", side_effect=_SimulatedTimeout("timed out")):
            value = _fetch_btc_dominance_cached(client)
        self.assertIsNone(value)
        # Cache must not have been populated with garbage.
        self.assertIsNone(client.get(BTC_DOMINANCE_CACHE_KEY))

    def test_malformed_response_returns_none(self) -> None:
        """Non-JSON or schema-incompatible response → None, no crash."""
        client = fakeredis.FakeRedis(decode_responses=True)

        # Branch a: response.json() raises.
        with patch(
            "requests.get",
            return_value=_StubResponse(raises_on_json=True),
        ):
            value = _fetch_btc_dominance_cached(client)
        self.assertIsNone(value)

        # Branch b: JSON parses but schema is wrong.
        with patch(
            "requests.get",
            return_value=_StubResponse(json_data={"unexpected": "shape"}),
        ):
            value = _fetch_btc_dominance_cached(client)
        self.assertIsNone(value)

        # Branch c: btc field is non-numeric.
        with patch(
            "requests.get",
            return_value=_StubResponse(
                json_data={"data": {"market_cap_percentage": {"btc": "abc"}}}
            ),
        ):
            value = _fetch_btc_dominance_cached(client)
        self.assertIsNone(value)

        # Branch d: btc field out of range.
        with patch(
            "requests.get",
            return_value=_StubResponse(
                json_data={"data": {"market_cap_percentage": {"btc": 150.0}}}
            ),
        ):
            value = _fetch_btc_dominance_cached(client)
        self.assertIsNone(value)

    def test_no_redis_client_skips_cache_but_still_returns_value(self) -> None:
        """``redis_client=None`` → live fetch runs, no caching attempted."""
        with patch(
            "requests.get",
            return_value=_StubResponse(json_data=_coingecko_payload(50.0)),
        ):
            value = _fetch_btc_dominance_cached(None)
        self.assertAlmostEqual(value, 50.0, places=3)


class BtcDominanceAgentIntegrationTests(unittest.TestCase):
    """A4 must consume the BTC dominance fetcher without ever crashing."""

    def test_agent_completes_when_btc_dominance_fetcher_returns_none(self) -> None:
        """A failing fetcher must NOT prevent A4 from emitting other findings."""
        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
            btc_dominance_fetcher=lambda _client: None,
        )
        finding = agent.investigate("trade-A4")
        self.assertEqual(finding.verdict, "innocent")
        self.assertIsNone(finding.error)

    def test_agent_surfaces_dominance_value_when_no_prev(self) -> None:
        """Fresh dominance + no prev sample → informational evidence bullet."""
        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
            redis_client=fakeredis.FakeRedis(decode_responses=True),
            btc_dominance_fetcher=lambda _client: 51.5,
        )
        finding = agent.investigate("trade-A4")
        joined = " | ".join(finding.evidence)
        self.assertIn("BTC dominance", joined)
        self.assertIn("51.50%", joined)
        # No prev sample → no red flag.
        self.assertEqual(finding.verdict, "innocent")

    def test_agent_flags_btc_dominance_shift_as_contributing(self) -> None:
        """A 1.5pp 1h shift → red flag → contributing tier (not primary)."""
        client = fakeredis.FakeRedis(decode_responses=True)
        # Seed a prior cached value 1.5pp lower than the fresh fetch.
        client.set(BTC_DOMINANCE_PREV_KEY, "50.000", ex=BTC_DOMINANCE_CACHE_TTL_S)

        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
            redis_client=client,
            btc_dominance_fetcher=lambda _client: 51.5,
        )
        finding = agent.investigate("trade-A4")
        joined = " | ".join(finding.evidence)
        # Red flag bullet present.
        self.assertIn("BTC dominance shifted", joined)
        # Contributing-only — not promoted to primary on this signal alone.
        self.assertEqual(finding.verdict, "contributing")

    def test_btc_dominance_signal_alone_does_not_promote_to_primary(self) -> None:
        """A 5pp shift is large but A4 must NOT promote on this signal alone."""
        client = fakeredis.FakeRedis(decode_responses=True)
        client.set(BTC_DOMINANCE_PREV_KEY, "45.000", ex=BTC_DOMINANCE_CACHE_TTL_S)

        store = _store()
        _seed_signal(store, _signal_snapshot())

        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
            redis_client=client,
            btc_dominance_fetcher=lambda _client: 50.0,  # +5pp shift
        )
        finding = agent.investigate("trade-A4")
        # 1 red flag from BTC dominance only → contributing (not primary).
        self.assertEqual(finding.verdict, "contributing")


class XSentimentScaffoldTests(unittest.TestCase):
    """The X/Twitter sentiment helper is a stub until API access is wired."""

    def test_returns_none_when_env_var_unset(self) -> None:
        """No X_API_KEY → silent None, no NotImplementedError."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("X_API_KEY", None)
            value = _fetch_x_sentiment("BTC/USD")
        self.assertIsNone(value)


# ---------------------------------------------------------------------------
# safe_investigate integration (timeout / crash protection from base class)
# ---------------------------------------------------------------------------


class ContextForensicsSafeInvestigateTests(unittest.TestCase):
    def test_safe_investigate_returns_finding_for_clean_run(self) -> None:
        store = _store()
        _seed_signal(store, _signal_snapshot())
        agent = ContextForensicsAgent(
            context_store=store,
            news_fetcher_factory=_news_factory([]),
            markets_fetcher=_markets_fetcher([]),
            gemini_caller=_stub_gemini(),
        )
        finding = agent.safe_investigate("trade-A4")
        self.assertEqual(finding.agent, "context")
        self.assertEqual(finding.verdict, "innocent")
        self.assertGreaterEqual(finding.runtime_s, 0.0)


if __name__ == "__main__":
    unittest.main()
