"""Fixture 4: News-event miss loss.

A losing trade where high-impact news broke in the 1h window before the
trade timestamp. The model fired without that context — A4
ContextForensicsAgent should detect an extreme headline cluster (>= 10
headlines in the 1h window) and promote the verdict directly to
``primary_cause`` so the swarm root_cause label resolves to "Context".

W1A's verdict-ladder tightening introduced ``_VERY_HIGH_NEWS_CLUSTER =
10`` — at or above that count A4 promotes regardless of other red
flags. This fixture seeds 11 headlines by default so the canonical
integration scenario exercises the new tier.

To keep the fixture deterministic and self-contained, ``build_fixture``
returns the trade_id PLUS a callable factory that the test wires into
the ContextForensicsAgent constructor:

- ``news_fetcher_factory(query) -> object with .fetch_news()``
  returning a list of 11 canned headlines, all timestamped within the 1h
  window before ``captured_at``. Crossing the ``_VERY_HIGH_NEWS_CLUSTER``
  threshold (10) promotes A4 to ``primary_cause`` directly.
- ``markets_fetcher() -> []`` — no Polymarket macro shifts (so the
  verdict is at-least-contributing, not necessarily primary).
- ``gemini_caller(prompt) -> str`` — returns a fixed one-line summary
  so the LLM bullet is reproducible.

The test calls ``build_fixture(...)`` and then constructs the agent as::

    agent = ContextForensicsAgent(
        context_store=ctx_store,
        news_fetcher_factory=info["news_fetcher_factory"],
        markets_fetcher=info["markets_fetcher"],
        gemini_caller=info["gemini_caller"],
    )

so the swarm path stays in-process and offline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from state.position_store import Position, PositionStore
from state.trade_context_store import TradeContextSnapshot, TradeContextStore


SYMBOL = "BTC/USD"
TRADE_ID = "fixture-news-event-miss-trade-1"


class _StubNewsFetcher:
    """In-memory stand-in for ``GoogleNewsRSSFetcher``.

    Exposes ``fetch_news()`` returning the canned headlines passed at
    construction. Matches the duck-typed shape the
    ContextForensicsAgent expects (the agent only calls ``.fetch_news()``
    and reads ``title`` / ``published_iso``).
    """

    def __init__(self, headlines: Sequence[Dict[str, Any]]) -> None:
        self._headlines = list(headlines)

    def fetch_news(self) -> List[Dict[str, Any]]:
        return list(self._headlines)


def build_fixture(
    *,
    context_store: TradeContextStore,
    position_store: Optional[PositionStore] = None,
    meta_base_dir: Optional[str] = None,
    trade_id: str = TRADE_ID,
    symbol: str = SYMBOL,
) -> Dict[str, Any]:
    """Populate stores + return ``trade_id`` plus injectable factories.

    Returns a dict::
        {
          "trade_id": str,
          "news_fetcher_factory": Callable[[str], _StubNewsFetcher],
          "markets_fetcher": Callable[[], list],
          "gemini_caller": Callable[[str], str],
        }
    """

    captured_at = datetime(2026, 5, 8, 15, 0, 0, tzinfo=timezone.utc)

    # ----- healthy meta so signal forensics stays clean -----
    if meta_base_dir is not None:
        slug = (
            symbol.lower()
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        slug_dir = os.path.join(meta_base_dir, slug)
        os.makedirs(slug_dir, exist_ok=True)
        meta_payload = {
            "optimal_threshold": 0.5,
            "feature_means": {"return_1": 0.0, "atr_14": 1.0},
            "feature_stds": {"return_1": 1.0, "atr_14": 0.5},
            "metrics_test": {"reliability_slope": 0.85},
            "threshold_metrics": {"0.5": {"reliability_slope": 0.85}},
        }
        with open(os.path.join(slug_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh)

    # ----- signal snapshot: clean features so only A4 lights up -----
    signal_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=captured_at.isoformat(),
        phase="signal",
        feature_buffer={"return_1": 0.03, "atr_14": 1.05},
        feature_window=[
            {"regime": "trend", "return_1": 0.02},
            {"regime": "trend", "return_1": 0.03},
            {"regime": "trend", "return_1": 0.03},
            {"regime": "trend", "return_1": 0.03},
        ],
        model_probs={"long": 0.78, "short": 0.22},
        model_confidence=0.78,
        risk_metrics_input={
            "side": "long",
            "proposed_notional_usd": 100.0,
            "bankroll": 10_000.0,
        },
        risk_metrics_output={
            "adjusted_position_size_pct": 1.0,
            "expected_value_estimate": 0.4,
        },
        breaker_context={"recommended_action": "allow", "tripped": []},
        ticker_buffer=[
            {
                "symbol": symbol,
                "bid": 30_000.0,
                "ask": 30_001.0,
                "last": 30_000.5,
                "as_of_utc": captured_at.isoformat(),
            }
        ],
        notes=None,
    )
    context_store.record_snapshot(signal_snap)

    # ----- fill snapshot: clean -----
    fill_t = captured_at + timedelta(seconds=1)
    fill_snap = TradeContextSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        captured_at_utc=fill_t.isoformat(),
        phase="fill",
        feature_buffer={},
        model_confidence=0.0,
        risk_metrics_output={
            "fill_price": 30_001.0,
            "fill_size": 0.003,
            "exchange": "coinbase",
            "status": "open",
        },
        ticker_buffer=[],
        notes=None,
    )
    context_store.record_snapshot(fill_snap)

    # ----- position record -----
    if position_store is not None:
        position = Position(
            position_id=trade_id,
            exchange="coinbase",
            symbol=symbol,
            side="long",
            status="closed",
            entry_price=30_001.0,
            entry_quote_usd=30_001.0 * 0.003,
            base_size=0.003,
            exit_price=29_700.0,
            exit_quote_usd=29_700.0 * 0.003,
            realized_pnl_usd=-0.9,
            opened_at_utc=captured_at.isoformat(),
            closed_at_utc=(captured_at + timedelta(minutes=10)).isoformat(),
            notes=None,
            model_meta={},
        )
        position_store.record_open(position)
        position_store.record_close(
            trade_id, exit_price=29_700.0, exit_quote_usd=29_700.0 * 0.003
        )

    # ----- canned headlines: 11 entries, all in the 1h pre-trade window -----
    # Spread across the 55 min before captured_at so they fall inside
    # ``(captured_at - 1h, captured_at]``. 11 entries crosses the
    # ``_VERY_HIGH_NEWS_CLUSTER`` threshold (10) so A4 promotes directly
    # to primary_cause.
    headline_titles = [
        "Federal Reserve signals emergency rate decision overnight",
        "BTC plunges 8% on macro shock; analysts warn of cascade",
        "Major exchange halts withdrawals citing liquidity stress",
        "Treasury announces unscheduled press conference for tonight",
        "Hedge fund liquidation rumours hit crypto markets",
        "Stablecoin issuer reveals reserve audit discrepancy",
        "Inflation print exceeds consensus; risk-off across the board",
        "ECB hints at coordinated central bank intervention",
        "Crypto whale wallet drains $200M to exchange in last hour",
        "Geopolitical risk premium surges as Mideast tensions flare",
        "On-chain analytics flag rapid leverage unwind in perp markets",
    ]
    headlines: List[Dict[str, Any]] = []
    n_headlines = len(headline_titles)
    for i, title in enumerate(headline_titles):
        # Place the i-th headline at staggered offsets across the
        # window so all entries land in (captured_at - 60min, captured_at].
        # Offsets evenly spaced from ~58 min ago down to ~2 min ago.
        offset_min = 58 - int(56 * i / max(1, n_headlines - 1))
        published = captured_at - timedelta(minutes=offset_min)
        headlines.append(
            {
                "title": title,
                "link": f"https://example.test/n{i}",
                "summary": "",
                "published": published.strftime("%Y-%m-%d %H:%M"),
                "published_iso": published.isoformat(),
            }
        )

    def _news_fetcher_factory(query: str) -> _StubNewsFetcher:
        # The agent passes ``symbol`` as the query — we ignore it and
        # return the canned pack.
        return _StubNewsFetcher(headlines)

    def _markets_fetcher() -> List[Any]:
        # No macro shifts — the news density alone should drive the
        # verdict to at least ``contributing``.
        return []

    def _gemini_caller(prompt: str) -> str:
        return "Macro shock: rate-decision rumour and liquidity stress hit risk assets"

    return {
        "trade_id": trade_id,
        "news_fetcher_factory": _news_fetcher_factory,
        "markets_fetcher": _markets_fetcher,
        "gemini_caller": _gemini_caller,
    }
