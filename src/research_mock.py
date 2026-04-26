"""Shared helpers for research-agent mock-data mode.

When the ``RESEARCH_MOCK`` environment variable is truthy, the Reddit and
Google News fetchers should return deterministic offline payloads instead
of hitting real APIs. This module owns the env-var detection and the
deterministic mock-context builders so both fetchers stay consistent and
tests can exercise the toggle without monkey-patching multiple modules.

The mock check intentionally happens at fetch time (not construction time)
so that:
  * the env var can be flipped within a single process during tests, and
  * constructing a fetcher remains side-effect free (no PRAW client built,
    no feedparser invocation) when running in mock mode.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

LOGGER = logging.getLogger(__name__)

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

_REDDIT_LOG_EMITTED = False
_NEWS_LOG_EMITTED = False


def is_research_mock_enabled() -> bool:
    """Return ``True`` when the research-mock env toggle is enabled.

    Recognised truthy values (case-insensitive, surrounding whitespace
    ignored): ``"1"``, ``"true"``, ``"yes"``, ``"on"``.
    """

    value = os.environ.get("RESEARCH_MOCK", "")
    return value.strip().lower() in _TRUTHY_VALUES


def _emit_reddit_log_once() -> None:
    global _REDDIT_LOG_EMITTED
    if _REDDIT_LOG_EMITTED:
        return
    LOGGER.info(
        "RESEARCH_MOCK=true — Reddit fetcher returning deterministic mock data."
    )
    _REDDIT_LOG_EMITTED = True


def _emit_news_log_once() -> None:
    global _NEWS_LOG_EMITTED
    if _NEWS_LOG_EMITTED:
        return
    LOGGER.info(
        "RESEARCH_MOCK=true — News fetcher returning deterministic mock data."
    )
    _NEWS_LOG_EMITTED = True


def reset_mock_log_state() -> None:
    """Reset the once-per-process log flags. Intended for test isolation."""

    global _REDDIT_LOG_EMITTED, _NEWS_LOG_EMITTED
    _REDDIT_LOG_EMITTED = False
    _NEWS_LOG_EMITTED = False


def _format_subreddits(subreddits: Iterable[str] | None) -> str:
    if not subreddits:
        return "all"
    cleaned = [str(item).strip() for item in subreddits if str(item or "").strip()]
    if not cleaned:
        return "all"
    if any(item.lower() == "all" for item in cleaned):
        return "all"
    return "+".join(cleaned)


def build_mock_reddit_context(
    search_query: str,
    *,
    subreddits: Iterable[str] | None = None,
) -> str:
    """Return a deterministic mock Reddit discussion context.

    The output mirrors the layout produced by
    :class:`reddit_research_agent.fetcher.RedditDeepDiver.fetch_discussion_context`
    so it remains a valid input to ``RedditAgent.analyze_discussion`` without
    any other code changes.
    """

    _emit_reddit_log_once()

    query = (search_query or "").strip() or "mock query"
    subreddit_target = _format_subreddits(subreddits)

    lines = [
        "REDDIT DISCUSSION CONTEXT",
        f"QUERY: {query}",
        f"SUBREDDITS: {subreddit_target}",
        "",
        "[MOCK DATA] RESEARCH_MOCK=true; offline deterministic fixture.",
        "",
        f"THREAD 1 | r/PredictionMarkets | score=412 | comments=24 | author=u/mock_alpha",
        f"TITLE: Mock thread A discussing '{query}'",
        f"POST: Deterministic synthetic post body about '{query}' for offline tests.",
        "TOP COMMENTS:",
        f"- u/mock_pro | score=180 [signal]: Bullish thesis grounded in fixture data for '{query}'.",
        "  - -> u/mock_reply | score=42: Replying with nuance and a counterpoint citation.",
        f"- u/mock_skeptic | score=95 [signal]: Bearish thesis citing prior base rates for '{query}'.",
        "",
        f"THREAD 2 | r/Politics | score=158 | comments=11 | author=u/mock_beta",
        f"TITLE: Mock thread B revisits '{query}'",
        f"POST: Second deterministic body referencing '{query}' as the central topic.",
        "TOP COMMENTS:",
        "- u/mock_observer | score=64 [signal]: Neutral observation enumerating evidence both ways.",
        "  - -> u/mock_followup | score=18: Adds a clarifying source link.",
    ]
    return "\n".join(lines)


def build_mock_news_entries(search_query: str) -> list[dict[str, str]]:
    """Return deterministic mock news entries (Google News RSS shape)."""

    _emit_news_log_once()
    query = (search_query or "").strip() or "mock query"
    return [
        {
            "title": f"Mock primary report: '{query}' explained",
            "link": "https://example.com/mock-news/primary",
            "published": "2026-01-01 09:00 UTC",
            "published_iso": "2026-01-01T09:00:00+00:00",
            "summary": (
                f"Deterministic offline summary covering '{query}' for use under "
                "RESEARCH_MOCK=true."
            ),
        },
        {
            "title": f"Mock follow-up: market reaction to '{query}'",
            "link": "https://example.com/mock-news/followup",
            "published": "2026-01-02 14:30 UTC",
            "published_iso": "2026-01-02T14:30:00+00:00",
            "summary": (
                f"Synthetic follow-up article describing reactions to '{query}' "
                "with no external API call."
            ),
        },
        {
            "title": f"Mock analysis: implications of '{query}' for resolution",
            "link": "https://example.com/mock-news/analysis",
            "published": "2026-01-03 18:45 UTC",
            "published_iso": "2026-01-03T18:45:00+00:00",
            "summary": (
                f"Offline analyst commentary about '{query}' including bullish "
                "and bearish considerations for prediction-market traders."
            ),
        },
    ]


def build_mock_news_context(search_query: str) -> str:
    """Return a deterministic mock news context formatted for the news agent."""

    # Local import to avoid an import cycle: the fetcher module imports this
    # module at top level, so we cannot import it at module scope here.
    from news_research_agent.fetcher import GoogleNewsRSSFetcher

    return GoogleNewsRSSFetcher.format_for_llm(build_mock_news_entries(search_query))
