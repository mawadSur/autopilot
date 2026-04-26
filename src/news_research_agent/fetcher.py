from __future__ import annotations

from datetime import datetime, timezone
from time import struct_time
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote_plus

import feedparser

from research_mock import build_mock_news_entries, is_research_mock_enabled


GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
DEFAULT_ARTICLE_LIMIT = 10


class GoogleNewsRSSFetcher:
    def __init__(
        self,
        search_query: str,
        *,
        article_limit: int = DEFAULT_ARTICLE_LIMIT,
        feedparser_module: Any | None = None,
    ) -> None:
        query = str(search_query or "").strip()
        if not query:
            raise ValueError("search_query must be a non-empty string")
        self.search_query = query
        self.article_limit = max(1, int(article_limit))
        self._feedparser = feedparser_module if feedparser_module is not None else feedparser

    def fetch_news(self) -> List[Dict[str, str]]:
        if is_research_mock_enabled():
            mock_entries = build_mock_news_entries(self.search_query)
            return mock_entries[: self.article_limit]
        feed = self._feedparser.parse(self._build_feed_url())
        entries = list(getattr(feed, "entries", []) or [])
        parsed_entries = [self._parse_entry(entry) for entry in entries]
        parsed_entries = [entry for entry in parsed_entries if entry["title"]]
        parsed_entries.sort(key=lambda item: item["published_iso"])
        return parsed_entries[: self.article_limit]

    def fetch_news_context(self) -> str:
        return self.format_for_llm(self.fetch_news())

    def fetch_context(self) -> str:
        return self.fetch_news_context()

    def _build_feed_url(self) -> str:
        return GOOGLE_NEWS_RSS_URL.format(query=quote_plus(self.search_query))

    @staticmethod
    def format_for_llm(entries: Sequence[Dict[str, str]]) -> str:
        lines = ["NEWS COVERAGE CONTEXT"]
        if not entries:
            lines.append("No recent Google News RSS articles found.")
            return "\n".join(lines)

        for index, entry in enumerate(entries, start=1):
            lines.append(
                f"ARTICLE {index} | {entry['published']} | {entry['title']}"
            )
            if entry.get("summary"):
                lines.append(f"Summary: {entry['summary']}")
            if entry.get("link"):
                lines.append(f"Link: {entry['link']}")
            lines.append("")
        return "\n".join(lines).strip()

    def _parse_entry(self, entry: Any) -> Dict[str, str]:
        title = self._clean_text(getattr(entry, "title", "") or self._get_mapping_value(entry, "title"))
        link = str(getattr(entry, "link", "") or self._get_mapping_value(entry, "link") or "").strip()
        summary = self._clean_text(
            getattr(entry, "summary", "") or self._get_mapping_value(entry, "summary") or ""
        )
        published_label, published_iso = self._extract_published(entry)
        return {
            "title": title,
            "link": link,
            "published": published_label,
            "published_iso": published_iso,
            "summary": summary,
        }

    def _extract_published(self, entry: Any) -> tuple[str, str]:
        raw_published = str(
            getattr(entry, "published", "") or self._get_mapping_value(entry, "published") or ""
        ).strip()
        parsed = getattr(entry, "published_parsed", None)
        if parsed is None:
            parsed = self._get_mapping_value(entry, "published_parsed")
        dt = self._coerce_datetime(parsed)
        if dt is None and raw_published:
            dt = self._coerce_iso_datetime(raw_published)
        if dt is None:
            return (raw_published or "unknown", "9999-12-31T23:59:59+00:00")
        label = raw_published or dt.strftime("%Y-%m-%d %H:%M UTC")
        return (label, dt.isoformat())

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, struct_time):
            return datetime(*value[:6], tzinfo=timezone.utc)
        if isinstance(value, tuple) and len(value) >= 6:
            return datetime(*value[:6], tzinfo=timezone.utc)
        return None

    @staticmethod
    def _coerce_iso_datetime(value: str) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _get_mapping_value(entry: Any, key: str) -> Any:
        if isinstance(entry, dict):
            return entry.get(key)
        get_fn = getattr(entry, "get", None)
        if callable(get_fn):
            return get_fn(key)
        return None

    @staticmethod
    def _clean_text(value: Any) -> str:
        return " ".join(str(value or "").split())


NewsAggregator = GoogleNewsRSSFetcher
