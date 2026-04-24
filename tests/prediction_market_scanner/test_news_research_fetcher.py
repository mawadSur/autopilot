import unittest
from time import struct_time

from news_research_agent.fetcher import GoogleNewsRSSFetcher


class FakeFeedParser:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def parse(self, url):
        self.calls.append(url)
        return type("Feed", (), {"entries": self.entries})()


class GoogleNewsRSSFetcherTests(unittest.TestCase):
    def test_fetch_news_encodes_query_and_parses_entries(self):
        entries = [
            {
                "title": "Article B",
                "link": "https://example.com/b",
                "published": "2026-04-20T12:00:00+00:00",
                "published_parsed": struct_time((2026, 4, 20, 12, 0, 0, 0, 0, 0)),
                "summary": "Summary B",
            },
            {
                "title": "Article A",
                "link": "https://example.com/a",
                "published": "2026-04-19T09:00:00+00:00",
                "published_parsed": struct_time((2026, 4, 19, 9, 0, 0, 0, 0, 0)),
                "summary": "Summary A",
            },
        ]
        fake_feedparser = FakeFeedParser(entries)
        fetcher = GoogleNewsRSSFetcher(
            "GPT-5 release date",
            feedparser_module=fake_feedparser,
        )

        parsed = fetcher.fetch_news()

        self.assertEqual(fake_feedparser.calls[0], "https://news.google.com/rss/search?q=GPT-5+release+date&hl=en-US&gl=US&ceid=US:en")
        self.assertEqual(parsed[0]["title"], "Article A")
        self.assertEqual(parsed[1]["title"], "Article B")
        self.assertEqual(parsed[0]["summary"], "Summary A")

    def test_format_for_llm_returns_chronological_context(self):
        context = GoogleNewsRSSFetcher.format_for_llm(
            [
                {
                    "title": "Article A",
                    "link": "https://example.com/a",
                    "published": "2026-04-19T09:00:00+00:00",
                    "published_iso": "2026-04-19T09:00:00+00:00",
                    "summary": "Summary A",
                },
                {
                    "title": "Article B",
                    "link": "https://example.com/b",
                    "published": "2026-04-20T12:00:00+00:00",
                    "published_iso": "2026-04-20T12:00:00+00:00",
                    "summary": "Summary B",
                },
            ]
        )

        self.assertIn("NEWS COVERAGE CONTEXT", context)
        self.assertLess(context.index("Article A"), context.index("Article B"))
        self.assertIn("Summary: Summary A", context)
        self.assertIn("Link: https://example.com/b", context)

    def test_empty_feed_formats_cleanly(self):
        fetcher = GoogleNewsRSSFetcher("election winner", feedparser_module=FakeFeedParser([]))
        self.assertIn("No recent Google News RSS articles found.", fetcher.fetch_news_context())


if __name__ == "__main__":
    unittest.main()
