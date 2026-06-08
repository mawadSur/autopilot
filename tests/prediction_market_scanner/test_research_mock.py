import logging
import os
import unittest
from unittest.mock import patch

from news_research_agent.fetcher import GoogleNewsRSSFetcher
from reddit_research_agent.fetcher import RedditDeepDiver
from research_mock import (
    build_mock_news_context,
    build_mock_reddit_context,
    is_research_mock_enabled,
    reset_mock_log_state,
)


class _ExplodingFeedParser:
    """Stand-in feedparser whose ``parse`` always raises.

    Used to prove the news fetcher never reaches the network path under mock
    mode (any call would surface as a test failure).
    """

    def parse(self, url):  # pragma: no cover - guarded via assertions
        raise AssertionError(
            f"feedparser.parse should not be invoked under RESEARCH_MOCK; "
            f"got url={url!r}"
        )


class _ExplodingReddit:
    """Stand-in PRAW client whose ``subreddit`` always raises."""

    def subreddit(self, target):  # pragma: no cover - guarded via assertions
        raise AssertionError(
            f"PRAW client should not be invoked under RESEARCH_MOCK; "
            f"got target={target!r}"
        )


class IsResearchMockEnabledTests(unittest.TestCase):
    def test_is_research_mock_enabled_true(self):
        for raw_value in ("1", "true", "TRUE", "Yes", "yes", "ON", " on ", "  true  "):
            with self.subTest(value=raw_value):
                with patch.dict(os.environ, {"RESEARCH_MOCK": raw_value}, clear=False):
                    self.assertTrue(is_research_mock_enabled())

    def test_is_research_mock_enabled_false(self):
        # Unset
        env_without_var = {
            key: value for key, value in os.environ.items() if key != "RESEARCH_MOCK"
        }
        with patch.dict(os.environ, env_without_var, clear=True):
            self.assertFalse(is_research_mock_enabled())

        # Explicit falsy values
        for raw_value in ("", "0", "false", "no", "off", "False", "NO", "  ", "maybe"):
            with self.subTest(value=raw_value):
                with patch.dict(os.environ, {"RESEARCH_MOCK": raw_value}, clear=False):
                    self.assertFalse(is_research_mock_enabled())


class RedditFetcherMockTests(unittest.TestCase):
    def setUp(self):
        reset_mock_log_state()

    def tearDown(self):
        reset_mock_log_state()

    def test_reddit_fetcher_returns_mock_when_enabled(self):
        with patch.dict(os.environ, {"RESEARCH_MOCK": "true"}, clear=False):
            # Construction must NOT require PRAW credentials when mock is on.
            diver = RedditDeepDiver(
                "election outcome 2026", subreddits=["politics", "economics"]
            )
            self.assertIsNone(
                diver.reddit,
                "reddit client must not be built in mock mode",
            )

            # Even if a deliberately-broken client is injected, the fetch path
            # must not call into it under mock mode.
            diver_with_explosive_client = RedditDeepDiver(
                "election outcome 2026",
                subreddits=["politics"],
                reddit_client=_ExplodingReddit(),
            )

            with self.assertLogs("research_mock", level=logging.INFO) as captured:
                context_default = diver.fetch_threads()
                context_alias = diver.fetch_discussion_context()
                _ = diver_with_explosive_client.fetch_threads()

        # All three calls returned the deterministic mock string.
        expected = build_mock_reddit_context(
            "election outcome 2026", subreddits=["politics", "economics"]
        )
        self.assertEqual(context_default, expected)
        self.assertEqual(context_alias, expected)
        self.assertIn("REDDIT DISCUSSION CONTEXT", context_default)
        self.assertIn("[MOCK DATA]", context_default)
        self.assertIn("election outcome 2026", context_default)

        # Log emitted exactly once across the multiple fetch calls.
        reddit_log_lines = [
            record
            for record in captured.records
            if "Reddit fetcher returning deterministic mock data" in record.getMessage()
        ]
        self.assertEqual(len(reddit_log_lines), 1)

    def test_reddit_fetcher_unchanged_when_mock_disabled(self):
        env_without_var = {
            key: value for key, value in os.environ.items() if key != "RESEARCH_MOCK"
        }
        with patch.dict(os.environ, env_without_var, clear=True):
            # Use an explicit fake reddit client to avoid touching PRAW.
            class FakeSubreddit:
                def search(self, *_args, **_kwargs):
                    return []

            class FakeReddit:
                def __init__(self):
                    self.targets = []

                def subreddit(self, target):
                    self.targets.append(target)
                    return FakeSubreddit()

            fake_reddit = FakeReddit()
            diver = RedditDeepDiver("standard query", reddit_client=fake_reddit)
            context = diver.fetch_threads()

        self.assertNotIn("[MOCK DATA]", context)
        self.assertIn("QUERY: standard query", context)
        self.assertEqual(fake_reddit.targets, ["all"])


class NewsFetcherMockTests(unittest.TestCase):
    def setUp(self):
        reset_mock_log_state()

    def tearDown(self):
        reset_mock_log_state()

    def test_news_fetcher_returns_mock_when_enabled(self):
        with patch.dict(os.environ, {"RESEARCH_MOCK": "yes"}, clear=False):
            fetcher = GoogleNewsRSSFetcher(
                "fed rate decision", feedparser_module=_ExplodingFeedParser()
            )

            with self.assertLogs("research_mock", level=logging.INFO) as captured:
                entries = fetcher.fetch_news()
                context = fetcher.fetch_news_context()
                also_context = fetcher.fetch_context()

        self.assertGreaterEqual(len(entries), 1)
        for entry in entries:
            self.assertIn("fed rate decision", entry["title"])
            self.assertTrue(entry["link"].startswith("https://example.com/mock-news/"))

        expected_context = build_mock_news_context("fed rate decision")
        self.assertEqual(context, expected_context)
        self.assertEqual(also_context, expected_context)
        self.assertIn("NEWS COVERAGE CONTEXT", context)
        self.assertIn("fed rate decision", context)

        news_log_lines = [
            record
            for record in captured.records
            if "News fetcher returning deterministic mock data" in record.getMessage()
        ]
        self.assertEqual(len(news_log_lines), 1)

    def test_news_fetcher_unchanged_when_mock_disabled(self):
        env_without_var = {
            key: value for key, value in os.environ.items() if key != "RESEARCH_MOCK"
        }
        with patch.dict(os.environ, env_without_var, clear=True):
            class CapturingFeedParser:
                def __init__(self):
                    self.calls = []

                def parse(self, url):
                    self.calls.append(url)
                    return type("Feed", (), {"entries": []})()

            fake = CapturingFeedParser()
            fetcher = GoogleNewsRSSFetcher("no mock query", feedparser_module=fake)
            entries = fetcher.fetch_news()

        # Real path was exercised: feedparser was called, no mock entries.
        self.assertEqual(entries, [])
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("no+mock+query", fake.calls[0])


class MockStringDeterminismTests(unittest.TestCase):
    def setUp(self):
        reset_mock_log_state()

    def tearDown(self):
        reset_mock_log_state()

    def test_mock_string_includes_search_query(self):
        # Reddit
        reddit_a = build_mock_reddit_context("alpha query", subreddits=["news"])
        reddit_b = build_mock_reddit_context("beta query", subreddits=["news"])

        self.assertIn("QUERY: alpha query", reddit_a)
        self.assertIn("alpha query", reddit_a)
        self.assertNotIn("alpha query", reddit_b)
        self.assertIn("QUERY: beta query", reddit_b)

        # Determinism: same inputs produce the same string twice in a row.
        self.assertEqual(
            build_mock_reddit_context("alpha query", subreddits=["news"]),
            reddit_a,
        )

        # News
        news_a = build_mock_news_context("alpha news")
        news_b = build_mock_news_context("beta news")

        self.assertIn("alpha news", news_a)
        self.assertNotIn("alpha news", news_b)
        self.assertIn("beta news", news_b)
        self.assertEqual(build_mock_news_context("alpha news"), news_a)


if __name__ == "__main__":
    unittest.main()
