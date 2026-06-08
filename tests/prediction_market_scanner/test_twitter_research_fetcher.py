import logging
import unittest
from unittest import mock

import twitter_research_agent.fetcher as fetcher_module
from twitter_research_agent.fetcher import TwitterFetcher


class TwitterFetcherMockModeTests(unittest.TestCase):
    def test_mock_mode_returns_formatted_string(self):
        fetcher = TwitterFetcher("election outcome")
        context = fetcher.fetch_tweets()

        self.assertIn("TWITTER DISCUSSION CONTEXT", context)
        self.assertIn("QUERY: election outcome", context)
        self.assertIn("SOURCE: mock", context)
        self.assertIn("TWEET 1 | @analyst_one | likes=120 | retweets=40", context)
        self.assertIn("REPLIES:", context)
        self.assertIn('@counterview', context)

    def test_mock_mode_honors_search_query(self):
        fetcher = TwitterFetcher("Federal Reserve cut")
        context = fetcher.fetch_tweets()

        self.assertIn("QUERY: Federal Reserve cut", context)
        self.assertIn("Federal Reserve cut", context)

    def test_constructor_rejects_blank_search_query(self):
        with self.assertRaises(ValueError):
            TwitterFetcher("   ")

    def test_constructor_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            TwitterFetcher("query", mode="scrape")  # type: ignore[arg-type]


class TwitterFetcherApiModeTests(unittest.TestCase):
    def test_api_mode_without_bearer_token_logs_warning_and_falls_back(self):
        with mock.patch.object(fetcher_module, "tweepy", object()), \
             mock.patch.dict("os.environ", {}, clear=False):
            # Make sure no env-provided token leaks in.
            fetcher_module.os.environ.pop("TWITTER_BEARER_TOKEN", None)
            fetcher = TwitterFetcher("regulatory ruling", mode="api")

            with self.assertLogs(fetcher_module.logger, level="WARNING") as captured:
                context = fetcher.fetch_tweets()

        self.assertTrue(
            any("TWITTER_BEARER_TOKEN" in record for record in captured.output),
            captured.output,
        )
        self.assertIn("SOURCE: mock", context)
        self.assertIn("QUERY: regulatory ruling", context)

    def test_api_mode_without_tweepy_logs_warning_and_falls_back(self):
        with mock.patch.object(fetcher_module, "tweepy", None):
            fetcher = TwitterFetcher(
                "rate decision",
                mode="api",
                bearer_token="dummy-token",
            )

            with self.assertLogs(fetcher_module.logger, level="WARNING") as captured:
                context = fetcher.fetch_tweets()

        self.assertTrue(
            any("tweepy is not installed" in record for record in captured.output),
            captured.output,
        )
        self.assertIn("SOURCE: mock", context)
        self.assertIn("QUERY: rate decision", context)

    def test_api_mode_with_token_and_tweepy_uses_api_results(self):
        class _FakeTweet:
            def __init__(self, text, likes, retweets, author_id):
                self.text = text
                self.public_metrics = {
                    "like_count": likes,
                    "retweet_count": retweets,
                }
                self.author_id = author_id
                self.created_at = "2026-01-01T00:00:00Z"

        class _FakeResponse:
            def __init__(self, tweets):
                self.data = tweets

        class _FakeClient:
            def __init__(self, *, bearer_token):
                self.bearer_token = bearer_token
                self.calls = []

            def search_recent_tweets(self, query, *, max_results, tweet_fields):
                self.calls.append(
                    {"query": query, "max_results": max_results, "tweet_fields": tweet_fields}
                )
                return _FakeResponse(
                    [
                        _FakeTweet("Primary-source link supports YES.", 50, 10, "user_alpha"),
                        _FakeTweet("Counter-take from credentialed analyst.", 33, 4, "user_beta"),
                    ]
                )

        captured_clients = []

        class _FakeTweepy:
            def __init__(self):
                pass

            def Client(self, *, bearer_token):
                client = _FakeClient(bearer_token=bearer_token)
                captured_clients.append(client)
                return client

        fake_tweepy = _FakeTweepy()
        with mock.patch.object(fetcher_module, "tweepy", fake_tweepy):
            fetcher = TwitterFetcher(
                "supreme court ruling",
                mode="api",
                bearer_token="real-token",
                max_results=200,
            )
            context = fetcher.fetch_tweets()

        self.assertEqual(len(captured_clients), 1)
        self.assertEqual(captured_clients[0].bearer_token, "real-token")
        # max_results clamped to 100 by Twitter API limit.
        self.assertEqual(captured_clients[0].calls[0]["max_results"], 100)
        self.assertIn("SOURCE: twitter-api-v2 (2 tweets)", context)
        self.assertIn("@user_alpha", context)
        self.assertIn("Primary-source link supports YES.", context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
