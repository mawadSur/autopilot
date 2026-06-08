from __future__ import annotations

import logging
import os
from typing import Any, List, Literal, Optional

logger = logging.getLogger(__name__)


DEFAULT_MAX_RESULTS = 50

# Twitter/X's API requires paid access (Basic = $100/mo as of 2024+) and
# scraping twitter.com violates their ToS. We default to a deterministic mock
# payload so downstream agents (calibration, synthesis) get a stable structure
# without provisioning paid credentials. `api` mode is best-effort and falls
# back to mock if the optional `tweepy` dependency is missing or no bearer
# token is available.
try:  # pragma: no cover - exercised through patching in tests
    import tweepy  # type: ignore
except ImportError:  # pragma: no cover - exercised through patching in tests
    tweepy = None


class TwitterFetcher:
    def __init__(
        self,
        search_query: str,
        *,
        mode: Literal["mock", "api"] = "mock",
        bearer_token: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        query = (search_query or "").strip()
        if not query:
            raise ValueError("search_query must be a non-empty string")

        if mode not in ("mock", "api"):
            raise ValueError("mode must be either 'mock' or 'api'")

        self.search_query = query
        self.mode = mode
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self.max_results = max(1, int(max_results))

    def fetch_tweets(self) -> str:
        if self.mode == "api":
            api_context = self._fetch_via_api()
            if api_context is not None:
                return api_context
        return self._build_mock_context()

    def _fetch_via_api(self) -> Optional[str]:
        if tweepy is None:
            logger.warning(
                "tweepy is not installed; falling back to mock Twitter context for query=%r.",
                self.search_query,
            )
            return None
        if not self.bearer_token:
            logger.warning(
                "TWITTER_BEARER_TOKEN is missing; falling back to mock Twitter context for query=%r.",
                self.search_query,
            )
            return None

        try:
            client = tweepy.Client(bearer_token=self.bearer_token)
            response = client.search_recent_tweets(
                query=self.search_query,
                max_results=min(self.max_results, 100),
                tweet_fields=["public_metrics", "author_id", "created_at"],
            )
        except Exception as exc:  # pragma: no cover - real API failures only
            logger.warning(
                "Twitter API request failed (%s); falling back to mock context for query=%r.",
                exc,
                self.search_query,
            )
            return None

        tweets = list(getattr(response, "data", None) or [])
        if not tweets:
            logger.warning(
                "Twitter API returned no tweets for query=%r; falling back to mock context.",
                self.search_query,
            )
            return None

        return self._format_api_tweets(tweets)

    def _format_api_tweets(self, tweets: List[Any]) -> str:
        lines: List[str] = [
            "TWITTER DISCUSSION CONTEXT",
            f"QUERY: {self.search_query}",
            f"SOURCE: twitter-api-v2 ({len(tweets)} tweets)",
            "",
        ]
        for index, tweet in enumerate(tweets, start=1):
            metrics = getattr(tweet, "public_metrics", None) or {}
            likes = int(metrics.get("like_count", 0) or 0)
            retweets = int(metrics.get("retweet_count", 0) or 0)
            author_id = getattr(tweet, "author_id", "unknown")
            text = (getattr(tweet, "text", "") or "").replace("\n", " ").strip()
            lines.append(
                f"TWEET {index} | @{author_id} | likes={likes} | retweets={retweets}"
            )
            lines.append(f'"{text}"')
            if index < len(tweets):
                lines.append("")
        return "\n".join(lines)

    def _build_mock_context(self) -> str:
        query = self.search_query
        lines = [
            "TWITTER DISCUSSION CONTEXT",
            f"QUERY: {query}",
            "SOURCE: mock (no live Twitter/X API call)",
            "",
            "TWEET 1 | @analyst_one | likes=120 | retweets=40",
            f'"Mock tweet about {query}: signal strength looks moderate based on recent reporting."',
            "REPLIES:",
            '- @counterview | likes=22: "Counter-take: this is overhyped, the catalyst is already priced in."',
            '- @data_nerd | likes=14: "Worth noting the historical base rate for similar setups is closer to 35%."',
            "",
            "TWEET 2 | @market_watcher | likes=87 | retweets=12",
            f'"Following the {query} story closely; primary sources are split but lean cautiously bullish."',
            "REPLIES:",
            '- @longtimer | likes=9: "Disagree on the lean — see thread linking the latest filing."',
            "",
            "TWEET 3 | @quant_takes | likes=205 | retweets=66",
            f'"Quant view on {query}: implied probability looks slightly underpriced versus our model."',
            "REPLIES:",
            '- @skeptic42 | likes=31: "Your model has overfit on the last three analogues."',
            '- @analyst_one | likes=18: "Same direction here — corroborates the earlier signal."',
        ]
        return "\n".join(lines)
