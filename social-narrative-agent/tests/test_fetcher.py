import sys
import unittest
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetcher import SocialAggregator
from models import SocialPost


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        for key, payload in self.responses.items():
            if key in url:
                return FakeResponse(payload)
        raise AssertionError(f"Unexpected URL: {url}")


class SocialAggregatorTests(unittest.TestCase):
    def test_fetch_reddit_threads_returns_posts_and_comments(self):
        search_payload = {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "title": "GPT-5 release date speculation",
                            "selftext": "People think it lands this quarter.",
                            "author": "alpha",
                            "score": 120,
                            "permalink": "/r/OpenAI/comments/abc123/gpt5_release_date_speculation/",
                            "url_overridden_by_dest": "https://openai.com/index/hello-gpt-5",
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "title": "Second thread",
                            "selftext": "Lower priority chatter.",
                            "author": "echo",
                            "score": 15,
                            "permalink": "/r/OpenAI/comments/def456/second_thread/",
                            "url": "https://reddit.com/r/OpenAI/comments/def456/second_thread/",
                        },
                    },
                ]
            }
        }
        comments_payload_one = [
            {"data": {"children": []}},
            {
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "author": "bravo",
                                "body": "Could be announced at the keynote.",
                                "score": 45,
                                "replies": "",
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "author": "charlie",
                                "body": "> prior leak\nNo official date yet https://example.com/leak",
                                "score": 32,
                                "replies": {
                                    "data": {
                                        "children": [
                                            {
                                                "kind": "t1",
                                                "data": {
                                                    "author": "delta",
                                                    "body": "Fair point.",
                                                    "score": 8,
                                                    "replies": "",
                                                },
                                            }
                                        ]
                                    }
                                },
                            },
                        },
                    ]
                }
            },
        ]
        comments_payload_two = [
            {"data": {"children": []}},
            {
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "author": "foxtrot",
                                "body": "This one has less traction.",
                                "score": 3,
                                "replies": "",
                            },
                        }
                    ]
                }
            },
        ]
        session = FakeSession(
            {
                "search.json": search_payload,
                "abc123": comments_payload_one,
                "def456": comments_payload_two,
            }
        )
        aggregator = SocialAggregator("GPT-5 release date", session=session, max_posts=5, max_comments_per_post=4)

        posts = aggregator.fetch_reddit_threads(limit=5)

        self.assertEqual(len(posts), 5)
        self.assertFalse(posts[0].is_reply)
        self.assertEqual(posts[0].platform, "reddit")
        self.assertEqual(posts[0].author_id, "alpha")
        self.assertIn("openai.com", str(posts[0].linked_urls[0]))
        self.assertTrue(posts[2].is_quote)
        self.assertIn("example.com", str(posts[2].linked_urls[0]))
        self.assertEqual(session.calls[0]["params"]["q"], "GPT-5 release date")
        self.assertEqual(session.calls[0]["headers"]["User-Agent"], "social-narrative-agent/1.0")

    def test_format_for_llm_filters_zero_engagement_and_preserves_thread_shape(self):
        aggregator = SocialAggregator("Election winner")
        posts = [
            SocialPost(
                platform="reddit",
                author_id="alpha",
                text="Base thread on the race.",
                is_reply=False,
                is_quote=False,
                linked_urls=["https://example.com/article"],
                engagement_score=100,
            ),
            SocialPost(
                platform="reddit",
                author_id="bravo",
                text="Noise with no traction.",
                is_reply=True,
                is_quote=False,
                linked_urls=[],
                engagement_score=0,
            ),
            SocialPost(
                platform="reddit",
                author_id="charlie",
                text="> Poll leak\nMomentum is shifting.",
                is_reply=True,
                is_quote=True,
                linked_urls=["https://news.example.com/poll"],
                engagement_score=40,
            ),
        ]

        formatted = aggregator.format_for_llm(posts)

        self.assertIn("TOPIC: Election winner", formatted)
        self.assertIn("T1 HOT | post | 100 | alpha | Base thread on the race.", formatted)
        self.assertIn("-> QUOTE | 40 | charlie | > Poll leak Momentum is shifting. | links:news.example.com", formatted)
        self.assertNotIn("Noise with no traction.", formatted)

    def test_format_for_llm_returns_no_engaged_posts_message(self):
        aggregator = SocialAggregator("GPT-5 release date")
        formatted = aggregator.format_for_llm([])
        self.assertEqual(formatted, "TOPIC: GPT-5 release date\nNo engaged posts found.")


if __name__ == "__main__":
    unittest.main()
