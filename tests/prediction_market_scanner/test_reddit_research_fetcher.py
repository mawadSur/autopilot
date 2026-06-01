import logging
import os
import unittest
from unittest.mock import patch

from reddit_research_agent import fetcher as reddit_fetcher
from reddit_research_agent.fetcher import (
    RedditDeepDiver,
    _reset_missing_credentials_log_state,
)


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeComment:
    def __init__(self, comment_id, body, score, depth=0, author="commenter"):
        self.id = comment_id
        self.body = body
        self.score = score
        self.depth = depth
        self.author = FakeAuthor(author)


class FakeCommentForest:
    def __init__(self, comments):
        self._comments = list(comments)
        self.replace_more_calls = []

    def replace_more(self, limit=0):
        self.replace_more_calls.append(limit)

    def list(self):
        return list(self._comments)


class FakeSubreddit:
    def __init__(self, submissions):
        self.submissions = list(submissions)
        self.search_calls = []

    def search(self, query, sort=None, time_filter=None, limit=None):
        self.search_calls.append(
            {
                "query": query,
                "sort": sort,
                "time_filter": time_filter,
                "limit": limit,
            }
        )
        return list(self.submissions)[:limit]


class FakeReddit:
    def __init__(self, subreddit):
        self._subreddit = subreddit
        self.targets = []

    def subreddit(self, target):
        self.targets.append(target)
        return self._subreddit


class FakeSubmission:
    def __init__(self, *, title, selftext, score, num_comments, subreddit, author, comments):
        self.title = title
        self.selftext = selftext
        self.score = score
        self.num_comments = num_comments
        self.subreddit = type("Subreddit", (), {"display_name": subreddit})()
        self.author = FakeAuthor(author)
        self.comments = FakeCommentForest(comments)


class RedditDeepDiverTests(unittest.TestCase):
    def test_fetch_discussion_context_defaults_to_all_and_formats_reply_chains(self):
        submission = FakeSubmission(
            title="Will event X happen?",
            selftext="Thread starter context with a few grounded details for the LLM.",
            score=410,
            num_comments=3,
            subreddit="PredictionMarkets",
            author="submitter",
            comments=[
                FakeComment("c1", "Top-level thesis with evidence and enough words to matter.", 85, depth=0, author="alpha"),
                FakeComment("c2", "Reply chain adds nuance and references a source.", 31, depth=1, author="beta"),
                FakeComment("c3", "Another top-level counterpoint.", 14, depth=0, author="gamma"),
            ],
        )
        reddit = FakeReddit(FakeSubreddit([submission]))

        diver = RedditDeepDiver("event x", reddit_client=reddit)
        context = diver.fetch_discussion_context()

        self.assertEqual(reddit.targets, ["all"])
        self.assertIn("REDDIT DISCUSSION CONTEXT", context)
        self.assertIn("QUERY: event x", context)
        self.assertIn("THREAD 1 | r/PredictionMarkets | score=410 | comments=3 | author=u/submitter", context)
        self.assertIn("TOP COMMENTS:", context)
        self.assertIn("- u/alpha | score=85 [signal]:", context)
        self.assertIn("  - -> u/beta | score=31:", context)
        self.assertEqual(submission.comments.replace_more_calls, [0])

    def test_fetch_discussion_context_joins_subreddits_and_prioritizes_long_or_high_score_comments(self):
        filler_comments = [
            FakeComment(f"f{index}", f"short filler {index}", 0, depth=0, author=f"filler{index}")
            for index in range(16)
        ]
        long_comment = FakeComment(
            "long1",
            " ".join(["Detailed"] * 180),
            1,
            depth=0,
            author="deepdive",
        )
        high_score_comment = FakeComment(
            "hot1",
            "Short but heavily upvoted comment.",
            220,
            depth=0,
            author="signalboost",
        )
        submission = FakeSubmission(
            title="Market discussion",
            selftext="Submission body",
            score=120,
            num_comments=18,
            subreddit="Politics",
            author="poster",
            comments=filler_comments + [long_comment, high_score_comment],
        )
        subreddit = FakeSubreddit([submission])
        reddit = FakeReddit(subreddit)

        diver = RedditDeepDiver(
            "market discussion",
            subreddits=["politics", "economics"],
            reddit_client=reddit,
        )
        context = diver.fetch_discussion_context()

        self.assertEqual(reddit.targets, ["politics+economics"])
        self.assertEqual(
            subreddit.search_calls,
            [{"query": "market discussion", "sort": "relevance", "time_filter": "month", "limit": 5}],
        )
        self.assertIn("u/deepdive | score=1 [signal]:", context)
        self.assertIn("u/signalboost | score=220 [signal]:", context)
        self.assertNotIn("u/filler15", context)

    def test_fetch_threads_alias_matches_discussion_context(self):
        submission = FakeSubmission(
            title="Alias check",
            selftext="Body",
            score=10,
            num_comments=1,
            subreddit="TestSub",
            author="poster",
            comments=[FakeComment("c1", "Useful comment", 12, depth=0, author="alpha")],
        )
        reddit = FakeReddit(FakeSubreddit([submission]))
        diver = RedditDeepDiver("alias query", reddit_client=reddit)

        self.assertEqual(diver.fetch_threads(), diver.fetch_discussion_context())


class GracefulDegradationTests(unittest.TestCase):
    """Behaviour when REDDIT_CLIENT_ID is absent (Devvit-MCP-driven workflows).

    With no PRAW credentials AND no explicit ``RESEARCH_MOCK`` toggle, the
    fetcher should quietly degrade to deterministic mock data and surface a
    one-time INFO log pointing the operator at the /reddit-research skill.
    Legacy users with credentials set keep getting the live PRAW path.
    """

    def setUp(self):
        _reset_missing_credentials_log_state()

    def tearDown(self):
        _reset_missing_credentials_log_state()

    def _env_without_reddit_or_mock(self) -> dict:
        return {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "REDDIT_CLIENT_ID",
                "REDDIT_CLIENT_SECRET",
                "REDDIT_USER_AGENT",
                "RESEARCH_MOCK",
            }
        }

    def test_no_reddit_client_id_falls_back_to_mock(self):
        with patch.dict(
            os.environ, self._env_without_reddit_or_mock(), clear=True
        ):
            with self.assertLogs(
                "reddit_research_agent.fetcher", level=logging.INFO
            ) as captured:
                diver = RedditDeepDiver(
                    "polymarket sample question",
                    subreddits=["politics", "news"],
                )
                self.assertIsNone(
                    diver.reddit,
                    "reddit client must not be built when credentials are missing",
                )

                context = diver.fetch_threads()

        # Mock context shape and content checks.
        self.assertIn("REDDIT DISCUSSION CONTEXT", context)
        self.assertIn("[MOCK DATA]", context)
        self.assertIn("polymarket sample question", context)
        self.assertIn("SUBREDDITS: politics+news", context)

        # Exactly one INFO log mentioning the /reddit-research skill.
        skill_log_lines = [
            record.getMessage()
            for record in captured.records
            if "/reddit-research skill" in record.getMessage()
        ]
        self.assertEqual(
            len(skill_log_lines),
            1,
            f"expected exactly one skill-pointer log, got: {skill_log_lines}",
        )
        self.assertIn("REDDIT_CLIENT_ID not set", skill_log_lines[0])
        self.assertIn("Devvit MCP", skill_log_lines[0])

    def test_existing_credentials_still_use_praw(self):
        """When REDDIT_CLIENT_ID is set, the real PRAW path is taken."""

        captured_kwargs = {}

        class FakePraw:
            class Reddit:
                def __init__(self, **kwargs):
                    captured_kwargs.update(kwargs)
                    # Mark this instance so the test can identify it later.
                    self._is_fake_praw = True

        env = self._env_without_reddit_or_mock()
        env.update(
            {
                "REDDIT_CLIENT_ID": "fake_client_id",
                "REDDIT_CLIENT_SECRET": "fake_client_secret",
                "REDDIT_USER_AGENT": "fake-user-agent/1.0",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            with patch.object(reddit_fetcher, "praw", FakePraw):
                diver = RedditDeepDiver(
                    "credentialed query", subreddits=["politics"]
                )

        # The constructor should have built a PRAW client (not None).
        self.assertIsNotNone(diver.reddit)
        self.assertTrue(getattr(diver.reddit, "_is_fake_praw", False))

        # PRAW received the env-derived configuration.
        self.assertEqual(captured_kwargs.get("client_id"), "fake_client_id")
        self.assertEqual(
            captured_kwargs.get("client_secret"), "fake_client_secret"
        )
        self.assertEqual(
            captured_kwargs.get("user_agent"), "fake-user-agent/1.0"
        )


if __name__ == "__main__":
    unittest.main()
