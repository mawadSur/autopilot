from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Sequence, Set

try:
    import praw
except ImportError:  # pragma: no cover - exercised through injected fake clients in tests
    praw = None


DEFAULT_SUBREDDIT = "all"
DEFAULT_THREAD_LIMIT = 5
DEFAULT_COMMENT_LIMIT = 15


def _normalize_text(value: str, *, max_chars: int) -> str:
    collapsed = " ".join((value or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _author_name(author: Any) -> str:
    if author is None:
        return "[deleted]"
    return getattr(author, "name", str(author))


class RedditDeepDiver:
    def __init__(
        self,
        search_query: str,
        *,
        subreddits: Optional[Sequence[str]] = None,
        reddit_client: Any | None = None,
        thread_limit: int = DEFAULT_THREAD_LIMIT,
        comment_limit: int = DEFAULT_COMMENT_LIMIT,
    ) -> None:
        query = (search_query or "").strip()
        if not query:
            raise ValueError("search_query must be a non-empty string")

        if isinstance(subreddits, str):
            normalized_subreddits = [subreddits]
        else:
            normalized_subreddits = list(subreddits or [DEFAULT_SUBREDDIT])

        self.search_query = query
        self.subreddits = normalized_subreddits
        self.thread_limit = max(1, int(thread_limit))
        self.comment_limit = max(1, int(comment_limit))
        self.reddit = reddit_client if reddit_client is not None else self._build_reddit_client()

    def fetch_threads(self) -> str:
        return self.fetch_discussion_context()

    def fetch_discussion_context(self) -> str:
        submissions = list(self._search_submissions())
        lines = [
            "REDDIT DISCUSSION CONTEXT",
            f"QUERY: {self.search_query}",
            f"SUBREDDITS: {self._subreddit_target()}",
            "",
        ]

        if not submissions:
            lines.append("No relevant Reddit submissions found in the past month.")
            return "\n".join(lines)

        for index, submission in enumerate(submissions, start=1):
            lines.extend(self._format_submission(submission, index=index))
            if index < len(submissions):
                lines.append("")
        return "\n".join(lines)

    def _build_reddit_client(self) -> Any:
        if praw is None:
            raise RuntimeError(
                "praw is required to fetch Reddit discussion context. Install dependencies first."
            )

        client_id = (os.getenv("REDDIT_CLIENT_ID") or "").strip()
        client_secret = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
        user_agent = (os.getenv("REDDIT_USER_AGENT") or "autopilot-reddit-research-agent/1.0").strip()

        if not client_id:
            raise RuntimeError(
                "Missing Reddit API configuration. Set REDDIT_CLIENT_ID before constructing RedditDeepDiver."
            )

        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def _subreddit_target(self) -> str:
        cleaned = [subreddit.strip() for subreddit in self.subreddits if subreddit and subreddit.strip()]
        if not cleaned or DEFAULT_SUBREDDIT in {item.lower() for item in cleaned}:
            return DEFAULT_SUBREDDIT
        return "+".join(cleaned)

    def _search_submissions(self) -> Iterable[Any]:
        subreddit = self.reddit.subreddit(self._subreddit_target())
        return subreddit.search(
            self.search_query,
            sort="relevance",
            time_filter="month",
            limit=self.thread_limit,
        )

    def _format_submission(self, submission: Any, *, index: int) -> List[str]:
        subreddit_name = getattr(getattr(submission, "subreddit", None), "display_name", "unknown")
        title = _normalize_text(getattr(submission, "title", ""), max_chars=220)
        body = _normalize_text(getattr(submission, "selftext", ""), max_chars=700)
        score = int(getattr(submission, "score", 0) or 0)
        num_comments = int(getattr(submission, "num_comments", 0) or 0)
        author = _author_name(getattr(submission, "author", None))

        lines = [
            f"THREAD {index} | r/{subreddit_name} | score={score} | comments={num_comments} | author=u/{author}",
            f"TITLE: {title or '[no title]'}",
        ]
        if body:
            lines.append(f"POST: {body}")

        comment_lines = self._format_top_comments(submission)
        if comment_lines:
            lines.append("TOP COMMENTS:")
            lines.extend(comment_lines)
        else:
            lines.append("TOP COMMENTS: [none retained]")
        return lines

    def _format_top_comments(self, submission: Any) -> List[str]:
        comments = getattr(submission, "comments", None)
        if comments is None:
            return []

        if hasattr(comments, "replace_more"):
            comments.replace_more(limit=0)

        if hasattr(comments, "list"):
            flattened = list(comments.list())
        else:
            flattened = list(comments)

        filtered_comments = [comment for comment in flattened if self._is_useful_comment(comment)]
        if not filtered_comments:
            return []

        selected_ids = self._select_comment_ids(filtered_comments)
        lines: List[str] = []
        for comment in filtered_comments:
            comment_id = getattr(comment, "id", None)
            if comment_id not in selected_ids:
                continue
            lines.append(self._comment_to_line(comment))
        return lines

    def _select_comment_ids(self, comments: Sequence[Any]) -> Set[str]:
        ranked = sorted(comments, key=self._comment_priority, reverse=True)
        return {getattr(comment, "id", "") for comment in ranked[: self.comment_limit]}

    def _comment_priority(self, comment: Any) -> float:
        score = float(getattr(comment, "score", 0) or 0)
        body = getattr(comment, "body", "") or ""
        word_count = len(body.split())
        return score + min(word_count / 12.0, 15.0)

    def _is_useful_comment(self, comment: Any) -> bool:
        body = (getattr(comment, "body", "") or "").strip()
        if not body or body in {"[deleted]", "[removed]"}:
            return False
        return True

    def _comment_to_line(self, comment: Any) -> str:
        depth = max(0, int(getattr(comment, "depth", 0) or 0))
        score = int(getattr(comment, "score", 0) or 0)
        author = _author_name(getattr(comment, "author", None))
        raw_body = getattr(comment, "body", "") or ""
        normalized_body = _normalize_text(raw_body, max_chars=360)
        word_count = len(raw_body.split())
        emphasis = ""
        if score >= 50 or word_count >= 80:
            emphasis = " [signal]"
        indent = "  " * min(depth, 4)
        branch = "-> " if depth else ""
        return f"{indent}- {branch}u/{author} | score={score}{emphasis}: {normalized_body}"
