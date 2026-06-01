from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import requests

from models import SocialPost


URL_PATTERN = re.compile(r"https?://[^\s\]\[\)\(<>\"']+")
QUOTE_LINE_PATTERN = re.compile(r"(?m)^\s*>")
DEFAULT_TIMEOUT_S = 15
DEFAULT_MAX_POSTS = 10
DEFAULT_MAX_COMMENTS_PER_POST = 8
DEFAULT_USER_AGENT = "social-narrative-agent/1.0"


class SocialAggregator:
    SEARCH_URL = "https://www.reddit.com/search.json"
    REDDIT_BASE_URL = "https://www.reddit.com"

    def __init__(
        self,
        topic_query: str,
        *,
        session: Optional[requests.Session] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        max_posts: int = DEFAULT_MAX_POSTS,
        max_comments_per_post: int = DEFAULT_MAX_COMMENTS_PER_POST,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        normalized_topic = str(topic_query or "").strip()
        if not normalized_topic:
            raise ValueError("topic_query must be a non-empty string")
        self.topic_query = normalized_topic
        self.session = session or requests.Session()
        self.timeout_s = int(timeout_s)
        self.max_posts = max(1, int(max_posts))
        self.max_comments_per_post = max(1, int(max_comments_per_post))
        self.user_agent = str(user_agent or DEFAULT_USER_AGENT).strip() or DEFAULT_USER_AGENT

    def fetch_reddit_threads(self, topic: Optional[str] = None, *, limit: int = 50) -> List[SocialPost]:
        normalized_topic = str(topic or self.topic_query).strip()
        if not normalized_topic:
            raise ValueError("topic must be a non-empty string")
        max_items = max(0, int(limit))
        if max_items == 0:
            return []

        submissions = self._search_reddit_posts(normalized_topic, limit=min(self.max_posts, max_items))
        posts: List[SocialPost] = []
        for submission in submissions:
            if len(posts) >= max_items:
                break
            posts.append(self._social_post_from_submission(submission))
            remaining = max_items - len(posts)
            if remaining <= 0:
                break
            comment_budget = min(
                remaining,
                self._comment_budget(max_items=max_items, submission_count=len(submissions)),
            )
            posts.extend(self._fetch_submission_comments(submission, max_count=comment_budget))
        return posts[:max_items]

    def format_for_llm(self, posts: List[SocialPost]) -> str:
        filtered_posts = [post for post in posts if post.engagement_score > 0 and post.text.strip()]
        if not filtered_posts:
            return f"TOPIC: {self.topic_query}\nNo engaged posts found."

        high_engagement_threshold = self._high_engagement_threshold(filtered_posts)
        lines = [f"TOPIC: {self.topic_query}"]
        thread_index = 0
        for post in filtered_posts:
            engagement_marker = " HOT" if post.engagement_score >= high_engagement_threshold else ""
            quote_marker = " QUOTE" if post.is_quote else ""
            domains = self._format_link_domains(post.linked_urls)
            compact_text = self._compact_text(post.text, limit=220 if not post.is_reply else 160)
            if not post.is_reply:
                thread_index += 1
                lines.append(
                    f"T{thread_index}{engagement_marker} | post | {post.engagement_score} | {post.author_id} | {compact_text}{domains}"
                )
            else:
                lines.append(
                    f"  ->{engagement_marker}{quote_marker} | {post.engagement_score} | {post.author_id} | {compact_text}{domains}"
                )
        return "\n".join(lines)

    def _comment_budget(self, *, max_items: int, submission_count: int) -> int:
        if submission_count <= 0:
            return self.max_comments_per_post
        balanced_budget = max(1, math.ceil(max(0, max_items - submission_count) / submission_count))
        return max(self.max_comments_per_post, balanced_budget)

    def _search_reddit_posts(self, topic: str, *, limit: int) -> List[Dict[str, Any]]:
        payload = self._request_json(
            self.SEARCH_URL,
            params={
                "q": topic,
                "sort": "top",
                "t": "month",
                "type": "link",
                "limit": limit,
                "raw_json": 1,
            },
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        children = data.get("children") if isinstance(data, dict) else None
        if not isinstance(children, list):
            return []
        submissions: List[Dict[str, Any]] = []
        for child in children:
            if not isinstance(child, dict) or child.get("kind") != "t3":
                continue
            child_data = child.get("data")
            if isinstance(child_data, dict):
                submissions.append(child_data)
        return submissions

    def _fetch_submission_comments(self, submission: Dict[str, Any], *, max_count: int) -> List[SocialPost]:
        permalink = str(submission.get("permalink") or "").strip()
        if not permalink or max_count <= 0:
            return []
        comments_url = f"{self.REDDIT_BASE_URL}{permalink.rstrip('/')}" + ".json"
        payload = self._request_json(
            comments_url,
            params={
                "raw_json": 1,
                "sort": "top",
                "depth": 3,
                "limit": max_count,
            },
        )
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        comments_listing = payload[1]
        listing_data = comments_listing.get("data") if isinstance(comments_listing, dict) else None
        children = listing_data.get("children") if isinstance(listing_data, dict) else None
        if not isinstance(children, list):
            return []

        comments: List[SocialPost] = []
        self._walk_comment_tree(children, comments=comments, max_count=max_count)
        return comments[:max_count]

    def _walk_comment_tree(
        self,
        children: Iterable[Any],
        *,
        comments: List[SocialPost],
        max_count: int,
    ) -> None:
        for child in children:
            if len(comments) >= max_count:
                return
            if not isinstance(child, dict) or child.get("kind") != "t1":
                continue
            data = child.get("data")
            if not isinstance(data, dict):
                continue
            body = self._normalize_whitespace(str(data.get("body") or ""))
            if body in {"", "[deleted]", "[removed]"}:
                continue
            comments.append(
                SocialPost(
                    platform="reddit",
                    author_id=str(data.get("author") or "[deleted]"),
                    text=body,
                    is_reply=True,
                    is_quote=bool(QUOTE_LINE_PATTERN.search(body)),
                    linked_urls=self._extract_urls(body),
                    engagement_score=max(0, int(data.get("score") or 0)),
                )
            )
            replies = data.get("replies")
            if isinstance(replies, dict):
                reply_data = replies.get("data")
                reply_children = reply_data.get("children") if isinstance(reply_data, dict) else None
                if isinstance(reply_children, list):
                    self._walk_comment_tree(reply_children, comments=comments, max_count=max_count)

    def _social_post_from_submission(self, submission: Dict[str, Any]) -> SocialPost:
        text = self._build_submission_text(submission)
        linked_urls = self._extract_submission_urls(submission, text=text)
        return SocialPost(
            platform="reddit",
            author_id=str(submission.get("author") or "[deleted]"),
            text=text,
            is_reply=False,
            is_quote=False,
            linked_urls=linked_urls,
            engagement_score=max(0, int(submission.get("score") or 0)),
        )

    def _request_json(self, url: str, *, params: Dict[str, Any]) -> Any:
        response = self.session.get(
            url,
            params=params,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def _build_submission_text(self, submission: Dict[str, Any]) -> str:
        title = self._normalize_whitespace(str(submission.get("title") or ""))
        selftext = self._normalize_whitespace(str(submission.get("selftext") or ""))
        if title and selftext:
            return f"{title} :: {self._compact_text(selftext, limit=240)}"
        if title:
            return title
        if selftext:
            return self._compact_text(selftext, limit=240)
        url = str(submission.get("url_overridden_by_dest") or submission.get("url") or "").strip()
        return url or "[link post]"

    def _extract_submission_urls(self, submission: Dict[str, Any], *, text: str) -> List[str]:
        urls = list(self._extract_urls(text))
        candidate = str(submission.get("url_overridden_by_dest") or submission.get("url") or "").strip()
        if candidate.startswith("http") and "reddit.com" not in urlparse(candidate).netloc.lower():
            urls.append(candidate)
        return self._dedupe_urls(urls)

    def _extract_urls(self, text: str) -> List[str]:
        return self._dedupe_urls(match.group(0).rstrip('.,!?)]') for match in URL_PATTERN.finditer(text or ""))

    def _dedupe_urls(self, urls: Iterable[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for url in urls:
            normalized = str(url or "").strip()
            parsed = urlparse(normalized)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _format_link_domains(self, urls: Sequence[Any]) -> str:
        domains: List[str] = []
        for url in urls[:2]:
            parsed = urlparse(str(url))
            domain = parsed.netloc.lower().removeprefix("www.")
            if domain and domain not in domains:
                domains.append(domain)
        if not domains:
            return ""
        return f" | links:{','.join(domains)}"

    def _high_engagement_threshold(self, posts: Sequence[SocialPost]) -> int:
        scores = sorted(post.engagement_score for post in posts if post.engagement_score > 0)
        if not scores:
            return 1
        percentile_index = max(0, math.ceil(len(scores) * 0.8) - 1)
        return scores[percentile_index]

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _compact_text(text: str, *, limit: int) -> str:
        normalized = SocialAggregator._normalize_whitespace(text)
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."
