from __future__ import annotations

from typing import Any, List

from news_research_agent.models import NewsResearchReport


class NewsAgent:
    def analyze_news(
        self,
        *,
        market_title: str,
        implied_prob: float,
        news_context: str,
        **_: Any,
    ) -> NewsResearchReport:
        title = str(market_title or "").strip()
        context = str(news_context or "").strip()
        probability = float(implied_prob)

        if not title:
            raise ValueError("market_title must be a non-empty string")
        if not context:
            raise ValueError("news_context must be a non-empty string")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("implied_prob must be between 0.0 and 1.0")

        timeline: List[str] = []
        key_facts: List[str] = []
        current_article: str | None = None
        for line in context.splitlines():
            stripped = line.strip()
            if stripped.startswith("ARTICLE "):
                article_body = stripped.split("|", 1)[1].strip() if "|" in stripped else stripped
                current_article = article_body
                timeline.append(article_body)
                continue
            if stripped.startswith("Summary:"):
                summary = stripped.removeprefix("Summary:").strip()
                if summary:
                    key_facts.append(summary)
                    continue
            if stripped.startswith("Link:"):
                continue
            if current_article and stripped:
                key_facts.append(stripped)

        timeline = timeline[:5]
        key_facts = key_facts[:5]
        article_count = max(len(timeline), 1)
        source_quality = min(10, 2 + article_count)
        if timeline:
            summary = (
                f"Google News RSS returned {article_count} recent articles related to {title}. "
                f"The feed centers on: {timeline[0]}."
            )
        else:
            summary = f"No recent Google News RSS articles were found for {title}."
        return NewsResearchReport(
            timeline=timeline,
            key_facts=key_facts,
            source_quality_score=source_quality,
            summary=summary,
        )
