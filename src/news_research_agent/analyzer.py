from __future__ import annotations

from typing import Any, List

from news_research_agent.models import NewsResearchReport


SYSTEM_PROMPT = (
    "You are a news research agent for prediction markets. Given a market title, the "
    "current implied probability, and a packet of recent Google News RSS articles, "
    "extract structured signal that downstream calibration and risk agents can consume.\n\n"
    "Return ONLY a JSON object that matches the required NewsResearchReport schema with "
    "ALL of the following fields populated:\n"
    "- timeline: chronological list of headline strings (max 5).\n"
    "- key_facts: short factual claims (max 5, non-empty strings).\n"
    "- source_quality_score: legacy int 0-10 reflecting source reputation.\n"
    "- bullish_thesis: non-empty string explaining the strongest case the YES side resolves.\n"
    "- bearish_thesis: non-empty string explaining the strongest case the NO side resolves.\n"
    "- evidence_quality_score: int 0-100; higher = stronger primary sources.\n"
    "- misinformation_risk_score: int 0-100; higher = higher misinformation risk.\n"
    "- sentiment_score: int -100..100; negative=bearish, positive=bullish, 0=neutral.\n"
    "- key_sources: list of up to 10 URLs or named outlets cited in the analysis.\n"
    "- summary: non-empty string summarizing the news landscape.\n"
    "Be calibrated, not promotional. If evidence is thin, lower confidence scores accordingly."
)


def _build_user_prompt(market_title: str, implied_prob: float, news_context: str) -> str:
    return (
        "Analyze the following News Coverage Context and return only a JSON object that "
        "matches the required NewsResearchReport schema with ALL fields populated.\n\n"
        f"Market title: {market_title}\n"
        f"Implied probability: {implied_prob:.6f}\n\n"
        "News Coverage Context:\n"
        f"{news_context}"
    )


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
        key_sources: List[str] = []
        current_article: str | None = None
        for line in context.splitlines():
            stripped = line.strip()
            if stripped.startswith("ARTICLE "):
                article_body = stripped.split("|", 1)[1].strip() if "|" in stripped else stripped
                current_article = article_body
                timeline.append(article_body)
                continue
            if stripped.startswith("Summary:"):
                summary_line = stripped.removeprefix("Summary:").strip()
                if summary_line:
                    key_facts.append(summary_line)
                    continue
            if stripped.startswith("Link:"):
                link_value = stripped.removeprefix("Link:").strip()
                if link_value and len(key_sources) < 10:
                    key_sources.append(link_value)
                continue
            if current_article and stripped:
                key_facts.append(stripped)

        timeline = timeline[:5]
        key_facts = key_facts[:5]
        article_count = max(len(timeline), 1)
        source_quality = min(10, 2 + article_count)
        evidence_quality_score = min(100, 20 + article_count * 15)
        # Without an LLM in this deterministic stub, default misinformation risk to a
        # moderately-cautious value scaled inversely with article count.
        misinformation_risk_score = max(0, min(100, 60 - article_count * 8))
        sentiment_score = 0
        if timeline:
            summary = (
                f"Google News RSS returned {article_count} recent articles related to {title}. "
                f"The feed centers on: {timeline[0]}."
            )
            bullish_thesis = (
                f"Recent coverage of {title} surfaces concrete developments such as "
                f"'{timeline[0]}', which could support the YES side resolving."
            )
            bearish_thesis = (
                f"Coverage volume around {title} is limited to {article_count} headline(s); "
                "absence of broader confirmation leaves room for the NO side."
            )
        else:
            summary = f"No recent Google News RSS articles were found for {title}."
            bullish_thesis = (
                f"No fresh news directly supports the YES side of {title}; treat any positive case as weak."
            )
            bearish_thesis = (
                f"With no recent news coverage of {title}, status-quo / NO outcome remains the default expectation."
            )
        return NewsResearchReport(
            timeline=timeline,
            key_facts=key_facts,
            source_quality_score=source_quality,
            bullish_thesis=bullish_thesis,
            bearish_thesis=bearish_thesis,
            evidence_quality_score=evidence_quality_score,
            misinformation_risk_score=misinformation_risk_score,
            sentiment_score=sentiment_score,
            key_sources=key_sources[:10],
            summary=summary,
        )
