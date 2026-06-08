from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

from google import genai
from google.genai import types

from config import cfg
from reddit_research_agent.models import RedditResearchReport


DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TIMEOUT_S = 30
SYSTEM_PROMPT_TEMPLATE = (
    "You are a Reddit research agent for event-driven markets. Your job is to find informed "
    "discussion, identify expert-level comments, and distinguish substantive reasoning from "
    "crowd speculation regarding the market: {market_title}. "
    "Current market implied probability is {implied_probability_pct}%. "
    "Evaluate depth of reasoning, references to data, disagreement quality, view updating, "
    "and sentiment balance. Prefer thoughtful contrarian comments over popular shallow comments.\n\n"
    "Return ONLY a JSON object that matches the required RedditResearchReport schema with "
    "ALL of the following fields populated:\n"
    "- bullish_thesis: non-empty string explaining the strongest case the YES side resolves.\n"
    "- bearish_thesis: non-empty string explaining the strongest case the NO side resolves.\n"
    "- key_evidence: list of distinct, non-empty pieces of evidence cited in the discussion.\n"
    "- key_assumptions: list of non-empty assumptions the bullish/bearish theses depend on.\n"
    "- conviction_score: int 0-10 conveying how strongly the discussion converges on a view.\n"
    "- evidence_quality_score: int 0-100; higher = stronger primary sources.\n"
    "- misinformation_risk_score: int 0-100; higher = higher misinformation risk.\n"
    "- sentiment_score: int -100..100; negative=bearish, positive=bullish, 0=neutral.\n"
    "- key_sources: list of up to 10 URLs, subreddit names, or commenter handles cited.\n"
    "- summary: non-empty narrative summary of the Reddit discussion landscape.\n"
    "- pricing_assessment: one of 'underpriced', 'overpriced', 'fairly priced', 'unclear'.\n"
    "- assessment_reasoning: non-empty rationale tying evidence to the pricing_assessment."
)


def _resolve_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    return (
        explicit_api_key
        or os.getenv("GOOGLE_API_KEY")
        or getattr(cfg, "gemini_api_key", None)
        or os.getenv("GEMINI_API_KEY")
    )


def _build_system_prompt(market_title: str, implied_prob: float) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        market_title=market_title,
        implied_probability_pct=implied_prob * 100,
    )


def _extract_response_text(response: Any) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise ValueError("Gemini response did not include any text candidates")

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    texts = []
    for part in parts:
        part_text = str(getattr(part, "text", "") or "").strip()
        if part_text:
            texts.append(part_text)
    combined = "".join(texts).strip()
    if not combined:
        raise ValueError("Gemini response did not include text content")
    return combined


def _coerce_report(parsed: Any) -> RedditResearchReport:
    if isinstance(parsed, RedditResearchReport):
        return parsed
    if isinstance(parsed, dict):
        return RedditResearchReport.model_validate(parsed)
    if isinstance(parsed, str):
        return RedditResearchReport.model_validate_json(parsed)
    raise TypeError(f"Unsupported parsed response type: {type(parsed)!r}")


def _parse_report_response(response: Any) -> RedditResearchReport:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _coerce_report(parsed)

    raw_text = _extract_response_text(response)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.removesuffix("```").strip()

    try:
        return RedditResearchReport.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
        return RedditResearchReport.model_validate(payload)


class RedditAgent:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout_s: Optional[int] = None,
        client: Any | None = None,
        genai_module: Any | None = None,
        types_module: Any | None = None,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.model_name = (model_name or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        self.timeout_s = int(
            timeout_s if timeout_s is not None else getattr(cfg, "gemini_timeout_s", DEFAULT_TIMEOUT_S)
        )
        self._genai = genai_module if genai_module is not None else genai
        self._types = types_module if types_module is not None else types
        self.client = client if client is not None else self._build_client()

    async def analyze_discussion(
        self,
        market_title: str,
        implied_prob: float,
        reddit_context: str,
    ) -> RedditResearchReport:
        title = (market_title or "").strip()
        context = (reddit_context or "").strip()
        implied_probability = float(implied_prob)

        if not title:
            raise ValueError("market_title must be a non-empty string")
        if not context:
            raise ValueError("reddit_context must be a non-empty string")
        if not 0.0 <= implied_probability <= 1.0:
            raise ValueError("implied_prob must be between 0.0 and 1.0")

        prompt = (
            "Analyze the following Reddit Discussion Context and return only a JSON object that "
            "matches the required RedditResearchReport schema with ALL fields populated.\n\n"
            f"Market title: {title}\n"
            f"Implied probability: {implied_probability:.6f}\n\n"
            "Reddit Discussion Context:\n"
            f"{context}"
        )
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config=self._generation_config(title, implied_probability),
        )
        return _parse_report_response(response)

    def _build_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        return self._genai.Client(api_key=self.api_key)

    def _generation_config(self, market_title: str, implied_prob: float) -> Any:
        return self._types.GenerateContentConfig(
            system_instruction=_build_system_prompt(market_title, implied_prob),
            response_mime_type="application/json",
            response_schema=RedditResearchReport,
            temperature=0.2,
            http_options=self._types.HttpOptions(timeout=self.timeout_s),
        )
