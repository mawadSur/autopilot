from __future__ import annotations

import json
import os
from typing import Any, Optional

from google import genai
from google.genai import types

from config import cfg
from synthesis_agent.models import SynthesisReport


DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TIMEOUT_S = 30
SYSTEM_PROMPT = (
    "You are a synthesis agent combining market data, social discussion, and news coverage.\n"
    "Your task is to classify how well the current market price reflects the available narrative.\n\n"
    "Produce:\n"
    "- current market-implied probability\n"
    "- narrative-implied direction: bullish / bearish / mixed\n"
    "- whether the narrative contains unique evidence not yet reflected in price\n"
    "- top 3 reasons the market could still be right\n"
    "- top 3 reasons the market could be wrong\n"
    "- verdict: a market-efficiency classification\n"
    "- concise explanation in plain English\n\n"
    "VERDICT DEFINITIONS - pick exactly one of stale | efficient | overreactive | unclear:\n"
    "- 'stale': the market price has not updated against recent material news; the narrative\n"
    "  contains fresh evidence that the current odds clearly do not yet reflect.\n"
    "- 'efficient': the market price reflects the available evidence well; news and social\n"
    "  discussion are already priced in and there is no obvious dislocation.\n"
    "- 'overreactive': the market has moved farther than the evidence supports; the crowd\n"
    "  appears to have over-corrected to recent narrative or sentiment.\n"
    "- 'unclear': there is insufficient signal in the narrative to classify; evidence is\n"
    "  thin, contradictory, or too low-quality to judge market efficiency either way.\n\n"
    "CRITICAL RULE: This verdict is a market-efficiency classification, NOT a trading\n"
    "recommendation. Output exactly one of the four labels above and nothing else for the\n"
    "verdict field. When evidence is genuinely thin or contradictory, prefer 'unclear' over\n"
    "guessing one of the other three categories."
)


def _resolve_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    return (
        explicit_api_key
        or os.getenv("GOOGLE_API_KEY")
        or getattr(cfg, "gemini_api_key", None)
        or os.getenv("GEMINI_API_KEY")
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _build_prompt(market_data: dict, social_context: str, news_context: str) -> str:
    market_json = json.dumps(_json_safe(market_data), indent=2, sort_keys=True)
    return (
        f"--- MARKET DATA ---\n{market_json}\n\n"
        f"--- SOCIAL & REDDIT NARRATIVE ---\n{social_context}\n\n"
        f"--- NEWS & RSS COVERAGE ---\n{news_context}\n\n"
        "Analyze the above streams. Is the market mispricing this event? Return the structured synthesis."
    )


def _coerce_report(parsed: Any) -> SynthesisReport:
    if isinstance(parsed, SynthesisReport):
        return parsed
    if isinstance(parsed, dict):
        return SynthesisReport.model_validate(parsed)
    if isinstance(parsed, str):
        return SynthesisReport.model_validate_json(parsed)
    raise TypeError(f"Unsupported parsed response type: {type(parsed)!r}")


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


def _parse_synthesis_response(response: Any) -> SynthesisReport:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _coerce_report(parsed)

    cleaned = _extract_response_text(response).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.removesuffix("```").strip()

    try:
        return SynthesisReport.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return SynthesisReport.model_validate(json.loads(cleaned[start : end + 1]))


class SynthesisAgent:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout_s: Optional[int] = None,
        client: Any | None = None,
        genai_module: Any | None = None,
        types_module: Any | None = None,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing.")

        self.model_name = (model_name or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        self.timeout_s = int(
            timeout_s if timeout_s is not None else getattr(cfg, "gemini_timeout_s", DEFAULT_TIMEOUT_S)
        )
        self._genai = genai_module if genai_module is not None else genai
        self._types = types_module if types_module is not None else types
        self.client = client if client is not None else self._genai.Client(api_key=self.api_key)

    def synthesize_edge(
        self,
        market_data: dict,
        social_context: str,
        news_context: str,
    ) -> SynthesisReport:
        if not isinstance(market_data, dict) or not market_data:
            raise ValueError("market_data must be a non-empty dict")
        social_text = str(social_context or "").strip()
        news_text = str(news_context or "").strip()
        if not social_text:
            raise ValueError("social_context must be a non-empty string")
        if not news_text:
            raise ValueError("news_context must be a non-empty string")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=_build_prompt(market_data, social_text, news_text),
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=SynthesisReport,
                temperature=0.1,
                http_options=self._types.HttpOptions(timeout=self.timeout_s),
            ),
        )
        return _parse_synthesis_response(response)
