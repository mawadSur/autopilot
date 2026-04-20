from __future__ import annotations

import inspect
import os
from typing import Any, Optional

from models import NarrativeAnalysis


DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a social-narrative research agent. Analyze recent discussions related to a prediction market. "
    "Focus on major claims, whether they stem from primary sources or rumor chains, who is driving the narrative, "
    "and the underlying sentiment (confidence, fear, sarcasm, hype, informed conviction). Determine if the info is "
    "new or recycled opinion. Do not summarize noise. Separate evidence from repetition. Compare the crowd's "
    "sentiment against the current market odds ({current_market_odds}) to determine if the crowd is ahead, behind, "
    "or aligned with the market."
)


class NarrativeAnalyzer:
    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    async def analyze_narrative(self, social_text: str, current_market_odds: float) -> NarrativeAnalysis:
        normalized_text = str(social_text or "").strip()
        if not normalized_text:
            raise ValueError("social_text must be a non-empty string")
        try:
            odds = float(current_market_odds)
        except (TypeError, ValueError) as exc:
            raise ValueError("current_market_odds must be a float") from exc
        if not 0.0 <= odds <= 1.0:
            raise ValueError("current_market_odds must be between 0.0 and 1.0")

        client = self._client or self._build_client()
        system_prompt = SYSTEM_PROMPT.format(current_market_odds=odds)
        response = await self._run_structured_request(
            client=client,
            system_prompt=system_prompt,
            social_text=normalized_text,
        )
        parsed = self._extract_parsed_payload(response)
        if isinstance(parsed, NarrativeAnalysis):
            return parsed
        return NarrativeAnalysis.model_validate(parsed)

    def _build_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run NarrativeAnalyzer")
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for NarrativeAnalyzer. Install it with `pip install openai`."
            ) from exc
        return AsyncOpenAI(api_key=self.api_key)

    async def _run_structured_request(self, *, client: Any, system_prompt: str, social_text: str) -> Any:
        user_payload = (
            "Analyze the following social discussion dataset and return only the structured schema.\n\n"
            "SOCIAL_TEXT:\n"
            f"{social_text}"
        )

        if getattr(client, "responses", None) and hasattr(client.responses, "parse"):
            return await self._maybe_await(
                client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    text_format=NarrativeAnalysis,
                )
            )

        completions = getattr(getattr(client, "chat", None), "completions", None)
        if completions and hasattr(completions, "parse"):
            return await self._maybe_await(
                completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format=NarrativeAnalysis,
                )
            )

        raise RuntimeError(
            "Configured LLM client does not support structured parsing. Expected responses.parse or chat.completions.parse."
        )

    def _extract_parsed_payload(self, response: Any) -> Any:
        if hasattr(response, "output_parsed") and response.output_parsed is not None:
            return response.output_parsed

        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None and getattr(message, "parsed", None) is not None:
                return message.parsed

        raise RuntimeError("LLM response did not include a structured parsed payload")

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value
