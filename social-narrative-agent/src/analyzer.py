from __future__ import annotations

import inspect
import os
from typing import Any, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import NarrativeAnalysis


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CLAIM_EXTRACTION_MODEL = "gpt-4.1-nano"
SYSTEM_PROMPT = (
    "You are a social-narrative research agent. Analyze recent discussions related to a prediction market. "
    "Focus on major claims, whether they stem from primary sources or rumor chains, who is driving the narrative, "
    "and the underlying sentiment (confidence, fear, sarcasm, hype, informed conviction). Determine if the info is "
    "new or recycled opinion. Do not summarize noise. Separate evidence from repetition. Compare the crowd's "
    "sentiment against the current market odds ({current_market_odds}) to determine if the crowd is ahead, behind, "
    "or aligned with the market."
)
CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are a claim extraction agent for noisy social-media datasets. Extract only distinct claims relevant to the "
    "topic. Remove spam, insults, jokes without informational content, pure cheering, and off-topic banter. Collapse "
    "duplicate paraphrases into one claim. For each retained claim, provide short source context describing where the "
    "claim came from and why it mattered in the discussion. Return only the structured schema."
)


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim: str = Field(..., min_length=1)
    source_context: str = Field(..., min_length=1)


class ClaimExtractionResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claims: List[ExtractedClaim] = Field(default_factory=list)

    @field_validator("claims")
    @classmethod
    def _dedupe_claims(cls, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        seen = set()
        deduped: List[ExtractedClaim] = []
        for claim in claims:
            normalized = " ".join(claim.claim.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(claim)
        return deduped


class NarrativeAnalyzer:
    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        claim_extraction_model: str = DEFAULT_CLAIM_EXTRACTION_MODEL,
    ) -> None:
        self._client = client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.claim_extraction_model = (
            str(claim_extraction_model or DEFAULT_CLAIM_EXTRACTION_MODEL).strip() or DEFAULT_CLAIM_EXTRACTION_MODEL
        )

    async def extract_claims(self, social_text: str, *, client: Any | None = None) -> ClaimExtractionResult:
        normalized_text = str(social_text or "").strip()
        if not normalized_text:
            raise ValueError("social_text must be a non-empty string")

        resolved_client = client or self._client or self._build_client()
        response = await self._run_structured_request(
            client=resolved_client,
            model=self.claim_extraction_model,
            system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            user_payload=(
                "Extract distinct claims and source context from the following social discussion dataset. "
                "Return only the structured schema.\n\n"
                "SOCIAL_TEXT:\n"
                f"{normalized_text}"
            ),
            schema=ClaimExtractionResult,
        )
        parsed = self._extract_parsed_payload(response)
        if isinstance(parsed, ClaimExtractionResult):
            return parsed
        return ClaimExtractionResult.model_validate(parsed)

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
        claim_result = await self.extract_claims(normalized_text, client=client)
        claim_payload = self._format_claims_for_analysis(claim_result)
        system_prompt = SYSTEM_PROMPT.format(current_market_odds=odds)
        response = await self._run_structured_request(
            client=client,
            model=self.model,
            system_prompt=system_prompt,
            user_payload=(
                "Analyze the following cleaned list of distinct claims and their source context. Return only the structured schema.\n\n"
                "DISTINCT CLAIMS AND SOURCE CONTEXT:\n"
                f"{claim_payload}"
            ),
            schema=NarrativeAnalysis,
        )
        parsed = self._extract_parsed_payload(response)
        if isinstance(parsed, NarrativeAnalysis):
            return parsed
        return NarrativeAnalysis.model_validate(parsed)

    def _format_claims_for_analysis(self, claim_result: ClaimExtractionResult) -> str:
        if not claim_result.claims:
            return "1. CLAIM: No distinct claims extracted. | SOURCE: Social discussion contained mostly noise or repetition."
        return "\n".join(
            f"{index}. CLAIM: {claim.claim} | SOURCE: {claim.source_context}"
            for index, claim in enumerate(claim_result.claims, start=1)
        )

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

    async def _run_structured_request(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str,
        user_payload: str,
        schema: Type[BaseModel],
    ) -> Any:
        if getattr(client, "responses", None) and hasattr(client.responses, "parse"):
            return await self._maybe_await(
                client.responses.parse(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    text_format=schema,
                )
            )

        completions = getattr(getattr(client, "chat", None), "completions", None)
        if completions and hasattr(completions, "parse"):
            return await self._maybe_await(
                completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format=schema,
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
