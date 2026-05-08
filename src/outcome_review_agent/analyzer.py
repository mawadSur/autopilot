from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

from google import genai
from google.genai import types

from config import cfg
from outcome_review_agent.models import OutcomeReview
from utils import extract_json_object


DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TIMEOUT_S = 30
SYSTEM_PROMPT = (
    'You are an outcome-review agent for a prediction market trading system. Your job is to perform a post-mortem on settled trades to separate skill (process) from luck (outcome).\n\n'
    'Use the Process vs. Outcome Matrix:\n'
    '1. Deserved Success: Good Process (high confidence, strong evidence) + Good Outcome (Win).\n'
    '2. Good Failure: Good Process + Bad Outcome (Loss due to statistical noise or "Black Swan").\n'
    '3. Dumb Luck: Bad Process (weak evidence, low confidence, thin liquidity) + Good Outcome (Win).\n'
    '4. Poetic Justice: Bad Process + Bad Outcome (Loss).\n\n'
    'Analyze:\n'
    '- Did the original thesis hold?\n'
    '- Did new information emerge later that was impossible to know at entry?\n'
    '- Was the original calibration reasonable given the information available at the time?\n\n'
    'Your goal is to identify if the system is "resulting" or if it actually has a flaw in its research or risk modules.'
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


def _coerce_review(parsed: Any) -> OutcomeReview:
    if isinstance(parsed, OutcomeReview):
        return parsed
    if isinstance(parsed, dict):
        return OutcomeReview.model_validate(parsed)
    if isinstance(parsed, str):
        return OutcomeReview.model_validate_json(parsed)
    raise TypeError(f"Unsupported parsed response type: {type(parsed)!r}")


def _parse_review_response(response: Any) -> OutcomeReview:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _coerce_review(parsed)

    raw_text = _extract_response_text(response)
    return _coerce_review(extract_json_object(raw_text))


def _build_review_prompt(
    trade_log: Mapping[str, Any] | Any,
    final_outcome: bool,
    post_settlement_news: Mapping[str, Any] | list[Any] | str | Any,
) -> str:
    payload = {
        "trade_log": _json_safe(trade_log),
        "final_outcome": bool(final_outcome),
        "post_settlement_news": _json_safe(post_settlement_news),
    }
    return (
        "Perform a post-mortem outcome review for this settled trade. "
        "Return only a JSON object that matches the required schema.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


class OutcomeReviewAgent:
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

    def review_settled_trade(
        self,
        trade_log: Mapping[str, Any] | Any,
        final_outcome: bool,
        post_settlement_news: Mapping[str, Any] | list[Any] | str | Any,
    ) -> OutcomeReview:
        if trade_log in (None, "", [], {}):
            raise ValueError("trade_log must be non-empty")
        if not isinstance(final_outcome, bool):
            raise TypeError("final_outcome must be a bool")
        if post_settlement_news in (None, "", [], {}):
            raise ValueError("post_settlement_news must be non-empty")

        prompt = _build_review_prompt(
            trade_log=trade_log,
            final_outcome=final_outcome,
            post_settlement_news=post_settlement_news,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=OutcomeReview,
                temperature=0.2,
                http_options=self._types.HttpOptions(timeout=self.timeout_s),
            ),
        )
        return _parse_review_response(response)

    def review_trade(self, trade_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Adapter for PerformanceTracker-style payloads stored as JSON.

        Pulls ``final_outcome`` (required), ``post_settlement_news`` (optional),
        and ``trade_log`` (optional, defaults to the whole payload) out of a
        single dict and delegates to ``review_settled_trade``. Returns the
        review as a plain dict so it serializes straight into the audit file.
        """
        if not isinstance(trade_payload, Mapping):
            raise TypeError(f"trade_payload must be a Mapping, got {type(trade_payload)!r}")
        if "final_outcome" not in trade_payload:
            raise ValueError("trade_payload must include 'final_outcome' (bool)")

        final_outcome = trade_payload["final_outcome"]
        if not isinstance(final_outcome, bool):
            raise TypeError("trade_payload['final_outcome'] must be a bool")

        trade_log = trade_payload.get("trade_log", trade_payload)
        post_settlement_news = trade_payload.get(
            "post_settlement_news",
            "No post-settlement news provided.",
        )

        review = self.review_settled_trade(
            trade_log=trade_log,
            final_outcome=final_outcome,
            post_settlement_news=post_settlement_news,
        )
        return review.model_dump()

    def _build_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        return self._genai.Client(api_key=self.api_key)
