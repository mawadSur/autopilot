from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

from google import genai
from google.genai import types

from config import cfg
from data_quality_auditor.models import DataQualityAudit, FocusedAuditFinding


DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TIMEOUT_S = 30
SYSTEM_PROMPT = (
    "You are a senior Data-Quality Auditor for a quantitative prediction market trading system. "
    "Your goal is to perform a forensic post-mortem on trade executions to identify if data integrity issues compromised the trade's outcome.\n\n"
    "Evaluate the provided trade logs against these specific failure modes:\n"
    "1. Stale Data: Was the market price or news context outdated at the moment of entry?\n"
    "2. Duplicate Sources: Were multiple signals actually just the same report from different aggregators?\n"
    "3. Missing Primary Sources: Did the system rely on secondary commentary while ignoring official/primary data?\n"
    "4. Misleading Sentiment: Did NLP errors misinterpret sarcasm, nuance, or complex news as a directional signal?\n"
    "5. Scraping Gaps: Are there visible \"dead zones\" or missing intervals in the historical features?\n"
    "6. Timestamp Mismatches: Do the event occurrence time, the news publication time, and the trade entry time conflict?\n"
    "7. Incorrect Market Metadata: Was the trade based on wrong contract rules (e.g., wrong resolution date or strike)?\n\n"
    "Output Requirements:\n"
    "- Return ONLY valid JSON.\n"
    "- Provide a rigorous analysis for each failure mode.\n"
    "- Determine if a 'data failure' occurred (yes/no).\n"
    "- If 'yes', specify the exact failure mode and a severity from 1 (negligible) to 5 (critical/terminal).\n"
    "- Provide a technical 'recommended fix' to prevent recurrence."
)


_FOCUSED_OUTPUT_REQUIREMENTS = (
    "Output Requirements:\n"
    "- Return ONLY valid JSON matching the requested schema.\n"
    "- failure_modes must be a list whose values are drawn from the in-scope mode names below.\n"
    "- If any in-scope mode is detected, set data_failure=true, include the mode name(s) in failure_modes, "
    "and set primary_failure_mode to the most material one.\n"
    "- If no in-scope failure is detected, set data_failure=false, leave failure_modes as [], "
    "and set primary_failure_mode to null.\n"
    "- Always provide severity 1-5 (1=negligible even on clean audits, 5=critical/terminal).\n"
    "- Always provide a detailed audit_trail explaining the reasoning and a technical recommended_fix."
)

INTEGRITY_SYSTEM_PROMPT = (
    "You are a senior Data-Quality Auditor for a quantitative prediction market trading system. "
    "Focus exclusively on integrity & source analysis.\n\n"
    "Identify if the trade entry was based on 'hallucinated' or 'stale' information. "
    "Compare the 'entry_timestamp' against the 'source_timestamps'. "
    "Are there duplicate news signatures appearing as unique evidence? "
    "Is a primary source (e.g., official government site, direct exchange feed) missing?\n\n"
    "In-scope failure mode names: stale_data, duplicate_sources, missing_primary_sources.\n\n"
    + _FOCUSED_OUTPUT_REQUIREMENTS
)

INTERPRETATION_SYSTEM_PROMPT = (
    "You are a senior Data-Quality Auditor for a quantitative prediction market trading system. "
    "Focus exclusively on interpretation & metadata analysis.\n\n"
    "Review the sentiment scores and market metadata for this trade. "
    "Did the system interpret a news headline as 'Bullish' when the underlying text was neutral or sarcastic? "
    "Does the contract 'expiry' in the metadata align with the actual event date?\n\n"
    "In-scope failure mode names: misleading_sentiment, incorrect_market_metadata.\n\n"
    + _FOCUSED_OUTPUT_REQUIREMENTS
)

PIPELINE_SYSTEM_PROMPT = (
    "You are a senior Data-Quality Auditor for a quantitative prediction market trading system. "
    "Focus exclusively on structural & pipeline analysis.\n\n"
    "Analyze the time-series continuity. "
    "Are there gaps in the 1-minute bar data leading up to the trade? "
    "Do the server-received timestamps align with the provider-published timestamps, "
    "or is there a 'lag' failure mode?\n\n"
    "In-scope failure mode names: scraping_gaps, timestamp_mismatches.\n\n"
    + _FOCUSED_OUTPUT_REQUIREMENTS
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


def _coerce_to(parsed: Any, target_cls: type) -> Any:
    if isinstance(parsed, target_cls):
        return parsed
    if isinstance(parsed, dict):
        return target_cls.model_validate(parsed)
    if isinstance(parsed, str):
        return target_cls.model_validate_json(parsed)
    raise TypeError(f"Unsupported parsed response type: {type(parsed)!r}")


def _parse_response_as(response: Any, target_cls: type) -> Any:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _coerce_to(parsed, target_cls)

    raw_text = _extract_response_text(response)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.removesuffix("```").strip()

    try:
        return target_cls.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return target_cls.model_validate(json.loads(cleaned[start : end + 1]))


def _build_audit_prompt(
    trade_payload: Mapping[str, Any] | Any,
    news_context: Any,
    features_window: Any,
) -> str:
    payload = {
        "trade_payload": _json_safe(trade_payload),
        "news_context": _json_safe(news_context),
        "features_window": _json_safe(features_window),
    }
    return (
        "Review the following trade data for integrity failures:\n\n"
        f"Trade Context:\n{json.dumps(payload['trade_payload'], indent=2, sort_keys=True)}\n\n"
        f"Latest News at Entry:\n{json.dumps(payload['news_context'], indent=2, sort_keys=True)}\n\n"
        f"Historical Feature Snapshot:\n{json.dumps(payload['features_window'], indent=2, sort_keys=True)}\n\n"
        "Audit Request:\nPerform the data-quality audit now. "
        "Return the JSON object matching the requested schema."
    )


class DataQualityAuditor:
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

    def audit_trade(
        self,
        trade_payload: Mapping[str, Any] | Any,
        news_context: Any,
        features_window: Any,
    ) -> DataQualityAudit:
        if trade_payload in (None, "", [], {}):
            raise ValueError("trade_payload must be non-empty")
        if news_context in (None, "", [], {}):
            raise ValueError("news_context must be non-empty")
        if features_window in (None, "", [], {}):
            raise ValueError("features_window must be non-empty")

        prompt = _build_audit_prompt(
            trade_payload=trade_payload,
            news_context=news_context,
            features_window=features_window,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=DataQualityAudit,
                temperature=0.2,
                http_options=self._types.HttpOptions(timeout=self.timeout_s),
            ),
        )
        return _parse_response_as(response, DataQualityAudit)

    def audit_integrity(
        self,
        trade_payload: Mapping[str, Any] | Any,
        news_context: Any,
        features_window: Any,
    ) -> FocusedAuditFinding:
        """Focused audit: stale_data, duplicate_sources, missing_primary_sources."""
        return self._run_focused_audit(
            INTEGRITY_SYSTEM_PROMPT,
            trade_payload=trade_payload,
            news_context=news_context,
            features_window=features_window,
        )

    def audit_interpretation(
        self,
        trade_payload: Mapping[str, Any] | Any,
        news_context: Any,
        features_window: Any,
    ) -> FocusedAuditFinding:
        """Focused audit: misleading_sentiment, incorrect_market_metadata."""
        return self._run_focused_audit(
            INTERPRETATION_SYSTEM_PROMPT,
            trade_payload=trade_payload,
            news_context=news_context,
            features_window=features_window,
        )

    def audit_pipeline(
        self,
        trade_payload: Mapping[str, Any] | Any,
        news_context: Any,
        features_window: Any,
    ) -> FocusedAuditFinding:
        """Focused audit: scraping_gaps, timestamp_mismatches."""
        return self._run_focused_audit(
            PIPELINE_SYSTEM_PROMPT,
            trade_payload=trade_payload,
            news_context=news_context,
            features_window=features_window,
        )

    def _run_focused_audit(
        self,
        system_prompt: str,
        *,
        trade_payload: Any,
        news_context: Any,
        features_window: Any,
    ) -> FocusedAuditFinding:
        if trade_payload in (None, "", [], {}):
            raise ValueError("trade_payload must be non-empty")
        if news_context in (None, "", [], {}):
            raise ValueError("news_context must be non-empty")
        if features_window in (None, "", [], {}):
            raise ValueError("features_window must be non-empty")

        prompt = _build_audit_prompt(
            trade_payload=trade_payload,
            news_context=news_context,
            features_window=features_window,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=FocusedAuditFinding,
                temperature=0.2,
                http_options=self._types.HttpOptions(timeout=self.timeout_s),
            ),
        )
        return _parse_response_as(response, FocusedAuditFinding)

    def review_trade(self, trade_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Adapter for PerformanceTracker-style payloads.

        Pulls ``news_context`` (with sensible fallbacks: explicit slot →
        ``research.news_report`` → sentinel) and ``features_window`` (explicit
        slot → ``scanner`` → sentinel) out of a single payload and delegates to
        ``audit_trade``. Returns a plain dict so it serializes straight into
        the audit file.
        """
        if not isinstance(trade_payload, Mapping):
            raise TypeError(f"trade_payload must be a Mapping, got {type(trade_payload)!r}")

        news_context = trade_payload.get("news_context")
        if not news_context:
            research = trade_payload.get("research") if isinstance(trade_payload.get("research"), Mapping) else None
            if research is not None:
                news_context = research.get("news_report")
        if not news_context:
            news_context = "No news context available."

        features_window = trade_payload.get("features_window")
        if not features_window:
            features_window = trade_payload.get("scanner")
        if not features_window:
            features_window = "No feature snapshot available."

        audit = self.audit_trade(
            trade_payload=trade_payload,
            news_context=news_context,
            features_window=features_window,
        )
        return audit.model_dump()

    def _build_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        return self._genai.Client(api_key=self.api_key)
