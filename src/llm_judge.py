from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from config import cfg
from utils import extract_json_object


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_S = 1.0
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
SUBJECTIVE_LANGUAGE_PATTERNS = (
    re.compile(r"\bsubstantial evidence\b", re.IGNORECASE),
    re.compile(r"\bcredible reporting\b", re.IGNORECASE),
    re.compile(r"\boverwhelming consensus\b", re.IGNORECASE),
    re.compile(r"\bconsensus\b", re.IGNORECASE),
    re.compile(r"\bsole discretion\b", re.IGNORECASE),
    re.compile(r"\bin our judgment\b", re.IGNORECASE),
    re.compile(r"\bunless\b.{0,80}\botherwise\b", re.IGNORECASE | re.DOTALL),
)
SYSTEM_PROMPT = (
    "You are a prediction market analyst. Score the market in a strict, skeptical way. "
    "Clarity score: 0-100 where 100 means fully objective, time-bounded, and mechanically resolvable, "
    "and 0 means highly subjective or likely to be disputed. Narrative momentum: 0-100 where 100 means the "
    "topic has strong recent news acceleration and public attention, and 0 means the topic appears dormant. "
    "If fresh news is unavailable or weak, keep narrative_momentum conservative. Return only valid JSON, with no markdown fences, "
    "and keep reasoning to one short sentence under 200 characters."
)


@dataclass
class LLMJudgeResult:
    clarity_score: int
    narrative_momentum: int
    anomaly_flags: List[str]
    reasoning: str = ""
    search_queries: List[str] = field(default_factory=list)
    source_titles: List[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return "AMBIGUOUS" in self.anomaly_flags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarity_score": self.clarity_score,
            "narrative_momentum": self.narrative_momentum,
            "anomaly_flags": list(self.anomaly_flags),
            "reasoning": self.reasoning,
            "search_queries": list(self.search_queries),
            "source_titles": list(self.source_titles),
        }


def _coerce_score(value: Any, default: int = 50) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = default
    return max(0, min(100, numeric))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _resolve_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    return (
        explicit_api_key
        or getattr(cfg, "gemini_api_key", None)
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _resolve_model(explicit_model: Optional[str]) -> str:
    return str(
        explicit_model
        or getattr(cfg, "gemini_model", None)
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()


def _resolve_timeout(explicit_timeout_s: Optional[int]) -> int:
    if explicit_timeout_s is not None:
        return int(explicit_timeout_s)
    configured_timeout = getattr(cfg, "gemini_timeout_s", None)
    if configured_timeout is not None:
        return int(configured_timeout)
    return DEFAULT_TIMEOUT_S


def _resolve_search_grounding(explicit_value: Optional[bool]) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    configured_value = getattr(cfg, "gemini_use_search_grounding", None)
    if configured_value is not None:
        return bool(configured_value)
    return True


def _subjective_language_matches(rules_text: str) -> List[str]:
    matches: List[str] = []
    for pattern in SUBJECTIVE_LANGUAGE_PATTERNS:
        match = pattern.search(rules_text or "")
        if not match:
            continue
        snippet = match.group(0).strip()
        if snippet not in matches:
            matches.append(snippet)
    return matches


def _extract_response_text(response_json: Dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response did not include any candidates")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [str(part.get("text", "")) for part in parts if part.get("text")]
    text = "".join(texts).strip()
    if not text:
        raise ValueError("Gemini response did not include text content")
    return text


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Thin shim around :func:`utils.extract_json_object` for back-compat.

    The shared helper handles Markdown fences, bracket extraction, and
    field-level regex fallback. We reuse it here so all three LLM call
    sites have consistent JSON-recovery behavior.
    """
    return extract_json_object(text)


_MISSING = object()


def _safe_get(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        value = _MISSING
        if isinstance(obj, dict):
            value = obj.get(name, _MISSING)
        else:
            value = getattr(obj, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return default


def _extract_grounding(candidate: Any) -> tuple[List[str], List[str]]:
    metadata = _safe_get(candidate, "groundingMetadata", "grounding_metadata")

    raw_queries = _safe_get(metadata, "webSearchQueries", "web_search_queries", default=[])
    if isinstance(raw_queries, (list, tuple)):
        search_queries = [str(query).strip() for query in raw_queries if str(query).strip()]
    else:
        query = str(raw_queries).strip()
        search_queries = [query] if query else []

    source_titles: List[str] = []
    raw_chunks = _safe_get(metadata, "groundingChunks", "grounding_chunks", default=[])
    if not isinstance(raw_chunks, (list, tuple)):
        raw_chunks = []
    for chunk in raw_chunks:
        web = _safe_get(chunk, "web")
        title = str(_safe_get(web, "title", "uri", default="") or "").strip()
        if title and title not in source_titles:
            source_titles.append(title)
    return search_queries, source_titles


def _build_payload(title: str, rules_text: str, *, use_search_grounding: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "system_instruction": {
            "parts": [
                {"text": SYSTEM_PROMPT}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Evaluate this prediction market and return only a JSON object with the keys "
                            "clarity_score, narrative_momentum, ambiguous, reasoning.\n\n"
                            "Market title:\n"
                            f"{title.strip()}\n\n"
                            "Rules text:\n"
                            f"{rules_text.strip()}\n\n"
                            "For clarity_score, focus on whether the resolution terms are objective, observable, "
                            "time-bounded, and low-discretion. For narrative_momentum, compare the market title "
                            "against recent news and public developments."
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
        },
    }
    if use_search_grounding:
        payload["tools"] = [{"google_search": {}}]
    return payload


def _extract_error_summary(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                return message
        message = str(payload.get("message") or "").strip()
        if message:
            return message
    text = response.text.strip()
    return text[:300] if text else f"HTTP {response.status_code}"


def _request_gemini_json(
    session: requests.Session,
    *,
    api_key: str,
    model: str,
    payload: Dict[str, Any],
    timeout_s: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_response_summary = ""
    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                f"{GEMINI_API_BASE_URL}/{model}:generateContent",
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout_s,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                last_response_summary = _extract_error_summary(response)
                if attempt >= max_retries:
                    response.raise_for_status()
                time.sleep(DEFAULT_RETRY_BACKOFF_S * (2 ** attempt))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(DEFAULT_RETRY_BACKOFF_S * (2 ** attempt))
    if last_error is not None:
        if last_response_summary:
            raise RuntimeError(f"Gemini request failed after retries: {last_response_summary}") from last_error
        raise RuntimeError("Gemini request failed after retries") from last_error
    raise RuntimeError("Gemini request failed without a response")


def judge_market(
    title: str,
    rules_text: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_search_grounding: Optional[bool] = None,
    session: Optional[requests.Session] = None,
    timeout_s: Optional[int] = None,
) -> LLMJudgeResult:
    normalized_title = str(title or "").strip()
    normalized_rules = str(rules_text or "").strip()
    if not normalized_title:
        raise ValueError("title is required")
    if not normalized_rules:
        raise ValueError("rules_text is required")

    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        raise RuntimeError("Set GEMINI_API_KEY in src/.env, .env, or the current environment before calling judge_market")

    resolved_model = _resolve_model(model)
    resolved_timeout_s = _resolve_timeout(timeout_s)
    resolved_use_search_grounding = _resolve_search_grounding(use_search_grounding)
    payload = _build_payload(normalized_title, normalized_rules, use_search_grounding=resolved_use_search_grounding)
    subjective_matches = _subjective_language_matches(normalized_rules)
    own_session = session is None
    http = session or requests.Session()

    fallback_notice = ""
    try:
        try:
            response_json = _request_gemini_json(
                http,
                api_key=resolved_api_key,
                model=resolved_model,
                payload=payload,
                timeout_s=resolved_timeout_s,
            )
        except RuntimeError:
            if not resolved_use_search_grounding:
                raise
            fallback_notice = (
                "Google Search grounding was unavailable; narrative momentum may rely on model knowledge instead of live search."
            )
            response_json = _request_gemini_json(
                http,
                api_key=resolved_api_key,
                model=resolved_model,
                payload=_build_payload(normalized_title, normalized_rules, use_search_grounding=False),
                timeout_s=resolved_timeout_s,
            )
    finally:
        if own_session:
            http.close()

    text = _extract_response_text(response_json)
    parsed = _extract_json_object(text)
    candidates = response_json.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    search_queries, source_titles = _extract_grounding(candidate)

    anomaly_flags: List[str] = []
    if subjective_matches or _coerce_bool(parsed.get("ambiguous")):
        anomaly_flags.append("AMBIGUOUS")

    reasoning = str(parsed.get("reasoning") or parsed.get("rationale") or "").strip()
    if fallback_notice:
        reasoning = f"{fallback_notice} {reasoning}".strip()
    if subjective_matches:
        matched_text = ", ".join(subjective_matches)
        reasoning = f"{reasoning} Subjective language matched: {matched_text}.".strip()

    return LLMJudgeResult(
        clarity_score=_coerce_score(parsed.get("clarity_score")),
        narrative_momentum=_coerce_score(parsed.get("narrative_momentum")),
        anomaly_flags=anomaly_flags,
        reasoning=reasoning,
        search_queries=search_queries,
        source_titles=source_titles,
    )
