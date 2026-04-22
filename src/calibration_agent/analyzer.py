from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional

from google import genai
from google.genai import types

from calibration_agent.models import CalibrationReport
from config import cfg
from models import Market
from news_research_agent.models import NewsResearchReport


DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TIMEOUT_S = 30
SYSTEM_PROMPT = (
    'You are a probability-calibration agent for event-driven markets. You are given market features, event metadata, research summaries, and a baseline probability from an XGBoost model. Your job is to estimate a calibrated "true probability" for the market outcome.\n\n'
    'Method:\n'
    '1. Review the structured features.\n'
    '2. Use the XGBoost probability as the baseline.\n'
    '3. Adjust only when the qualitative evidence clearly justifies it.\n'
    '4. Explain what evidence supports any adjustment.\n'
    '5. Quantify uncertainty explicitly.\n\n'
    'Rules:\n'
    '- Do not force a prediction when uncertainty is high.\n'
    '- Small edges with low confidence should be rejected (action: pass).\n'
    '- If the evidence quality is weak, keep the final probability close to the model baseline (llm_adjustment = 0).\n'
    '- Never confuse narrative intensity with informational value. Be a cold, analytical skeptic.'
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


def _coerce_probability(value: Any, *, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return probability


def _coerce_report(parsed: Any) -> CalibrationReport:
    if isinstance(parsed, CalibrationReport):
        return parsed
    if isinstance(parsed, dict):
        return CalibrationReport.model_validate(parsed)
    if isinstance(parsed, str):
        return CalibrationReport.model_validate_json(parsed)
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


def _parse_report_response(response: Any) -> CalibrationReport:
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
        return CalibrationReport.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return CalibrationReport.model_validate(json.loads(cleaned[start : end + 1]))


def _serialize_market(market: Market) -> dict[str, Any]:
    if not isinstance(market, Market):
        raise TypeError("market must be a Market instance")
    market.refresh_derived_fields()
    resolution_date = market.resolution_date.isoformat() if hasattr(market.resolution_date, "isoformat") else str(market.resolution_date)
    return {
        "market_id": market.market_id,
        "title": market.title,
        "category": market.category,
        "implied_prob": float(market.implied_prob),
        "bid_price": float(market.bid_price),
        "ask_price": float(market.ask_price),
        "spread": float(market.spread),
        "volume_24h": float(market.volume_24h),
        "price_history": dict(market.price_history),
        "open_interest": float(market.open_interest),
        "resolution_date": resolution_date,
        "days_to_resolution": float(market.days_to_resolution),
        "rules_text": market.rules_text,
        "avg_volume_7d": market.avg_volume_7d,
        "volume_change_1h": market.volume_change_1h,
        "category_avg_spread": market.category_avg_spread,
    }


def _default_news_report(market: Market) -> NewsResearchReport:
    return NewsResearchReport(
        timeline=[],
        key_facts=[],
        source_quality_score=0,
        summary=(
            f"No news research report was provided for {market.title}. "
            "Treat qualitative evidence as weak unless the structured market metadata clearly justifies caution."
        ),
    )


def _coerce_news_report(news_report: Any, market: Market) -> NewsResearchReport:
    if news_report in (None, "", {}, []):
        return _default_news_report(market)
    if isinstance(news_report, NewsResearchReport):
        return news_report
    if isinstance(news_report, Mapping):
        return NewsResearchReport.model_validate(dict(news_report))
    model_dump = getattr(news_report, "model_dump", None)
    if callable(model_dump):
        return NewsResearchReport.model_validate(model_dump())
    raise TypeError("news_report must be a NewsResearchReport, a mapping, or None")


def _build_calibrate_prompt(market: Market, news_report: NewsResearchReport, xgboost_prob: float, reddit_report: Any | None = None) -> str:
    payload = {
        "market": _serialize_market(market),
        "reddit_research_report": _json_safe(reddit_report),
        "news_research_report": _json_safe(news_report),
        "xgboost_prob": xgboost_prob,
    }
    return (
        "Calibrate the true probability for the following market inputs. "
        "Return only a JSON object that matches the required schema.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _build_legacy_prompt(
    market_features: Mapping[str, Any],
    research_summaries: Any,
    xgboost_baseline: float,
    market_implied_prob: float,
) -> str:
    payload = {
        "market_features": _json_safe(dict(market_features)),
        "research_summaries": _json_safe(research_summaries),
        "xgboost_baseline": xgboost_baseline,
        "market_implied_prob": market_implied_prob,
    }
    return (
        "Calibrate the true probability for the following market inputs. "
        "Return only a JSON object that matches the required schema.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


class CalibrationAgent:
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

    def calibrate(
        self,
        market: Market,
        news_report: NewsResearchReport | Mapping[str, Any] | None,
        xgboost_prob: float,
        *,
        reddit_report: Any | None = None,
    ) -> CalibrationReport:
        if not isinstance(market, Market):
            raise TypeError("market must be a Market instance")

        baseline_probability = _coerce_probability(xgboost_prob, field_name="xgboost_prob")
        normalized_news_report = _coerce_news_report(news_report, market)
        prompt = _build_calibrate_prompt(market, normalized_news_report, baseline_probability, reddit_report=reddit_report)
        return self._generate(prompt)

    def calibrate_probability(
        self,
        *,
        market_features: dict,
        research_summaries: Any,
        xgboost_baseline: float,
        market_implied_prob: float,
    ) -> CalibrationReport:
        if not isinstance(market_features, dict) or not market_features:
            raise ValueError("market_features must be a non-empty dict")
        if research_summaries in (None, "", [], {}):
            raise ValueError("research_summaries must be non-empty")

        baseline_probability = _coerce_probability(xgboost_baseline, field_name="xgboost_baseline")
        implied_probability = _coerce_probability(market_implied_prob, field_name="market_implied_prob")
        prompt = _build_legacy_prompt(
            market_features,
            research_summaries,
            baseline_probability,
            implied_probability,
        )
        return self._generate(prompt)

    def _generate(self, prompt: str) -> CalibrationReport:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CalibrationReport,
                temperature=0.2,
                http_options=self._types.HttpOptions(timeout=self.timeout_s),
            ),
        )
        return _parse_report_response(response)

    def _build_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        return self._genai.Client(api_key=self.api_key)
