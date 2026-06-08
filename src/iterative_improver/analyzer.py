from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

from google import genai
from google.genai import types

from config import cfg
from iterative_improver.models import RetrainingRecommendation


DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_TIMEOUT_S = 30
SYSTEM_PROMPT = (
    "You are an Iterative Improver agent for a quantitative prediction market trading system. "
    "You receive trades that already passed an outcome review and were flagged as 'Good Process / Bad Outcome' "
    "(model failure rather than statistical noise). Your job is to generate the next retraining experiment.\n\n"
    "For every input, diagnose which of these specific failure modes occurred:\n"
    "- regime_shift: the market structure changed (volatility regime, participant mix, liquidity profile) "
    "in a way the training data did not represent.\n"
    "- calibration_error: the model's probability outputs were systematically miscalibrated relative to realized frequencies.\n"
    "- narrative_overfit: the model learned spurious patterns from news/social text that did not generalize.\n\n"
    "Output Requirements:\n"
    "- Return ONLY valid JSON matching the requested schema.\n"
    "- failure_diagnosis must be exactly one of: regime_shift, calibration_error, narrative_overfit.\n"
    "- retraining_priority is an integer 0-10 (0=defer indefinitely, 10=block trading until retrained). "
    "Take a stand on how broken the current model is.\n"
    "- new_features must contain EXACTLY three feature recommendations, each with snake_case name, "
    "description (what it computes from available inputs), and rationale (why it addresses the blind spot).\n"
    "- reasoning ties the diagnosis to specific evidence in the trade log."
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


def _coerce_recommendation(parsed: Any) -> RetrainingRecommendation:
    if isinstance(parsed, RetrainingRecommendation):
        return parsed
    if isinstance(parsed, dict):
        return RetrainingRecommendation.model_validate(parsed)
    if isinstance(parsed, str):
        return RetrainingRecommendation.model_validate_json(parsed)
    raise TypeError(f"Unsupported parsed response type: {type(parsed)!r}")


def _parse_recommendation_response(response: Any) -> RetrainingRecommendation:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _coerce_recommendation(parsed)

    raw_text = _extract_response_text(response)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.removesuffix("```").strip()

    try:
        return RetrainingRecommendation.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return RetrainingRecommendation.model_validate(json.loads(cleaned[start : end + 1]))


def _build_prompt(
    *,
    prediction: float,
    outcome: bool,
    model_meta: Any,
    trade_log: Any,
) -> str:
    payload = {
        "prediction_pct": round(float(prediction) * 100.0, 2)
        if 0.0 <= float(prediction) <= 1.0
        else float(prediction),
        "actual_outcome": "win" if outcome else "loss",
        "model_meta": _json_safe(model_meta),
        "trade_log": _json_safe(trade_log),
    }
    return (
        "I just had a 'Good Process / Bad Outcome' trade that looks like a model failure rather than just bad luck.\n\n"
        f"Model Prediction: {payload['prediction_pct']}%\n"
        f"Actual Outcome: {payload['actual_outcome']}\n\n"
        "Model metadata:\n"
        f"{json.dumps(payload['model_meta'], indent=2, sort_keys=True)}\n\n"
        "Trade log:\n"
        f"{json.dumps(payload['trade_log'], indent=2, sort_keys=True)}\n\n"
        "Diagnose which failure mode occurred (regime_shift | calibration_error | narrative_overfit), "
        "set the Retraining Priority (0-10), and return EXACTLY 3 new features that would close this blind spot. "
        "Return only the JSON object matching the requested schema."
    )


class IterativeImproverAgent:
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

    def improve_after_trade(
        self,
        *,
        prediction: float,
        outcome: bool,
        model_meta: Any,
        trade_log: Any,
    ) -> RetrainingRecommendation:
        if prediction is None:
            raise ValueError("prediction is required")
        if not isinstance(prediction, (int, float)):
            raise TypeError("prediction must be numeric (probability 0..1 or percent)")
        if not isinstance(outcome, bool):
            raise TypeError("outcome must be a bool")
        if model_meta in (None, "", [], {}):
            raise ValueError("model_meta must be non-empty")
        if trade_log in (None, "", [], {}):
            raise ValueError("trade_log must be non-empty")

        prompt = _build_prompt(
            prediction=prediction,
            outcome=outcome,
            model_meta=model_meta,
            trade_log=trade_log,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RetrainingRecommendation,
                temperature=0.2,
                http_options=self._types.HttpOptions(timeout=self.timeout_s),
            ),
        )
        return _parse_recommendation_response(response)

    def review_trade(self, trade_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Adapter for PerformanceTracker conditional integration.

        Pulls ``prediction`` (from ``calibration.calibrated_true_prob``),
        ``outcome`` (from ``final_outcome``), ``model_meta`` (from explicit slot,
        sentinel fallback), and uses the whole payload as the trade_log.
        Returns a plain dict so it serializes straight into the audit file.
        """
        if not isinstance(trade_payload, Mapping):
            raise TypeError(f"trade_payload must be a Mapping, got {type(trade_payload)!r}")

        outcome = trade_payload.get("final_outcome")
        if not isinstance(outcome, bool):
            raise ValueError(
                "trade_payload['final_outcome'] must be a bool for IterativeImproverAgent.review_trade"
            )

        prediction = self._extract_prediction(trade_payload)
        model_meta = trade_payload.get("model_meta") or "No model metadata available."

        recommendation = self.improve_after_trade(
            prediction=prediction,
            outcome=outcome,
            model_meta=model_meta,
            trade_log=trade_payload,
        )
        return recommendation.model_dump()

    @staticmethod
    def _extract_prediction(trade_payload: Mapping[str, Any]) -> float:
        explicit = trade_payload.get("prediction")
        if isinstance(explicit, (int, float)):
            return float(explicit)

        calibration = trade_payload.get("calibration")
        if isinstance(calibration, Mapping):
            for key in ("calibrated_true_prob", "calibrated_probability", "probability"):
                value = calibration.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        elif calibration is not None:
            # Pydantic model or similar — try attribute access for the same keys.
            for key in ("calibrated_true_prob", "calibrated_probability", "probability"):
                value = getattr(calibration, key, None)
                if isinstance(value, (int, float)):
                    return float(value)

        raise ValueError(
            "trade_payload must include a 'prediction' or 'calibration.calibrated_true_prob' value."
        )

    def _build_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        return self._genai.Client(api_key=self.api_key)
