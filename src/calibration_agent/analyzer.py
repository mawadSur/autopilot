from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Mapping, Optional, Tuple

from google import genai
from google.genai import types

from calibration_agent.models import CalibrationReport
from config import cfg
from models import Market
from news_research_agent.models import NewsResearchReport
from utils import extract_json_object

LOGGER = logging.getLogger(__name__)


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


# Cap on the effective sample size we'll claim from confidence. Without this,
# a confidence value of 1.0 yields posterior_var ~ 0 and the adjuster's signal
# saturates immediately. 50 keeps the math well-behaved while still letting a
# unanimous, high-confidence pair drive posterior_confidence > 0.95.
_MAX_PSEUDO_OBSERVATIONS: float = 50.0


def _clamp_unit(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _coerce_confidence_pseudo_count(confidence: float, weight: float = 1.0) -> float:
    """Map a [0, 1] confidence into a pseudo-observation count for Beta math.

    confidence=0 -> 0 pseudo-observations (the prior wins).
    confidence=1 -> ``_MAX_PSEUDO_OBSERVATIONS`` pseudo-observations (very tight).
    Optional ``weight`` scales the contribution -- used by the outcome-weight
    adjuster to dial down the LLM's influence after a string of bad reviews.
    """
    c = _clamp_unit(confidence)
    w = max(0.0, float(weight))
    return c * w * _MAX_PSEUDO_OBSERVATIONS


def bayesian_fusion(
    xgb_prob: float,
    xgb_confidence: float,
    llm_prob: float,
    llm_confidence: float,
    *,
    llm_weight: float = 1.0,
) -> Tuple[float, float]:
    """Fuse XGBoost and LLM probability estimates via a Beta-posterior.

    Treats each estimator as if it observed ``confidence * MAX`` pseudo-trials
    that landed YES at rate ``prob``. The posterior is the conjugate
    combination of the two Beta likelihoods (with the implicit Beta(1,1)
    prior absorbed into the +1 terms).

    Args:
        xgb_prob: XGBoost's probability estimate, in [0, 1].
        xgb_confidence: XGBoost confidence proxy in [0, 1] (see notes below).
        llm_prob: LLM's probability estimate, in [0, 1].
        llm_confidence: LLM confidence in [0, 1] (rescale [0, 100] -> [0, 1]
            at the call site).
        llm_weight: Optional [0, +inf) multiplier on the LLM pseudo-count.
            The OutcomeWeightAdjuster uses this to dial back LLM influence
            after Dumb Luck or Poetic Justice classifications.

    Returns:
        ``(posterior_prob, posterior_confidence)`` where ``posterior_prob``
        is the fused probability and ``posterior_confidence`` is in [0, 1]
        (1 = posterior is very tight, 0 = nearly uniform).

    Notes on c_xgb (proxy choice):
        The XGBoost predictor doesn't expose a per-bar entropy directly.
        We use a coarse proxy: shrink raw confidence by half the distance to
        0.5 (so a 50/50 prediction = 0 confidence, a 99/1 prediction = ~1).
        Callers that have a better calibration-slope-based confidence can
        pass it directly in ``xgb_confidence``.
    """
    p_xgb = _clamp_unit(xgb_prob)
    p_llm = _clamp_unit(llm_prob)
    # Clamp slightly inside (0, 1) to avoid degenerate alphas/betas at the
    # endpoints. Anything closer than 1e-6 rounds to the bounds anyway.
    p_xgb = _clamp_unit(p_xgb, lo=1e-9, hi=1.0 - 1e-9)
    p_llm = _clamp_unit(p_llm, lo=1e-9, hi=1.0 - 1e-9)

    c_xgb = _coerce_confidence_pseudo_count(xgb_confidence)
    c_llm = _coerce_confidence_pseudo_count(llm_confidence, weight=llm_weight)

    alpha_xgb = c_xgb * p_xgb + 1.0
    beta_xgb = c_xgb * (1.0 - p_xgb) + 1.0
    alpha_llm = c_llm * p_llm + 1.0
    beta_llm = c_llm * (1.0 - p_llm) + 1.0

    alpha = alpha_xgb + alpha_llm - 1.0
    beta = beta_xgb + beta_llm - 1.0

    if alpha + beta <= 0.0:
        # Both estimators degenerate; fall back to the midpoint of priors.
        return 0.5, 0.0

    posterior_prob = alpha / (alpha + beta)
    denom_var = (alpha + beta) ** 2 * (alpha + beta + 1.0)
    posterior_var = (alpha * beta) / denom_var if denom_var > 0.0 else 0.0
    if math.isnan(posterior_prob) or math.isnan(posterior_var):
        return 0.5, 0.0
    posterior_confidence = 1.0 / (1.0 + posterior_var)
    return _clamp_unit(posterior_prob), _clamp_unit(posterior_confidence)


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


def _validate_invariants(report: CalibrationReport) -> CalibrationReport:
    """Enforce the spec invariant: calibrated_true_prob >= xgboost_prob whenever
    llm_adjustment_pct_points is non-negative. Negative adjustments are permitted
    to drop the calibrated probability below the baseline.
    """
    if report.llm_adjustment_pct_points >= 0 and report.calibrated_true_prob < report.xgboost_prob:
        raise ValueError(
            "calibrated_true_prob ("
            f"{report.calibrated_true_prob}) must be >= xgboost_prob ("
            f"{report.xgboost_prob}) when llm_adjustment_pct_points ("
            f"{report.llm_adjustment_pct_points}) is non-negative."
        )
    return report


def _coerce_report(parsed: Any) -> CalibrationReport:
    if isinstance(parsed, CalibrationReport):
        return _validate_invariants(parsed)
    if isinstance(parsed, dict):
        return _validate_invariants(CalibrationReport.model_validate(parsed))
    if isinstance(parsed, str):
        return _validate_invariants(CalibrationReport.model_validate_json(parsed))
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
    return _coerce_report(extract_json_object(raw_text))


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
        bullish_thesis=(
            f"No news evidence was provided for {market.title}; treat any bullish case as weak."
        ),
        bearish_thesis=(
            f"No news evidence was provided for {market.title}; status-quo / NO outcome remains the default."
        ),
        evidence_quality_score=0,
        misinformation_risk_score=50,
        sentiment_score=0,
        key_sources=[],
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
        xgboost_confidence: float | None = None,
        use_bayesian_fusion: bool = False,
        llm_weight: float = 1.0,
    ) -> CalibrationReport:
        if not isinstance(market, Market):
            raise TypeError("market must be a Market instance")

        baseline_probability = _coerce_probability(xgboost_prob, field_name="xgboost_prob")
        normalized_news_report = _coerce_news_report(news_report, market)
        prompt = _build_calibrate_prompt(market, normalized_news_report, baseline_probability, reddit_report=reddit_report)
        report = self._generate(prompt)

        if use_bayesian_fusion:
            report = self._apply_bayesian_fusion(
                report,
                xgboost_prob=baseline_probability,
                xgboost_confidence=xgboost_confidence,
                market_implied_prob=float(market.implied_prob),
                llm_weight=llm_weight,
            )
        return report

    def _apply_bayesian_fusion(
        self,
        report: CalibrationReport,
        *,
        xgboost_prob: float,
        xgboost_confidence: float | None,
        market_implied_prob: float,
        llm_weight: float = 1.0,
    ) -> CalibrationReport:
        """Replace the additive shift the LLM emitted with Bayesian fusion.

        The LLM's emitted ``calibrated_true_prob`` is treated as the LLM's
        own probability estimate; its ``confidence_score`` (0-100) is
        rescaled to [0, 1] and used as the LLM pseudo-count. We then fuse
        with the XGBoost baseline using ``bayesian_fusion`` and rebuild
        the report.
        """
        # Default XGB confidence proxy: shrink raw distance from 0.5.
        # An 80/20 prediction -> 0.6 confidence; a 99/1 prediction -> ~1.
        if xgboost_confidence is None:
            xgboost_confidence = min(1.0, 2.0 * abs(float(xgboost_prob) - 0.5))
        llm_confidence_unit = _clamp_unit(float(report.confidence_score) / 100.0)

        posterior_prob, posterior_confidence = bayesian_fusion(
            xgboost_prob,
            xgboost_confidence,
            float(report.calibrated_true_prob),
            llm_confidence_unit,
            llm_weight=llm_weight,
        )

        adjustment_pct_points = (posterior_prob - xgboost_prob) * 100.0
        new_confidence_score = int(round(_clamp_unit(posterior_confidence) * 100.0))
        edge_vs_market = posterior_prob - market_implied_prob

        # Audit log: keep a structured trail of the inputs and the output so
        # downstream debugging and the OutcomeWeightAdjuster can replay
        # decisions later.
        LOGGER.info(
            "bayesian_fusion: xgb=(%.4f, c=%.3f) llm=(%.4f, c=%.3f) llm_weight=%.3f -> "
            "posterior=(%.4f, c=%.3f)",
            xgboost_prob,
            xgboost_confidence,
            float(report.calibrated_true_prob),
            llm_confidence_unit,
            llm_weight,
            posterior_prob,
            posterior_confidence,
        )

        fused = CalibrationReport(
            xgboost_prob=xgboost_prob,
            llm_adjustment_pct_points=adjustment_pct_points,
            calibrated_true_prob=posterior_prob,
            confidence_score=new_confidence_score,
            key_drivers=list(report.key_drivers),
            key_uncertainties=list(report.key_uncertainties),
            edge_vs_market=_clamp_unit(edge_vs_market, lo=-1.0, hi=1.0),
            action=report.action,
            reasoning=(
                f"{report.reasoning} [bayesian_fusion: posterior={posterior_prob:.4f}, "
                f"posterior_confidence={posterior_confidence:.3f}, llm_weight={llm_weight:.2f}]"
            ),
        )
        return fused

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
