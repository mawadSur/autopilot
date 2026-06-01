"""
XGBoost baseline service for the calibration agent.

STATUS: MOCK by default. The function ``get_xgboost_probability`` returns the
market's implied probability plus uniform noise unless a real model is wired in
via ``XGBOOST_MODEL_PATH``. All downstream "calibrated" probabilities are
garbage-in/garbage-out while the mock is active.

To ship a real model and unblock the calibration agent, the team needs:

    (a) Labeled dataset: settled prediction-market trades paired with the
        feature snapshot (market state + context) captured at decision time.
        Without settled outcomes there is no supervision signal.
    (b) Feature engineering pipeline: a deterministic transformer that maps a
        ``Market`` (and any external context) into the exact feature vector the
        model was trained on. Document the column order alongside the model
        artifact.
    (c) Training script: trains an XGBoost classifier on (a) using (b),
        produces a calibrated probability output (e.g. isotonic / Platt), and
        serializes via ``joblib.dump`` to the path pointed at by
        ``XGBOOST_MODEL_PATH``.
    (d) Evaluation gate: out-of-time / out-of-market holdout reporting log-loss
        and Brier score vs. the implied-prob baseline. Promote only if the
        model beats the market price baseline on held-out data.

Until (a)-(d) land, the mock fallback emits a loud WARNING on first call so
operators are not misled by suspiciously-real-looking calibration output.

VERSIONING NOTE - FEATURE VECTOR EXPANSION:
    The model-facing feature vector was expanded from 8 market-microstructure
    columns (FEATURE_COLUMNS) to 14 columns by appending 6 research-signal
    columns (RESEARCH_FEATURE_COLUMNS) for a combined ALL_FEATURE_COLUMNS.
    Models trained against the OLD 8-feature vector will produce wrong
    outputs (shape mismatch or silent garbage) when invoked through
    ``_full_feature_vector`` / ``_predict_with_model``. Re-train against
    ALL_FEATURE_COLUMNS or pin the model path to a legacy artifact only after
    wrapping the model in a shim that ignores the trailing 6 features.
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from models import Market

try:  # pragma: no cover - import guard exercised indirectly
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:  # pragma: no cover - cfg is optional from this module's perspective
    from config import cfg as _cfg  # type: ignore
except Exception:  # pragma: no cover
    _cfg = None


logger = logging.getLogger(__name__)

DEFAULT_JITTER_RANGE = 0.02

# Module-level cache for the loaded XGBoost model. ``_MODEL_LOADED`` is a
# tri-state: ``None`` means "not yet attempted", any other value means "load
# attempt completed" (the value itself is the model, or ``False`` if load
# failed and we should stay on the mock path for the rest of the process).
_MODEL: Any = None
_MODEL_LOADED: bool = False
_MOCK_WARNED: bool = False


def _resolve_model_path() -> Optional[str]:
    """Return the configured XGBoost model path, or ``None`` if unset."""

    env_path = os.environ.get("XGBOOST_MODEL_PATH")
    if env_path:
        return env_path
    if _cfg is not None:
        cfg_path = getattr(_cfg, "xgboost_model_path", None)
        if cfg_path:
            return str(cfg_path)
    return None


def _load_real_model() -> Any:
    """Attempt to load and cache the real XGBoost model.

    Returns the loaded model object on success, or ``None`` if no path is
    configured / the file is missing / the load fails. After the first
    invocation the result (success or failure) is cached for the lifetime of
    the process so we don't retry joblib.load on every prediction.
    """

    global _MODEL, _MODEL_LOADED

    if _MODEL_LOADED:
        return _MODEL

    path = _resolve_model_path()
    if not path:
        _MODEL_LOADED = True
        _MODEL = None
        return None

    if joblib is None:
        logger.error(
            "XGBOOST_MODEL_PATH=%s is set but joblib is not importable; "
            "falling back to mock baseline.",
            path,
        )
        _MODEL_LOADED = True
        _MODEL = None
        return None

    if not Path(path).is_file():
        logger.error(
            "XGBOOST_MODEL_PATH=%s does not point to an existing file; "
            "falling back to mock baseline.",
            path,
        )
        _MODEL_LOADED = True
        _MODEL = None
        return None

    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
        logger.error(
            "Failed to load XGBoost model from %s: %s; falling back to mock "
            "baseline.",
            path,
            exc,
        )
        _MODEL_LOADED = True
        _MODEL = None
        return None

    logger.info("Loaded XGBoost baseline model from %s", path)
    _MODEL = model
    _MODEL_LOADED = True
    return model


FEATURE_COLUMNS: tuple[str, ...] = (
    "implied_prob",
    "spread",
    "volume_24h",
    "open_interest",
    "days_to_resolution",
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
)

# Design choice (Option A): research-signal features live in a parallel constant
# and are extracted by a separate function ``extract_research_features`` so
# ``extract_market_features`` keeps its single Market-only signature; the
# orchestrator merges both dicts before writing ``features_window``.
RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    "news_sentiment_score",
    "news_evidence_quality_score",
    "news_misinformation_risk_score",
    "reddit_sentiment_score",
    "reddit_evidence_quality_score",
    "reddit_misinformation_risk_score",
)

ALL_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + RESEARCH_FEATURE_COLUMNS


def extract_market_features(market: Market) -> Dict[str, Any]:
    """Public feature extractor for the calibration baseline.

    Returns a labeled dict suitable for logging into a trade execution log's
    ``features_window`` slot at decision time. The orchestrator uses this so
    that, once trades settle, the (features, outcome) pairs can train the
    real XGBoost model.

    Feature contract (numeric columns must stay in sync with FEATURE_COLUMNS):

        ``implied_prob``           - mid-price implied probability (0..1)
        ``spread``                 - ask - bid (price space)
        ``volume_24h``             - 24h notional volume in USD
        ``open_interest``          - open interest in USD (0.0 if unknown)
        ``days_to_resolution``     - time-to-resolution in days (0.0 if N/A)
        ``price_change_1h``        - 1h price delta (0.0 if missing)
        ``price_change_6h``        - 6h price delta (0.0 if missing)
        ``price_change_24h``       - 24h price delta (0.0 if missing)
        ``captured_at_utc``        - ISO-8601 timestamp the snapshot was taken
                                     (NOT a model input — for audit/dataset assembly)
    """

    market.refresh_derived_fields()
    price_history = getattr(market, "price_history", {}) or {}
    return {
        "implied_prob": float(market.implied_prob),
        "spread": float(market.spread or 0.0),
        "volume_24h": float(market.volume_24h or 0.0),
        "open_interest": float(getattr(market, "open_interest", 0.0) or 0.0),
        "days_to_resolution": float(market.days_to_resolution or 0.0),
        "price_change_1h": float(price_history.get("1h", 0.0) or 0.0),
        "price_change_6h": float(price_history.get("6h", 0.0) or 0.0),
        "price_change_24h": float(price_history.get("24h", 0.0) or 0.0),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _safe_research_value(report: Any, attr: str) -> float:
    """Return ``float(report.attr)`` if available, else 0.0 (neutral default).

    Tolerates None reports, dicts (legacy callsites pass ``{}``), pydantic
    models, and arbitrary objects. Any extraction failure (missing attr,
    non-numeric value) collapses to ``0.0`` to match the documented contract.
    """

    if report is None:
        return 0.0
    value: Any = None
    if isinstance(report, dict):
        value = report.get(attr)
    else:
        value = getattr(report, attr, None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_research_features(news_report: Any, reddit_report: Any) -> Dict[str, Any]:
    """Public extractor for narrative / research-signal features.

    Returns a labeled dict aligned with :data:`RESEARCH_FEATURE_COLUMNS` so
    the orchestrator can merge it with :func:`extract_market_features` to form
    the full ``features_window`` payload.

    Feature contract (numeric columns must stay in sync with
    RESEARCH_FEATURE_COLUMNS):

        ``news_sentiment_score``               - news ``sentiment_score`` (-100..100)
        ``news_evidence_quality_score``        - news ``evidence_quality_score`` (0..100)
        ``news_misinformation_risk_score``     - news ``misinformation_risk_score`` (0..100)
        ``reddit_sentiment_score``             - reddit ``sentiment_score`` (-100..100)
        ``reddit_evidence_quality_score``      - reddit ``evidence_quality_score`` (0..100)
        ``reddit_misinformation_risk_score``   - reddit ``misinformation_risk_score`` (0..100)

    Defaults: if a report is ``None`` or missing a field (or the field is
    non-numeric), the corresponding feature defaults to ``0.0`` (neutral).
    Both ``NewsResearchReport`` and ``RedditResearchReport`` (pydantic v2)
    are accepted; raw dicts are also tolerated for legacy callsites.
    """

    return {
        "news_sentiment_score": _safe_research_value(news_report, "sentiment_score"),
        "news_evidence_quality_score": _safe_research_value(
            news_report, "evidence_quality_score"
        ),
        "news_misinformation_risk_score": _safe_research_value(
            news_report, "misinformation_risk_score"
        ),
        "reddit_sentiment_score": _safe_research_value(reddit_report, "sentiment_score"),
        "reddit_evidence_quality_score": _safe_research_value(
            reddit_report, "evidence_quality_score"
        ),
        "reddit_misinformation_risk_score": _safe_research_value(
            reddit_report, "misinformation_risk_score"
        ),
    }


def _full_feature_vector(
    market: Market,
    news_report: Any = None,
    reddit_report: Any = None,
) -> list[float]:
    """Order-fixed numeric feature vector for the XGBoost model.

    Reads from :data:`ALL_FEATURE_COLUMNS` so the training pipeline must
    produce a dataframe with these exact columns in this exact order. The
    model is expected to consume a 2D array of shape
    ``(n_samples, len(ALL_FEATURE_COLUMNS))`` and expose a ``predict_proba``
    method returning probabilities in ``[0, 1]`` for the positive (Yes)
    outcome.

    ``news_report`` / ``reddit_report`` may be ``None`` at call time (e.g. the
    legacy mock path); missing fields collapse to ``0.0`` per
    :func:`extract_research_features`.
    """

    feats = {
        **extract_market_features(market),
        **extract_research_features(news_report, reddit_report),
    }
    return [float(feats[col]) for col in ALL_FEATURE_COLUMNS]


def _market_features(market: Market) -> list[float]:
    """Compat shim: delegates to :func:`_full_feature_vector` with no research.

    Kept so any external caller that imported the old ``_market_features``
    name keeps working; for production inference prefer
    ``_full_feature_vector`` (or pass research reports through this shim is
    NOT supported — use the new function instead).
    """

    return _full_feature_vector(market, None, None)


def _predict_with_model(
    model: Any,
    market: Market,
    news_report: Any = None,
    reddit_report: Any = None,
) -> float:
    """Run inference against the real model and return a clamped probability."""

    features = [_full_feature_vector(market, news_report, reddit_report)]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        # Expect shape (1, 2) for binary classifiers; positive class is index 1.
        try:
            value = float(proba[0][1])
        except (IndexError, TypeError):
            value = float(proba[0])
    elif hasattr(model, "predict"):
        value = float(model.predict(features)[0])
    else:
        raise AttributeError(
            "Loaded XGBoost model exposes neither predict_proba nor predict"
        )
    return max(0.0, min(1.0, value))


def _mock_probability(market: Market) -> float:
    """Mock fallback: implied prob plus uniform jitter."""

    global _MOCK_WARNED

    if not _MOCK_WARNED:
        logger.warning(
            "Using mock XGBoost baseline - real model not loaded. Calibrated "
            "probabilities are NOT trustworthy until a real model is wired in."
        )
        _MOCK_WARNED = True
    else:
        logger.debug("Using mock XGBoost baseline (jitter=+/-%.3f)", DEFAULT_JITTER_RANGE)

    baseline = float(market.implied_prob)
    jitter = random.uniform(-DEFAULT_JITTER_RANGE, DEFAULT_JITTER_RANGE)
    return max(0.0, min(1.0, baseline + jitter))


def get_xgboost_probability(
    market: Market,
    *,
    news_report: Any = None,
    reddit_report: Any = None,
) -> float:
    """Return the XGBoost baseline probability for ``market``.

    If ``XGBOOST_MODEL_PATH`` (env var) or ``cfg.xgboost_model_path`` points at
    a readable joblib file, the cached real model is used for inference. The
    expected feature contract is documented on :func:`_full_feature_vector`
    (8 market-microstructure columns followed by 6 research-signal columns,
    in the order of :data:`ALL_FEATURE_COLUMNS`).

    ``news_report`` / ``reddit_report`` are optional kwargs; when omitted the
    research-signal slots default to ``0.0`` (neutral) per
    :func:`extract_research_features`. The orchestrator can pass full reports
    once they are available so the real model sees the narrative signals too.

    If no model is configured, or the load/predict fails, the function falls
    back to a mock that returns ``market.implied_prob + uniform(+/-0.02)``.
    The mock path emits a single WARNING on first use per process and DEBUG
    logs thereafter so operators are not silently misled.
    """

    if not isinstance(market, Market):
        raise TypeError("market must be a Market instance")

    market.refresh_derived_fields()
    _ = market.volume_24h
    _ = market.spread
    _ = market.days_to_resolution

    model = _load_real_model()
    if model is not None:
        try:
            return _predict_with_model(
                model,
                market,
                news_report=news_report,
                reddit_report=reddit_report,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "XGBoost prediction failed (%s); falling back to mock baseline "
                "for this call.",
                exc,
            )
            return _mock_probability(market)

    return _mock_probability(market)
