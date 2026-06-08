"""SignalForensicsAgent — A1 of the Loss Postmortem Swarm.

Investigates whether the model's signal was the primary cause of a loss.

Reads:
- ``TradeContextSnapshot`` at phase=``signal`` (features, model_probs,
  model_confidence, risk_metrics_input)
- ``model_meta.json`` for the trade's symbol (training-distribution stats,
  ``optimal_threshold``, ``reliability_slope`` / ``threshold_metrics``)

Checks (each may emit an evidence bullet + a red-flag count):
1. Confidence-vs-threshold margin: signal barely above the floor.
2. Feature NaN/inf or zero-quality.
3. Distance from training distribution (Mahalanobis on key features against
   training-set means/stds in meta.json) — flag if > 2σ.
4. Reliability at this probability bin (anti-calibrated bin → strong signal).
5. Regime mismatch within 3 bars before signal.

Verdict logic:
- 3+ red flags → ``primary_cause`` (confidence ≥ 0.7)
- 1-2 red flags → ``contributing`` (confidence 0.4-0.7)
- 0 red flags → ``innocent``
- agent crash / missing snapshot → handled by :class:`BaseForensicsAgent`'s
  safety wrappers (``verdict="unknown"``)

Critical caveat (Lane E foundation): signal snapshots may currently have
empty ``feature_buffer`` and ``model_probs`` because the predictor surface
only returns ``(side, confidence)``. This agent gracefully degrades — when
those fields are empty it skips the corresponding checks and emits an
evidence bullet noting the limitation rather than crashing or asserting
``primary_cause`` purely on missing data.

Mahalanobis implementation note: we use the diagonal approximation
(z-score per feature, aggregate as ``sqrt(sum(z^2))``) rather than a full
covariance-aware Mahalanobis. The diagonal form is fine when feature
correlations aren't dominant and is what the meta.json shape currently
supports (means + stds per feature). If meta.json later carries a
covariance matrix, replace ``_diag_mahalanobis`` with a covariance-aware
version.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from loss_postmortem.base import (
    DEFAULT_AGENT_TIMEOUT_S,
    BaseForensicsAgent,
    ForensicsFinding,
)
from state.trade_context_store import TradeContextSnapshot, TradeContextStore

LOGGER = logging.getLogger(__name__)

# A confidence within this gap of the threshold counts as "barely above floor".
DEFAULT_MARGIN_FLOOR_PCT = 0.05  # 5 % of the threshold value
# Mahalanobis distance flagged as out-of-distribution.
DEFAULT_MAHALANOBIS_FLAG_SIGMA = 2.0
# Default threshold floor used when meta has no optimal_threshold.
DEFAULT_THRESHOLD_FLOOR = 0.5
# Cap how many features we'll Mahalanobis-score; keeps cost bounded.
MAX_MAHALANOBIS_FEATURES = 64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_finite_number(x: Any) -> bool:
    """True iff ``x`` is a finite numeric scalar (rejects NaN/inf/None)."""
    if x is None:
        return False
    if isinstance(x, bool):
        # bool is a subclass of int — reject it explicitly so a stray True
        # doesn't sneak through Mahalanobis as 1.0.
        return False
    if not isinstance(x, (int, float)):
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _diag_mahalanobis(
    features: Mapping[str, Any],
    means: Mapping[str, Any],
    stds: Mapping[str, Any],
    *,
    max_features: int = MAX_MAHALANOBIS_FEATURES,
) -> Tuple[Optional[float], int]:
    """Diagonal Mahalanobis distance with finite-only inputs.

    Returns ``(distance, n_features_used)``. ``distance`` is ``None`` if no
    eligible features were found.
    """
    sum_sq = 0.0
    used = 0
    for feat, mu in means.items():
        if used >= max_features:
            break
        sigma = stds.get(feat)
        val = features.get(feat)
        if not (
            _is_finite_number(val)
            and _is_finite_number(mu)
            and _is_finite_number(sigma)
        ):
            continue
        sigma_f = float(sigma)
        if sigma_f <= 0.0:
            # Degenerate feature — skip rather than divide by zero.
            continue
        z = (float(val) - float(mu)) / sigma_f
        sum_sq += z * z
        used += 1
    if used == 0:
        return None, 0
    return math.sqrt(sum_sq), used


def _load_meta_for_symbol(
    symbol: str,
    *,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Best-effort load of ``model_crypto/<symbol_slug>/meta.json``.

    Returns ``{}`` (not None) when the file is missing or unparseable so
    downstream checks can probe optional keys without try/except.
    """
    if not symbol:
        return {}
    # Map e.g. "ETH/USD" -> "eth_usd". This matches the existing dir layout
    # under ``model_crypto/``. We tolerate any version suffix (eth_usd_v2,
    # eth_usd_v1, etc.) — we pick whatever directories begin with the slug
    # and prefer the highest numeric suffix.
    slug = (
        symbol.lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "_")
        .strip()
    )
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "model_crypto",
        )
        base_dir = os.path.normpath(base_dir)
    if not os.path.isdir(base_dir):
        return {}
    candidates: List[str] = []
    try:
        for name in os.listdir(base_dir):
            if name.startswith(slug):
                candidates.append(name)
    except OSError as exc:
        LOGGER.debug("signal_forensics: meta listing failed: %r", exc)
        return {}
    if not candidates:
        return {}
    # Sort so a "v2" beats "v1" lexicographically; tie-break on full name.
    candidates.sort(reverse=True)
    for cand in candidates:
        meta_path = os.path.join(base_dir, cand, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.debug(
                "signal_forensics: failed to parse %s: %r", meta_path, exc
            )
            continue
        if isinstance(data, dict):
            return data
    return {}


def _extract_means_stds(
    meta: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pull per-feature training means + stds from a meta.json blob.

    Tolerant of multiple shapes:
    - ``meta["feature_means"]`` / ``meta["feature_stds"]`` (mapping form)
    - ``meta["training_distribution"] = {"means": ..., "stds": ...}``
    - ``meta["feature_stats"][feat] = {"mean": ..., "std": ...}``
    Anything else returns empty dicts and the Mahalanobis check skips.
    """
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    fm = meta.get("feature_means")
    fs = meta.get("feature_stds")
    if isinstance(fm, Mapping) and isinstance(fs, Mapping):
        for k, v in fm.items():
            if _is_finite_number(v):
                means[str(k)] = float(v)
        for k, v in fs.items():
            if _is_finite_number(v):
                stds[str(k)] = float(v)
        if means and stds:
            return means, stds

    td = meta.get("training_distribution")
    if isinstance(td, Mapping):
        td_means = td.get("means")
        td_stds = td.get("stds")
        if isinstance(td_means, Mapping) and isinstance(td_stds, Mapping):
            for k, v in td_means.items():
                if _is_finite_number(v):
                    means[str(k)] = float(v)
            for k, v in td_stds.items():
                if _is_finite_number(v):
                    stds[str(k)] = float(v)
            if means and stds:
                return means, stds

    fstats = meta.get("feature_stats")
    if isinstance(fstats, Mapping):
        for feat, blob in fstats.items():
            if not isinstance(blob, Mapping):
                continue
            mu = blob.get("mean")
            sd = blob.get("std")
            if _is_finite_number(mu) and _is_finite_number(sd):
                means[str(feat)] = float(mu)
                stds[str(feat)] = float(sd)

    return means, stds


def _reliability_at_bin(
    threshold_metrics: Mapping[str, Any], probability: float
) -> Optional[float]:
    """Return the reliability_slope for the bin closest to ``probability``.

    ``threshold_metrics`` is the ``meta.threshold_metrics`` dict — keys are
    threshold strings like ``"0.55"`` and values are per-bin metric dicts
    that may contain ``reliability_slope``. We pick the key numerically
    closest to ``probability``. Returns ``None`` if no bin has a finite
    slope.
    """
    if not isinstance(threshold_metrics, Mapping):
        return None
    best_bin: Optional[Tuple[float, float]] = None  # (distance, slope)
    for k, v in threshold_metrics.items():
        try:
            bin_prob = float(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, Mapping):
            continue
        slope = v.get("reliability_slope")
        if not _is_finite_number(slope):
            continue
        dist = abs(bin_prob - probability)
        if best_bin is None or dist < best_bin[0]:
            best_bin = (dist, float(slope))
    return None if best_bin is None else best_bin[1]


def _detect_regime_shift(
    feature_window: Optional[Sequence[Mapping[str, Any]]],
    *,
    lookback_bars: int = 3,
) -> Optional[bool]:
    """Heuristic regime-shift detector over the last ``lookback_bars`` bars.

    Looks for boolean/categorical regime fields ("regime", "regime_id",
    "trend_regime") and flags True if the regime in the most recent bar
    differs from any of the prior ``lookback_bars`` bars.

    Returns:
    - ``True`` if a shift was detected.
    - ``False`` if a regime field exists and is stable.
    - ``None`` if no usable regime field is present (caller skips the check).
    """
    if not feature_window:
        return None
    candidate_keys = ("regime", "regime_id", "trend_regime", "market_regime")
    # Look at the last (lookback_bars + 1) bars.
    window = list(feature_window)[-(lookback_bars + 1):]
    if len(window) < 2:
        return None
    for key in candidate_keys:
        values = []
        for bar in window:
            if not isinstance(bar, Mapping):
                continue
            if key in bar and bar[key] is not None:
                values.append(bar[key])
        if len(values) < 2:
            continue
        # Compare the most recent value against any prior value.
        latest = values[-1]
        priors = values[:-1]
        if any(p != latest for p in priors):
            return True
        return False
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SignalForensicsAgent(BaseForensicsAgent):
    """Agent A1 — investigates the model's signal as a loss cause."""

    agent_name = "signal"

    def __init__(
        self,
        *,
        context_store: TradeContextStore,
        timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        meta_base_dir: Optional[str] = None,
        margin_floor_pct: float = DEFAULT_MARGIN_FLOOR_PCT,
        mahalanobis_flag_sigma: float = DEFAULT_MAHALANOBIS_FLAG_SIGMA,
    ) -> None:
        super().__init__(context_store=context_store, timeout_s=timeout_s)
        self.meta_base_dir = meta_base_dir
        self.margin_floor_pct = float(margin_floor_pct)
        self.mahalanobis_flag_sigma = float(mahalanobis_flag_sigma)

    # ------------------------------------------------------------------
    # public contract
    # ------------------------------------------------------------------
    def investigate(self, trade_id: str) -> ForensicsFinding:
        signal = self.context_store.get_signal_snapshot(trade_id)
        if signal is None:
            # No snapshot to analyse — be honest, don't crash. This is
            # distinct from "I crashed" so we return an explicit innocent
            # verdict with a single evidence bullet rather than ``unknown``.
            # The synthesizer can still see we ran cleanly.
            return ForensicsFinding(
                agent="signal",
                verdict="unknown",
                confidence=0.0,
                evidence=[
                    f"no signal snapshot recorded for trade_id={trade_id!r}"
                ],
                severity=1,
                error="missing_signal_snapshot",
            )

        meta = _load_meta_for_symbol(
            signal.symbol, base_dir=self.meta_base_dir
        )

        evidence: List[str] = []
        red_flags = 0
        suggested_action: Optional[Dict[str, Any]] = None

        # ----- Check 1: confidence vs threshold margin -----
        conf = float(signal.model_confidence or 0.0)
        threshold = meta.get("optimal_threshold")
        if not _is_finite_number(threshold):
            threshold = DEFAULT_THRESHOLD_FLOOR
        threshold_f = float(threshold)
        margin = conf - threshold_f
        margin_floor = max(self.margin_floor_pct * threshold_f, 1e-6)
        if 0.0 <= margin <= margin_floor:
            red_flags += 1
            evidence.append(
                f"confidence {conf:.4f} only {margin:.4f} above threshold "
                f"{threshold_f:.4f} (within {self.margin_floor_pct:.0%} margin)"
            )
            if suggested_action is None:
                suggested_action = {
                    "type": "raise_floor",
                    "target": "supervisor.confidence_floor",
                    "from": threshold_f,
                    "to": round(threshold_f + margin_floor, 6),
                    "scope": f"{signal.symbol} only",
                }
        elif margin < 0.0:
            evidence.append(
                f"confidence {conf:.4f} below threshold {threshold_f:.4f} "
                f"(should not have fired)"
            )
            # Below-threshold firing is a different bug class (signal pipe
            # corruption); count as a red flag too.
            red_flags += 1

        # ----- Check 2: feature NaN / inf / zero-quality -----
        feature_buffer = signal.feature_buffer or {}
        if not feature_buffer:
            evidence.append(
                "feature_buffer empty — NaN/inf check skipped "
                "(predictor surface limitation)"
            )
        else:
            bad_features: List[str] = []
            for feat, val in feature_buffer.items():
                # Snapshots round-trip NaN/Inf as None; non-numeric junk is
                # also caught here.
                if val is None:
                    bad_features.append(f"{feat}=None")
                    continue
                if not _is_finite_number(val):
                    bad_features.append(f"{feat}={val!r}")
            if bad_features:
                red_flags += 1
                shown = ", ".join(bad_features[:6])
                evidence.append(
                    f"{len(bad_features)} feature(s) NaN/inf/missing: {shown}"
                    + (" …" if len(bad_features) > 6 else "")
                )
                if suggested_action is None:
                    suggested_action = {
                        "type": "feature_request",
                        "target": "compute_features",
                        "feature": "nan_inf_guard",
                        "details": (
                            f"{len(bad_features)} non-finite features at signal time"
                        ),
                    }

        # ----- Check 3: Mahalanobis distance from training distribution -----
        if not feature_buffer:
            evidence.append(
                "feature_buffer empty — Mahalanobis check skipped"
            )
        else:
            means, stds = _extract_means_stds(meta)
            if not means or not stds:
                evidence.append(
                    "training feature means/stds not present in meta.json — "
                    "Mahalanobis check skipped"
                )
            else:
                dist, n_used = _diag_mahalanobis(
                    feature_buffer, means, stds
                )
                if dist is None or n_used == 0:
                    evidence.append(
                        "no overlapping numeric features between snapshot "
                        "and training stats — Mahalanobis check skipped"
                    )
                else:
                    if dist > self.mahalanobis_flag_sigma * math.sqrt(
                        max(1, n_used)
                    ):
                        red_flags += 1
                        evidence.append(
                            f"Mahalanobis distance {dist:.2f} over {n_used} "
                            f"features exceeds "
                            f"{self.mahalanobis_flag_sigma:.1f}σ threshold "
                            f"({self.mahalanobis_flag_sigma * math.sqrt(n_used):.2f})"
                        )
                        if suggested_action is None:
                            suggested_action = {
                                "type": "retrain",
                                "target": f"{signal.symbol} model",
                                "reason": (
                                    f"feature drift > {self.mahalanobis_flag_sigma:.1f}σ"
                                ),
                            }

        # ----- Check 4: reliability at this probability bin -----
        threshold_metrics = meta.get("threshold_metrics")
        # Try the model_probs dict first (if present); otherwise fall back
        # to model_confidence as the "winning class" probability.
        probs = signal.model_probs or {}
        active_prob: Optional[float] = None
        if probs:
            # Pick the largest finite probability as the model's bet.
            finite = [
                float(v) for v in probs.values() if _is_finite_number(v)
            ]
            if finite:
                active_prob = max(finite)
        if active_prob is None and _is_finite_number(conf):
            active_prob = conf
        if not probs:
            evidence.append(
                "model_probs empty — reliability bin check used confidence "
                "as fallback (predictor surface limitation)"
            )
        if active_prob is None:
            evidence.append(
                "no usable probability for reliability bin lookup — "
                "calibration check skipped"
            )
        else:
            slope = _reliability_at_bin(
                threshold_metrics if isinstance(threshold_metrics, Mapping)
                else {},
                active_prob,
            )
            if slope is None:
                # Fall back to the model-level reliability_slope if available.
                metrics_blob = meta.get("metrics_test") or meta.get("metrics_val")
                if isinstance(metrics_blob, Mapping):
                    fallback = metrics_blob.get("reliability_slope")
                    if _is_finite_number(fallback):
                        slope = float(fallback)
            if slope is None:
                evidence.append(
                    "no reliability_slope available in meta.json — "
                    "calibration check skipped"
                )
            elif slope < 0.0:
                red_flags += 1
                evidence.append(
                    f"model anti-calibrated near p={active_prob:.3f} "
                    f"(reliability_slope={slope:.3f} < 0)"
                )
                if suggested_action is None:
                    suggested_action = {
                        "type": "retrain",
                        "target": f"{signal.symbol} model",
                        "reason": "anti-calibrated reliability slope at trade prob bin",
                    }

        # ----- Check 5: regime mismatch within 3 bars before signal -----
        shifted = _detect_regime_shift(signal.feature_window)
        if shifted is None:
            if signal.feature_window is None:
                evidence.append(
                    "feature_window not captured — regime-shift check skipped"
                )
            else:
                evidence.append(
                    "no regime classifier feature in window — "
                    "regime-shift check skipped"
                )
        elif shifted is True:
            red_flags += 1
            evidence.append(
                "regime shifted within 3 bars before signal (regime classifier "
                "feature changed across recent bars)"
            )
            if suggested_action is None:
                suggested_action = {
                    "type": "feature_request",
                    "target": "compute_features",
                    "feature": "regime_distance",
                }

        # ------------------------------------------------------------------
        # verdict
        # ------------------------------------------------------------------
        if red_flags >= 3:
            verdict = "primary_cause"
            # Map 3 → 0.7, 4 → 0.8, 5 → 0.9 (capped at 0.95).
            confidence = min(0.95, 0.7 + 0.1 * (red_flags - 3))
            severity = 5 if red_flags >= 4 else 4
        elif red_flags >= 1:
            verdict = "contributing"
            # Map 1 → 0.4, 2 → 0.55.
            confidence = 0.4 + 0.15 * (red_flags - 1)
            severity = 3 if red_flags >= 2 else 2
        else:
            verdict = "innocent"
            confidence = 0.0
            severity = 1

        return ForensicsFinding(
            agent="signal",
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            suggested_action=suggested_action,
            severity=severity,
        )


__all__ = [
    "SignalForensicsAgent",
    "DEFAULT_MARGIN_FLOOR_PCT",
    "DEFAULT_MAHALANOBIS_FLAG_SIGMA",
    "DEFAULT_THRESHOLD_FLOOR",
]
