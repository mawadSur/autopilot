"""Deterministic, auditable regime router (one model, per-regime routing).

Chosen approach (see branch audit): keep the single calibrated ``P(long)``
model, but let a *deterministic* regime classifier switch the decision
threshold — and whether we trade the long side at all — per market regime.

Why deterministic (and not the k-NN ``regime_memory`` store): the k-NN store's
per-window ``optimal_threshold`` / ``regime_label`` metadata is explicitly
synthetic-v0 (see ``regime_memory/backfill.py``) and unvalidated. A rule over
features the model already trusts (``adx`` for trend strength, ``close_over_ema_50``
for direction) is transparent, cheap, leak-free, and — crucially — directly
ablatable by the post-fee branch audit. If the audit shows routing does not pay
for itself post-fee, it stays off (``cfg.use_regime_routing = False``).

Long-only note: the crypto XGBoost models emit binary ``P(long)`` only. So
"routing" here means, per regime: (a) is the long side enabled at all, and
(b) what long threshold to require. There is no short side to disable.

Pure functions + a small dataclass so this is unit-testable in isolation and
usable both in the streaming backtest and at live decision time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

__all__ = [
    "Regime",
    "RegimeParams",
    "RegimeRouter",
    "classify_regime",
]


# Regime labels. Kept as plain strings (not an Enum) so they serialize cleanly
# into the audit JSON and config without custom encoders.
class Regime:
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    ALL = ("trend_up", "trend_down", "chop")


def _get(row: Any, key: str, default: float = float("nan")) -> float:
    """Read a float from a dict-like or pandas Series row; NaN on miss/bad."""
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except (KeyError, IndexError, TypeError):
        return float(default)
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def classify_regime(
    row: Any,
    *,
    adx_trend_min: float = 25.0,
    dir_band: float = 0.0005,
) -> str:
    """Classify one feature row into a trend regime.

    Uses two columns present and fully populated in the crypto datasets:

    * ``adx``               -- trend *strength* (0..100). ``>= adx_trend_min``
                               means "there is a trend"; below it is chop.
    * ``close_over_ema_50`` -- trend *direction*: ``close / ema_50 - 1`` (a
                               fraction). Sign past ``dir_band`` gives up/down.

    Fail-safe: if either input is missing/NaN we return ``CHOP`` — the most
    conservative regime (highest threshold in the default router), never a
    spuriously aggressive one.
    """
    adx = _get(row, "adx")
    direction = _get(row, "close_over_ema_50")
    if not (np.isfinite(adx) and np.isfinite(direction)):
        return Regime.CHOP
    if adx < float(adx_trend_min):
        return Regime.CHOP
    if direction > float(dir_band):
        return Regime.TREND_UP
    if direction < -float(dir_band):
        return Regime.TREND_DOWN
    return Regime.CHOP


@dataclass
class RegimeParams:
    """Per-regime routing knobs for a long-only model."""

    enabled: bool = True
    thr_long: float = 0.5


@dataclass
class RegimeRouter:
    """Route a per-bar long threshold (and enable flag) by detected regime.

    The defaults below are a *hypothesis* to be confirmed by the post-fee branch
    audit, not tuned constants:

    * ``trend_up``   -- with-trend longs, so require a slightly lower threshold.
    * ``chop``       -- no trend edge; demand a higher threshold (trade less).
    * ``trend_down`` -- do not buy into a downtrend at all (``enabled=False``).

    ``adx_trend_min`` / ``dir_band`` are the classifier thresholds (see
    :func:`classify_regime`).
    """

    params: Dict[str, RegimeParams] = field(
        default_factory=lambda: {
            Regime.TREND_UP: RegimeParams(enabled=True, thr_long=0.55),
            Regime.CHOP: RegimeParams(enabled=True, thr_long=0.65),
            Regime.TREND_DOWN: RegimeParams(enabled=False, thr_long=0.99),
        }
    )
    adx_trend_min: float = 25.0
    dir_band: float = 0.0005

    def classify(self, row: Any) -> str:
        return classify_regime(
            row, adx_trend_min=self.adx_trend_min, dir_band=self.dir_band
        )

    def route(self, row: Any) -> tuple[bool, float, str]:
        """Return ``(enabled, thr_long, regime)`` for a feature row."""
        regime = self.classify(row)
        p = self.params.get(regime, RegimeParams())
        return bool(p.enabled), float(p.thr_long), regime

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any]]) -> "RegimeRouter":
        """Build a router from a plain dict (e.g. loaded from JSON/env).

        Shape::

            {
              "adx_trend_min": 25.0,
              "dir_band": 0.0005,
              "params": {
                "trend_up":   {"enabled": true,  "thr_long": 0.55},
                "chop":       {"enabled": true,  "thr_long": 0.65},
                "trend_down": {"enabled": false, "thr_long": 0.99}
              }
            }

        Missing keys fall back to the dataclass defaults.
        """
        router = cls()
        if not cfg:
            return router
        router.adx_trend_min = float(cfg.get("adx_trend_min", router.adx_trend_min))
        router.dir_band = float(cfg.get("dir_band", router.dir_band))
        raw_params = cfg.get("params") or {}
        for regime, spec in raw_params.items():
            if regime not in Regime.ALL or not isinstance(spec, Mapping):
                continue
            base = router.params.get(regime, RegimeParams())
            router.params[regime] = RegimeParams(
                enabled=bool(spec.get("enabled", base.enabled)),
                thr_long=float(spec.get("thr_long", base.thr_long)),
            )
        return router
