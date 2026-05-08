"""Cross-Asset Alpha Discovery Lab — nightly correlation miner skeleton.

Phase 4 / E2 (CEO plan, 2026-05-07). The module is purely additive — nothing
in ``src/`` imports it yet; integration into the live feature pipeline is a
future PR (see ``INTEGRATION.md`` next to this file).

Public API
----------
* :class:`FeaturePair` / :class:`CorrelationResult` — frozen result types
  emitted by the miner.
* :class:`CorrelationMiner` — runs a cartesian product over feature pairs
  across asset classes and horizons, computing rank-IC (Spearman) for each
  tuple over a rolling window.
* :class:`FeatureSource` — :pep:`544` Protocol implemented by data adapters
  (crypto OHLCV, Polymarket macro-market timeseries, etc.). Adapters live in
  :mod:`alpha_lab.feature_sources`.
* :class:`AutoPromotionGate` / :class:`PromotionCandidate` — rolling-window
  rank-IC threshold gate that emits human-review candidates rather than
  auto-applying. Mirrors :mod:`calibration_agent.outcome_weight_adjuster`'s
  bounded + audit-log discipline.
* :class:`NightlyRunner` — orchestrator entrypoint that runs miner -> gate ->
  daily summary JSON. CLI: ``python -m alpha_lab.nightly_runner``.

The full backfill + production wiring (Coinbase REST, Polymarket macro fetch,
cron scheduling, promotion-queue advancement) is a future PR. This module is
the SKELETON: modules, interfaces, smoke-test-level functionality.
"""

from __future__ import annotations

from alpha_lab.correlation_miner import (
    CorrelationMiner,
    CorrelationResult,
    FeaturePair,
    FeatureSource,
)

__all__ = [
    "CorrelationMiner",
    "CorrelationResult",
    "FeaturePair",
    "FeatureSource",
]

# Lazy submodule re-exports — keep the package importable even when an optional
# downstream module is being iterated on. Mirrors ``regime_memory/__init__.py``.
try:
    from alpha_lab.auto_promotion_gate import (  # noqa: F401
        AutoPromotionGate,
        PromotionCandidate,
    )

    __all__.extend(["AutoPromotionGate", "PromotionCandidate"])
except Exception:  # pragma: no cover - optional submodule
    pass

try:
    from alpha_lab.nightly_runner import NightlyRunner  # noqa: F401

    __all__.append("NightlyRunner")
except Exception:  # pragma: no cover - optional submodule
    pass

try:
    from alpha_lab.feature_sources import (  # noqa: F401
        CryptoFeatureSource,
        PolymarketFeatureSource,
    )

    __all__.extend(["CryptoFeatureSource", "PolymarketFeatureSource"])
except Exception:  # pragma: no cover - optional submodule
    pass
