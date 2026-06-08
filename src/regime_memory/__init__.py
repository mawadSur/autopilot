"""Regime memory: nearest-neighbor parameter resolution from historical windows.

Phase 3 / E4 (CEO plan, 2026-05-07). The module is purely additive — nothing
in ``src/`` imports it yet; integration into :class:`predictor.XGBoostPredictor`
and :class:`risk_management_agent.calculator.RiskCalculator` is a future PR
(see ``INTEGRATION.md`` next to this file).

Public API
----------
* :class:`RegimeWindow` — frozen dataclass: a single (embedded, metadata)
  record stored in the regime memory.
* :class:`RegimeEncoder` — turns a ``window_size``-bar feature DataFrame into
  a fixed-dim float vector via summary statistics (mean / std / last /
  pct_change). v0 — a learned encoder is a future upgrade.
* :class:`RegimeStore` / :class:`NaiveRegimeStore` — vector store with k-NN
  cosine-similarity lookup. Uses ``faiss-cpu`` when available, else a numpy
  brute-force fallback. Persists to disk as a single ``.npz``.
* :class:`RegimeLookup` — combines the encoder + store + a default-param
  fallback, returns a similarity-weighted dict of resolved params for the
  current window plus a ``_regime_confidence`` score.

The :mod:`backfill` submodule is a CLI / library entry point for ingesting a
historical OHLCV parquet into a persisted store. It is intentionally not
imported here — running it triggers heavy pandas IO that no test needs.
"""

from __future__ import annotations

from regime_memory.encoder import RegimeEncoder, RegimeWindow

__all__ = [
    "RegimeEncoder",
    "RegimeWindow",
]

# Lazy submodule re-exports so the package is importable even if
# optional submodules (or their deps) aren't present yet — handy for
# the four-commit roll-out where ``store`` and ``lookup`` land later.
try:
    from regime_memory.store import NaiveRegimeStore, RegimeStore  # noqa: F401

    __all__.extend(["NaiveRegimeStore", "RegimeStore"])
except Exception:  # pragma: no cover - optional submodule
    pass

try:
    from regime_memory.lookup import RegimeLookup  # noqa: F401

    __all__.append("RegimeLookup")
except Exception:  # pragma: no cover - optional submodule
    pass
