"""Protocol definitions shared across the autopilot stacks.

Currently exposes the :class:`Tradeable` Protocol (Lane D Sub-agent D1) plus
its supporting dataclasses :class:`AssetClass`, :class:`FeeModel`, and
:class:`RiskAttributes`. Adapters that wrap a venue-specific connector
(``CoinbaseExchange``, ``HyperliquidExchange``, Polymarket fetcher, etc.)
to fit this Protocol live under ``src/exchanges/adapters/``.
"""

from __future__ import annotations

from .tradeable import (
    AssetClass,
    FeeModel,
    RiskAttributes,
    Tradeable,
)

__all__ = [
    "AssetClass",
    "FeeModel",
    "RiskAttributes",
    "Tradeable",
]
