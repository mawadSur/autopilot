"""Tradeable Protocol adapters for venue-specific exchange connectors.

Each adapter wraps an existing connector (``CoinbaseExchange``,
``HyperliquidExchange``, etc.) plus a single symbol/market id, exposing
the :class:`protocols.Tradeable` surface so the supervisor can drive
heterogeneous venues uniformly.

Lane D Sub-agent D1 ships the spot crypto + perp crypto adapters here.
Lane D Sub-agent D2 will add the prediction-market adapter.
"""

from __future__ import annotations

from .coinbase_tradeable import CoinbaseTradeable

__all__ = [
    "CoinbaseTradeable",
]
