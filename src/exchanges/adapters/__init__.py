"""Tradeable Protocol adapters for venue-specific exchange connectors.

Each adapter wraps an existing connector (``CoinbaseExchange``,
``HyperliquidExchange``, the Polymarket Gamma fetcher, etc.) plus a single
symbol/market id, exposing the :class:`protocols.Tradeable` surface so the
supervisor can drive heterogeneous venues uniformly.

Lane D Sub-agent D1 shipped the spot crypto + perp crypto adapters
(``CoinbaseTradeable``, ``HyperliquidTradeable``).
Lane D Sub-agent D2 adds the prediction-market adapter
(``PolymarketTradeable``).
"""

from __future__ import annotations

from .coinbase_tradeable import CoinbaseTradeable
from .hyperliquid_tradeable import HyperliquidTradeable
from .polymarket_tradeable import PolymarketTradeable

__all__ = [
    "CoinbaseTradeable",
    "HyperliquidTradeable",
    "PolymarketTradeable",
]
