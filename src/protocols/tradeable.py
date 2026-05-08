"""Tradeable Protocol — single execution surface for the supervisor.

Lane D Sub-agent D1. The Protocol abstracts the differences between spot
crypto venues (Coinbase), perpetual-futures venues (Hyperliquid), and
prediction markets (Polymarket) so the supervisor can iterate over a
heterogeneous list of ``Tradeable`` instances without venue-specific
branching.

A Tradeable instance is bound to ONE symbol/market — adapters wrap a
venue connector (e.g. :class:`CoinbaseExchange`) plus a symbol string,
exposing the methods below. The Protocol is :func:`runtime_checkable`
so call sites can guard with ``isinstance(obj, Tradeable)`` for tests
and dynamic config paths.

This module is purely additive: nothing in the legacy live-trading stack
imports from ``src/protocols/`` today. Wiring into the supervisor is
Lane D Sub-agent D2's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    # Reuse the existing OrderResult / Ticker shapes defined alongside the
    # Coinbase connector. Hyperliquid imports the same types from
    # ``exchanges.coinbase`` (see src/exchanges/hyperliquid.py), so a single
    # forward reference here is correct for both adapter families.
    from exchanges.coinbase import OrderResult, Ticker


__all__ = [
    "AssetClass",
    "FeeModel",
    "RiskAttributes",
    "Tradeable",
]


class AssetClass(Enum):
    """High-level taxonomy of what a Tradeable represents.

    Used by the risk engine + position store to decide which sizing logic,
    Kelly divisor, and bookkeeping conventions apply.
    """

    SPOT_CRYPTO = "spot_crypto"
    PERP_CRYPTO = "perp_crypto"
    PREDICTION_BINARY = "prediction_binary"


@dataclass(frozen=True)
class FeeModel:
    """Per-Tradeable fee schedule.

    All rates are decimal (e.g. ``0.001`` for 10 bps). ``settlement_fee_bps``
    is integer basis points charged on settlement (Polymarket levies a
    settlement fee on winning shares; Coinbase / Hyperliquid both default to
    zero). ``gas_fee_usd`` covers on-chain venues that bill gas separately
    from maker/taker (e.g. on-chain Polymarket order placement).
    """

    maker: float
    taker: float
    settlement_fee_bps: int = 0
    gas_fee_usd: float = 0.0


@dataclass(frozen=True)
class RiskAttributes:
    """Per-tick risk-shape data returned by a Tradeable.

    Returned by :meth:`Tradeable.risk_attributes` so the risk engine can
    size + monitor positions without venue-specific branching.

    Attributes:
        kelly_divisor: The denominator the risk engine plugs into a Kelly
            sizing formula. Conventions per asset class:
              * Spot crypto: ``1.0`` (no implied probability — Kelly degenerates
                to direct edge sizing).
              * Perp crypto: ``1.0`` (same as spot; the leverage knob lives on
                the position object, not the divisor).
              * Prediction-binary: ``p * (1 - p)`` — variance of a Bernoulli
                outcome with implied probability ``p``.
        notional_exposure_usd: USD-denominated notional of ``size_base`` units
            at ``entry_price``. Used by the supervisor for portfolio caps.
        liquidation_price: Perp-only. Price at which the position would be
            liquidated by the exchange. ``None`` for spot + binary.
        margin_used_usd: Perp/margin-only. USD margin currently held against
            this hypothetical position. ``None`` for spot + cash-funded
            binary markets.
    """

    kelly_divisor: float
    notional_exposure_usd: float
    liquidation_price: Optional[float] = None
    margin_used_usd: Optional[float] = None


@runtime_checkable
class Tradeable(Protocol):
    """Uniform execution + risk surface across spot, perp, and binary venues.

    Implementers live under ``src/exchanges/adapters/``. Each adapter is
    bound to ONE symbol/market — a single ``CoinbaseExchange`` instance can
    back many ``CoinbaseTradeable`` instances (one per symbol).

    The Protocol is :func:`runtime_checkable`, so call sites can use
    ``isinstance(obj, Tradeable)`` for guard rails. ``runtime_checkable``
    only verifies attribute presence — not signatures — so adapters are
    still expected to honour the documented method shapes.
    """

    # ---- identity / static metadata -----------------------------------

    @property
    def symbol(self) -> str: ...

    @property
    def asset_class(self) -> AssetClass: ...

    @property
    def tick_size(self) -> float: ...

    @property
    def min_size(self) -> float: ...

    @property
    def fee_model(self) -> FeeModel: ...

    # ---- read-only market + account data ------------------------------

    def get_ticker(self) -> "Ticker": ...

    def get_balances(self) -> Dict[str, float]: ...

    # ---- order placement + management ---------------------------------

    def place_market_order(
        self,
        side: Literal["buy", "sell"],
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> "OrderResult": ...

    def place_limit_order(
        self,
        side: Literal["buy", "sell"],
        *,
        base_size: float,
        limit_price: float,
    ) -> "OrderResult": ...

    def cancel_order(self, order_id: str) -> "OrderResult": ...

    def get_open_orders(self) -> List["OrderResult"]: ...

    # ---- risk shape ---------------------------------------------------

    def risk_attributes(
        self,
        *,
        side: Literal["buy", "sell"],
        size_base: float,
        entry_price: float,
    ) -> RiskAttributes: ...
