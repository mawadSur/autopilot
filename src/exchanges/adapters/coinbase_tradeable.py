"""CoinbaseTradeable — Tradeable Protocol adapter for CoinbaseExchange.

Lane D Sub-agent D1, Commit 2. Wraps an existing
:class:`exchanges.coinbase.CoinbaseExchange` plus a single symbol so the
supervisor can treat Coinbase spot the same way as Hyperliquid perp or
Polymarket binary markets.

A single ``CoinbaseExchange`` instance can back many ``CoinbaseTradeable``
instances (one per symbol). This adapter does NOT modify the underlying
connector; it only delegates with the symbol baked in.

Conventions baked into this adapter:
  * ``asset_class = AssetClass.SPOT_CRYPTO``.
  * Fee schedule defaults to Coinbase Advanced retail (``maker=0.0040``,
    ``taker=0.0060``). Operators can override via :class:`CoinbaseTradeable`'s
    ``fee_model`` constructor kwarg if they're on a tier with different
    rates. The underlying ``CoinbaseExchange`` does not currently expose a
    fee-tier query API; wiring that through is a TODO for a future PR.
  * Spot has no implied probability, so ``risk_attributes.kelly_divisor``
    is ``1.0`` (Kelly degenerates to direct edge sizing). This is the
    documented neutral divisor — the risk engine should treat spot Kelly
    as ``edge`` rather than ``edge / (p * (1 - p))``.
  * ``tick_size`` and ``min_size`` are constructor kwargs because
    ``CoinbaseExchange`` does not surface a public ``products`` metadata
    fetch helper today. Defaults (``0.01`` price tick, ``1e-8`` size)
    are conservative; callers with real product metadata should pass it
    in. TODO: extend ``CoinbaseExchange`` with a ``get_product()`` helper
    in a follow-up so the adapter can populate these defaults from the
    venue.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from exchanges.coinbase import (
    Balance,
    CoinbaseExchange,
    OrderResult,
    Ticker,
)
from protocols import (
    AssetClass,
    FeeModel,
    RiskAttributes,
)


__all__ = ["CoinbaseTradeable"]


# Coinbase Advanced retail tier fee schedule. Operators on a higher
# volume tier should override via the constructor.
_DEFAULT_COINBASE_FEE_MODEL = FeeModel(
    maker=0.0040,  # 40 bps
    taker=0.0060,  # 60 bps
    settlement_fee_bps=0,
    gas_fee_usd=0.0,
)

# Conservative defaults. Real values per product live in Coinbase's
# product metadata; ``CoinbaseExchange`` does not expose a fetch helper
# for them today.
_DEFAULT_TICK_SIZE = 0.01
_DEFAULT_MIN_SIZE = 1e-8


class CoinbaseTradeable:
    """Tradeable Protocol adapter bound to ONE Coinbase spot symbol.

    Args:
        exchange:    A constructed :class:`CoinbaseExchange` to delegate to.
        symbol:      The market symbol (e.g. ``"ETH/USD"``, ``"BTC-USD"``).
                     Stored verbatim and passed through to the underlying
                     connector, which handles its own normalization.
        fee_model:   Optional override of the fee schedule. Defaults to the
                     Coinbase Advanced retail tier (40 bps maker / 60 bps
                     taker).
        tick_size:   Optional override of price tick size. Defaults to
                     ``0.01``.
        min_size:    Optional override of minimum order size in base units.
                     Defaults to ``1e-8``.
    """

    def __init__(
        self,
        exchange: CoinbaseExchange,
        symbol: str,
        *,
        fee_model: Optional[FeeModel] = None,
        tick_size: float = _DEFAULT_TICK_SIZE,
        min_size: float = _DEFAULT_MIN_SIZE,
    ) -> None:
        self._exchange = exchange
        self._symbol = symbol
        self._fee_model = fee_model if fee_model is not None else _DEFAULT_COINBASE_FEE_MODEL
        self._tick_size = float(tick_size)
        self._min_size = float(min_size)

    # ------------------------------------------------------------------
    # Identity / static metadata
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.SPOT_CRYPTO

    @property
    def tick_size(self) -> float:
        return self._tick_size

    @property
    def min_size(self) -> float:
        return self._min_size

    @property
    def fee_model(self) -> FeeModel:
        return self._fee_model

    @property
    def exchange(self) -> CoinbaseExchange:
        """Escape hatch for callers that need the raw connector (logging, debugging)."""
        return self._exchange

    # ------------------------------------------------------------------
    # Read-only market + account data
    # ------------------------------------------------------------------

    def get_ticker(self) -> Ticker:
        return self._exchange.get_ticker(self._symbol)

    def get_balances(self) -> Dict[str, float]:
        """Return total balance per currency as a flat dict.

        The Tradeable Protocol contract is ``Dict[str, float]`` keyed by
        currency. The underlying ``CoinbaseExchange.get_balances()`` returns
        a richer ``List[Balance]`` (``free``, ``locked``, ``total`` per
        currency); we collapse to ``total`` here. Callers that need the
        granular shape can reach through ``self.exchange``.
        """
        balances: List[Balance] = self._exchange.get_balances()
        return {b.currency: b.total for b in balances}

    # ------------------------------------------------------------------
    # Order placement + management
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        side: Literal["buy", "sell"],
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        return self._exchange.place_market_order(
            self._symbol,
            side,
            quote_size_usd=quote_size_usd,
            base_size=base_size,
        )

    def place_limit_order(
        self,
        side: Literal["buy", "sell"],
        *,
        base_size: float,
        limit_price: float,
    ) -> OrderResult:
        return self._exchange.place_limit_order(
            self._symbol,
            side,
            base_size=base_size,
            limit_price=limit_price,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        return self._exchange.cancel_order(order_id, self._symbol)

    def get_open_orders(self) -> List[OrderResult]:
        return self._exchange.get_open_orders(self._symbol)

    # ------------------------------------------------------------------
    # Risk shape
    # ------------------------------------------------------------------

    def risk_attributes(
        self,
        *,
        side: Literal["buy", "sell"],
        size_base: float,
        entry_price: float,
    ) -> RiskAttributes:
        """Return the risk shape for a hypothetical spot position.

        Spot crypto has no implied probability, no liquidation, and no
        margin. We surface:
          * ``kelly_divisor = 1.0`` — Kelly degenerates to direct edge
            sizing for continuous-price spot trades.
          * ``notional_exposure_usd = size_base * entry_price``.
          * ``liquidation_price = None`` and ``margin_used_usd = None``
            (perp-only fields).
        """
        notional = float(size_base) * float(entry_price)
        return RiskAttributes(
            kelly_divisor=1.0,
            notional_exposure_usd=notional,
            liquidation_price=None,
            margin_used_usd=None,
        )
