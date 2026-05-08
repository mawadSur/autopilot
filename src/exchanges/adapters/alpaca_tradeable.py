"""AlpacaTradeable — Tradeable Protocol adapter for AlpacaExchange.

Phase P3 — unlocks the spot-equity asset class on top of D1's Tradeable
Protocol + D2's adapter scaffolding. Wraps an existing
:class:`exchanges.alpaca.AlpacaExchange` plus a single stock symbol so
the supervisor can drive Alpaca the same way as Coinbase spot or
Polymarket binary markets.

Conventions baked into this adapter:
  * ``asset_class = AssetClass.SPOT_EQUITY`` — added to the enum in this
    PR (the only existing-source modification; everything else is purely
    additive). Backward compatible with all existing values.
  * Fee schedule defaults to ``FeeModel(maker=0.0, taker=0.0)``. Alpaca
    is commission-free for retail US equities; SEC + TAF settlement fees
    do exist but they are sub-bp and typically waived/absorbed by the
    broker for casual flows. Operators with a high-volume routing tier
    can override via the ``fee_model`` kwarg.
  * ``tick_size = 0.01`` (US equities trade in cent ticks above $1.00 per
    Reg NMS Rule 612). Sub-penny tick markets exist but are not exposed.
  * ``min_size`` defaults to ``1.0`` (whole share). Callers wiring
    fractional-share symbols should pass ``min_size=0.0001`` explicitly,
    or use :meth:`AlpacaTradeable.from_asset` (a future helper) once
    asset metadata caching is wired.
  * Read methods (``get_ticker``, ``get_balances``, ``get_open_orders``)
    delegate to the wrapped client.
  * Write methods (``place_market_order``, ``place_limit_order``,
    ``cancel_order``) delegate — gating is owned by the underlying
    :class:`AlpacaExchange` (env var ``ALPACA_TRADING_ENABLED``). The
    adapter does NOT re-check the flag; it merely passes through, so a
    single source-of-truth governs the gate.
  * ``risk_attributes`` returns ``kelly_divisor=1.0`` (matches the
    spot-crypto convention — Kelly degenerates to direct edge sizing for
    continuous-price markets). ``liquidation_price`` and
    ``margin_used_usd`` are ``None`` because the adapter treats every
    position as cash equity. Margin-account support exists on Alpaca
    (``account.margin_multiplier``) but estimating per-position margin
    requires per-symbol initial-margin lookups that the venue does not
    expose for pre-trade sizing — TODO for a future PR.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from exchanges.alpaca import AlpacaExchange
from exchanges.coinbase import OrderResult, Ticker
from protocols import (
    AssetClass,
    FeeModel,
    RiskAttributes,
)


__all__ = ["AlpacaTradeable"]


# Alpaca is commission-free for retail US equities. Sub-penny SEC/TAF
# settlement fees exist but are typically absorbed by the broker.
_DEFAULT_ALPACA_FEE_MODEL = FeeModel(
    maker=0.0,
    taker=0.0,
    settlement_fee_bps=0,
    gas_fee_usd=0.0,
)

# US equities trade in cent ticks (Reg NMS Rule 612).
_DEFAULT_TICK_SIZE = 0.01
# Whole-share default. Fractional symbols should override.
_DEFAULT_MIN_SIZE = 1.0


class AlpacaTradeable:
    """Tradeable Protocol adapter bound to ONE Alpaca equity symbol.

    Args:
        exchange:    A constructed :class:`AlpacaExchange` to delegate to.
        symbol:      The stock symbol (e.g. ``"AAPL"``, ``"MSFT"``). The
                     adapter uppercases at construction for downstream
                     consistency; the public ``symbol`` property returns
                     a namespaced ``"alpaca:AAPL"`` to mirror the
                     ``PolymarketTradeable`` ``polymarket:...`` shape.
        fee_model:   Optional override of the fee schedule. Defaults to
                     ``maker=0.0, taker=0.0`` (Alpaca commission-free).
        tick_size:   Optional override of price tick size. Defaults to
                     ``0.01`` (cent ticks per Reg NMS).
        min_size:    Optional override of minimum order size in base
                     units. Defaults to ``1.0`` (whole share). Pass
                     ``0.0001`` for fractional-share symbols.
    """

    def __init__(
        self,
        exchange: AlpacaExchange,
        symbol: str,
        *,
        fee_model: Optional[FeeModel] = None,
        tick_size: float = _DEFAULT_TICK_SIZE,
        min_size: float = _DEFAULT_MIN_SIZE,
    ) -> None:
        self._exchange = exchange
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")
        self._raw_symbol = symbol.strip().upper()
        self._fee_model = (
            fee_model if fee_model is not None else _DEFAULT_ALPACA_FEE_MODEL
        )
        self._tick_size = float(tick_size)
        self._min_size = float(min_size)

    # ------------------------------------------------------------------
    # Identity / static metadata
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        """Namespaced symbol — ``"alpaca:<TICKER>"``.

        Mirrors ``PolymarketTradeable``'s ``"polymarket:<id>"`` shape so
        the supervisor's heterogeneous-symbol routing can disambiguate
        venues from a single string.
        """
        return f"alpaca:{self._raw_symbol}"

    @property
    def raw_symbol(self) -> str:
        """The bare ticker (no namespace) for delegating to the connector."""
        return self._raw_symbol

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.SPOT_EQUITY

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
    def exchange(self) -> AlpacaExchange:
        """Escape hatch for callers that need the raw connector."""
        return self._exchange

    # ------------------------------------------------------------------
    # Read-only market + account data
    # ------------------------------------------------------------------

    def get_ticker(self) -> Ticker:
        return self._exchange.get_ticker(self._raw_symbol)

    def get_balances(self) -> Dict[str, float]:
        """Return USD cash balance as ``Dict[str, float]``.

        :class:`AlpacaExchange.get_balances` already returns the flat
        ``Dict[currency, float]`` shape required by the Tradeable
        Protocol, so this is a straight delegate.
        """
        return self._exchange.get_balances()

    def get_open_orders(self) -> List[OrderResult]:
        """Return open orders. Alpaca's ``get_open_orders`` is account-wide
        (no per-symbol filter exposed today); we delegate verbatim."""
        return self._exchange.get_open_orders()

    # ------------------------------------------------------------------
    # Order placement + management — gating owned by AlpacaExchange
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        side: Literal["buy", "sell"],
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        """Place a market order. Gated by ``ALPACA_TRADING_ENABLED`` env var
        on the underlying connector — the adapter does NOT re-check, so
        a single source-of-truth governs the flag."""
        return self._exchange.place_market_order(
            self._raw_symbol,
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
        """Place a limit order. Gated by ``ALPACA_TRADING_ENABLED``."""
        return self._exchange.place_limit_order(
            self._raw_symbol,
            side,
            base_size=base_size,
            limit_price=limit_price,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order. Gated by ``ALPACA_TRADING_ENABLED``."""
        return self._exchange.cancel_order(order_id)

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
        """Return the risk shape for a hypothetical equity position.

        Spot equity has no implied probability and no liquidation; we
        treat it as cash for V1:
          * ``kelly_divisor = 1.0`` — matches CoinbaseTradeable spot
            convention (Kelly degenerates to direct edge sizing for
            continuous-price markets).
          * ``notional_exposure_usd = size_base * entry_price``.
          * ``liquidation_price = None`` and ``margin_used_usd = None``.
            Alpaca margin accounts have a non-trivial margin model
            (Reg-T 50%, FINRA 25% maintenance, day-trading 25%) but
            populating per-position margin pre-trade requires symbol-
            level initial-margin lookups that Alpaca does not expose
            today. TODO: when the venue surfaces an estimator, derive
            ``margin_used_usd = notional / account.margin_multiplier``
            for short-and-margined sides.
        """
        notional = float(size_base) * float(entry_price)
        return RiskAttributes(
            kelly_divisor=1.0,
            notional_exposure_usd=notional,
            liquidation_price=None,
            margin_used_usd=None,
        )
