"""HyperliquidTradeable — Tradeable Protocol adapter for HyperliquidExchange.

Lane D Sub-agent D1, Commit 3. Wraps an existing
:class:`exchanges.hyperliquid.HyperliquidExchange` plus a single perp
symbol so the supervisor can drive Hyperliquid the same way as Coinbase
spot or Polymarket binary markets.

Conventions baked into this adapter:
  * ``asset_class = AssetClass.PERP_CRYPTO``.
  * Fee schedule defaults to Hyperliquid's published perp tier
    (``maker=0.0002``, ``taker=0.0005`` — 2 bps / 5 bps). Operators with
    a different tier can override via the ``fee_model`` kwarg.
  * Read methods (``get_ticker``, ``get_balances``, ``get_open_orders``)
    delegate to the wrapped client.
  * Write methods (``place_market_order``, ``place_limit_order``,
    ``cancel_order``) raise :class:`NotImplementedError`. This mirrors
    the current state of :class:`HyperliquidExchange` — write actions
    require EIP-712 signing of the order body, which is intentionally
    deferred (see ``HyperliquidExchange._NOT_IMPL_MSG``).
  * ``risk_attributes`` returns ``kelly_divisor=1.0`` (perps + spot share
    the divisor convention; leverage lives on the position object, not
    the divisor). ``liquidation_price`` and ``margin_used_usd`` are
    pulled from the wrapped client's ``get_open_positions`` call when a
    matching position exists; otherwise both fields are ``None`` because
    the bare ``HyperliquidExchange`` does not surface an estimator for
    a hypothetical (not-yet-open) position. Populating those fields for
    pre-trade sizing requires a margin-tier API call that is not wired
    through today — that's a TODO for D2/D3 or a follow-up.

This adapter does NOT modify the underlying connector. ``tick_size`` and
``min_size`` come from constructor kwargs because the existing client
does not expose a ``meta`` query helper today; conservative defaults
are used.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from exchanges.coinbase import OrderResult, Ticker
from exchanges.hyperliquid import HyperliquidExchange
from protocols import (
    AssetClass,
    FeeModel,
    RiskAttributes,
)


__all__ = ["HyperliquidTradeable"]


_NOT_IMPL_MSG = (
    "Hyperliquid write methods require EIP-712 signing — intentionally "
    "deferred. The wrapped HyperliquidExchange raises the same error; "
    "this adapter mirrors that behaviour."
)


# Hyperliquid published perp fee schedule (default tier).
_DEFAULT_HYPERLIQUID_FEE_MODEL = FeeModel(
    maker=0.0002,  # 2 bps
    taker=0.0005,  # 5 bps
    settlement_fee_bps=0,
    gas_fee_usd=0.0,
)

# Conservative defaults — Hyperliquid's per-asset metadata isn't surfaced
# by the existing client today.
_DEFAULT_TICK_SIZE = 0.01
_DEFAULT_MIN_SIZE = 0.0001


class HyperliquidTradeable:
    """Tradeable Protocol adapter bound to ONE Hyperliquid perp symbol.

    Args:
        client:      A constructed :class:`HyperliquidExchange` to delegate to.
        symbol:      The perp symbol (e.g. ``"ETH"``, ``"BTC"``, ``"ETH-PERP"``).
                     Stored verbatim and passed through to the underlying
                     connector, which handles its own normalization.
        fee_model:   Optional override of the fee schedule. Defaults to the
                     published Hyperliquid perp tier (2 bps maker / 5 bps
                     taker).
        tick_size:   Optional override of price tick size. Defaults to ``0.01``.
        min_size:    Optional override of minimum order size in base units.
                     Defaults to ``0.0001``.
    """

    def __init__(
        self,
        client: HyperliquidExchange,
        symbol: str,
        *,
        fee_model: Optional[FeeModel] = None,
        tick_size: float = _DEFAULT_TICK_SIZE,
        min_size: float = _DEFAULT_MIN_SIZE,
    ) -> None:
        self._client = client
        self._symbol = symbol
        self._fee_model = (
            fee_model if fee_model is not None else _DEFAULT_HYPERLIQUID_FEE_MODEL
        )
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
        return AssetClass.PERP_CRYPTO

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
    def client(self) -> HyperliquidExchange:
        """Escape hatch for callers that need the raw connector."""
        return self._client

    # ------------------------------------------------------------------
    # Read-only market + account data
    # ------------------------------------------------------------------

    def get_ticker(self) -> Ticker:
        return self._client.get_ticker(self._symbol)

    def get_balances(self) -> Dict[str, float]:
        """Return total balance per currency as a flat dict.

        Hyperliquid is USDC-margined, so this typically returns a single
        ``USDC`` row. The Tradeable Protocol contract is
        ``Dict[str, float]`` keyed by currency; we collapse to ``total``.
        """
        balances = self._client.get_balances()
        return {b.currency: b.total for b in balances}

    def get_open_orders(self) -> List[OrderResult]:
        return self._client.get_open_orders(self._symbol)

    # ------------------------------------------------------------------
    # Write methods — V1: not implemented (require EIP-712 signing)
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        side: Literal["buy", "sell"],
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        raise NotImplementedError(_NOT_IMPL_MSG)

    def place_limit_order(
        self,
        side: Literal["buy", "sell"],
        *,
        base_size: float,
        limit_price: float,
    ) -> OrderResult:
        raise NotImplementedError(_NOT_IMPL_MSG)

    def cancel_order(self, order_id: str) -> OrderResult:
        raise NotImplementedError(_NOT_IMPL_MSG)

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
        """Return the risk shape for a hypothetical perp position.

        For perps:
          * ``kelly_divisor = 1.0`` (Kelly degenerates to direct edge sizing;
            the leverage knob lives on the position object, not the
            divisor).
          * ``notional_exposure_usd = size_base * entry_price``.
          * ``liquidation_price`` and ``margin_used_usd`` are populated
            from an existing open position on the same symbol when one
            exists. For pre-trade sizing of a not-yet-open position both
            fields are ``None`` because the existing
            :class:`HyperliquidExchange` does not surface a margin-tier
            estimator. That gap is a TODO for D2/D3 (or whoever wires
            ``meta`` query support into the client).
        """
        notional = float(size_base) * float(entry_price)
        liquidation_price: Optional[float] = None
        margin_used_usd: Optional[float] = None

        # Best-effort: if the wallet already holds an open position on this
        # symbol, surface its real liquidation_price + margin_used_usd. The
        # call is wrapped in a broad except so a missing/unset wallet
        # address (clearinghouseState requires it) does not break risk
        # sizing for callers that just want notional exposure.
        try:
            positions = self._client.get_open_positions()
        except Exception:
            positions = []

        norm_target = self._symbol.upper().split("-", 1)[0].split("/", 1)[0]
        for pos in positions:
            if pos.symbol.upper() == norm_target:
                liquidation_price = pos.liquidation_price
                margin_used_usd = pos.margin_used_usd
                break

        return RiskAttributes(
            kelly_divisor=1.0,
            notional_exposure_usd=notional,
            liquidation_price=liquidation_price,
            margin_used_usd=margin_used_usd,
        )
