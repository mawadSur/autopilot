"""PolymarketTradeable — Tradeable Protocol adapter for Polymarket binary markets.

Lane D Sub-agent D2, Commit 1. Wraps a Polymarket Gamma fetcher (the
existing :mod:`fetcher` module's ``fetch_active_markets``-style client) plus
a single ``market_id`` so the supervisor can drive Polymarket binary markets
the same way as Coinbase spot or Hyperliquid perps.

Conventions baked into this adapter:
  * ``asset_class = AssetClass.PREDICTION_BINARY``.
  * ``symbol`` returns ``f"polymarket:{market_id}"`` so it cannot collide
    with crypto symbols (``ETH/USD``, ``BTC-USD``) when the supervisor
    iterates a heterogeneous tradeables list.
  * ``tick_size = 0.01`` — Polymarket prices trade in cents per share.
  * ``min_size = 1.0`` — one share. The Gamma API does not currently
    surface a per-market floor; this is the conservative floor used by
    the existing scanner.
  * ``fee_model.settlement_fee_bps`` defaults to ``POLYMARKET_FEE_BPS``
    (200 bps, sourced from :mod:`config`). ``maker``/``taker`` are zero —
    Polymarket charges no spread fee, only the settlement fee on winning
    shares (see :func:`risk_management_agent.risk_engine.apply_polymarket_fees`).
  * ``risk_attributes.kelly_divisor = p * (1 - p)`` — the variance of a
    Bernoulli outcome at implied probability ``p``. This is the
    Beta-distribution Kelly divisor for binary markets.

Order placement:
  * ``place_market_order`` writes a ``trade_execution_<market_id>.json``
    log via the same writer used by ``orchestrator.run_final_risk_gate``
    (see :func:`orchestrator._write_trade_execution_log`). No real broker
    connector exists today — the JSON log is the authoritative record.
    The returned ``OrderResult.order_id`` is the basename of the
    written log file so callers can correlate.
  * ``place_limit_order`` and ``cancel_order`` raise
    :class:`NotImplementedError`. Polymarket limit-order signing requires
    the on-chain CLOB client which is intentionally deferred.

TODO follow-ups (broker work):
  * ``get_balances`` returns ``{}`` because the Gamma fetcher does not
    expose an authenticated USDC balance endpoint. A future broker-API
    PR should populate this.
  * ``get_open_orders`` reads the local ``trade_execution_*.json`` files
    for this market_id and returns OrderResult-shaped rows. The "real"
    open-orders concept lives on Polymarket's CLOB; we surface the
    JSON-log view as a best-effort proxy until the broker is wired.
  * ``get_ticker`` calls ``fetcher.fetch_market(market_id)``-style helper
    if available; otherwise returns a stubbed ``Ticker`` based on the
    market's last-known implied probability. Documented as a stub.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from config import POLYMARKET_FEE_BPS
from exchanges.coinbase import OrderResult, Ticker
from protocols import (
    AssetClass,
    FeeModel,
    RiskAttributes,
)


LOGGER = logging.getLogger(__name__)


__all__ = ["PolymarketTradeable"]


# Polymarket trades binary outcome shares in [0, 1] dollars-per-share.
_DEFAULT_TICK_SIZE = 0.01
# One whole share; the Gamma API doesn't surface a per-market minimum.
_DEFAULT_MIN_SIZE = 1.0
# Repo root for trade_execution_*.json. AUTOPILOT_TRADE_STORE overrides.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TRADE_STORE_ENV_VAR = "AUTOPILOT_TRADE_STORE"


def _resolve_trade_store_dir() -> Path:
    """Return the directory where ``trade_execution_<id>.json`` files live.

    Mirrors :func:`storage.sync._default_trade_store_dir`: the
    ``AUTOPILOT_TRADE_STORE`` env var wins when set, otherwise the repo
    root is used (where ``orchestrator._write_trade_execution_log``
    already writes).
    """
    raw = (os.environ.get(_TRADE_STORE_ENV_VAR) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _REPO_ROOT


class PolymarketTradeable:
    """Tradeable Protocol adapter bound to ONE Polymarket binary market.

    Args:
        market_id:    The Polymarket Gamma market id (string). Stored
                      verbatim and used as the namespace component of
                      ``symbol`` and the trade-execution log filename.
        fetcher:      Any object that can answer Polymarket market-data
                      queries — typically the module :mod:`fetcher` (or
                      a stub in tests). The adapter prefers a
                      ``fetch_market(market_id)`` method if available,
                      and otherwise falls back to a stubbed ticker.
        fee_bps:      Settlement fee in basis points. Defaults to
                      :data:`config.POLYMARKET_FEE_BPS` (200 bps).
        gas_fee_usd:  Per-trade gas surcharge. Defaults to ``0.0``.
        trade_store_dir: Override the directory where trade_execution
                      logs are written/read. Defaults to
                      ``$AUTOPILOT_TRADE_STORE`` or the repo root.
        log_writer:   Optional callable ``(*, event_payload, market_id)
                      -> Path`` to override the JSON log writer (used in
                      tests). Defaults to a writer that writes
                      ``<trade_store_dir>/trade_execution_<market_id>.json``.
    """

    def __init__(
        self,
        market_id: str,
        fetcher: Any,
        *,
        fee_bps: int = POLYMARKET_FEE_BPS,
        gas_fee_usd: float = 0.0,
        trade_store_dir: Optional[Path] = None,
        log_writer: Optional[Any] = None,
    ) -> None:
        if not market_id:
            raise ValueError("market_id must be a non-empty string")
        self._market_id = str(market_id)
        self._fetcher = fetcher
        self._fee_model = FeeModel(
            maker=0.0,
            taker=0.0,
            settlement_fee_bps=int(fee_bps),
            gas_fee_usd=float(gas_fee_usd),
        )
        self._trade_store_dir = (
            Path(trade_store_dir) if trade_store_dir is not None else None
        )
        self._log_writer = log_writer

    # ------------------------------------------------------------------
    # Identity / static metadata
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return f"polymarket:{self._market_id}"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.PREDICTION_BINARY

    @property
    def tick_size(self) -> float:
        return _DEFAULT_TICK_SIZE

    @property
    def min_size(self) -> float:
        return _DEFAULT_MIN_SIZE

    @property
    def fee_model(self) -> FeeModel:
        return self._fee_model

    @property
    def market_id(self) -> str:
        """Bare Gamma market id (no ``polymarket:`` prefix). Convenience for callers."""
        return self._market_id

    @property
    def fetcher(self) -> Any:
        """Escape hatch for callers needing the raw Polymarket fetcher."""
        return self._fetcher

    # ------------------------------------------------------------------
    # Read-only market + account data
    # ------------------------------------------------------------------

    def get_ticker(self) -> Ticker:
        """Return a Ticker for this market.

        Tries the fetcher's ``fetch_market(market_id)`` method first
        (returns a :class:`models.Market` with bid/ask/implied prob);
        falls back to a stubbed ticker if no such method exists. The
        stub path is documented because the Gamma API does not expose
        a single-market endpoint; the production fetcher today only
        pages active markets, so callers may need to pre-warm a market
        snapshot before constructing the adapter.
        """
        market = None
        fetch_market = getattr(self._fetcher, "fetch_market", None)
        if callable(fetch_market):
            try:
                market = fetch_market(self._market_id)
            except Exception as exc:  # noqa: BLE001 - read is best-effort
                LOGGER.debug(
                    "PolymarketTradeable.get_ticker: fetch_market(%s) failed: %s",
                    self._market_id,
                    exc,
                )
                market = None

        if market is None:
            # Stubbed fallback. ``last`` is set to 0.5 (max-uncertainty)
            # so callers can detect the absence of a real quote.
            return Ticker(
                symbol=self.symbol,
                bid=0.0,
                ask=0.0,
                last=0.5,
                volume_24h_base=0.0,
                as_of_utc=datetime.now(timezone.utc).isoformat(),
            )

        bid = float(getattr(market, "bid_price", 0.0) or 0.0)
        ask = float(getattr(market, "ask_price", 0.0) or 0.0)
        last = float(getattr(market, "implied_prob", 0.0) or 0.0)
        volume_24h = float(getattr(market, "volume_24h", 0.0) or 0.0)
        return Ticker(
            symbol=self.symbol,
            bid=bid,
            ask=ask,
            last=last,
            volume_24h_base=volume_24h,
            as_of_utc=datetime.now(timezone.utc).isoformat(),
        )

    def get_balances(self) -> Dict[str, float]:
        """Return the operator's USDC balance from Polymarket.

        The Gamma fetcher in this repo does not currently expose an
        authenticated balances endpoint; we therefore probe the fetcher
        for an optional ``get_balances()`` method and fall back to ``{}``
        with a debug log. Returning an empty dict (rather than raising)
        keeps the supervisor's tick loop alive — this is consistent with
        Hyperliquid's own balance fallback when the wallet is unset.
        """
        get_balances = getattr(self._fetcher, "get_balances", None)
        if not callable(get_balances):
            LOGGER.debug(
                "PolymarketTradeable.get_balances: fetcher has no get_balances(); "
                "returning empty mapping (broker integration is a TODO)."
            )
            return {}
        try:
            raw = get_balances()
        except Exception as exc:  # noqa: BLE001 - never crash on read
            LOGGER.debug(
                "PolymarketTradeable.get_balances: fetcher.get_balances raised %s",
                exc,
            )
            return {}
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        return {}

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
        """Write a trade_execution log entry and return an OrderResult.

        Polymarket has no real broker connector in this repo today —
        the JSON log file IS the authoritative trade record (see
        :func:`orchestrator._write_trade_execution_log`). This method
        constructs a minimal event_payload, writes it via the same path
        pattern, and returns an OrderResult whose ``order_id`` is the
        log filename so callers can correlate.

        Sizing semantics:
          * ``quote_size_usd`` is the USD stake (preferred input).
          * ``base_size`` is the share count (alternative).
          * If both are given, ``quote_size_usd`` wins.
          * Either-but-not-both must be set (matches Coinbase/Hyperliquid
            convention).
        """
        if quote_size_usd is None and base_size is None:
            raise ValueError(
                "place_market_order requires quote_size_usd or base_size"
            )

        ticker = self.get_ticker()
        # Use ask for buys, bid for sells, falling back to last when the
        # quote side is empty (the stub-ticker path).
        if str(side).lower() == "buy":
            entry_price = float(ticker.ask) if ticker.ask > 0.0 else float(ticker.last)
        else:
            entry_price = float(ticker.bid) if ticker.bid > 0.0 else float(ticker.last)

        if quote_size_usd is not None:
            sized_quote = float(quote_size_usd)
            sized_base = (sized_quote / entry_price) if entry_price > 0.0 else 0.0
        else:
            sized_base = float(base_size or 0.0)
            sized_quote = sized_base * entry_price

        now_iso = datetime.now(timezone.utc).isoformat()
        event_payload: Dict[str, Any] = {
            "event_id": self._market_id,
            "trade_id": self._market_id,
            "status": "open",
            "created_at_utc": now_iso,
            "settled_at": None,
            "final_outcome": None,
            "market_outcome": None,
            "post_settlement_news": None,
            "scanner": None,
            "features_window": {},
            "model_meta": None,
            "research": None,
            "calibration": None,
            "risk": None,
            "entry_price": entry_price,
            "position_size_usd": sized_quote,
            "exit_price": None,
            "realized_pnl_usd": None,
            "max_loss_usd": sized_quote,
            "source": "supervisor_polymarket_tradeable",
            "notes": f"market_order side={side}",
        }
        log_path = self._write_log(event_payload=event_payload)

        return OrderResult(
            order_id=log_path.name,
            symbol=self.symbol,
            side=str(side),  # type: ignore[arg-type]
            type="market",
            quote_size_usd=sized_quote,
            base_size=sized_base,
            limit_price=None,
            status="open",
            filled_base=sized_base,
            filled_quote_usd=sized_quote,
            avg_fill_price=entry_price,
            fee_usd=0.0,
            created_at_utc=now_iso,
            raw_payload={"log_path": str(log_path)},
        )

    def place_limit_order(
        self,
        side: Literal["buy", "sell"],
        *,
        base_size: float,
        limit_price: float,
    ) -> OrderResult:
        raise NotImplementedError(
            "Polymarket limit orders are not implemented (CLOB client deferred)."
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        raise NotImplementedError(
            "Polymarket order cancellation is not implemented (CLOB client deferred)."
        )

    def get_open_orders(self) -> List[OrderResult]:
        """Return the local trade_execution_*.json entries for THIS market.

        This is a best-effort proxy for "open orders" — the Polymarket
        CLOB tracks the canonical order book, but this repo's broker
        integration is a TODO. We scan the trade-store directory for
        ``trade_execution_<market_id>.json`` (single file per market id)
        and return at most one OrderResult.
        """
        store_dir = self._trade_store_dir or _resolve_trade_store_dir()
        target = store_dir / f"trade_execution_{self._market_id}.json"
        if not target.exists():
            return []
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - never crash on parse
            LOGGER.debug(
                "PolymarketTradeable.get_open_orders: parse %s failed: %s",
                target,
                exc,
            )
            return []
        if not isinstance(payload, dict):
            return []
        # Skip already-settled trades.
        status = str(payload.get("status") or "open").lower()
        if status not in {"open", "pending"}:
            return []
        size_usd = float(payload.get("position_size_usd") or 0.0)
        entry_price = float(payload.get("entry_price") or 0.0)
        base_size = (size_usd / entry_price) if entry_price > 0.0 else 0.0
        return [
            OrderResult(
                order_id=target.name,
                symbol=self.symbol,
                side="buy",  # always-long-YES convention; see orchestrator
                type="market",
                quote_size_usd=size_usd,
                base_size=base_size,
                limit_price=None,
                status="open",
                filled_base=base_size,
                filled_quote_usd=size_usd,
                avg_fill_price=entry_price,
                fee_usd=0.0,
                created_at_utc=str(payload.get("created_at_utc") or ""),
                raw_payload={"log_path": str(target)},
            )
        ]

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
        """Return the risk shape for a hypothetical Polymarket position.

        For binary markets:
          * ``kelly_divisor = p * (1 - p)`` — Bernoulli outcome variance,
            i.e. the Beta-distribution Kelly divisor at implied probability
            ``p = entry_price``. Stays positive but small at the edges
            (e.g. 0.0196 at p=0.02, ~0.0196 at p=0.98).
          * ``notional_exposure_usd = size_base * entry_price``.
          * ``liquidation_price`` and ``margin_used_usd`` are ``None`` —
            cash-funded binary markets have neither.
        """
        p = float(entry_price)
        kelly_divisor = max(0.0, p * (1.0 - p))
        notional = float(size_base) * p
        return RiskAttributes(
            kelly_divisor=kelly_divisor,
            notional_exposure_usd=notional,
            liquidation_price=None,
            margin_used_usd=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_log(self, *, event_payload: Dict[str, Any]) -> Path:
        """Persist the event_payload as a trade_execution log.

        Test seams: callers may inject ``log_writer`` (returns Path) or
        ``trade_store_dir`` (overrides the default repo root) to keep
        unit tests hermetic. The default writer mirrors the path layout
        used by :func:`orchestrator._write_trade_execution_log`.
        """
        if self._log_writer is not None:
            return Path(
                self._log_writer(
                    event_payload=event_payload, market_id=self._market_id
                )
            )
        store_dir = self._trade_store_dir or _resolve_trade_store_dir()
        store_dir.mkdir(parents=True, exist_ok=True)
        output_path = store_dir / f"trade_execution_{self._market_id}.json"
        output_path.write_text(
            json.dumps(event_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return output_path
