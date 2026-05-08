"""Alpaca stocks connector for the autopilot trading stack.

Thin, hermetic wrapper around Alpaca's HTTPS REST API (Trading + Market
Data). Mirrors the ``CoinbaseExchange`` / ``HyperliquidExchange`` shape
where it makes sense (``get_ticker``, ``get_balances``, ``get_open_orders``,
``cancel_order``, ``place_*_order``) so the supervisor can drive Alpaca
the same way as the existing crypto venues.

Phase P3 (stocks adapter) — landed on top of D1's Tradeable Protocol +
D2's adapter scaffolding.

V1 scope — read-only by default, write behind ``ALPACA_TRADING_ENABLED``:
    Read endpoints (account, clock, calendar, asset metadata, latest
    quotes/bars, open orders, positions) are wired and fully usable.
    Order placement / cancellation are gated behind the
    ``ALPACA_TRADING_ENABLED=true`` env var; until set, write methods
    raise :class:`NotImplementedError` with a clear pointer to the flag.
    This mirrors D1's Hyperliquid pattern (writes deferred until signing
    lands) but with the pivot that Alpaca's auth model is plain HTTPS
    headers — no EIP-712, no SDK dependency — so the operator just needs
    to flip the flag once they've validated paper-trading credentials.

Defense-in-depth:
  - Default ``paper=True`` — operators must explicitly opt into live.
  - Order placement double-checks ``ALPACA_TRADING_ENABLED`` at call time
    (env-var read on each call, not at construction) so toggling the
    flag flips behaviour without restarting the process.
  - All HTTP calls go through :meth:`AlpacaExchange._request`, which
    wraps any error and re-raises as :class:`AlpacaError` with the
    underlying exception preserved on ``__cause__``.
  - Hermetic: the ``requests`` module is the only network surface; tests
    patch ``exchanges.alpaca.requests.get`` / ``.post`` to keep the suite
    offline.

Alpaca API gaps surfaced by this connector:
  - Margin / short-sale support: ``get_account()`` returns
    ``margin_multiplier`` but the adapter currently treats every position
    as cash (``margin_used_usd=None``). Wiring real margin estimation
    requires per-symbol initial-margin lookups that Alpaca does not
    expose for pre-trade sizing — TODO for a future PR.
  - Fractional-share min size: surfaced via ``AlpacaAsset.min_order_size``
    but Alpaca's docs are inconsistent here; the connector trusts the
    asset payload verbatim.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import requests
from pydantic import BaseModel, ConfigDict

from exchanges.coinbase import OrderResult, Ticker

logger = logging.getLogger(__name__)


__all__ = [
    "AlpacaError",
    "AlpacaExchange",
    "AlpacaAccount",
    "AlpacaClock",
    "AlpacaCalendarDay",
    "AlpacaAsset",
    "Position",
    "is_trading_enabled",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AlpacaError(RuntimeError):
    """Raised for any failure originating from the Alpaca connector.

    The underlying exception (network failure, non-2xx HTTP, JSON decode,
    etc.) is preserved on ``__cause__`` via ``raise ... from exc``.
    """


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class AlpacaAccount(BaseModel):
    """Snapshot of the trading account.

    Mirrors the subset of Alpaca's ``GET /v2/account`` response we need
    for sizing + risk decisions. ``margin_multiplier`` is surfaced for
    future margin-aware risk estimation; today the adapter treats all
    positions as cash equity (see module docstring).
    """

    model_config = ConfigDict(extra="ignore")

    account_number: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    pattern_day_trader: bool = False
    margin_multiplier: float = 1.0
    status: str = "ACTIVE"
    currency: str = "USD"


class AlpacaClock(BaseModel):
    """Market-clock snapshot from ``GET /v2/clock``."""

    model_config = ConfigDict(extra="ignore")

    is_open: bool
    next_open: str
    next_close: str
    timestamp: str


class AlpacaCalendarDay(BaseModel):
    """One trading-calendar entry from ``GET /v2/calendar``."""

    model_config = ConfigDict(extra="ignore")

    date: str
    open: str
    close: str


class AlpacaAsset(BaseModel):
    """Asset metadata for a tradeable symbol from ``GET /v2/assets/{symbol}``.

    ``min_order_size`` is exposed by Alpaca as a string; we coerce to
    float. Defaults to ``1.0`` (whole-share) when the venue does not
    surface a value — the asset's ``fractionable`` flag should be the
    deciding factor for callers that care.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    symbol: str
    name: str = ""
    asset_class: str = "us_equity"
    exchange: str = ""
    status: str = "active"
    tradable: bool = True
    marginable: bool = False
    shortable: bool = False
    easy_to_borrow: bool = False
    fractionable: bool = False
    min_order_size: float = 1.0


class Position(BaseModel):
    """Open equity position for the configured account.

    Aligned with Alpaca's ``GET /v2/positions/{symbol}`` payload — we
    keep the shape narrow (symbol, qty, avg entry, market value, pnl)
    because that's all the supervisor's risk engine consumes today.
    """

    model_config = ConfigDict(extra="ignore")

    symbol: str
    qty: float
    side: Literal["long", "short"]
    avg_entry_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float = 0.0
    current_price: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"
_LIVE_BASE_URL = "https://api.alpaca.markets/v2"
_DATA_BASE_URL = "https://data.alpaca.markets/v2"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "on")
    return bool(value)


def is_trading_enabled() -> bool:
    """Return True iff ``ALPACA_TRADING_ENABLED`` is set to a truthy value.

    Read on every call (not cached) so operators can flip the flag
    without restarting the process. The check is the SOLE gate on
    Alpaca write methods — paper vs. live is a separate ``paper=`` flag
    on the connector.
    """
    return _coerce_bool(os.getenv("ALPACA_TRADING_ENABLED", ""))


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class AlpacaExchange:
    """Hermetic wrapper around Alpaca's REST API.

    Args:
        api_key:      Alpaca API key id. Required.
        api_secret:   Alpaca API secret. Required.
        paper:        If True (default), routes trading endpoints to
                      paper-api.alpaca.markets. Set to False for live.
                      Market-data endpoints live on data.alpaca.markets
                      regardless.
        base_url:     Optional override of the trading API base URL.
                      Defaults to paper or live based on ``paper=``.
        data_base_url: Optional override of the market-data base URL.
                      Defaults to ``https://data.alpaca.markets/v2``.
        timeout_s:    Per-request timeout in seconds (default 10s).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        paper: bool = True,
        base_url: Optional[str] = None,
        data_base_url: Optional[str] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._api_secret = (api_secret or "").strip()
        self._paper = bool(paper)
        if base_url is not None:
            self._base_url = base_url.rstrip("/")
        else:
            self._base_url = (_PAPER_BASE_URL if self._paper else _LIVE_BASE_URL)
        self._data_base_url = (data_base_url or _DATA_BASE_URL).rstrip("/")
        self._timeout_s = float(timeout_s)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def is_paper(self) -> bool:
        return self._paper

    def is_trading_enabled(self) -> bool:
        """Convenience wrapper around the module-level env-flag check."""
        return is_trading_enabled()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def data_base_url(self) -> str:
        return self._data_base_url

    # ------------------------------------------------------------------
    # Read-only: account
    # ------------------------------------------------------------------

    def get_account(self) -> AlpacaAccount:
        """Return account equity / buying power / cash snapshot."""
        data = self._request("GET", "/account") or {}
        return AlpacaAccount(
            account_number=str(data.get("account_number") or data.get("id") or ""),
            equity=_coerce_float(data.get("equity")),
            cash=_coerce_float(data.get("cash")),
            buying_power=_coerce_float(data.get("buying_power")),
            portfolio_value=_coerce_float(
                data.get("portfolio_value") or data.get("equity")
            ),
            pattern_day_trader=_coerce_bool(data.get("pattern_day_trader")),
            margin_multiplier=_coerce_float(
                data.get("multiplier") or data.get("margin_multiplier"),
                default=1.0,
            ),
            status=str(data.get("status") or "ACTIVE"),
            currency=str(data.get("currency") or "USD"),
        )

    def get_balances(self) -> Dict[str, float]:
        """Return a flat ``Dict[currency, float]`` of cash balances.

        Alpaca is USD-denominated; this returns ``{"USD": cash}`` (and
        nothing else). For total equity including holdings, callers should
        use :meth:`get_account`.
        """
        acct = self.get_account()
        return {acct.currency: acct.cash}

    # ------------------------------------------------------------------
    # Read-only: market clock + calendar
    # ------------------------------------------------------------------

    def get_clock(self) -> AlpacaClock:
        """Return market-open clock from ``/v2/clock``."""
        data = self._request("GET", "/clock") or {}
        return AlpacaClock(
            is_open=_coerce_bool(data.get("is_open")),
            next_open=str(data.get("next_open") or ""),
            next_close=str(data.get("next_close") or ""),
            timestamp=str(data.get("timestamp") or _utcnow_iso()),
        )

    def get_calendar(
        self, start: str, end: str
    ) -> List[AlpacaCalendarDay]:
        """Return the trading-calendar window between ``start`` and ``end``.

        Both dates are ISO ``YYYY-MM-DD``; Alpaca returns one entry per
        trading day in the inclusive window.
        """
        data = (
            self._request(
                "GET",
                "/calendar",
                params={"start": start, "end": end},
            )
            or []
        )
        out: List[AlpacaCalendarDay] = []
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                out.append(
                    AlpacaCalendarDay(
                        date=str(entry.get("date") or ""),
                        open=str(entry.get("open") or ""),
                        close=str(entry.get("close") or ""),
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Read-only: asset metadata
    # ------------------------------------------------------------------

    def get_asset(self, symbol: str) -> AlpacaAsset:
        """Return asset metadata (tradable, fractionable, shortable, ...)."""
        if not symbol or not isinstance(symbol, str):
            raise AlpacaError(f"Invalid symbol: {symbol!r}")
        sym = symbol.strip().upper()
        data = self._request("GET", f"/assets/{sym}") or {}
        return AlpacaAsset(
            id=str(data.get("id") or ""),
            symbol=str(data.get("symbol") or sym),
            name=str(data.get("name") or ""),
            asset_class=str(data.get("class") or data.get("asset_class") or "us_equity"),
            exchange=str(data.get("exchange") or ""),
            status=str(data.get("status") or "active"),
            tradable=_coerce_bool(data.get("tradable"), default=True),
            marginable=_coerce_bool(data.get("marginable")),
            shortable=_coerce_bool(data.get("shortable")),
            easy_to_borrow=_coerce_bool(data.get("easy_to_borrow")),
            fractionable=_coerce_bool(data.get("fractionable")),
            min_order_size=_coerce_float(
                data.get("min_order_size"),
                default=0.0001 if _coerce_bool(data.get("fractionable")) else 1.0,
            ),
        )

    # ------------------------------------------------------------------
    # Read-only: market data (quotes + bars)
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> Ticker:
        """Top-of-book + last-trade snapshot for ``symbol``.

        Pulls the latest quote from ``/v2/stocks/{symbol}/quotes/latest``
        for bid/ask, then the latest bar from
        ``/v2/stocks/{symbol}/bars/latest`` for last price + 1-minute volume.
        Falls back to the quote's mid for ``last`` if the bar endpoint
        is unavailable. Volume comes from the latest bar (1-minute window),
        which is a coarser signal than 24h volume but is what Alpaca
        provides without rolling our own aggregation.
        """
        if not symbol or not isinstance(symbol, str):
            raise AlpacaError(f"Invalid symbol: {symbol!r}")
        sym = symbol.strip().upper()

        # Latest quote: /v2/stocks/{symbol}/quotes/latest -> {"quote": {...}}
        quote_payload = (
            self._data_request(f"/stocks/{sym}/quotes/latest") or {}
        )
        quote = quote_payload.get("quote") or quote_payload
        bid = _coerce_float(quote.get("bp") or quote.get("bid_price"))
        ask = _coerce_float(quote.get("ap") or quote.get("ask_price"))

        # Latest bar: /v2/stocks/{symbol}/bars/latest -> {"bar": {...}}
        last = 0.0
        volume = 0.0
        try:
            bar_payload = (
                self._data_request(f"/stocks/{sym}/bars/latest") or {}
            )
            bar = bar_payload.get("bar") or bar_payload
            last = _coerce_float(bar.get("c") or bar.get("close"))
            volume = _coerce_float(bar.get("v") or bar.get("volume"))
        except AlpacaError:
            # Bars endpoint can be unavailable on certain accounts/symbols —
            # don't fail the whole ticker fetch over it. Fall back to mid.
            pass

        if last <= 0 and (bid > 0 or ask > 0):
            last = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)

        return Ticker(
            symbol=sym,
            bid=bid,
            ask=ask,
            last=last,
            volume_24h_base=volume,
            as_of_utc=_utcnow_iso(),
        )

    # ------------------------------------------------------------------
    # Read-only: orders + positions
    # ------------------------------------------------------------------

    def get_open_orders(self) -> List[OrderResult]:
        """Return open orders via ``GET /v2/orders?status=open``."""
        data = self._request("GET", "/orders", params={"status": "open"}) or []
        out: List[OrderResult] = []
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                out.append(self._order_from_raw(entry))
        return out

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for ``symbol`` or ``None`` if flat.

        Alpaca returns 404 when no position exists; we map that to
        ``None`` rather than raising. Other 4xx/5xx still raise.
        """
        if not symbol or not isinstance(symbol, str):
            raise AlpacaError(f"Invalid symbol: {symbol!r}")
        sym = symbol.strip().upper()
        try:
            data = self._request("GET", f"/positions/{sym}", _allow_404=True)
        except AlpacaError:
            raise
        if data is None:
            return None
        if not isinstance(data, dict):
            return None
        qty = _coerce_float(data.get("qty"))
        side_raw = str(data.get("side") or ("long" if qty >= 0 else "short")).lower()
        side: Literal["long", "short"] = "long" if side_raw == "long" else "short"
        return Position(
            symbol=str(data.get("symbol") or sym),
            qty=abs(qty),
            side=side,
            avg_entry_price=_coerce_float(data.get("avg_entry_price")),
            market_value=_coerce_float(data.get("market_value")),
            cost_basis=_coerce_float(data.get("cost_basis")),
            unrealized_pl=_coerce_float(data.get("unrealized_pl")),
            current_price=_coerce_float(data.get("current_price")),
        )

    # ------------------------------------------------------------------
    # Write methods — gated behind ALPACA_TRADING_ENABLED
    # ------------------------------------------------------------------

    _NOT_ENABLED_MSG = (
        "Alpaca trading is feature-flag-gated; set ALPACA_TRADING_ENABLED=true "
        "to enable order placement / cancellation. The flag is read on every "
        "call so you can toggle without restarting the process."
    )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        """Place a market order. Pass exactly one of quote_size_usd / base_size.

        Gated behind ``ALPACA_TRADING_ENABLED=true`` — until that env var
        is set, this raises :class:`NotImplementedError` with a clear
        pointer to the flag. Once enabled, calls
        ``POST /v2/orders`` with ``type=market``, ``side=buy|sell``, and
        either ``qty`` (base shares) or ``notional`` (USD).

        Order placement failures DO raise — operators must know if a
        live order was rejected. Compare with the read-only paths above,
        which surface failures as typed exceptions but do not silently
        swallow.
        """
        if not is_trading_enabled():
            raise NotImplementedError(self._NOT_ENABLED_MSG)
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if (quote_size_usd is None) == (base_size is None):
            raise ValueError(
                "Exactly one of quote_size_usd or base_size must be provided"
            )
        if quote_size_usd is not None and quote_size_usd <= 0:
            raise ValueError("quote_size_usd must be positive")
        if base_size is not None and base_size <= 0:
            raise ValueError("base_size must be positive")

        sym = symbol.strip().upper()
        body: Dict[str, Any] = {
            "symbol": sym,
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        if base_size is not None:
            body["qty"] = str(float(base_size))
        else:
            body["notional"] = str(float(quote_size_usd))  # type: ignore[arg-type]

        raw = self._request("POST", "/orders", json=body) or {}
        return self._order_from_raw(
            raw,
            fallback_symbol=sym,
            fallback_side=side,
            fallback_type="market",
            quote_size_usd=quote_size_usd,
            base_size=base_size,
        )

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        *,
        base_size: float,
        limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        """Place a limit order. Gated behind ``ALPACA_TRADING_ENABLED=true``."""
        if not is_trading_enabled():
            raise NotImplementedError(self._NOT_ENABLED_MSG)
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if base_size <= 0:
            raise ValueError("base_size must be positive")
        if limit_price <= 0:
            raise ValueError("limit_price must be positive")

        sym = symbol.strip().upper()
        body: Dict[str, Any] = {
            "symbol": sym,
            "side": side,
            "type": "limit",
            "qty": str(float(base_size)),
            "limit_price": str(float(limit_price)),
            "time_in_force": time_in_force,
        }
        raw = self._request("POST", "/orders", json=body) or {}
        return self._order_from_raw(
            raw,
            fallback_symbol=sym,
            fallback_side=side,
            fallback_type="limit",
            base_size=base_size,
            limit_price=limit_price,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order. Gated behind ``ALPACA_TRADING_ENABLED=true``.

        Alpaca's cancel endpoint returns 204 No Content on success; we
        synthesize an ``OrderResult`` with ``status="cancelled"`` so the
        Tradeable Protocol contract is honoured.
        """
        if not is_trading_enabled():
            raise NotImplementedError(self._NOT_ENABLED_MSG)
        if not order_id or not isinstance(order_id, str):
            raise ValueError(f"order_id must be a non-empty string, got {order_id!r}")
        # DELETE returns 204 with an empty body; we don't parse the response.
        self._request("DELETE", f"/orders/{order_id}", _allow_empty=True)
        return OrderResult(
            order_id=order_id,
            symbol="",
            side="buy",
            type="market",
            status="cancelled",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc=_utcnow_iso(),
            raw_payload={},
        )

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        _allow_404: bool = False,
        _allow_empty: bool = False,
    ) -> Any:
        """Issue a request to the trading API. Returns parsed JSON.

        Wraps requests-level + HTTP-level errors as :class:`AlpacaError`.
        When ``_allow_404`` is True, a 404 returns ``None`` instead of
        raising — used for ``GET /positions/{symbol}`` (404 = flat).
        """
        url = f"{self._base_url}{path}"
        try:
            if method == "GET":
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self._timeout_s,
                )
            elif method == "POST":
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=json,
                    timeout=self._timeout_s,
                )
            elif method == "DELETE":
                resp = requests.delete(
                    url,
                    headers=self._headers(),
                    timeout=self._timeout_s,
                )
            else:
                raise AlpacaError(f"Unsupported HTTP method: {method!r}")
        except AlpacaError:
            raise
        except Exception as exc:
            raise AlpacaError(
                f"Alpaca {method} {path} request failed: {exc}"
            ) from exc

        return self._parse_response(
            resp,
            method=method,
            path=path,
            allow_404=_allow_404,
            allow_empty=_allow_empty,
        )

    def _data_request(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Issue a GET against the market-data API base URL."""
        url = f"{self._data_base_url}{path}"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self._timeout_s,
            )
        except Exception as exc:
            raise AlpacaError(
                f"Alpaca data GET {path} request failed: {exc}"
            ) from exc

        return self._parse_response(resp, method="GET", path=path)

    def _parse_response(
        self,
        resp: Any,
        *,
        method: str,
        path: str,
        allow_404: bool = False,
        allow_empty: bool = False,
    ) -> Any:
        status = getattr(resp, "status_code", 0)
        if status == 404 and allow_404:
            return None
        if status == 204 or (allow_empty and 200 <= int(status or 0) < 300):
            text = getattr(resp, "text", "") or ""
            if not text.strip():
                return None
        if not 200 <= int(status or 0) < 300:
            # Try to surface the API's structured error message.
            api_msg = ""
            try:
                body = resp.json()
                if isinstance(body, dict):
                    api_msg = (
                        body.get("message")
                        or body.get("error")
                        or body.get("code")
                        or ""
                    )
            except Exception:
                api_msg = (getattr(resp, "text", "") or "")[:200]
            raise AlpacaError(
                f"Alpaca {method} {path} returned HTTP {status}: {api_msg}"
            )
        try:
            return resp.json()
        except Exception as exc:
            raise AlpacaError(
                f"Alpaca {method} {path} returned non-JSON body: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Order parsing
    # ------------------------------------------------------------------

    def _order_from_raw(
        self,
        raw: Dict[str, Any],
        *,
        fallback_symbol: str = "",
        fallback_side: str = "buy",
        fallback_type: str = "market",
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        raw = dict(raw or {})
        order_id = str(raw.get("id") or raw.get("client_order_id") or "")
        if not order_id:
            raise AlpacaError("Alpaca order response missing id")

        side = str(raw.get("side") or fallback_side).lower()
        if side not in ("buy", "sell"):
            side = "buy"
        otype = str(raw.get("type") or raw.get("order_type") or fallback_type).lower()
        if otype not in ("market", "limit"):
            otype = fallback_type if fallback_type in ("market", "limit") else "market"

        status = self._map_status(raw.get("status"))
        filled_qty = _coerce_float(raw.get("filled_qty"))
        avg_fill_price_raw = raw.get("filled_avg_price")
        avg_fill_price: Optional[float] = (
            float(avg_fill_price_raw) if avg_fill_price_raw not in (None, "", "null") else None
        )
        cost = (avg_fill_price or 0.0) * filled_qty

        return OrderResult(
            order_id=order_id,
            symbol=str(raw.get("symbol") or fallback_symbol),
            side=side,  # type: ignore[arg-type]
            type=otype,  # type: ignore[arg-type]
            quote_size_usd=quote_size_usd
            if quote_size_usd is not None
            else (_coerce_float(raw.get("notional")) or None),
            base_size=base_size
            if base_size is not None
            else (_coerce_float(raw.get("qty")) or None),
            limit_price=limit_price
            if limit_price is not None
            else (_coerce_float(raw.get("limit_price")) or None),
            status=status,
            filled_base=filled_qty,
            filled_quote_usd=cost,
            avg_fill_price=avg_fill_price,
            fee_usd=0.0,  # Alpaca is commission-free for retail; SEC/TAF fees are negligible.
            created_at_utc=str(raw.get("created_at") or _utcnow_iso()),
            raw_payload=raw,
        )

    @staticmethod
    def _map_status(raw_status: Any) -> Literal["pending", "open", "filled", "cancelled", "rejected"]:
        """Map Alpaca order status onto our normalized vocabulary."""
        if raw_status is None:
            return "pending"
        s = str(raw_status).lower()
        if s in ("new", "accepted", "pending_new", "accepted_for_bidding", "held"):
            return "open"
        if s in ("filled", "done_for_day", "closed"):
            return "filled"
        if s in ("partially_filled",):
            return "open"
        if s in ("canceled", "cancelled", "expired", "replaced", "pending_cancel"):
            return "cancelled"
        if s in ("rejected", "suspended"):
            return "rejected"
        if s in ("pending", "pending_replace"):
            return "pending"
        return "pending"
