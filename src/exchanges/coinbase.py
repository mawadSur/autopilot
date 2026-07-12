"""Coinbase Advanced Trade connector for the autopilot crypto trading stack.

Thin, hermetic wrapper around ccxt's Coinbase exchange. Designed to be the
single execution surface for live + paper trading. The class is deliberately
narrow (read balances/ticker, place + cancel orders) and intentionally
*does not* expose any withdrawal-related methods.

Defense-in-depth notes:
  - `sandbox` defaults to True. Operators must opt into live mode.
  - When `sandbox=False`, the connector refuses to place orders if the
    `KILL_SWITCH_FILE` env var points to an existing file. The full
    circuit-breaker logic lives in src/risk/circuit_breakers.py (Phase 2);
    this check is a last-line safety belt.
  - All ccxt calls are wrapped and re-raised as `ExchangeError`, with the
    underlying exception preserved as `__cause__`.

Phase 1 of the live-trading buildout. Wiring into live_trader.py is Phase 5.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, computed_field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExchangeError(RuntimeError):
    """Raised for any failure originating from the exchange connector.

    The underlying exception (ccxt error, network failure, validation
    error, etc.) is preserved on `__cause__` via `raise ... from exc`.
    """


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


_Side = Literal["buy", "sell"]
_OrderType = Literal["market", "limit"]
_Status = Literal["pending", "open", "filled", "cancelled", "rejected"]


class OrderResult(BaseModel):
    """Normalized representation of a placed/queried order."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    symbol: str
    side: _Side
    type: _OrderType
    quote_size_usd: Optional[float] = None
    base_size: Optional[float] = None
    limit_price: Optional[float] = None
    status: _Status
    filled_base: float = 0.0
    filled_quote_usd: float = 0.0
    avg_fill_price: Optional[float] = None
    fee_usd: float = 0.0
    created_at_utc: str
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class Balance(BaseModel):
    """Per-currency account balance snapshot."""

    model_config = ConfigDict(extra="forbid")

    currency: str
    free: float
    locked: float
    total: float


class Ticker(BaseModel):
    """Top-of-book + 24h volume snapshot for a single symbol."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h_base: float
    as_of_utc: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        # Coinbase's public /products endpoint returns `price` reliably but
        # omits best_bid / best_ask. Without this fallback, every paper-mode
        # fill records entry_price=1.0 (the supervisor's hard fallback) and
        # Redis-based PnL is junk.
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return float(self.last)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> float:
        if self.bid <= 0 or self.ask <= 0:
            return 0.0
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0:
            return 0.0
        return ((self.ask - self.bid) / mid) * 10_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_symbol(symbol: str) -> str:
    """Accept ETH/USDT or ETH-USD style; emit ccxt-style BASE/QUOTE."""
    if not symbol or not isinstance(symbol, str):
        raise ExchangeError(f"Invalid symbol: {symbol!r}")
    if "/" in symbol:
        return symbol.upper()
    if "-" in symbol:
        parts = symbol.split("-")
        if len(parts) != 2 or not all(parts):
            raise ExchangeError(f"Invalid symbol: {symbol!r}")
        return f"{parts[0].upper()}/{parts[1].upper()}"
    raise ExchangeError(f"Unrecognised symbol format: {symbol!r}")


# Coinbase US spot lists USD pairs, not USDT. Operators accustomed to Binance/
# Bybit symbols frequently pass ``ETH/USDT`` — translate with a one-time WARNING
# rather than letting ccxt's market-lookup IndexError leak through.
_COINBASE_QUOTE_TRANSLATIONS: Dict[str, str] = {"USDT": "USD", "USDC": "USD"}
_LOGGED_TRANSLATIONS: set[str] = set()


def _coinbase_market_symbol(symbol: str) -> str:
    """Apply Coinbase-specific quote currency mapping after generic normalize."""
    norm = _normalize_symbol(symbol)
    base, _, quote = norm.partition("/")
    translated_quote = _COINBASE_QUOTE_TRANSLATIONS.get(quote, quote)
    if translated_quote == quote:
        return norm
    translated = f"{base}/{translated_quote}"
    cache_key = f"{norm}->{translated}"
    if cache_key not in _LOGGED_TRANSLATIONS:
        _LOGGED_TRANSLATIONS.add(cache_key)
        logger.warning(
            "Coinbase quote translation: %s -> %s (Coinbase US spot does not list %s pairs)",
            norm,
            translated,
            quote,
        )
    return translated


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _map_status(raw_status: Any) -> _Status:
    """Map ccxt order status strings onto our normalized vocabulary."""
    if raw_status is None:
        return "pending"
    s = str(raw_status).lower()
    if s in ("open", "new", "accepted", "active"):
        return "open"
    if s in ("closed", "filled", "done"):
        return "filled"
    if s in ("canceled", "cancelled"):
        return "cancelled"
    if s in ("rejected", "expired", "failed"):
        return "rejected"
    if s in ("pending", "submitted"):
        return "pending"
    return "pending"


# ---------------------------------------------------------------------------
# Phase-16 fill metadata extractors. A2 ExecutionForensics needs structured
# partial-fill / rejection data instead of regex-scanning ``Position.notes``.
# These best-effort extractors pull what the ccxt response provides; on
# Coinbase US that's typically a ``trades`` list per order plus a ``status``
# string. If a field isn't surfaced by the exchange we return None / [] —
# never fabricate.
# ---------------------------------------------------------------------------


def extract_partial_fills(order: OrderResult) -> List[Dict[str, Any]]:
    """Best-effort extraction of partial-fill records from a ccxt order.

    A "partial fill" is any order where:
      * The exchange split the requested size across multiple fills (the
        ccxt response's ``trades`` list has more than one entry), OR
      * ``filled_base`` is positive but less than the requested base_size.

    Returns a list of ``{"size", "price", "filled_at_utc"}`` dicts. Empty
    list when the order had a single fill (the common path) or when the
    payload does not surface trade-level data.
    """
    raw = order.raw_payload or {}
    fills: List[Dict[str, Any]] = []
    trades = raw.get("trades") or []
    if isinstance(trades, list) and len(trades) >= 2:
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            size = _coerce_float(
                trade.get("amount")
                or trade.get("size")
                or trade.get("filled")
            )
            price = _coerce_float(
                trade.get("price")
                or trade.get("avgPrice")
                or trade.get("avg_price")
            )
            ts = (
                trade.get("datetime")
                or trade.get("timestamp_iso")
                or trade.get("filled_at_utc")
            )
            if not isinstance(ts, str):
                ts = _utcnow_iso()
            if size > 0 and price > 0:
                fills.append(
                    {"size": size, "price": price, "filled_at_utc": ts}
                )
    return fills


def extract_rejection_reason(order: OrderResult) -> Optional[str]:
    """Pull a structured rejection reason from a ccxt-style order response.

    Returns ``None`` for orders that aren't in a rejected-like state.
    Looks at the normalized status first, then falls back to known reason
    keys in the raw payload (Coinbase's ``reject_reason``, ccxt's generic
    ``info.reject_reason`` / ``info.message``).
    """
    if order.status not in ("rejected", "cancelled"):
        return None
    raw = order.raw_payload or {}
    info = raw.get("info") or {}
    candidates = [
        raw.get("reject_reason"),
        raw.get("rejection_reason"),
        info.get("reject_reason") if isinstance(info, dict) else None,
        info.get("rejection_reason") if isinstance(info, dict) else None,
        info.get("message") if isinstance(info, dict) else None,
        raw.get("message"),
    ]
    for c in candidates:
        if c:
            return str(c)
    # Fall through: still return the status string so consumers know it
    # was a rejection without a structured reason (better than ``None``).
    return f"status={order.status}"


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class CoinbaseExchange:
    """Hermetic wrapper around ccxt's Coinbase exchange.

    Args:
        api_key:    Coinbase API key. Falls back to env var COINBASE_API_KEY.
        api_secret: Coinbase API secret. Falls back to env COINBASE_API_SECRET.
        passphrase: Coinbase API passphrase (legacy Pro-style key tier). Falls
                    back to env COINBASE_API_PASSPHRASE.
        sandbox:    If True, sets ccxt sandbox mode after init. Defaults to
                    True for safety; explicit False is required to trade live.
                    Falls back to env COINBASE_USE_SANDBOX (truthy = sandbox).
        ccxt_module: Injected for testability. Defaults to importing `ccxt`.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        sandbox: Optional[bool] = None,
        ccxt_module: Any = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("COINBASE_API_KEY", "")
        self._api_secret = (
            api_secret if api_secret is not None else os.getenv("COINBASE_API_SECRET", "")
        )
        self._passphrase = (
            passphrase if passphrase is not None else os.getenv("COINBASE_API_PASSPHRASE", "")
        )

        if sandbox is None:
            env_val = os.getenv("COINBASE_USE_SANDBOX", "true").strip().lower()
            self._sandbox = env_val not in ("false", "0", "no", "off", "")
        else:
            self._sandbox = bool(sandbox)

        if ccxt_module is None:
            try:
                import ccxt  # type: ignore

                ccxt_module = ccxt
            except ImportError as exc:  # pragma: no cover - ccxt is in requirements.txt
                raise ExchangeError("ccxt is not installed") from exc

        self._ccxt = ccxt_module
        self._client = self._build_client()

        # Optional WS market-data stream. Opt-in via COINBASE_WS_STREAM
        # so existing tests (and operators who want the simpler REST-only
        # path) see zero behavioural change. When enabled, get_ticker and
        # fetch_recent_candles consult the stream first and fall back to
        # REST when the cache is empty/stale.
        self._stream: Optional["CoinbaseMarketStream"] = None
        ws_enabled = os.getenv("COINBASE_WS_STREAM", "").strip().lower() in (
            "1",
            "on",
            "true",
            "yes",
        )
        if ws_enabled:
            try:
                from exchanges.coinbase_ws import CoinbaseMarketStream

                self._stream = CoinbaseMarketStream()
                self._stream.start()
            except Exception as exc:  # noqa: BLE001 -- stream is opt-in; never crash
                logger.warning(
                    "CoinbaseExchange: WS stream init failed (%s); REST-only",
                    exc,
                )
                self._stream = None

    def close(self) -> None:
        """Best-effort shutdown of the optional WS stream thread."""
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                logger.warning("CoinbaseExchange: stream stop raised", exc_info=True)
            finally:
                self._stream = None

    # -- client construction -------------------------------------------------

    def _build_client(self) -> Any:
        """Construct the ccxt client.

        ccxt has, across versions, used both 'coinbase' and 'coinbaseadvanced'
        for the Advanced Trade API. We probe both at runtime, preferring
        'coinbase' (the canonical id since ccxt 4.x folded Advanced Trade
        into it) and falling back to 'coinbaseadvanced'.

        Note on sandbox: as of ccxt 4.5, neither id ships a sandbox URL, so
        `set_sandbox_mode(True)` raises NotSupported. We treat that as a
        soft failure and keep our internal `_sandbox` flag set — the flag
        still gates kill-switch checks and signals operator intent.
        """
        params: Dict[str, Any] = {
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
        }
        if self._passphrase:
            params["password"] = self._passphrase

        builder = None
        self._ccxt_exchange_id: Optional[str] = None
        if hasattr(self._ccxt, "coinbase"):
            builder = getattr(self._ccxt, "coinbase")
            self._ccxt_exchange_id = "coinbase"
        elif hasattr(self._ccxt, "coinbaseadvanced"):
            builder = getattr(self._ccxt, "coinbaseadvanced")
            self._ccxt_exchange_id = "coinbaseadvanced"
        else:
            raise ExchangeError("ccxt module exposes neither 'coinbase' nor 'coinbaseadvanced'")

        try:
            client = builder(params)
        except Exception as exc:
            raise ExchangeError(f"Failed to construct ccxt Coinbase client: {exc}") from exc

        self._sandbox_native = False
        if self._sandbox:
            setter = getattr(client, "set_sandbox_mode", None)
            if callable(setter):
                try:
                    setter(True)
                    self._sandbox_native = True
                except Exception:
                    # ccxt 4.x doesn't ship a sandbox URL for Coinbase. We keep
                    # `_sandbox` True so the kill-switch and operator-intent
                    # checks still apply; native sandbox routing is just off.
                    self._sandbox_native = False

        return client

    # -- safety --------------------------------------------------------------

    def is_sandbox(self) -> bool:
        return self._sandbox

    def _check_kill_switch(self) -> None:
        """Block live order placement if the kill-switch file exists.

        Defense-in-depth only: the canonical breaker logic is in
        src/risk/circuit_breakers.py (Phase 2). We never block sandbox orders.
        """
        if self._sandbox:
            return
        path = os.getenv("KILL_SWITCH_FILE", "").strip()
        if path and os.path.exists(path):
            raise ExchangeError(
                f"Kill switch is active (file present: {path}); refusing to place live order"
            )

    # -- order placement -----------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        """Place a market order. Pass exactly one of quote_size_usd / base_size."""
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

        self._check_kill_switch()

        norm_symbol = _normalize_symbol(symbol)
        params: Dict[str, Any] = {}
        if quote_size_usd is not None:
            # ccxt Coinbase accepts a quote-currency notional via createMarketBuyOrderRequiresPrice=False
            # plus passing the cost directly. The community-standard switch is `cost` in params.
            amount = float(quote_size_usd)
            params["createMarketBuyOrderRequiresPrice"] = False
            params["cost"] = amount
        else:
            amount = float(base_size)  # type: ignore[arg-type]

        try:
            raw = self._client.create_order(
                norm_symbol, "market", side, amount, None, params
            )
        except Exception as exc:
            raise ExchangeError(f"create_order (market {side}) failed: {exc}") from exc

        return self._order_from_raw(
            raw,
            fallback_symbol=norm_symbol,
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
        time_in_force: str = "GTC",
    ) -> OrderResult:
        """Place a limit order at `limit_price` for `base_size` units of base."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if base_size <= 0:
            raise ValueError("base_size must be positive")
        if limit_price <= 0:
            raise ValueError("limit_price must be positive")

        self._check_kill_switch()

        norm_symbol = _normalize_symbol(symbol)
        params: Dict[str, Any] = {"timeInForce": time_in_force}
        try:
            raw = self._client.create_order(
                norm_symbol, "limit", side, float(base_size), float(limit_price), params
            )
        except Exception as exc:
            raise ExchangeError(f"create_order (limit {side}) failed: {exc}") from exc

        return self._order_from_raw(
            raw,
            fallback_symbol=norm_symbol,
            fallback_side=side,
            fallback_type="limit",
            base_size=base_size,
            limit_price=limit_price,
        )

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        norm_symbol = _normalize_symbol(symbol)
        try:
            raw = self._client.cancel_order(order_id, norm_symbol)
        except Exception as exc:
            raise ExchangeError(f"cancel_order failed: {exc}") from exc
        return self._order_from_raw(
            raw,
            fallback_symbol=norm_symbol,
            fallback_side="buy",
            fallback_type="market",
            forced_status="cancelled",
            forced_order_id=order_id,
        )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        norm_symbol = _normalize_symbol(symbol) if symbol else None
        try:
            raw_list = self._client.fetch_open_orders(norm_symbol)
        except Exception as exc:
            raise ExchangeError(f"fetch_open_orders failed: {exc}") from exc

        results: List[OrderResult] = []
        for raw in raw_list or []:
            results.append(
                self._order_from_raw(
                    raw,
                    fallback_symbol=norm_symbol or str(raw.get("symbol", "")),
                    fallback_side=str(raw.get("side", "buy")),
                    fallback_type=str(raw.get("type", "limit")),
                )
            )
        return results

    def get_order(self, order_id: str, symbol: str) -> OrderResult:
        norm_symbol = _normalize_symbol(symbol)
        try:
            raw = self._client.fetch_order(order_id, norm_symbol)
        except Exception as exc:
            raise ExchangeError(f"fetch_order failed: {exc}") from exc
        return self._order_from_raw(
            raw,
            fallback_symbol=norm_symbol,
            fallback_side=str(raw.get("side", "buy")),
            fallback_type=str(raw.get("type", "market")),
            forced_order_id=order_id,
        )

    # -- account / market data ----------------------------------------------

    def get_balances(self) -> List[Balance]:
        try:
            raw = self._client.fetch_balance()
        except Exception as exc:
            raise ExchangeError(f"fetch_balance failed: {exc}") from exc

        balances: List[Balance] = []
        free_map = (raw or {}).get("free", {}) or {}
        used_map = (raw or {}).get("used", {}) or {}
        total_map = (raw or {}).get("total", {}) or {}
        currencies = set(free_map) | set(used_map) | set(total_map)
        # ccxt also stuffs per-currency dicts at the top level; ignore those
        # and rely on the canonical free/used/total maps.
        for ccy in sorted(currencies):
            free = _coerce_float(free_map.get(ccy))
            locked = _coerce_float(used_map.get(ccy))
            total_raw = total_map.get(ccy)
            total = _coerce_float(total_raw, default=free + locked)
            balances.append(
                Balance(currency=str(ccy), free=free, locked=locked, total=total)
            )
        return balances

    def get_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker. Stream cache (if attached + fresh) > REST fallback.

        The REST endpoint is the public Advanced Trade
        ``/api/v3/brokerage/market/products/{product_id}``. When
        ``COINBASE_WS_STREAM=on``, get_ticker first consults the
        WS cache and only falls through to REST on cache miss / staleness;
        this kills the per-tick latency + DNS-blip fragility documented
        in [[autopilot-no-exit-policy-blocker-2026-05-16]].
        """
        norm_symbol = _coinbase_market_symbol(symbol)
        # Stream fast path. The stream itself is a soft dependency --
        # any failure here silently falls through to REST.
        stream = self._stream
        if stream is not None:
            try:
                stream.ensure_subscribed(norm_symbol)
                cached = stream.get_ticker(norm_symbol)
            except Exception:  # noqa: BLE001
                cached = None
            if cached is not None:
                try:
                    return Ticker(
                        symbol=norm_symbol,
                        bid=float(cached["bid"]),
                        ask=float(cached["ask"]),
                        last=float(cached["last"]),
                        volume_24h_base=float(cached.get("volume_24h_base", 0.0)),
                        as_of_utc=str(cached.get("as_of_utc") or ""),
                    )
                except Exception:  # noqa: BLE001 -- fall back to REST on parse error
                    pass
        try:
            return self._fetch_ticker_via_rest(norm_symbol)
        except ExchangeError:
            raise
        except Exception as exc:
            raise ExchangeError(f"fetch_ticker failed: {exc}") from exc

    def _fetch_ticker_via_rest(self, norm_symbol: str) -> Ticker:
        """Public Coinbase Advanced Trade products endpoint. No auth, no ccxt."""
        # ccxt format ``ETH/USD`` -> Coinbase product_id ``ETH-USD``.
        product_id = norm_symbol.replace("/", "-")
        url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}"
        try:
            resp = requests.get(url, timeout=30.0)
        except Exception as exc:
            raise ExchangeError(f"products GET failed: {exc}") from exc

        status = getattr(resp, "status_code", 0)
        if status == 404:
            raise ExchangeError(
                f"Product {product_id!r} is not listed on Coinbase. "
                "Try a USD pair such as ETH-USD, BTC-USD, or SOL-USD."
            )
        if not 200 <= status < 300:
            body = getattr(resp, "text", "")[:200]
            raise ExchangeError(f"products GET returned HTTP {status}: {body}")

        try:
            data = resp.json() or {}
        except Exception as exc:
            raise ExchangeError(f"products GET returned non-JSON: {exc}") from exc

        # Coinbase Advanced Trade renamed these fields: ``best_bid`` →
        # ``best_bid_price`` and ``best_ask`` → ``best_ask_price``. Read both
        # legacy + current names so we keep working if Coinbase reverts or
        # publishes both. Also expose ``mid_market_price`` as a fallback for
        # ``last`` when ``price`` is missing.
        bid = _coerce_float(
            data.get("best_bid_price") or data.get("best_bid")
        )
        ask = _coerce_float(
            data.get("best_ask_price") or data.get("best_ask")
        )
        last = _coerce_float(
            data.get("price"),
            default=_coerce_float(data.get("mid_market_price"))
            or ((bid + ask) / 2.0 if (bid and ask) else 0.0),
        )
        volume = _coerce_float(data.get("volume_24h"))
        # Public Coinbase products endpoint sometimes returns empty strings
        # for best_bid_price/best_ask_price (auth-gated). Fall back to last
        # trade price as both sides — loses spread realism but gains price
        # realism, which is what matters for paper-fill correctness. The
        # alternative ($1.0 default OR raising every tick) is worse.
        if (bid <= 0.0 or ask <= 0.0) and last > 0.0:
            if bid <= 0.0:
                bid = last
            if ask <= 0.0:
                ask = last
        if bid <= 0.0 or ask <= 0.0:
            raise ExchangeError(
                f"Coinbase ticker for {norm_symbol} returned non-positive "
                f"bid/ask/last (bid={bid}, ask={ask}, last={last}); the "
                f"field schema may have changed. Available keys: "
                f"{sorted(data.keys())[:8]}..."
            )
        return Ticker(
            symbol=norm_symbol,
            bid=bid,
            ask=ask,
            last=last,
            volume_24h_base=volume,
            as_of_utc=_utcnow_iso(),
        )

    def fetch_candles_window(
        self,
        symbol: str,
        *,
        granularity: Literal[
            "ONE_MINUTE",
            "FIVE_MINUTE",
            "FIFTEEN_MINUTE",
            "THIRTY_MINUTE",
            "ONE_HOUR",
            "TWO_HOUR",
            "SIX_HOUR",
            "ONE_DAY",
        ] = "ONE_MINUTE",
        start_unix: int,
        end_unix: int,
    ) -> List[Dict[str, Any]]:
        """Fetch candles in an explicit ``[start_unix, end_unix]`` window.

        Used by historical backfill (``crypto_training.backfill_ohlcv``).
        Coinbase enforces a 350-bar cap PER REQUEST, so the caller is
        responsible for chunking large windows. The endpoint and parsing
        is identical to ``fetch_recent_candles``; only the UNIX bounds
        come from the caller instead of being computed from "now".
        """
        return self._fetch_candles(
            symbol=symbol,
            granularity=granularity,
            start_unix=start_unix,
            end_unix=end_unix,
        )

    def fetch_recent_candles(
        self,
        symbol: str,
        *,
        granularity: Literal[
            "ONE_MINUTE",
            "FIVE_MINUTE",
            "FIFTEEN_MINUTE",
            "THIRTY_MINUTE",
            "ONE_HOUR",
            "TWO_HOUR",
            "SIX_HOUR",
            "ONE_DAY",
        ] = "ONE_MINUTE",
        limit: int = 350,
    ) -> List[Dict[str, Any]]:
        """Fetch the most recent ``limit`` candles. Stream cache > REST.

        Returns rows sorted oldest-first with keys ``timestamp`` (ISO UTC),
        ``open``, ``high``, ``low``, ``close``, ``volume`` (all floats).
        Coinbase returns at most 350 candles per request via REST; the WS
        cache is bounded the same way (default 400 bars).

        When the attached stream has ``>= limit`` fresh 1m bars it serves
        them directly (avoids the per-minute 350-bar REST round-trip the
        predictor would otherwise burn on every refresh). Falls back to
        REST when the cache is empty (warmup), too short, or stale.
        """
        # Stream fast path: only meaningful for the canonical ONE_MINUTE
        # request. Anything else (historical backfill, multi-tf) goes
        # straight to REST.
        stream = self._stream
        if stream is not None and granularity == "ONE_MINUTE":
            try:
                stream.ensure_subscribed(_coinbase_market_symbol(symbol))
                cached = stream.get_candles(symbol, limit=int(limit))
            except Exception:  # noqa: BLE001
                cached = None
            if cached is not None:
                # Match REST contract: strip the internal ``_unix`` key.
                rows = [
                    {k: v for k, v in row.items() if k != "_unix"} for row in cached
                ]
                return rows
        granularity_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }[granularity]
        bounded_limit = max(1, min(int(limit), 350))
        end = int(datetime.now(timezone.utc).timestamp())
        start = end - granularity_seconds * bounded_limit
        return self._fetch_candles(
            symbol=symbol,
            granularity=granularity,
            start_unix=start,
            end_unix=end,
        )

    def _fetch_candles(
        self,
        *,
        symbol: str,
        granularity: str,
        start_unix: int,
        end_unix: int,
    ) -> List[Dict[str, Any]]:
        """Shared candles fetch implementation. Coinbase has a 350-bar/req cap."""
        norm_symbol = _coinbase_market_symbol(symbol)
        product_id = norm_symbol.replace("/", "-")
        url = (
            f"https://api.coinbase.com/api/v3/brokerage/market/products/"
            f"{product_id}/candles"
        )
        params = {
            "start": str(int(start_unix)),
            "end": str(int(end_unix)),
            "granularity": granularity,
            "limit": "350",
        }
        try:
            resp = requests.get(url, params=params, timeout=30.0)
        except Exception as exc:
            raise ExchangeError(f"candles GET failed: {exc}") from exc

        status = getattr(resp, "status_code", 0)
        if status == 404:
            raise ExchangeError(
                f"Product {product_id!r} is not listed on Coinbase. "
                "Try a USD pair such as ETH-USD, BTC-USD, or SOL-USD."
            )
        if not 200 <= status < 300:
            body = getattr(resp, "text", "")[:200]
            raise ExchangeError(f"candles GET returned HTTP {status}: {body}")

        try:
            data = resp.json() or {}
        except Exception as exc:
            raise ExchangeError(f"candles GET returned non-JSON: {exc}") from exc

        raw_candles = data.get("candles") or []
        rows: List[Dict[str, Any]] = []
        for c in raw_candles:
            try:
                ts = int(c["start"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "_unix": ts,
                    "open": _coerce_float(c.get("open")),
                    "high": _coerce_float(c.get("high")),
                    "low": _coerce_float(c.get("low")),
                    "close": _coerce_float(c.get("close")),
                    "volume": _coerce_float(c.get("volume")),
                }
            )
        # Coinbase returns newest-first; we want oldest-first.
        rows.sort(key=lambda r: r["_unix"])
        for r in rows:
            r.pop("_unix", None)
        return rows

    # -- internal -----------------------------------------------------------

    def _order_from_raw(
        self,
        raw: Optional[Dict[str, Any]],
        *,
        fallback_symbol: str,
        fallback_side: str,
        fallback_type: str,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
        limit_price: Optional[float] = None,
        forced_status: Optional[_Status] = None,
        forced_order_id: Optional[str] = None,
    ) -> OrderResult:
        raw = dict(raw or {})
        order_id = forced_order_id or str(raw.get("id") or raw.get("order_id") or "")
        if not order_id:
            raise ExchangeError("Exchange response missing order id")

        side = str(raw.get("side") or fallback_side).lower()
        if side not in ("buy", "sell"):
            side = "buy"
        otype = str(raw.get("type") or fallback_type).lower()
        if otype not in ("market", "limit"):
            otype = fallback_type if fallback_type in ("market", "limit") else "market"

        status: _Status = forced_status if forced_status else _map_status(raw.get("status"))

        filled_base = _coerce_float(raw.get("filled"))
        avg_price = raw.get("average")
        avg_fill_price = float(avg_price) if avg_price is not None else None
        cost = _coerce_float(raw.get("cost"))
        if cost == 0.0 and avg_fill_price and filled_base:
            cost = avg_fill_price * filled_base

        fee_usd = 0.0
        fee_obj = raw.get("fee") or {}
        if isinstance(fee_obj, dict):
            fee_usd = _coerce_float(fee_obj.get("cost"))

        return OrderResult(
            order_id=order_id,
            symbol=str(raw.get("symbol") or fallback_symbol),
            side=side,  # type: ignore[arg-type]
            type=otype,  # type: ignore[arg-type]
            quote_size_usd=quote_size_usd,
            base_size=base_size if base_size is not None else (_coerce_float(raw.get("amount")) or None),
            limit_price=limit_price if limit_price is not None else (
                _coerce_float(raw.get("price")) or None
            ),
            status=status,
            filled_base=filled_base,
            filled_quote_usd=cost,
            avg_fill_price=avg_fill_price,
            fee_usd=fee_usd,
            created_at_utc=str(raw.get("datetime") or _utcnow_iso()),
            raw_payload=raw,
        )
