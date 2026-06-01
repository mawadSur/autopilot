"""Hyperliquid perpetual-futures connector for the autopilot trading stack.

Thin, hermetic wrapper around Hyperliquid's HTTPS REST API. Mirrors the
``CoinbaseExchange`` interface where it makes sense (``get_ticker``,
``get_balances``, ``get_open_orders``, ``cancel_order``, ``place_*_order``)
so the supervisor can call either exchange polymorphically. Adds
perp-specific concepts on top: leverage, margin, funding rate, mark vs.
oracle price, liquidation price, open interest.

Phase 6 of the live-trading buildout (multi-asset + perps).

V1 scope — read-only:
    Read endpoints (``/info``) work with plain JSON over HTTPS and are
    fully implemented here. The signed write endpoint (``/exchange``)
    requires EIP-712 signing of the order action by the wallet's private
    key. We deliberately do NOT take a dependency on
    ``hyperliquid-python-sdk`` (which pulls heavy crypto deps); instead,
    the write methods raise :class:`NotImplementedError` with a clear
    pointer to the SDK / docs. The supervisor will continue to use
    Coinbase for spot live trades; Hyperliquid is read-only until the
    signing path is wired (future phase).

Defense-in-depth notes:
  - All HTTP calls go through :meth:`HyperliquidExchange._post_info`,
    which wraps any error and re-raises it as :class:`ExchangeError` with
    the underlying exception preserved on ``__cause__``.
  - Read-only endpoints work without a private key. ``is_signing_enabled``
    reports whether write methods are wired (always ``False`` in V1, even
    when the key is set, until signing lands).
  - Hermetic by construction: a ``requests.Session`` is injected via the
    constructor for tests, so no network call ever leaves the box.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field

from exchanges.coinbase import (
    Balance,
    ExchangeError,
    OrderResult,
    Ticker,
)


__all__ = [
    "HyperliquidExchange",
    "PerpTicker",
    "PerpPosition",
    "MarginAccount",
]


_DEFAULT_BASE_URL = "https://api.hyperliquid.xyz"


# ---------------------------------------------------------------------------
# Perp-specific result models
# ---------------------------------------------------------------------------


class PerpTicker(BaseModel):
    """Perp-specific top-of-book + funding snapshot for a single symbol.

    Kept separate from the spot :class:`Ticker` so that the spot model
    stays free of perp-only fields (funding rate, oracle price, OI).
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    mark_price: float
    oracle_price: float
    funding_rate_8h: float
    """Hyperliquid 8-hour funding rate. Positive => longs pay shorts."""
    open_interest_base: float
    volume_24h_quote_usd: float
    as_of_utc: str


class PerpPosition(BaseModel):
    """A single open perpetual-futures position."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: Literal["long", "short"]
    size_base: float
    entry_price: float
    mark_price: float
    unrealized_pnl_usd: float
    liquidation_price: Optional[float] = None
    leverage: float
    margin_used_usd: float


class MarginAccount(BaseModel):
    """Account-level margin / equity snapshot for the Hyperliquid clearinghouse."""

    model_config = ConfigDict(extra="forbid")

    account_value_usd: float
    total_margin_used_usd: float
    withdrawable_usd: float
    leverage: float
    """Account-level leverage = sum(notional) / account_value."""
    as_of_utc: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_perp_symbol(symbol: str) -> str:
    """Hyperliquid uses bare base-asset symbols for perps (e.g. 'ETH', 'BTC').

    Accepts 'ETH', 'ETH-PERP', 'ETH/USD', 'ETH-USD' style and emits the
    bare base. Symbol matching against Hyperliquid metadata is case-sensitive
    on their side; we upper-case for safety.
    """
    if not symbol or not isinstance(symbol, str):
        raise ExchangeError(f"Invalid symbol: {symbol!r}")
    s = symbol.strip().upper()
    for sep in ("-", "/"):
        if sep in s:
            s = s.split(sep, 1)[0]
    if s.endswith("PERP"):
        s = s[:-4].rstrip("-/")
    if not s:
        raise ExchangeError(f"Invalid symbol: {symbol!r}")
    return s


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class HyperliquidExchange:
    """Hermetic wrapper around Hyperliquid's REST API.

    Args:
        private_key:    EVM wallet private key. Required for the signed
                        ``/exchange`` endpoint (write methods). Falls
                        back to env var ``HYPERLIQUID_PRIVATE_KEY``.
                        When unset, read-only methods still work but
                        write methods raise :class:`NotImplementedError`.
        wallet_address: 0x-prefixed EVM address used as the ``user`` field
                        on ``/info`` calls that need it (clearinghouse
                        state, fills). If omitted, attempted to derive
                        from ``private_key``; if derivation isn't wired
                        (no web3.py dependency), falls back to env
                        ``HYPERLIQUID_WALLET_ADDRESS``.
        base_url:       API base URL. Falls back to env
                        ``HYPERLIQUID_BASE_URL`` then to
                        ``https://api.hyperliquid.xyz``.
        session:        Injected ``requests.Session`` for tests. A fresh
                        session is constructed if omitted.
        timeout_s:      Per-request timeout in seconds (default 10s).
    """

    def __init__(
        self,
        *,
        private_key: Optional[str] = None,
        wallet_address: Optional[str] = None,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self._private_key: str = (
            private_key
            if private_key is not None
            else os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        ).strip()

        addr = wallet_address if wallet_address is not None else os.getenv(
            "HYPERLIQUID_WALLET_ADDRESS", ""
        )
        addr = (addr or "").strip()
        if not addr and self._private_key:
            try:
                addr = self._derive_address_from_private_key(self._private_key)
            except NotImplementedError:
                # Address derivation requires web3.py / eth-account, which
                # we deliberately don't depend on. Operator must supply
                # HYPERLIQUID_WALLET_ADDRESS explicitly.
                addr = ""
        self._wallet_address: str = addr

        self._base_url: str = (
            base_url
            if base_url is not None
            else os.getenv("HYPERLIQUID_BASE_URL", "") or _DEFAULT_BASE_URL
        ).rstrip("/")

        self._session: requests.Session = session if session is not None else requests.Session()
        self._timeout_s: float = float(timeout_s)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def is_signing_enabled(self) -> bool:
        """True only when ``private_key`` is set AND signing is wired.

        V1 always returns ``False`` even with a key, because the EIP-712
        signing path is not implemented. We keep the key-presence check
        here so the field still surfaces in logs/metrics; flipping to
        ``True`` requires the signing path landing in a future phase.
        """
        # V1: signing not wired yet. Even with a key, we cannot place
        # orders — return False so callers route writes elsewhere.
        return False

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def wallet_address(self) -> str:
        return self._wallet_address

    # ------------------------------------------------------------------
    # Read-only: market data
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> Ticker:
        """Spot-shaped ticker for a Hyperliquid perp.

        LIMITATION: Hyperliquid's ``/info`` ``metaAndAssetCtxs`` payload
        does NOT publish top-of-book bid/ask directly. We synthesize a
        bid/ask from the mark and oracle prices: the half-spread is
        ``abs(mark - oracle) / 2`` (a coarse proxy for execution slippage),
        and the mid is the mark price. For tighter execution-quality
        estimates, query the L2 book endpoint directly. This shape exists
        so the supervisor can treat HL like a spot venue when only mid +
        24h volume are needed.
        """
        norm = _normalize_perp_symbol(symbol)
        perp = self._get_perp_ticker_raw(norm)

        mark = perp["mark_price"]
        oracle = perp["oracle_price"]
        # Half-spread: |mark - oracle| / 2. When mark == oracle (steady
        # state), spread collapses to zero — that's an honest answer:
        # we have no book data, so we don't pretend a spread exists.
        half_spread = abs(mark - oracle) / 2.0
        bid = mark - half_spread
        ask = mark + half_spread
        # Volume is in quote (USD) on HL; divide by mark for an approximate
        # base-volume figure to keep the Ticker schema consistent with
        # Coinbase. When mark is zero, fall back to the quote figure.
        vol_quote = perp["volume_24h_quote_usd"]
        vol_base = vol_quote / mark if mark > 0 else vol_quote

        return Ticker(
            symbol=norm,
            bid=bid,
            ask=ask,
            last=mark,
            volume_24h_base=vol_base,
            as_of_utc=_utcnow_iso(),
        )

    def get_perp_ticker(self, symbol: str) -> PerpTicker:
        """Perp-native ticker including funding, OI, and oracle price."""
        norm = _normalize_perp_symbol(symbol)
        d = self._get_perp_ticker_raw(norm)
        return PerpTicker(
            symbol=norm,
            mark_price=d["mark_price"],
            oracle_price=d["oracle_price"],
            funding_rate_8h=d["funding_rate_8h"],
            open_interest_base=d["open_interest_base"],
            volume_24h_quote_usd=d["volume_24h_quote_usd"],
            as_of_utc=_utcnow_iso(),
        )

    def _get_perp_ticker_raw(self, symbol: str) -> Dict[str, float]:
        """Fetch + parse metaAndAssetCtxs, return the row for `symbol`.

        The payload shape is ``[meta, asset_ctxs]`` where:
          - ``meta["universe"]`` is a list of ``{"name": "BTC", ...}`` dicts.
          - ``asset_ctxs`` is a parallel list of dicts with mark/oracle/funding.
        """
        payload = self._post_info({"type": "metaAndAssetCtxs"})

        meta: Dict[str, Any] = {}
        ctxs: List[Dict[str, Any]] = []
        if isinstance(payload, list) and len(payload) >= 2:
            meta = payload[0] if isinstance(payload[0], dict) else {}
            ctxs = payload[1] if isinstance(payload[1], list) else []
        elif isinstance(payload, dict):
            meta = payload.get("meta", {}) or {}
            ctxs = payload.get("assetCtxs", []) or []

        universe = (meta or {}).get("universe", []) or []

        idx: Optional[int] = None
        for i, asset in enumerate(universe):
            name = str((asset or {}).get("name", "")).upper()
            if name == symbol:
                idx = i
                break
        if idx is None or idx >= len(ctxs):
            raise ExchangeError(f"Hyperliquid symbol not found: {symbol!r}")

        ctx = ctxs[idx] or {}
        # Funding key on HL is "funding" (8h rate as a decimal). OI is
        # "openInterest" in base units. Volume is "dayNtlVlm" in quote
        # (USD) terms. Mark is "markPx", oracle is "oraclePx".
        return {
            "mark_price": _coerce_float(ctx.get("markPx") or ctx.get("mark")),
            "oracle_price": _coerce_float(ctx.get("oraclePx") or ctx.get("oracle")),
            "funding_rate_8h": _coerce_float(ctx.get("funding")),
            "open_interest_base": _coerce_float(
                ctx.get("openInterest") or ctx.get("oi")
            ),
            "volume_24h_quote_usd": _coerce_float(
                ctx.get("dayNtlVlm") or ctx.get("dayBaseVlm") or ctx.get("volume24h")
            ),
        }

    # ------------------------------------------------------------------
    # Read-only: account
    # ------------------------------------------------------------------

    def get_balances(self) -> List[Balance]:
        """Return USDC margin balance only.

        Hyperliquid is a USDC-margined venue; there is no per-asset spot
        balance to enumerate. We surface a single ``USDC`` row using the
        clearinghouse ``withdrawable`` (free) and ``totalMarginUsed``
        (locked) figures, with ``total = accountValue``.
        """
        state = self._fetch_clearinghouse_state()
        margin_summary = (state or {}).get("marginSummary", {}) or {}
        cross_margin_summary = (state or {}).get("crossMarginSummary", {}) or {}

        free = _coerce_float(state.get("withdrawable"))
        locked = _coerce_float(
            margin_summary.get("totalMarginUsed")
            or cross_margin_summary.get("totalMarginUsed")
        )
        total = _coerce_float(
            margin_summary.get("accountValue")
            or cross_margin_summary.get("accountValue"),
            default=free + locked,
        )

        return [Balance(currency="USDC", free=free, locked=locked, total=total)]

    def get_margin_account(self) -> MarginAccount:
        """Return the account-level margin snapshot."""
        state = self._fetch_clearinghouse_state()
        margin_summary = (state or {}).get("marginSummary", {}) or {}
        cross_margin_summary = (state or {}).get("crossMarginSummary", {}) or {}

        account_value = _coerce_float(
            margin_summary.get("accountValue")
            or cross_margin_summary.get("accountValue")
        )
        margin_used = _coerce_float(
            margin_summary.get("totalMarginUsed")
            or cross_margin_summary.get("totalMarginUsed")
        )
        withdrawable = _coerce_float(state.get("withdrawable"))
        notional = _coerce_float(
            margin_summary.get("totalNtlPos")
            or cross_margin_summary.get("totalNtlPos")
        )
        leverage = (notional / account_value) if account_value > 0 else 0.0

        return MarginAccount(
            account_value_usd=account_value,
            total_margin_used_usd=margin_used,
            withdrawable_usd=withdrawable,
            leverage=leverage,
            as_of_utc=_utcnow_iso(),
        )

    def get_open_positions(self) -> List[PerpPosition]:
        """Return all open perp positions for the configured wallet."""
        state = self._fetch_clearinghouse_state()
        asset_positions = (state or {}).get("assetPositions", []) or []

        results: List[PerpPosition] = []
        for entry in asset_positions:
            pos = (entry or {}).get("position") or {}
            size_signed = _coerce_float(pos.get("szi"))
            if size_signed == 0.0:
                # Hyperliquid sometimes returns flat rows; skip.
                continue
            side: Literal["long", "short"] = "long" if size_signed > 0 else "short"
            entry_price = _coerce_float(pos.get("entryPx"))
            unrealized = _coerce_float(pos.get("unrealizedPnl"))
            margin_used = _coerce_float(pos.get("marginUsed"))
            liq_raw = pos.get("liquidationPx")
            liq_price: Optional[float] = (
                float(liq_raw) if liq_raw not in (None, "", "null") else None
            )
            leverage_obj = pos.get("leverage") or {}
            if isinstance(leverage_obj, dict):
                lev = _coerce_float(leverage_obj.get("value"))
            else:
                lev = _coerce_float(leverage_obj)

            # Mark price isn't on the position payload directly; derive
            # from entry + unrealized PnL when possible: pnl = size *
            # (mark - entry) for longs (sign-flipped for shorts using
            # signed size). Solve: mark = entry + pnl/size_signed.
            if size_signed != 0.0 and entry_price > 0:
                mark_price = entry_price + (unrealized / size_signed)
            else:
                mark_price = entry_price

            results.append(
                PerpPosition(
                    symbol=str(pos.get("coin") or "").upper(),
                    side=side,
                    size_base=abs(size_signed),
                    entry_price=entry_price,
                    mark_price=mark_price,
                    unrealized_pnl_usd=unrealized,
                    liquidation_price=liq_price,
                    leverage=lev,
                    margin_used_usd=margin_used,
                )
            )
        return results

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Return resting (open) orders for the configured wallet."""
        self._require_wallet("openOrders")
        body = {"type": "openOrders", "user": self._wallet_address}
        raw = self._post_info(body) or []

        results: List[OrderResult] = []
        for o in raw:
            if not isinstance(o, dict):
                continue
            coin = str(o.get("coin") or "").upper()
            if symbol is not None and coin != _normalize_perp_symbol(symbol):
                continue
            side_raw = str(o.get("side") or "B").upper()
            # HL uses "B"/"A" for buy/ask(=sell).
            side: Literal["buy", "sell"] = "buy" if side_raw in ("B", "BUY") else "sell"
            limit_price = _coerce_float(o.get("limitPx"))
            base_size = _coerce_float(o.get("sz") or o.get("origSz"))
            order_id = str(o.get("oid") or o.get("id") or "")
            if not order_id:
                continue
            ts = o.get("timestamp")
            created_at = (
                datetime.fromtimestamp(int(ts) / 1000.0, tz=timezone.utc).isoformat()
                if isinstance(ts, (int, float))
                else _utcnow_iso()
            )
            results.append(
                OrderResult(
                    order_id=order_id,
                    symbol=coin,
                    side=side,
                    type="limit",
                    base_size=base_size,
                    limit_price=limit_price,
                    status="open",
                    filled_base=0.0,
                    filled_quote_usd=0.0,
                    avg_fill_price=None,
                    fee_usd=0.0,
                    created_at_utc=created_at,
                    raw_payload=dict(o),
                )
            )
        return results

    def get_recent_fills(self, *, limit: int = 50) -> List[OrderResult]:
        """Return recent fills (filled orders) for the configured wallet.

        Hyperliquid's ``userFills`` returns one row per fill; we parse
        each into an :class:`OrderResult` with ``status="filled"``.
        """
        self._require_wallet("userFills")
        body = {"type": "userFills", "user": self._wallet_address}
        raw = self._post_info(body) or []
        if not isinstance(raw, list):
            return []

        results: List[OrderResult] = []
        for f in raw[: max(0, int(limit))]:
            if not isinstance(f, dict):
                continue
            coin = str(f.get("coin") or "").upper()
            side_raw = str(f.get("side") or "B").upper()
            side: Literal["buy", "sell"] = "buy" if side_raw in ("B", "BUY") else "sell"
            px = _coerce_float(f.get("px"))
            sz = _coerce_float(f.get("sz"))
            fee = _coerce_float(f.get("fee"))
            order_id = str(f.get("oid") or f.get("tid") or f.get("hash") or "")
            if not order_id:
                continue
            ts = f.get("time")
            created_at = (
                datetime.fromtimestamp(int(ts) / 1000.0, tz=timezone.utc).isoformat()
                if isinstance(ts, (int, float))
                else _utcnow_iso()
            )
            results.append(
                OrderResult(
                    order_id=order_id,
                    symbol=coin,
                    side=side,
                    type="market",
                    base_size=sz,
                    limit_price=None,
                    status="filled",
                    filled_base=sz,
                    filled_quote_usd=px * sz,
                    avg_fill_price=px,
                    fee_usd=fee,
                    created_at_utc=created_at,
                    raw_payload=dict(f),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Write: NOT IMPLEMENTED in V1 (require EIP-712 signing)
    # ------------------------------------------------------------------

    _NOT_IMPL_MSG = (
        "Hyperliquid order placement requires EIP-712 signing of the order "
        "action by the wallet's private key. V1 wires read-only access only. "
        "Use the official hyperliquid-python-sdk or implement signing per "
        "https://hyperliquid.gitbook.io/ before calling this method."
    )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        base_size: float,
        leverage: Optional[int] = None,
    ) -> OrderResult:
        """V1: not implemented — requires EIP-712 signing of the order action."""
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        *,
        base_size: float,
        limit_price: float,
        leverage: Optional[int] = None,
    ) -> OrderResult:
        """V1: not implemented — requires EIP-712 signing of the order action."""
        raise NotImplementedError(self._NOT_IMPL_MSG)

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """V1: not implemented — requires EIP-712 signing of the cancel action."""
        raise NotImplementedError(self._NOT_IMPL_MSG)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_info(self, body: Dict[str, Any]) -> Any:
        """POST a JSON body to ``/info`` and return the parsed response.

        Wraps any exception (network, non-2xx, non-JSON) as
        :class:`ExchangeError` with the original on ``__cause__``.
        """
        url = f"{self._base_url}/info"
        try:
            resp = self._session.post(url, json=body, timeout=self._timeout_s)
        except Exception as exc:
            raise ExchangeError(f"Hyperliquid /info POST failed: {exc}") from exc

        status = getattr(resp, "status_code", None)
        if status is not None and not (200 <= int(status) < 300):
            text = getattr(resp, "text", "")
            raise ExchangeError(
                f"Hyperliquid /info returned HTTP {status}: {text!r}"
            )
        try:
            return resp.json()
        except Exception as exc:
            raise ExchangeError(
                f"Hyperliquid /info returned non-JSON body: {exc}"
            ) from exc

    def _fetch_clearinghouse_state(self) -> Dict[str, Any]:
        self._require_wallet("clearinghouseState")
        body = {"type": "clearinghouseState", "user": self._wallet_address}
        out = self._post_info(body)
        if not isinstance(out, dict):
            raise ExchangeError(
                f"Hyperliquid clearinghouseState returned non-dict: {type(out).__name__}"
            )
        return out

    def _require_wallet(self, op_name: str) -> None:
        if not self._wallet_address:
            raise ExchangeError(
                f"Hyperliquid {op_name} requires a wallet address. Set "
                "HYPERLIQUID_WALLET_ADDRESS or pass wallet_address= to the "
                "constructor."
            )

    def _derive_address_from_private_key(self, private_key: str) -> str:
        """Derive the EVM address from a private key.

        Deliberately not implemented: doing this requires ``eth-account``
        / ``web3.py``, which would pull a heavy crypto stack we do not
        currently depend on. Operators must supply
        ``HYPERLIQUID_WALLET_ADDRESS`` (env or constructor) explicitly.
        """
        raise NotImplementedError(
            "Address derivation from private key requires eth-account/web3.py, "
            "which autopilot does not depend on. Set HYPERLIQUID_WALLET_ADDRESS "
            "explicitly (env var or constructor kwarg) instead."
        )
