"""Read-only Kalshi market-data client for cross-venue mispricing detection.

This module is a thin, hermetic wrapper around Kalshi's public ``trade-api``
v2 **market-data** endpoints. Its only job is to pull market metadata and
order books so a later stage can compare Polymarket vs. Kalshi prices and
flag cross-venue mispricing.

SCOPE — READ-ONLY, NO MONEY MOVES:
    This client intentionally exposes ONLY market-data reads
    (``get_markets``, ``get_market``, ``get_orderbook``). It performs no
    authentication-gated trade calls, places no orders, and writes nothing.
    Do not add order placement, balance, or any signed/portfolio endpoints
    here — those belong in a separate, deliberately opted-in trading client.

LIVE-VERIFICATION CAVEAT (operator must confirm before relying on this):
    * Base URL — defaults to ``https://api.elections.kalshi.com/trade-api/v2``.
      Kalshi has historically also served ``https://trading-api.kalshi.com``
      and demo hosts. The operator MUST confirm the *current* production base
      URL for public market data before wiring this into anything live; it is
      a constructor parameter precisely so it can be overridden without code
      changes.
    * API key requirement — public market-data endpoints have historically
      been readable without auth, but Kalshi can change this. The operator
      MUST confirm whether the current market-data endpoints require an API
      key / bearer token. If they do, an ``api_key`` can be passed and it is
      sent as a ``Bearer`` ``Authorization`` header; if they don't, leave it
      unset. Either way this client never sends credentials to a write path.
    * Endpoint paths / response shapes (``/markets``, ``/markets/{ticker}``,
      ``/markets/{ticker}/orderbook``) and the cents pricing convention must
      be re-checked against the live API docs.

Hermetic by construction: all HTTP goes through an injected
``requests.Session`` so tests never touch the network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


__all__ = [
    "KalshiAPIError",
    "KalshiMarketDataClient",
    "normalize_market",
]


_DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_DEFAULT_TIMEOUT_S = 10.0


class KalshiAPIError(Exception):
    """Raised when a Kalshi market-data request fails.

    Wraps network errors, non-2xx responses, and malformed payloads with
    context about which call (and, where relevant, which ticker) failed so
    callers never see a bare ``requests`` exception leak through.
    """


def _cents_to_prob(cents: Any) -> Optional[float]:
    """Convert a Kalshi cent price (1-99) to a [0, 1] probability.

    Kalshi quotes Yes/No prices in integer cents where 1c..99c maps to a
    1%..99% implied probability. Returns ``None`` when the input is missing
    or non-numeric so normalization never crashes on a partial payload.
    """
    if cents is None:
        return None
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return None


def normalize_market(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Project a raw Kalshi market dict onto a common cross-venue schema.

    Returns a dict with the keys:
        ticker, title, yes_bid, yes_ask, no_bid, no_ask, last_price,
        volume, close_ts, implied_prob

    All price fields are converted from Kalshi cents (1-99) to [0, 1]
    probabilities. Missing fields degrade to ``None`` (prices/ts) or ``0``
    (volume) rather than raising, so a partial payload still normalizes.

    ``implied_prob`` choice:
        We prefer the Yes mid-price ``(yes_bid + yes_ask) / 2`` because the
        midpoint is the least-biased single-number estimate of fair value
        and nets out the bid/ask spread. When only one side is present we
        fall back to whichever of ``yes_ask`` or ``yes_bid`` exists, and
        finally to ``last_price``. This mirrors how a maker would mark the
        book and keeps the number directly comparable to a Polymarket
        mid-price for arb detection.
    """
    if not isinstance(raw, dict):
        raise KalshiAPIError(
            f"normalize_market expected a dict, got {type(raw).__name__}"
        )

    yes_bid = _cents_to_prob(raw.get("yes_bid"))
    yes_ask = _cents_to_prob(raw.get("yes_ask"))
    no_bid = _cents_to_prob(raw.get("no_bid"))
    no_ask = _cents_to_prob(raw.get("no_ask"))
    last_price = _cents_to_prob(raw.get("last_price"))

    implied_prob = _implied_prob_from_yes(yes_bid, yes_ask, last_price)

    # Volume key varies across Kalshi payloads; accept the common ones.
    volume = raw.get("volume")
    if volume is None:
        volume = raw.get("volume_24h", 0)
    try:
        volume = int(volume) if volume is not None else 0
    except (TypeError, ValueError):
        volume = 0

    return {
        "ticker": raw.get("ticker"),
        "title": raw.get("title"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
        "volume": volume,
        "close_ts": raw.get("close_time") or raw.get("close_ts"),
        "implied_prob": implied_prob,
    }


def _implied_prob_from_yes(
    yes_bid: Optional[float],
    yes_ask: Optional[float],
    last_price: Optional[float],
) -> Optional[float]:
    """Derive the implied probability per the documented preference order.

    1. Yes mid-price ``(yes_bid + yes_ask) / 2`` when both sides exist.
    2. Whichever single Yes quote is present (ask preferred over bid).
    3. ``last_price`` as a last resort.
    4. ``None`` when nothing usable is present.
    """
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0
    if yes_ask is not None:
        return yes_ask
    if yes_bid is not None:
        return yes_bid
    return last_price


class KalshiMarketDataClient:
    """Read-only client for Kalshi public market-data endpoints.

    All HTTP is issued via an injected (or lazily-created)
    ``requests.Session`` so the client is fully offline-testable: tests
    patch ``session.get``.

    Args:
        base_url:   Override for the trade-api v2 base URL. Defaults to
                    :data:`_DEFAULT_BASE_URL`. See the module docstring's
                    live-verification caveat.
        session:    Injected ``requests.Session`` for tests; a fresh one is
                    created when omitted.
        timeout_s:  Per-request timeout in seconds (default 10s).
        api_key:    Optional bearer token, sent only if the operator has
                    confirmed market-data reads require auth. Never used for
                    any write/trade path (there are none here).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        api_key: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session: requests.Session = (
            session if session is not None else requests.Session()
        )
        self._timeout_s = float(timeout_s)
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Public read-only API
    # ------------------------------------------------------------------

    def get_markets(
        self,
        limit: int = 100,
        status: str = "open",
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a page of markets and normalize each one.

        Returns the list of normalized market dicts (see
        :func:`normalize_market`). Pagination is exposed via ``cursor`` —
        the caller passes back the ``cursor`` the API returned to walk
        subsequent pages.
        """
        params: Dict[str, Any] = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor

        payload = self._get("/markets", params=params, context="get_markets")
        raw_markets = payload.get("markets") if isinstance(payload, dict) else None
        if not isinstance(raw_markets, list):
            raw_markets = []
        return [normalize_market(m) for m in raw_markets if isinstance(m, dict)]

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Fetch a single market by ticker and normalize it."""
        if not ticker:
            raise KalshiAPIError("get_market called with empty ticker")
        path = f"/markets/{ticker}"
        payload = self._get(path, context="get_market", ticker=ticker)
        raw = payload.get("market") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            raise KalshiAPIError(
                f"get_market({ticker!r}): response missing 'market' object"
            )
        return normalize_market(raw)

    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        """Fetch the order book for a ticker.

        Returns the raw ``orderbook`` object from the API (Yes/No price
        levels). Kept un-normalized: depth/level structure is venue-specific
        and the arb stage consumes it directly.
        """
        if not ticker:
            raise KalshiAPIError("get_orderbook called with empty ticker")
        path = f"/markets/{ticker}/orderbook"
        payload = self._get(
            path,
            params={"depth": depth},
            context="get_orderbook",
            ticker=ticker,
        )
        book = payload.get("orderbook") if isinstance(payload, dict) else None
        if not isinstance(book, dict):
            raise KalshiAPIError(
                f"get_orderbook({ticker!r}): response missing 'orderbook' object"
            )
        return book

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        context: str = "",
        ticker: Optional[str] = None,
    ) -> Any:
        """GET ``path`` and return parsed JSON, wrapping all failures.

        Any network error, non-2xx status, or non-JSON body is re-raised as
        :class:`KalshiAPIError` with the call context (and ticker, where
        relevant) and the original exception on ``__cause__``.
        """
        url = f"{self._base_url}{path}"
        where = context or path
        if ticker:
            where = f"{where}({ticker!r})"

        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            raise KalshiAPIError(
                f"Kalshi {where} GET {url} failed: {exc}"
            ) from exc

        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(resp, "status_code", "?")
            raise KalshiAPIError(
                f"Kalshi {where} GET {url} returned HTTP {status}: {exc}"
            ) from exc

        try:
            return resp.json()
        except (ValueError, requests.RequestException) as exc:
            raise KalshiAPIError(
                f"Kalshi {where} GET {url} returned non-JSON body: {exc}"
            ) from exc
