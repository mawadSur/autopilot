"""Read-only Polymarket *data-api* client (SHADOW-ONLY).

This module is a thin, hermetic wrapper around Polymarket's public
**data-api** (``https://data-api.polymarket.com``) — the off-chain analytics
service that serves a wallet's trade history, a wallet's positions (with
realized PnL), and a market's current token holders. Its only job is to feed
the "smart-money wallet follower" edge: rank wallets by realized profit and
detect when several top wallets converge on the same outcome of a market.

SCOPE — READ-ONLY, NO MONEY MOVES (Constitution: safety first):
    This client exposes ONLY data reads (``get_trades``, ``get_positions``,
    ``get_holders``). It performs no authentication, signs nothing, places no
    orders, touches no wallet/web3 path, and writes nothing. Do NOT add order
    placement, signing, balance, allowance, or any wallet/web3 endpoint here —
    those are explicitly out of scope for the shadow stack.

LIVE-VERIFICATION CAVEAT:
    The data-api endpoints below were LIVE-VERIFIED against the production
    data-api on 2026-05-31 (real responses probed). The profit-leaderboard
    endpoint (``get_profit_leaderboard``) lives on a DIFFERENT host —
    ``https://lb-api.polymarket.com`` — and was LIVE-VERIFIED on 2026-06-01;
    only the windows ``'all'`` (all-time) and ``'1d'`` (today) are confirmed,
    any other window string returns HTTP 400. Polymarket can change paths,
    query-parameter names, or response shapes without notice, so an operator
    MUST re-confirm them before trusting this client in production. Both base
    URLs are constructor parameters (``DEFAULT_BASE_URL`` / ``DEFAULT_LB_BASE_URL``
    defaults) so they can be overridden without a code change.

Endpoints (shapes the parsers are built to, verbatim from the live probe):

    GET /trades?user=<addr>&market=<conditionId>&limit=N&offset=M
        -> JSON list of trade dicts:
           {proxyWallet, side:"BUY"|"SELL", asset:"<tokenId str>",
            conditionId, size:float, price:float, timestamp:int(unix s),
            title, slug, outcome:str, outcomeIndex:int(0/1), name,
            transactionHash}

    GET /positions?user=<addr>&limit=N
        -> JSON list of position dicts:
           {proxyWallet, asset, conditionId, size, avgPrice, initialValue,
            currentValue, cashPnl:float, percentPnl, totalBought,
            realizedPnl:float, percentRealizedPnl, curPrice,
            redeemable:bool, title, outcome, outcomeIndex, oppositeOutcome,
            oppositeAsset, endDate, negativeRisk}
        -> ``realizedPnl`` is the per-position realized profit;
           ``redeemable=true`` means the position is resolved/settled.

    GET /holders?market=<conditionId>&limit=N
        -> JSON list of per-token holder groups:
           {token:"<tokenId str>",
            holders:[ {proxyWallet, asset, amount:float, outcomeIndex:int,
                       name, pseudonym, verified} ]}

    GET /profit?window=<all|1d>&limit=N   (HOST: https://lb-api.polymarket.com)
        -> JSON list of profit-leaderboard rows, descending by profit:
           {proxyWallet, amount:float(profit USD), name, pseudonym}
        -> ``window`` confirmed values: ``'all'`` (all-time) and ``'1d'``
           (today). Any other window string -> HTTP 400 (surfaced as a
           :class:`PolymarketDataAPIError`, never a crash).

Hermetic by construction: all HTTP goes through an injected
``requests.Session`` so tests never touch the network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_LB_BASE_URL",
    "PolymarketDataAPIError",
    "PolymarketDataAPIClient",
]


DEFAULT_BASE_URL = "https://data-api.polymarket.com"
# The profit leaderboard lives on a SEPARATE host (note: lb-api, not data-api).
DEFAULT_LB_BASE_URL = "https://lb-api.polymarket.com"
_DEFAULT_TIMEOUT_S = 10.0


class PolymarketDataAPIError(Exception):
    """Raised when a Polymarket data-api request fails.

    Wraps network errors, non-2xx responses, and malformed payloads with the
    call context (which method, and which user/market where relevant) so
    callers never see a bare ``requests`` exception leak through. The
    originating exception is preserved on ``__cause__``.
    """


class PolymarketDataAPIClient:
    """Read-only client for the Polymarket public data-api.

    All HTTP is issued via an injected (or lazily-created) ``requests.Session``
    so the client is fully offline-testable: tests patch ``session.get``.

    Args:
        base_url:    Override for the data-api base URL. Defaults to
                     :data:`DEFAULT_BASE_URL`. See the module docstring's
                     live-verification caveat.
        lb_base_url: Override for the profit-leaderboard host (a DIFFERENT
                     service from the data-api). Defaults to
                     :data:`DEFAULT_LB_BASE_URL`.
        session:     Injected ``requests.Session`` for tests; a fresh one is
                     created when omitted.
        timeout_s:   Per-request timeout in seconds (default 10s).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        lb_base_url: str = DEFAULT_LB_BASE_URL,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._lb_base_url = lb_base_url.rstrip("/")
        self._session: requests.Session = (
            session if session is not None else requests.Session()
        )
        self._timeout_s = float(timeout_s)

    # ------------------------------------------------------------------
    # Public read-only API
    # ------------------------------------------------------------------

    def get_trades(
        self,
        user: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch a page of trades, optionally filtered by wallet and/or market.

        Issues ``GET /trades?user=<user>&market=<market>&limit=N&offset=M``.
        Both ``user`` (a proxy-wallet address) and ``market`` (a ``conditionId``)
        are optional filters; omit either to leave it unconstrained. Returns the
        raw list of trade dicts (see the module docstring for the shape).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if user:
            params["user"] = user
        if market:
            params["market"] = market

        payload = self._get(
            "/trades",
            params=params,
            context="get_trades",
            detail=user or market,
        )
        return _as_list(payload)

    def get_positions(self, user: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetch a wallet's positions (with realized PnL).

        Issues ``GET /positions?user=<user>&limit=N``. ``user`` is required (a
        proxy-wallet address). Returns the raw list of position dicts; each
        carries ``realizedPnl`` (per-position realized profit) and
        ``redeemable`` (``True`` once the position is resolved/settled).
        """
        if not user:
            raise PolymarketDataAPIError("get_positions called with empty user")
        params: Dict[str, Any] = {"user": user, "limit": limit}
        payload = self._get(
            "/positions",
            params=params,
            context="get_positions",
            detail=user,
        )
        return _as_list(payload)

    def get_holders(
        self,
        market_condition_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch a market's current token holders, grouped per outcome token.

        Issues ``GET /holders?market=<conditionId>&limit=N``. Returns the raw
        list of per-token holder groups, each ``{token, holders:[...]}`` (see
        the module docstring for the holder shape).
        """
        if not market_condition_id:
            raise PolymarketDataAPIError(
                "get_holders called with empty market_condition_id"
            )
        params: Dict[str, Any] = {"market": market_condition_id, "limit": limit}
        payload = self._get(
            "/holders",
            params=params,
            context="get_holders",
            detail=market_condition_id,
        )
        return _as_list(payload)

    def get_profit_leaderboard(
        self,
        window: str = "all",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch the historical *profit* leaderboard (real winners).

        Issues ``GET /profit?window=<window>&limit=N`` against the separate
        leaderboard host (:data:`DEFAULT_LB_BASE_URL`, note: ``lb-api``, NOT the
        ``data-api`` host the other methods use). Returns the raw list of rows,
        descending by realized profit; each row is
        ``{proxyWallet, amount(profit USD float), name, pseudonym}``.

        ``window`` is the leaderboard horizon. Only ``'all'`` (all-time) and
        ``'1d'`` (today) are live-verified; any other value returns HTTP 400,
        which is wrapped in :class:`PolymarketDataAPIError` (with the window in
        the context) rather than crashing — callers see a clear error.

        Read-only: this is a pure GET of a public leaderboard. It places no
        orders, signs nothing, and touches no wallet/web3 path.
        """
        params: Dict[str, Any] = {"window": window, "limit": limit}
        payload = self._get(
            "/profit",
            params=params,
            context="get_profit_leaderboard",
            detail=window,
            base_url=self._lb_base_url,
        )
        return _as_list(payload)

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        context: str = "",
        detail: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        """GET ``path`` and return parsed JSON, wrapping all failures.

        Any network error, non-2xx status, or non-JSON body is re-raised as
        :class:`PolymarketDataAPIError` with the call context (and the user or
        market detail, where relevant) and the original exception on
        ``__cause__``.

        ``base_url`` defaults to the data-api host; pass the leaderboard host
        (:data:`DEFAULT_LB_BASE_URL`) for the ``/profit`` endpoint.
        """
        root = (base_url if base_url is not None else self._base_url)
        url = f"{root}{path}"
        where = context or path
        if detail:
            where = f"{where}({detail!r})"

        headers: Dict[str, str] = {"Accept": "application/json"}

        try:
            resp = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            raise PolymarketDataAPIError(
                f"Polymarket data-api {where} GET {url} failed: {exc}"
            ) from exc

        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(resp, "status_code", "?")
            raise PolymarketDataAPIError(
                f"Polymarket data-api {where} GET {url} "
                f"returned HTTP {status}: {exc}"
            ) from exc

        try:
            return resp.json()
        except (ValueError, requests.RequestException) as exc:
            raise PolymarketDataAPIError(
                f"Polymarket data-api {where} GET {url} "
                f"returned non-JSON body: {exc}"
            ) from exc


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    """Coerce a data-api list payload to a list of dicts.

    The three endpoints all return a top-level JSON array. We defensively drop
    any non-dict element rather than raising, so one malformed row does not
    sink a whole page. A non-list payload yields an empty list.
    """
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]
