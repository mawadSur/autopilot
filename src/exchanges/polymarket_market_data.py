"""Read-only Polymarket CLOB order-book client (SHADOW-ONLY).

This module is a thin, hermetic wrapper around Polymarket's **CLOB**
order-book read endpoint. Its only job is to pull the live YES/NO best asks
for a binary market so the intra-market arbitrage detector
(``src/arb_detector.py``) can flag a model-free, risk-free edge when
``yes_ask + no_ask < 1`` net of costs.

SCOPE — READ-ONLY, NO MONEY MOVES (Constitution: safety first):
    This client exposes ONLY order-book reads (``get_order_book``,
    ``best_ask`` / ``best_bid``, ``get_yes_no_best_asks``). It performs no
    authentication, signs nothing, places no orders, touches no wallet, and
    writes nothing. Do NOT add order placement, signing, balance, allowance,
    or any wallet/web3 endpoint here — those are explicitly out of scope for
    the shadow stack and belong (if ever) in a separate, deliberately
    opted-in trading client.

Pricing convention:
    Polymarket CLOB prices are already dollars in the closed interval
    ``[0, 1]`` (a YES token trading at $0.62 implies a 62% probability). No
    cents->dollars conversion is needed here, unlike the Kalshi sibling
    client.

clobTokenIds ordering assumption:
    A Gamma binary market exposes ``clobTokenIds`` as a JSON-string list of
    exactly TWO ERC-1155 token ids. This module ASSUMES the ordering is
    ``[YES_token, NO_token]`` — index 0 is YES, index 1 is NO — matching the
    convention already used by ``fetcher._resolve_clob_token_id`` (which
    treats ``clobTokenIds[0]`` as the canonical YES side). If the ordering
    is ever wrong for a given market the detected "arb" would be spurious, so
    see the live-verification caveat below.

LIVE-VERIFICATION CAVEAT (operator MUST confirm before trusting in production):
    * Endpoint path & response shape — this client hits
      ``GET {CLOB_API_BASE_URL}/book?token_id=<id>`` and expects a JSON
      object shaped roughly::

          {
            "market": "...", "asset_id": "...",
            "bids": [{"price": "0.41", "size": "100"}, ...],
            "asks": [{"price": "0.43", "size": "250"}, ...]
          }

      The exact path (``/book`` vs. ``/orderbook``), the query-parameter name
      (``token_id``), and the bids/asks field names + price/size string-vs-
      float typing MUST be re-checked against the current Polymarket CLOB API
      docs before this is relied upon live.
    * clobTokenIds ordering — confirm that index 0 is genuinely the YES
      outcome and index 1 the NO outcome for the markets you scan. A flipped
      pair does not break the arb identity (YES+NO still pays $1), but it does
      mislabel which leg is which in the ledger.

Hermetic by construction: all HTTP goes through an injected
``requests.Session`` (or a fresh one carrying the shared scanner User-Agent)
so tests never touch the network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from fetcher import CLOB_API_BASE_URL, DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT
except Exception:  # pragma: no cover - only hit if fetcher is unavailable.
    # Fall back to the same literals fetcher defines, so this module is still
    # importable/usable in isolation. Keep these in sync with src/fetcher.py.
    CLOB_API_BASE_URL = "https://clob.polymarket.com"
    DEFAULT_TIMEOUT_SECONDS = 20
    DEFAULT_USER_AGENT = "autopilot-polymarket-scanner/1.0"


__all__ = [
    "PolymarketAPIError",
    "get_order_book",
    "get_market_resolution",
    "best_ask",
    "best_bid",
    "get_yes_no_best_asks",
]


# Tolerated field-name variants for the two clobTokenIds on a market object.
_CLOB_TOKEN_KEYS = ("clobTokenIds", "clob_token_ids", "clobTokenIDs")


class PolymarketAPIError(Exception):
    """Raised when a Polymarket CLOB market-data request fails.

    Wraps network errors, non-2xx responses, and malformed payloads with the
    call + token context so callers never see a bare ``requests`` exception
    leak through. The originating exception is preserved on ``__cause__``.
    """


def _session_with_ua(session: Optional[requests.Session]) -> requests.Session:
    """Return ``session`` (ensuring a UA) or a fresh one with the scanner UA.

    Reuses the same User-Agent / Accept convention as ``fetcher`` so the CLOB
    sees a consistent client. A session passed in by a caller (or a test) is
    used as-is apart from filling in default headers it is missing.
    """
    http = session if session is not None else requests.Session()
    if "Accept" not in http.headers:
        http.headers["Accept"] = "application/json"
    if "User-Agent" not in http.headers:
        http.headers["User-Agent"] = DEFAULT_USER_AGENT
    return http


def get_order_book(token_id: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch the CLOB order book for a single ERC-1155 outcome token.

    Issues ``GET {CLOB_API_BASE_URL}/book?token_id=<token_id>`` and returns
    the parsed JSON book, expected to carry ``bids`` and ``asks`` arrays of
    ``{"price": ..., "size": ...}`` levels (prices already in dollars,
    ``[0, 1]``). See the module docstring's live-verification caveat for the
    exact path/shape that must be confirmed against the live API.

    Args:
        token_id: The CLOB outcome token id (a long decimal string). Required;
            an empty value raises :class:`PolymarketAPIError` without making a
            network call.
        session: Optional injected ``requests.Session`` (tests patch its
            ``get``). A fresh session carrying the scanner User-Agent is built
            when omitted.

    Returns:
        The parsed book dict.

    Raises:
        PolymarketAPIError: On empty ``token_id``, any network error, a non-2xx
            response, or a non-JSON / non-dict body — never a bare ``requests``
            error. Failures are NOT swallowed.
    """
    if not token_id:
        raise PolymarketAPIError("get_order_book called with empty token_id")

    http = _session_with_ua(session)
    url = f"{CLOB_API_BASE_URL}/book"

    try:
        resp = http.get(url, params={"token_id": token_id}, timeout=DEFAULT_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise PolymarketAPIError(
            f"Polymarket get_order_book(token_id={token_id!r}) GET {url} failed: {exc}"
        ) from exc

    try:
        resp.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(resp, "status_code", "?")
        raise PolymarketAPIError(
            f"Polymarket get_order_book(token_id={token_id!r}) GET {url} "
            f"returned HTTP {status}: {exc}"
        ) from exc

    try:
        book = resp.json()
    except (ValueError, requests.RequestException) as exc:
        raise PolymarketAPIError(
            f"Polymarket get_order_book(token_id={token_id!r}) GET {url} "
            f"returned non-JSON body: {exc}"
        ) from exc

    if not isinstance(book, dict):
        raise PolymarketAPIError(
            f"Polymarket get_order_book(token_id={token_id!r}) GET {url} "
            f"returned a non-dict payload: {type(book).__name__}"
        )
    return book


def get_market_resolution(
    condition_id: str,
    session: Optional[requests.Session] = None,
) -> Optional[Dict[str, Any]]:
    """Read a CLOB market's resolution status for SHADOW settlement.

    Issues ``GET {CLOB_API_BASE_URL}/markets/<condition_id>`` and normalizes
    the subset of the response settlement needs into::

        {
          "closed": bool,            # market resolved? (do NOT settle if False)
          "tokens": [               # ordered to match outcomeIndex (0, 1, ...)
            {"outcome": str|None, "price": float|None, "winner": bool},
            ...
          ],
        }

    Resolution semantics (shape live-verified 2026-06-01):
        While a market is open the CLOB returns ``closed: false`` and the caller
        MUST leave the position open (no look-ahead settlement). Once
        ``closed: true``, each token in the ordered ``tokens`` list carries a
        boolean ``winner`` and a ``price`` of exactly 0 or 1. The token list is
        ordered to match ``outcomeIndex`` (index 0, 1, ...), the same
        ``[YES, NO]`` convention as ``clobTokenIds``, so the converged
        ``outcomeIndex`` indexes directly into ``tokens``.

    This is intentionally a degrade-gracefully read: settlement runs unattended
    in a loop, and one unreachable/garbled market must NOT raise. ANY failure —
    empty ``condition_id``, network error, non-2xx HTTP, non-JSON / non-dict
    body, or a missing ``tokens``/``closed`` field — returns ``None`` so the
    caller simply leaves that position open and retries on the next sweep.

    SHADOW-ONLY: this is a pure read — it places no order, signs nothing, and
    touches no wallet, consistent with the rest of this module.

    Args:
        condition_id: The market ``conditionId`` (a ``0x...`` string).
        session: Optional injected ``requests.Session`` (tests patch its
            ``get``). A fresh session carrying the scanner User-Agent is built
            when omitted.

    Returns:
        The normalized resolution dict, or ``None`` on any error.
    """
    if not condition_id:
        return None

    http = _session_with_ua(session)
    url = f"{CLOB_API_BASE_URL}/markets/{condition_id}"

    try:
        resp = http.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None
    except Exception:  # pragma: no cover - defensive: never let a read crash a sweep.
        return None

    if not isinstance(payload, dict):
        return None

    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, (list, tuple)):
        return None

    tokens: List[Dict[str, Any]] = []
    for raw in raw_tokens:
        if not isinstance(raw, dict):
            # Preserve positional ordering by index: a malformed token entry
            # would shift outcomeIndex alignment, so bail rather than mislabel.
            return None
        tokens.append(
            {
                "outcome": raw.get("outcome"),
                "price": _coerce_price(raw.get("price")),
                "winner": bool(raw.get("winner")),
            }
        )

    return {"closed": bool(payload.get("closed")), "tokens": tokens}


def _coerce_price(value: Any) -> Optional[float]:
    """Coerce a CLOB price/size value to ``float``; ``None`` if not numeric.

    The CLOB serializes book levels with string prices (e.g. ``"0.43"``), so
    we tolerate both strings and numbers and degrade unparseable values to
    ``None`` rather than raising — one bad level should not sink the book.
    """
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _level_prices(levels: Any) -> List[float]:
    """Extract the numeric ``price`` from each ``{"price", "size"}`` level.

    Accepts the documented list-of-dicts shape and is tolerant of bare
    ``[price, size]`` pair levels too. Non-numeric / malformed levels are
    skipped.
    """
    prices: List[float] = []
    if not isinstance(levels, (list, tuple)):
        return prices
    for level in levels:
        price: Optional[float]
        if isinstance(level, dict):
            price = _coerce_price(level.get("price"))
        elif isinstance(level, (list, tuple)) and level:
            price = _coerce_price(level[0])
        else:
            price = _coerce_price(level)
        if price is not None:
            prices.append(price)
    return prices


def best_ask(book: Dict[str, Any]) -> Optional[float]:
    """Lowest ask price in ``book``, or ``None`` if there are no asks.

    The best (cheapest) price to BUY at is the minimum of the ``asks`` side.
    """
    if not isinstance(book, dict):
        return None
    prices = _level_prices(book.get("asks"))
    return min(prices) if prices else None


def best_bid(book: Dict[str, Any]) -> Optional[float]:
    """Highest bid price in ``book``, or ``None`` if there are no bids.

    The best (richest) price to SELL at is the maximum of the ``bids`` side.
    """
    if not isinstance(book, dict):
        return None
    prices = _level_prices(book.get("bids"))
    return max(prices) if prices else None


def _resolve_clob_token_ids(market: Any) -> Optional[Sequence[str]]:
    """Extract the two clobTokenIds from a Gamma market dict or object.

    Accepts:
        * a Gamma market **dict** with ``clobTokenIds`` (or ``clob_token_ids``)
          as either a JSON-string of a 2-element list or an already-parsed
          list; or
        * an **object** exposing one of those attributes in the same shapes.

    Returns the ``[yes_token, no_token]`` sequence (length 2) or ``None`` when
    the field is missing, unparseable, or does not contain exactly two ids.
    """
    raw: Any = None
    for key in _CLOB_TOKEN_KEYS:
        if isinstance(market, dict):
            if key in market and market[key] is not None:
                raw = market[key]
                break
        elif getattr(market, key, None) is not None:
            raw = getattr(market, key)
            break

    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None

    tokens = [str(token).strip() for token in raw]
    if not all(tokens):
        return None
    return tokens


def get_yes_no_best_asks(
    market: Any,
    session: Optional[requests.Session] = None,
) -> Optional[Tuple[float, float]]:
    """Resolve a market's YES/NO best asks from the CLOB books.

    Reads the market's two ``clobTokenIds`` — ASSUMED ordered
    ``[YES_token, NO_token]`` (index 0 = YES, index 1 = NO); see the module
    docstring — fetches each token's order book, and returns
    ``(yes_best_ask, no_best_ask)`` in dollars ``[0, 1]``.

    Expected ``market`` fields:
        ``clobTokenIds`` (or ``clob_token_ids``) — a JSON-string list, or an
        already-parsed list, of exactly two outcome token ids. Either a Gamma
        market dict or an object exposing that attribute is accepted.

    Returns:
        ``(yes_best_ask, no_best_ask)`` when both books resolve and each has at
        least one ask; otherwise ``None`` — specifically when the
        ``clobTokenIds`` are missing or not exactly two, or when either token's
        book has no asks.

    Raises:
        PolymarketAPIError: Propagated from :func:`get_order_book` if a book
            fetch fails (network / HTTP / non-JSON). A *missing* or empty asks
            side is a normal "no quote" condition and returns ``None`` instead.
    """
    tokens = _resolve_clob_token_ids(market)
    if tokens is None:
        return None

    yes_token, no_token = tokens[0], tokens[1]

    yes_book = get_order_book(yes_token, session=session)
    yes_ask = best_ask(yes_book)
    if yes_ask is None:
        return None

    no_book = get_order_book(no_token, session=session)
    no_ask = best_ask(no_book)
    if no_ask is None:
        return None

    return (yes_ask, no_ask)
