"""Tests for src/exchanges/polymarket_market_data.py.

Hermetic — never makes a real HTTPS call. We patch ``session.get`` to return
a stub response whose ``.json()`` yields canned Polymarket CLOB ``/book``
payloads, and assert on best_ask/best_bid parsing, YES/NO ask resolution from
a market's two ``clobTokenIds`` (index 0 = YES, index 1 = NO), the
None-return conditions (missing / !=2 token ids, empty asks), and the
PolymarketAPIError wrapping of HTTP failures.

SHADOW-ONLY: the client under test exposes ONLY read methods — there is no
order/execution surface to exercise.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Optional
from unittest import mock

import requests

from exchanges.polymarket_market_data import (
    PolymarketAPIError,
    best_ask,
    best_bid,
    get_order_book,
    get_yes_no_best_asks,
)


# ---------------------------------------------------------------------------
# Stub HTTP scaffolding
# ---------------------------------------------------------------------------


class _StubResponse:
    """Minimal ``requests.Response`` shaped stub."""

    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
        raise_on_json: Optional[Exception] = None,
    ) -> None:
        self._json_data = json_data
        self.status_code = status_code
        self._raise_on_json = raise_on_json

    def raise_for_status(self) -> None:
        if not (200 <= int(self.status_code) < 300):
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._json_data


def _make_session(get_mock: mock.Mock) -> requests.Session:
    session = mock.Mock(spec=requests.Session)
    session.get = get_mock
    # ``_session_with_ua`` touches ``session.headers``; give it a real dict.
    session.headers = {}
    return session


# ---------------------------------------------------------------------------
# Sample CLOB /book payloads (prices already in dollars [0, 1])
# ---------------------------------------------------------------------------


def _book_payload(bids: Any = None, asks: Any = None) -> dict:
    return {
        "market": "0xcondition",
        "asset_id": "token-123",
        "bids": bids if bids is not None else [
            {"price": "0.40", "size": "100"},
            {"price": "0.41", "size": "250"},
        ],
        "asks": asks if asks is not None else [
            {"price": "0.45", "size": "300"},
            {"price": "0.43", "size": "120"},
        ],
    }


# ---------------------------------------------------------------------------
# best_ask / best_bid parsing
# ---------------------------------------------------------------------------


class BestAskBidTest(unittest.TestCase):
    def test_best_ask_is_lowest_ask(self) -> None:
        book = _book_payload()
        self.assertAlmostEqual(best_ask(book), 0.43)

    def test_best_bid_is_highest_bid(self) -> None:
        book = _book_payload()
        self.assertAlmostEqual(best_bid(book), 0.41)

    def test_best_ask_none_when_no_asks(self) -> None:
        self.assertIsNone(best_ask(_book_payload(asks=[])))
        self.assertIsNone(best_ask({"bids": [{"price": "0.4", "size": "1"}]}))

    def test_best_bid_none_when_no_bids(self) -> None:
        self.assertIsNone(best_bid(_book_payload(bids=[])))

    def test_tolerates_bare_pair_levels(self) -> None:
        # [price, size] pair-shaped levels are tolerated alongside dict levels.
        book = {"asks": [[0.55, 10], [0.52, 5]], "bids": [[0.30, 7], [0.31, 9]]}
        self.assertAlmostEqual(best_ask(book), 0.52)
        self.assertAlmostEqual(best_bid(book), 0.31)

    def test_non_dict_book_returns_none(self) -> None:
        self.assertIsNone(best_ask(None))  # type: ignore[arg-type]
        self.assertIsNone(best_bid("nope"))  # type: ignore[arg-type]

    def test_non_numeric_levels_skipped(self) -> None:
        book = {"asks": [{"price": "garbage", "size": "1"}, {"price": "0.6", "size": "1"}]}
        self.assertAlmostEqual(best_ask(book), 0.6)


# ---------------------------------------------------------------------------
# get_order_book: happy path + params + error wrapping
# ---------------------------------------------------------------------------


class GetOrderBookTest(unittest.TestCase):
    def test_returns_parsed_book_and_passes_token_param(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_book_payload()))
        session = _make_session(get_mock)

        book = get_order_book("token-abc", session=session)
        self.assertIn("asks", book)
        self.assertIn("bids", book)

        args, kwargs = get_mock.call_args
        self.assertTrue(args[0].endswith("/book"))
        self.assertEqual(kwargs["params"]["token_id"], "token-abc")

    def test_empty_token_id_raises_without_network_call(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_book_payload()))
        session = _make_session(get_mock)
        with self.assertRaises(PolymarketAPIError):
            get_order_book("", session=session)
        get_mock.assert_not_called()

    def test_network_error_raises_polymarket_api_error(self) -> None:
        get_mock = mock.Mock(side_effect=requests.ConnectionError("refused"))
        session = _make_session(get_mock)
        with self.assertRaises(PolymarketAPIError) as ctx:
            get_order_book("token-x", session=session)
        self.assertIn("token-x", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, requests.ConnectionError)

    def test_http_500_raises_polymarket_api_error(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({}, status_code=500))
        session = _make_session(get_mock)
        with self.assertRaises(PolymarketAPIError) as ctx:
            get_order_book("token-y", session=session)
        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, requests.HTTPError)

    def test_non_json_body_raises_polymarket_api_error(self) -> None:
        get_mock = mock.Mock(
            return_value=_StubResponse(raise_on_json=ValueError("no json"))
        )
        session = _make_session(get_mock)
        with self.assertRaises(PolymarketAPIError):
            get_order_book("token-z", session=session)

    def test_non_dict_payload_raises(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(["not", "a", "dict"]))
        session = _make_session(get_mock)
        with self.assertRaises(PolymarketAPIError):
            get_order_book("token-w", session=session)


# ---------------------------------------------------------------------------
# get_yes_no_best_asks: resolve two tokens -> two asks
# ---------------------------------------------------------------------------


class GetYesNoBestAsksTest(unittest.TestCase):
    def test_resolves_yes_then_no_asks(self) -> None:
        # YES book best ask 0.43, NO book best ask 0.55.
        yes_book = _book_payload(asks=[{"price": "0.45", "size": "1"}, {"price": "0.43", "size": "1"}])
        no_book = _book_payload(asks=[{"price": "0.55", "size": "1"}, {"price": "0.58", "size": "1"}])

        def _get(url, params=None, timeout=None):
            if params["token_id"] == "YES_TOKEN":
                return _StubResponse(yes_book)
            if params["token_id"] == "NO_TOKEN":
                return _StubResponse(no_book)
            raise AssertionError(f"unexpected token_id {params['token_id']!r}")

        get_mock = mock.Mock(side_effect=_get)
        session = _make_session(get_mock)

        market = {"clobTokenIds": json.dumps(["YES_TOKEN", "NO_TOKEN"])}
        asks = get_yes_no_best_asks(market, session=session)
        self.assertIsNotNone(asks)
        yes_ask, no_ask = asks
        self.assertAlmostEqual(yes_ask, 0.43)
        self.assertAlmostEqual(no_ask, 0.55)
        # YES queried first (index 0), NO second (index 1).
        self.assertEqual(get_mock.call_args_list[0].kwargs["params"]["token_id"], "YES_TOKEN")
        self.assertEqual(get_mock.call_args_list[1].kwargs["params"]["token_id"], "NO_TOKEN")

    def test_accepts_already_parsed_list_and_snake_case_key(self) -> None:
        book = _book_payload(asks=[{"price": "0.30", "size": "1"}])
        get_mock = mock.Mock(return_value=_StubResponse(book))
        session = _make_session(get_mock)

        market = {"clob_token_ids": ["YES", "NO"]}  # parsed list + snake_case key
        asks = get_yes_no_best_asks(market, session=session)
        self.assertIsNotNone(asks)
        self.assertAlmostEqual(asks[0], 0.30)
        self.assertAlmostEqual(asks[1], 0.30)

    def test_accepts_object_exposing_clob_token_ids(self) -> None:
        book = _book_payload(asks=[{"price": "0.20", "size": "1"}])
        get_mock = mock.Mock(return_value=_StubResponse(book))
        session = _make_session(get_mock)

        class _MarketObj:
            clobTokenIds = json.dumps(["YES", "NO"])

        asks = get_yes_no_best_asks(_MarketObj(), session=session)
        self.assertIsNotNone(asks)
        self.assertAlmostEqual(asks[0], 0.20)

    def test_missing_clob_token_ids_returns_none(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_book_payload()))
        session = _make_session(get_mock)
        self.assertIsNone(get_yes_no_best_asks({}, session=session))
        get_mock.assert_not_called()

    def test_wrong_token_count_returns_none(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_book_payload()))
        session = _make_session(get_mock)
        one = {"clobTokenIds": json.dumps(["ONLY_ONE"])}
        three = {"clobTokenIds": json.dumps(["A", "B", "C"])}
        self.assertIsNone(get_yes_no_best_asks(one, session=session))
        self.assertIsNone(get_yes_no_best_asks(three, session=session))
        get_mock.assert_not_called()

    def test_unparseable_token_json_returns_none(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_book_payload()))
        session = _make_session(get_mock)
        market = {"clobTokenIds": "{not valid json"}
        self.assertIsNone(get_yes_no_best_asks(market, session=session))
        get_mock.assert_not_called()

    def test_empty_yes_asks_returns_none(self) -> None:
        empty_book = _book_payload(asks=[])
        get_mock = mock.Mock(return_value=_StubResponse(empty_book))
        session = _make_session(get_mock)
        market = {"clobTokenIds": json.dumps(["YES", "NO"])}
        self.assertIsNone(get_yes_no_best_asks(market, session=session))

    def test_empty_no_asks_returns_none(self) -> None:
        yes_book = _book_payload(asks=[{"price": "0.40", "size": "1"}])
        no_book = _book_payload(asks=[])

        def _get(url, params=None, timeout=None):
            if params["token_id"] == "YES":
                return _StubResponse(yes_book)
            return _StubResponse(no_book)

        get_mock = mock.Mock(side_effect=_get)
        session = _make_session(get_mock)
        market = {"clobTokenIds": json.dumps(["YES", "NO"])}
        self.assertIsNone(get_yes_no_best_asks(market, session=session))

    def test_book_fetch_http_error_propagates(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({}, status_code=503))
        session = _make_session(get_mock)
        market = {"clobTokenIds": json.dumps(["YES", "NO"])}
        with self.assertRaises(PolymarketAPIError):
            get_yes_no_best_asks(market, session=session)


if __name__ == "__main__":
    unittest.main()
