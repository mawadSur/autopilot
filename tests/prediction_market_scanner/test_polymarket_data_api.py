"""Tests for src/exchanges/polymarket_data_api.py.

Hermetic — never makes a real HTTPS call. We patch ``session.get`` to return a
stub response whose ``.json()`` yields canned data-api-shaped payloads (the
EXACT shapes live-verified 2026-05-31), and assert on per-endpoint parsing,
query-param wiring, and error wrapping.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Optional
from unittest import mock

import requests

from exchanges.polymarket_data_api import (
    DEFAULT_BASE_URL,
    DEFAULT_LB_BASE_URL,
    PolymarketDataAPIClient,
    PolymarketDataAPIError,
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


def _make_client(get_mock: mock.Mock) -> PolymarketDataAPIClient:
    session = mock.Mock(spec=requests.Session)
    session.get = get_mock
    return PolymarketDataAPIClient(session=session)


# ---------------------------------------------------------------------------
# Sample payloads — EXACT live-verified shapes
# ---------------------------------------------------------------------------


def _trades_payload() -> List[dict]:
    return [
        {
            "proxyWallet": "0xWALLET_A",
            "side": "BUY",
            "asset": "11111111111111111111",
            "conditionId": "0xCOND1",
            "size": 250.0,
            "price": 0.62,
            "timestamp": 1748600000,
            "title": "Will X happen?",
            "slug": "will-x-happen",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "name": "alpha",
            "transactionHash": "0xTX1",
        },
        {
            "proxyWallet": "0xWALLET_B",
            "side": "SELL",
            "asset": "22222222222222222222",
            "conditionId": "0xCOND1",
            "size": 100.0,
            "price": 0.38,
            "timestamp": 1748600100,
            "title": "Will X happen?",
            "slug": "will-x-happen",
            "outcome": "No",
            "outcomeIndex": 1,
            "name": "bravo",
            "transactionHash": "0xTX2",
        },
    ]


def _positions_payload() -> List[dict]:
    return [
        {
            "proxyWallet": "0xWALLET_A",
            "asset": "11111111111111111111",
            "conditionId": "0xCOND1",
            "size": 250.0,
            "avgPrice": 0.55,
            "initialValue": 137.5,
            "currentValue": 250.0,
            "cashPnl": 112.5,
            "percentPnl": 0.818,
            "totalBought": 137.5,
            "realizedPnl": 112.5,
            "percentRealizedPnl": 0.818,
            "curPrice": 1.0,
            "redeemable": True,
            "title": "Will X happen?",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "oppositeOutcome": "No",
            "oppositeAsset": "22222222222222222222",
            "endDate": "2026-05-01T00:00:00Z",
            "negativeRisk": False,
        }
    ]


def _holders_payload() -> List[dict]:
    return [
        {
            "token": "11111111111111111111",
            "holders": [
                {
                    "proxyWallet": "0xWALLET_A",
                    "asset": "11111111111111111111",
                    "amount": 250.0,
                    "outcomeIndex": 0,
                    "name": "Yes",
                    "pseudonym": "alpha",
                    "verified": True,
                }
            ],
        },
        {
            "token": "22222222222222222222",
            "holders": [
                {
                    "proxyWallet": "0xWALLET_B",
                    "asset": "22222222222222222222",
                    "amount": 100.0,
                    "outcomeIndex": 1,
                    "name": "No",
                    "pseudonym": "bravo",
                    "verified": False,
                }
            ],
        },
    ]


def _profit_payload() -> List[dict]:
    return [
        {
            "proxyWallet": "0xWINNER_1",
            "amount": 1234567.89,
            "name": "Theo4",
            "pseudonym": "theo",
        },
        {
            "proxyWallet": "0xWINNER_2",
            "amount": 98765.43,
            "name": "WhaleTwo",
            "pseudonym": "wtwo",
        },
    ]


# ---------------------------------------------------------------------------
# get_trades
# ---------------------------------------------------------------------------


class GetTradesTest(unittest.TestCase):
    def test_parses_trade_list(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_trades_payload()))
        client = _make_client(get_mock)

        trades = client.get_trades(user="0xWALLET_A", market="0xCOND1", limit=50, offset=0)

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0]["proxyWallet"], "0xWALLET_A")
        self.assertEqual(trades[0]["side"], "BUY")
        self.assertEqual(trades[0]["outcomeIndex"], 0)
        self.assertEqual(trades[0]["conditionId"], "0xCOND1")

    def test_query_params_and_url(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_trades_payload()))
        client = _make_client(get_mock)

        client.get_trades(user="0xWALLET_A", market="0xCOND1", limit=25, offset=50)

        args, kwargs = get_mock.call_args
        self.assertEqual(args[0], f"{DEFAULT_BASE_URL}/trades")
        self.assertEqual(kwargs["params"]["user"], "0xWALLET_A")
        self.assertEqual(kwargs["params"]["market"], "0xCOND1")
        self.assertEqual(kwargs["params"]["limit"], 25)
        self.assertEqual(kwargs["params"]["offset"], 50)
        self.assertEqual(kwargs["timeout"], 10.0)

    def test_omits_optional_filters(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_trades_payload()))
        client = _make_client(get_mock)

        client.get_trades(limit=100, offset=0)

        _, kwargs = get_mock.call_args
        self.assertNotIn("user", kwargs["params"])
        self.assertNotIn("market", kwargs["params"])
        self.assertEqual(kwargs["params"]["limit"], 100)

    def test_non_list_payload_yields_empty(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({"unexpected": "object"}))
        client = _make_client(get_mock)
        self.assertEqual(client.get_trades(), [])


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------


class GetPositionsTest(unittest.TestCase):
    def test_parses_positions(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_positions_payload()))
        client = _make_client(get_mock)

        positions = client.get_positions("0xWALLET_A", limit=500)
        self.assertEqual(len(positions), 1)
        pos = positions[0]
        self.assertEqual(pos["realizedPnl"], 112.5)
        self.assertTrue(pos["redeemable"])
        self.assertEqual(pos["conditionId"], "0xCOND1")

    def test_query_params_and_url(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_positions_payload()))
        client = _make_client(get_mock)

        client.get_positions("0xWALLET_A", limit=250)

        args, kwargs = get_mock.call_args
        self.assertEqual(args[0], f"{DEFAULT_BASE_URL}/positions")
        self.assertEqual(kwargs["params"]["user"], "0xWALLET_A")
        self.assertEqual(kwargs["params"]["limit"], 250)

    def test_empty_user_rejected_without_network(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_positions_payload()))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError):
            client.get_positions("")
        get_mock.assert_not_called()


# ---------------------------------------------------------------------------
# get_holders
# ---------------------------------------------------------------------------


class GetHoldersTest(unittest.TestCase):
    def test_parses_holder_groups(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_holders_payload()))
        client = _make_client(get_mock)

        groups = client.get_holders("0xCOND1", limit=100)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["token"], "11111111111111111111")
        self.assertEqual(groups[0]["holders"][0]["proxyWallet"], "0xWALLET_A")
        self.assertEqual(groups[0]["holders"][0]["outcomeIndex"], 0)

    def test_query_params_and_url(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_holders_payload()))
        client = _make_client(get_mock)

        client.get_holders("0xCOND1", limit=42)

        args, kwargs = get_mock.call_args
        self.assertEqual(args[0], f"{DEFAULT_BASE_URL}/holders")
        self.assertEqual(kwargs["params"]["market"], "0xCOND1")
        self.assertEqual(kwargs["params"]["limit"], 42)

    def test_empty_market_rejected_without_network(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_holders_payload()))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError):
            client.get_holders("")
        get_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


class ErrorHandlingTest(unittest.TestCase):
    def test_network_error_wrapped(self) -> None:
        get_mock = mock.Mock(side_effect=requests.ConnectionError("refused"))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError) as ctx:
            client.get_trades(user="0xWALLET_A")
        self.assertIn("get_trades", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, requests.ConnectionError)

    def test_timeout_wrapped(self) -> None:
        get_mock = mock.Mock(side_effect=requests.Timeout("timed out"))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError):
            client.get_positions("0xWALLET_A")

    def test_http_500_wrapped_with_context(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse([], status_code=500))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError) as ctx:
            client.get_holders("0xCOND1")
        msg = str(ctx.exception)
        self.assertIn("get_holders", msg)
        self.assertIn("0xCOND1", msg)
        self.assertIsInstance(ctx.exception.__cause__, requests.HTTPError)

    def test_non_json_body_wrapped(self) -> None:
        get_mock = mock.Mock(
            return_value=_StubResponse(raise_on_json=ValueError("no json"))
        )
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError):
            client.get_trades()


# ---------------------------------------------------------------------------
# get_profit_leaderboard (DIFFERENT host: lb-api)
# ---------------------------------------------------------------------------


class GetProfitLeaderboardTest(unittest.TestCase):
    def test_parses_profit_list(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_profit_payload()))
        client = _make_client(get_mock)

        rows = client.get_profit_leaderboard(window="all", limit=100)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["proxyWallet"], "0xWINNER_1")
        self.assertEqual(rows[0]["amount"], 1234567.89)
        self.assertEqual(rows[0]["name"], "Theo4")
        self.assertEqual(rows[1]["proxyWallet"], "0xWINNER_2")

    def test_hits_lb_host_with_window_and_limit(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_profit_payload()))
        client = _make_client(get_mock)

        client.get_profit_leaderboard(window="1d", limit=25)

        args, kwargs = get_mock.call_args
        # The profit endpoint MUST go to the lb-api host, not the data-api host.
        self.assertEqual(args[0], f"{DEFAULT_LB_BASE_URL}/profit")
        self.assertNotEqual(DEFAULT_LB_BASE_URL, DEFAULT_BASE_URL)
        self.assertEqual(kwargs["params"]["window"], "1d")
        self.assertEqual(kwargs["params"]["limit"], 25)

    def test_default_window_is_all(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_profit_payload()))
        client = _make_client(get_mock)
        client.get_profit_leaderboard()
        _, kwargs = get_mock.call_args
        self.assertEqual(kwargs["params"]["window"], "all")
        self.assertEqual(kwargs["params"]["limit"], 100)

    def test_bad_window_400_wrapped_with_context(self) -> None:
        # An unsupported window string returns HTTP 400 -> wrapped, not crashed.
        get_mock = mock.Mock(return_value=_StubResponse([], status_code=400))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError) as ctx:
            client.get_profit_leaderboard(window="7d")
        msg = str(ctx.exception)
        self.assertIn("get_profit_leaderboard", msg)
        self.assertIn("7d", msg)  # the offending window is surfaced
        self.assertIn("400", msg)
        self.assertIsInstance(ctx.exception.__cause__, requests.HTTPError)

    def test_network_error_wrapped(self) -> None:
        get_mock = mock.Mock(side_effect=requests.ConnectionError("refused"))
        client = _make_client(get_mock)
        with self.assertRaises(PolymarketDataAPIError) as ctx:
            client.get_profit_leaderboard(window="all")
        self.assertIn("get_profit_leaderboard", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, requests.ConnectionError)

    def test_non_list_payload_yields_empty(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({"error": "nope"}))
        client = _make_client(get_mock)
        self.assertEqual(client.get_profit_leaderboard(), [])

    def test_lb_base_url_override(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_profit_payload()))
        session = mock.Mock(spec=requests.Session)
        session.get = get_mock
        client = PolymarketDataAPIClient(
            session=session, lb_base_url="https://lb.example.test/"
        )
        client.get_profit_leaderboard()
        args, _ = get_mock.call_args
        self.assertEqual(args[0], "https://lb.example.test/profit")


# ---------------------------------------------------------------------------
# base_url override
# ---------------------------------------------------------------------------


class BaseUrlTest(unittest.TestCase):
    def test_base_url_override(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_trades_payload()))
        session = mock.Mock(spec=requests.Session)
        session.get = get_mock
        client = PolymarketDataAPIClient(
            base_url="https://example.test/api/", session=session
        )
        client.get_trades()
        args, _ = get_mock.call_args
        # Trailing slash stripped; path appended cleanly.
        self.assertEqual(args[0], "https://example.test/api/trades")


if __name__ == "__main__":
    unittest.main()
