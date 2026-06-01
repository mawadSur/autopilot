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
