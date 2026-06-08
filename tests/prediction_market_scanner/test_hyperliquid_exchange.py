"""Tests for src/exchanges/hyperliquid.py.

Hermetic — never makes a real HTTPS call. We inject a stub
``requests.Session`` whose ``.post()`` returns a ``Mock`` whose
``.json()`` yields canned payloads.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from exchanges import (
    Balance,
    ExchangeError,
    HyperliquidExchange,
    MarginAccount,
    OrderResult,
    PerpPosition,
    PerpTicker,
    Ticker,
)


# ---------------------------------------------------------------------------
# Stub session scaffolding
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
        text: str = "",
        raise_on_json: Optional[Exception] = None,
    ) -> None:
        self._json_data = json_data
        self.status_code = status_code
        self.text = text
        self._raise_on_json = raise_on_json

    def json(self) -> Any:
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._json_data


class _StubSession:
    """Minimal ``requests.Session`` shaped stub.

    Records each ``.post()`` call and dispatches based on the JSON body's
    ``"type"`` field to whatever response was registered.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._responses_by_type: Dict[str, _StubResponse] = {}
        self._default_response: Optional[_StubResponse] = None
        self._raise_on_post: Optional[Exception] = None

    def register(self, info_type: str, response: _StubResponse) -> None:
        self._responses_by_type[info_type] = response

    def set_default(self, response: _StubResponse) -> None:
        self._default_response = response

    def raise_on_post(self, exc: Exception) -> None:
        self._raise_on_post = exc

    def post(self, url: str, json: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> _StubResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raise_on_post is not None:
            raise self._raise_on_post
        body = json or {}
        info_type = str(body.get("type", ""))
        if info_type in self._responses_by_type:
            return self._responses_by_type[info_type]
        if self._default_response is not None:
            return self._default_response
        return _StubResponse(json_data={})


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------


def _meta_and_ctxs(symbol: str = "ETH") -> List[Any]:
    return [
        {"universe": [{"name": "BTC"}, {"name": symbol}]},
        [
            {"markPx": "50000.0", "oraclePx": "49990.0", "funding": "0.00010", "openInterest": "1000.0", "dayNtlVlm": "100000000"},
            {
                "markPx": "2000.5",
                "oraclePx": "2000.0",
                "funding": "0.0000125",
                "openInterest": "5000.0",
                "dayNtlVlm": "75000000.0",
            },
        ],
    ]


def _clearinghouse_state() -> Dict[str, Any]:
    return {
        "withdrawable": "950.0",
        "marginSummary": {
            "accountValue": "1000.0",
            "totalMarginUsed": "50.0",
            "totalNtlPos": "2500.0",
        },
        "assetPositions": [
            {
                "position": {
                    "coin": "ETH",
                    "szi": "1.25",
                    "entryPx": "2000.0",
                    "unrealizedPnl": "12.5",
                    "marginUsed": "50.0",
                    "liquidationPx": "1500.0",
                    "leverage": {"value": "5"},
                }
            },
            {
                "position": {
                    "coin": "BTC",
                    "szi": "-0.1",
                    "entryPx": "50000.0",
                    "unrealizedPnl": "-100.0",
                    "marginUsed": "100.0",
                    "liquidationPx": None,
                    "leverage": {"value": "10"},
                }
            },
            # Flat row that should be skipped.
            {"position": {"coin": "SOL", "szi": "0", "entryPx": "0"}},
        ],
    }


def _open_orders() -> List[Dict[str, Any]]:
    return [
        {
            "coin": "ETH",
            "side": "B",
            "limitPx": "1900.0",
            "sz": "0.5",
            "oid": 12345,
            "timestamp": 1700000000000,
        },
        {
            "coin": "BTC",
            "side": "A",
            "limitPx": "60000.0",
            "sz": "0.01",
            "oid": 67890,
            "timestamp": 1700000001000,
        },
    ]


def _user_fills() -> List[Dict[str, Any]]:
    return [
        {
            "coin": "ETH",
            "side": "B",
            "px": "2000.0",
            "sz": "0.25",
            "fee": "0.5",
            "oid": 111,
            "time": 1700000000000,
        },
        {
            "coin": "ETH",
            "side": "A",
            "px": "2010.0",
            "sz": "0.25",
            "fee": "0.5",
            "oid": 222,
            "time": 1700000050000,
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class HyperliquidExchangeInitTests(unittest.TestCase):
    def test_init_reads_env_vars_when_args_omitted(self) -> None:
        env = {
            "HYPERLIQUID_PRIVATE_KEY": "0xabc",
            "HYPERLIQUID_WALLET_ADDRESS": "0xdeadbeef",
            "HYPERLIQUID_BASE_URL": "https://test.hyperliquid.example",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            client = HyperliquidExchange(session=_StubSession())
        self.assertEqual(client.wallet_address, "0xdeadbeef")
        self.assertEqual(client.base_url, "https://test.hyperliquid.example")
        # Signing not wired in V1 even when key is set.
        self.assertFalse(client.is_signing_enabled())

    def test_init_uses_default_base_url(self) -> None:
        env = {
            "HYPERLIQUID_PRIVATE_KEY": "",
            "HYPERLIQUID_WALLET_ADDRESS": "",
            "HYPERLIQUID_BASE_URL": "",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            client = HyperliquidExchange(session=_StubSession())
        self.assertEqual(client.base_url, "https://api.hyperliquid.xyz")


class HyperliquidExchangeReadTests(unittest.TestCase):
    def _client(self, session: Optional[_StubSession] = None, *, with_wallet: bool = True) -> HyperliquidExchange:
        sess = session or _StubSession()
        return HyperliquidExchange(
            private_key=None,
            wallet_address="0xWALLET" if with_wallet else None,
            base_url="https://api.hyperliquid.xyz",
            session=sess,
        )

    def test_get_ticker_synthesizes_bid_ask_from_mark_and_oracle(self) -> None:
        sess = _StubSession()
        sess.register("metaAndAssetCtxs", _StubResponse(json_data=_meta_and_ctxs("ETH")))
        client = self._client(sess)

        t = client.get_ticker("ETH")
        self.assertIsInstance(t, Ticker)
        self.assertEqual(t.symbol, "ETH")
        # mark=2000.5, oracle=2000.0 -> half_spread = 0.25
        self.assertAlmostEqual(t.ask - t.bid, 0.5, places=6)
        self.assertAlmostEqual(t.last, 2000.5, places=6)
        self.assertAlmostEqual(t.mid, 2000.5, places=6)

    def test_get_perp_ticker_parses_funding_rate_and_open_interest(self) -> None:
        sess = _StubSession()
        sess.register("metaAndAssetCtxs", _StubResponse(json_data=_meta_and_ctxs("ETH")))
        client = self._client(sess)

        pt = client.get_perp_ticker("ETH-PERP")
        self.assertIsInstance(pt, PerpTicker)
        self.assertEqual(pt.symbol, "ETH")
        self.assertAlmostEqual(pt.mark_price, 2000.5)
        self.assertAlmostEqual(pt.oracle_price, 2000.0)
        self.assertAlmostEqual(pt.funding_rate_8h, 0.0000125)
        self.assertAlmostEqual(pt.open_interest_base, 5000.0)
        self.assertAlmostEqual(pt.volume_24h_quote_usd, 75000000.0)

    def test_get_balances_returns_usdc_margin(self) -> None:
        sess = _StubSession()
        sess.register("clearinghouseState", _StubResponse(json_data=_clearinghouse_state()))
        client = self._client(sess)

        balances = client.get_balances()
        self.assertEqual(len(balances), 1)
        b = balances[0]
        self.assertIsInstance(b, Balance)
        self.assertEqual(b.currency, "USDC")
        self.assertAlmostEqual(b.free, 950.0)
        self.assertAlmostEqual(b.locked, 50.0)
        self.assertAlmostEqual(b.total, 1000.0)

    def test_get_margin_account_parses_clearinghouse_state(self) -> None:
        sess = _StubSession()
        sess.register("clearinghouseState", _StubResponse(json_data=_clearinghouse_state()))
        client = self._client(sess)

        ma = client.get_margin_account()
        self.assertIsInstance(ma, MarginAccount)
        self.assertAlmostEqual(ma.account_value_usd, 1000.0)
        self.assertAlmostEqual(ma.total_margin_used_usd, 50.0)
        self.assertAlmostEqual(ma.withdrawable_usd, 950.0)
        # Account leverage = totalNtlPos / accountValue = 2500/1000 = 2.5
        self.assertAlmostEqual(ma.leverage, 2.5)

    def test_get_open_positions_parses_active_positions(self) -> None:
        sess = _StubSession()
        sess.register("clearinghouseState", _StubResponse(json_data=_clearinghouse_state()))
        client = self._client(sess)

        positions = client.get_open_positions()
        self.assertEqual(len(positions), 2)  # SOL flat row skipped
        eth = next(p for p in positions if p.symbol == "ETH")
        btc = next(p for p in positions if p.symbol == "BTC")

        self.assertIsInstance(eth, PerpPosition)
        self.assertEqual(eth.side, "long")
        self.assertAlmostEqual(eth.size_base, 1.25)
        self.assertAlmostEqual(eth.entry_price, 2000.0)
        self.assertAlmostEqual(eth.unrealized_pnl_usd, 12.5)
        self.assertAlmostEqual(eth.liquidation_price or 0.0, 1500.0)
        self.assertAlmostEqual(eth.leverage, 5.0)
        self.assertAlmostEqual(eth.margin_used_usd, 50.0)
        # Mark = entry + pnl/size_signed = 2000 + 12.5/1.25 = 2010
        self.assertAlmostEqual(eth.mark_price, 2010.0)

        self.assertEqual(btc.side, "short")
        self.assertAlmostEqual(btc.size_base, 0.1)
        self.assertIsNone(btc.liquidation_price)

    def test_get_open_orders_returns_resting_orders(self) -> None:
        sess = _StubSession()
        sess.register("openOrders", _StubResponse(json_data=_open_orders()))
        client = self._client(sess)

        orders = client.get_open_orders()
        self.assertEqual(len(orders), 2)

        eth_order = next(o for o in orders if o.symbol == "ETH")
        btc_order = next(o for o in orders if o.symbol == "BTC")

        self.assertIsInstance(eth_order, OrderResult)
        self.assertEqual(eth_order.order_id, "12345")
        self.assertEqual(eth_order.side, "buy")
        self.assertEqual(eth_order.type, "limit")
        self.assertEqual(eth_order.status, "open")
        self.assertAlmostEqual(eth_order.limit_price or 0.0, 1900.0)
        self.assertAlmostEqual(eth_order.base_size or 0.0, 0.5)

        self.assertEqual(btc_order.side, "sell")
        self.assertEqual(btc_order.order_id, "67890")

        # Symbol filter
        only_eth = client.get_open_orders(symbol="ETH-PERP")
        self.assertEqual(len(only_eth), 1)
        self.assertEqual(only_eth[0].symbol, "ETH")

    def test_get_recent_fills_parses_user_fills(self) -> None:
        sess = _StubSession()
        sess.register("userFills", _StubResponse(json_data=_user_fills()))
        client = self._client(sess)

        fills = client.get_recent_fills(limit=10)
        self.assertEqual(len(fills), 2)

        first = fills[0]
        self.assertEqual(first.symbol, "ETH")
        self.assertEqual(first.side, "buy")
        self.assertEqual(first.status, "filled")
        self.assertAlmostEqual(first.filled_base, 0.25)
        self.assertAlmostEqual(first.filled_quote_usd, 500.0)
        self.assertAlmostEqual(first.avg_fill_price or 0.0, 2000.0)
        self.assertAlmostEqual(first.fee_usd, 0.5)
        self.assertEqual(first.order_id, "111")

        # Limit slicing
        sliced = client.get_recent_fills(limit=1)
        self.assertEqual(len(sliced), 1)


class HyperliquidExchangeWriteTests(unittest.TestCase):
    def _client(self) -> HyperliquidExchange:
        return HyperliquidExchange(
            private_key="0xabc",
            wallet_address="0xWALLET",
            base_url="https://api.hyperliquid.xyz",
            session=_StubSession(),
        )

    def test_place_market_order_raises_not_implemented(self) -> None:
        client = self._client()
        with self.assertRaises(NotImplementedError) as ctx:
            client.place_market_order("ETH", "buy", base_size=0.5)
        self.assertIn("EIP-712", str(ctx.exception))

    def test_place_limit_order_raises_not_implemented(self) -> None:
        client = self._client()
        with self.assertRaises(NotImplementedError) as ctx:
            client.place_limit_order("ETH", "sell", base_size=0.5, limit_price=2100.0)
        self.assertIn("EIP-712", str(ctx.exception))

    def test_cancel_order_raises_not_implemented(self) -> None:
        client = self._client()
        with self.assertRaises(NotImplementedError) as ctx:
            client.cancel_order("12345", "ETH")
        self.assertIn("EIP-712", str(ctx.exception))

    def test_is_signing_enabled_false_when_private_key_missing(self) -> None:
        env = {"HYPERLIQUID_PRIVATE_KEY": "", "HYPERLIQUID_WALLET_ADDRESS": "0xWALLET"}
        with mock.patch.dict("os.environ", env, clear=False):
            client = HyperliquidExchange(session=_StubSession())
        self.assertFalse(client.is_signing_enabled())


class HyperliquidExchangeErrorTests(unittest.TestCase):
    def test_post_info_wraps_errors_in_exchange_error(self) -> None:
        sess = _StubSession()
        underlying = ConnectionError("DNS lookup failed")
        sess.raise_on_post(underlying)
        client = HyperliquidExchange(
            wallet_address="0xWALLET",
            base_url="https://api.hyperliquid.xyz",
            session=sess,
        )
        with self.assertRaises(ExchangeError) as ctx:
            client.get_perp_ticker("ETH")
        self.assertIs(ctx.exception.__cause__, underlying)
        self.assertIn("/info POST failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
