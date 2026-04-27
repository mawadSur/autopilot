"""Tests for src/exchanges/coinbase.py.

Hermetic — no real network or ccxt instances are created. We inject a fake
ccxt module via the `ccxt_module` constructor kwarg.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from pydantic import ValidationError

from exchanges import (
    Balance,
    CoinbaseExchange,
    ExchangeError,
    OrderResult,
    Ticker,
)


# ---------------------------------------------------------------------------
# Fake ccxt scaffolding
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal ccxt-shaped client. Records calls and returns canned data."""

    def __init__(self, params: Dict[str, Any]):
        self.init_params = params
        self.sandbox_called: Optional[bool] = None
        self.create_order_calls: List[tuple] = []
        self.cancel_calls: List[tuple] = []
        self.fetch_order_calls: List[tuple] = []
        self.fetch_open_orders_calls: List[Optional[str]] = []
        self.fetch_balance_calls = 0
        self.fetch_ticker_calls: List[str] = []

        # Canned responses (override per-test)
        self.create_order_response: Dict[str, Any] = {
            "id": "order-1",
            "symbol": "ETH/USDT",
            "side": "buy",
            "type": "market",
            "status": "closed",
            "filled": 0.05,
            "amount": 0.05,
            "average": 2000.0,
            "cost": 100.0,
            "fee": {"cost": 0.5, "currency": "USDT"},
            "datetime": "2024-01-01T00:00:00.000Z",
        }
        self.balance_response: Dict[str, Any] = {
            "free": {"USDT": 1000.0, "ETH": 0.5},
            "used": {"USDT": 50.0, "ETH": 0.0},
            "total": {"USDT": 1050.0, "ETH": 0.5},
        }
        self.ticker_response: Dict[str, Any] = {
            "bid": 100.0,
            "ask": 100.2,
            "last": 100.1,
            "baseVolume": 12345.6,
        }

    def set_sandbox_mode(self, flag: bool) -> None:
        self.sandbox_called = flag

    def create_order(
        self,
        symbol: str,
        otype: str,
        side: str,
        amount: float,
        price: Optional[float],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.create_order_calls.append((symbol, otype, side, amount, price, dict(params)))
        return dict(self.create_order_response)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        self.cancel_calls.append((order_id, symbol))
        return {"id": order_id, "symbol": symbol, "status": "canceled"}

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        self.fetch_order_calls.append((order_id, symbol))
        return {
            "id": order_id,
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "status": "open",
            "amount": 1.0,
            "price": 99.0,
        }

    def fetch_open_orders(self, symbol: Optional[str]) -> List[Dict[str, Any]]:
        self.fetch_open_orders_calls.append(symbol)
        return [
            {
                "id": "open-1",
                "symbol": symbol or "ETH/USDT",
                "side": "buy",
                "type": "limit",
                "status": "open",
                "amount": 0.1,
                "price": 95.0,
            }
        ]

    def fetch_balance(self) -> Dict[str, Any]:
        self.fetch_balance_calls += 1
        return dict(self.balance_response)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        self.fetch_ticker_calls.append(symbol)
        return dict(self.ticker_response)


class _FakeCcxtModule:
    """Stand-in for `import ccxt`. Exposes a `coinbase` callable."""

    def __init__(self, *, expose_advanced: bool = False, expose_legacy: bool = True):
        self._created: List[_FakeClient] = []
        if expose_legacy:
            def coinbase(params: Dict[str, Any]) -> _FakeClient:
                client = _FakeClient(params)
                self._created.append(client)
                return client

            self.coinbase = coinbase  # type: ignore[assignment]
        if expose_advanced:
            def coinbaseadvanced(params: Dict[str, Any]) -> _FakeClient:
                client = _FakeClient(params)
                self._created.append(client)
                return client

            self.coinbaseadvanced = coinbaseadvanced  # type: ignore[assignment]

    def last_client(self) -> _FakeClient:
        return self._created[-1]


def _make_exchange(
    *,
    sandbox: bool = True,
    api_key: str = "k",
    api_secret: str = "s",
    passphrase: Optional[str] = None,
    expose_advanced: bool = False,
    expose_legacy: bool = True,
) -> tuple[CoinbaseExchange, _FakeCcxtModule]:
    fake = _FakeCcxtModule(expose_advanced=expose_advanced, expose_legacy=expose_legacy)
    ex = CoinbaseExchange(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        sandbox=sandbox,
        ccxt_module=fake,
    )
    return ex, fake


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class SchemaTests(unittest.TestCase):
    def test_order_result_round_trip_and_rejects_extras(self) -> None:
        payload = {
            "order_id": "abc",
            "symbol": "ETH/USDT",
            "side": "buy",
            "type": "market",
            "quote_size_usd": 100.0,
            "base_size": None,
            "limit_price": None,
            "status": "filled",
            "filled_base": 0.05,
            "filled_quote_usd": 100.0,
            "avg_fill_price": 2000.0,
            "fee_usd": 0.5,
            "created_at_utc": "2024-01-01T00:00:00+00:00",
            "raw_payload": {"foo": "bar"},
        }
        order = OrderResult(**payload)
        self.assertEqual(order.model_dump()["order_id"], "abc")
        with self.assertRaises(ValidationError):
            OrderResult(**{**payload, "extra_field": 1})

    def test_balance_round_trip_and_rejects_extras(self) -> None:
        b = Balance(currency="USDT", free=10.0, locked=1.0, total=11.0)
        self.assertEqual(b.total, 11.0)
        with self.assertRaises(ValidationError):
            Balance(currency="X", free=1.0, locked=0.0, total=1.0, foo="bar")  # type: ignore[call-arg]

    def test_ticker_round_trip_and_rejects_extras(self) -> None:
        t = Ticker(
            symbol="ETH/USDT",
            bid=100.0,
            ask=101.0,
            last=100.5,
            volume_24h_base=12.0,
            as_of_utc="2024-01-01T00:00:00+00:00",
        )
        dumped = t.model_dump()
        self.assertIn("mid", dumped)
        self.assertIn("spread_bps", dumped)
        with self.assertRaises(ValidationError):
            Ticker(  # type: ignore[call-arg]
                symbol="X",
                bid=1.0,
                ask=2.0,
                last=1.5,
                volume_24h_base=0.0,
                as_of_utc="2024-01-01T00:00:00+00:00",
                extra=True,
            )


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class InitTests(unittest.TestCase):
    def test_init_reads_env_vars_when_args_omitted(self) -> None:
        env = {
            "COINBASE_API_KEY": "env-key",
            "COINBASE_API_SECRET": "env-secret",
            "COINBASE_API_PASSPHRASE": "env-pass",
            "COINBASE_USE_SANDBOX": "true",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            fake = _FakeCcxtModule()
            ex = CoinbaseExchange(ccxt_module=fake)
            self.assertTrue(ex.is_sandbox())
            client = fake.last_client()
            self.assertEqual(client.init_params["apiKey"], "env-key")
            self.assertEqual(client.init_params["secret"], "env-secret")
            self.assertEqual(client.init_params["password"], "env-pass")

    def test_init_defaults_to_sandbox_for_safety(self) -> None:
        # When sandbox is not provided AND env var is unset, default True.
        env_no_sandbox = {k: v for k, v in os.environ.items() if k != "COINBASE_USE_SANDBOX"}
        with mock.patch.dict(os.environ, env_no_sandbox, clear=True):
            fake = _FakeCcxtModule()
            ex = CoinbaseExchange(api_key="k", api_secret="s", ccxt_module=fake)
            self.assertTrue(ex.is_sandbox())
            self.assertEqual(fake.last_client().sandbox_called, True)

    def test_init_prefers_coinbase_id_when_both_present(self) -> None:
        fake = _FakeCcxtModule(expose_legacy=True, expose_advanced=True)
        ex = CoinbaseExchange(api_key="k", api_secret="s", sandbox=True, ccxt_module=fake)
        # coinbase is the canonical id since ccxt 4.x
        self.assertEqual(ex._ccxt_exchange_id, "coinbase")


# ---------------------------------------------------------------------------
# Order placement tests
# ---------------------------------------------------------------------------


class OrderPlacementTests(unittest.TestCase):
    def test_place_market_order_with_quote_size_calls_ccxt_correctly(self) -> None:
        ex, fake = _make_exchange()
        result = ex.place_market_order("ETH-USD", "buy", quote_size_usd=100.0)
        client = fake.last_client()
        self.assertEqual(len(client.create_order_calls), 1)
        symbol, otype, side, amount, price, params = client.create_order_calls[0]
        self.assertEqual(symbol, "ETH/USD")  # normalized
        self.assertEqual(otype, "market")
        self.assertEqual(side, "buy")
        self.assertEqual(amount, 100.0)
        self.assertIsNone(price)
        self.assertEqual(params.get("cost"), 100.0)
        self.assertEqual(params.get("createMarketBuyOrderRequiresPrice"), False)
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.quote_size_usd, 100.0)

    def test_place_market_order_rejects_both_size_args(self) -> None:
        ex, _ = _make_exchange()
        with self.assertRaises(ValueError):
            ex.place_market_order("ETH/USDT", "buy", quote_size_usd=100.0, base_size=0.05)

    def test_place_market_order_rejects_neither_size_arg(self) -> None:
        ex, _ = _make_exchange()
        with self.assertRaises(ValueError):
            ex.place_market_order("ETH/USDT", "buy")

    def test_place_limit_order_calls_ccxt_correctly(self) -> None:
        ex, fake = _make_exchange()
        result = ex.place_limit_order(
            "ETH/USDT", "sell", base_size=0.1, limit_price=2100.0, time_in_force="GTC"
        )
        client = fake.last_client()
        symbol, otype, side, amount, price, params = client.create_order_calls[0]
        self.assertEqual(symbol, "ETH/USDT")
        self.assertEqual(otype, "limit")
        self.assertEqual(side, "sell")
        self.assertEqual(amount, 0.1)
        self.assertEqual(price, 2100.0)
        self.assertEqual(params.get("timeInForce"), "GTC")
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.limit_price, 2100.0)
        self.assertEqual(result.base_size, 0.1)


# ---------------------------------------------------------------------------
# Account / market data tests
# ---------------------------------------------------------------------------


class AccountAndMarketDataTests(unittest.TestCase):
    def test_get_balances_normalizes_response(self) -> None:
        ex, _ = _make_exchange()
        balances = ex.get_balances()
        as_dict = {b.currency: b for b in balances}
        self.assertIn("USDT", as_dict)
        self.assertIn("ETH", as_dict)
        self.assertEqual(as_dict["USDT"].free, 1000.0)
        self.assertEqual(as_dict["USDT"].locked, 50.0)
        self.assertEqual(as_dict["USDT"].total, 1050.0)
        self.assertEqual(as_dict["ETH"].free, 0.5)
        self.assertEqual(as_dict["ETH"].locked, 0.0)
        self.assertEqual(as_dict["ETH"].total, 0.5)

    def test_get_ticker_computes_mid_and_spread_bps(self) -> None:
        ex, _ = _make_exchange()

        # get_ticker now bypasses ccxt and hits Coinbase's public products
        # endpoint directly via requests. Stub requests.get accordingly.
        from unittest import mock

        fake_response = mock.MagicMock(
            status_code=200,
            json=lambda: {
                "best_bid": "100.0",
                "best_ask": "100.2",
                "price": "100.1",
                "volume_24h": "5000.0",
            },
        )
        with mock.patch(
            "exchanges.coinbase.requests.get",
            return_value=fake_response,
        ) as patched_get:
            ticker = ex.get_ticker("ETH-USD")

        self.assertEqual(ticker.symbol, "ETH/USD")
        # mid = 100.1; spread = 0.2 / 100.1 * 10000 ≈ 19.98 bps
        self.assertAlmostEqual(ticker.mid, 100.1, places=4)
        self.assertAlmostEqual(ticker.spread_bps, (0.2 / 100.1) * 10_000.0, places=4)
        self.assertEqual(ticker.volume_24h_base, 5000.0)
        # Confirm we called the public products endpoint with the dash-style id.
        called_url = patched_get.call_args.args[0]
        self.assertIn("/api/v3/brokerage/market/products/ETH-USD", called_url)

    def test_fetch_recent_candles_returns_oldest_first(self) -> None:
        ex, _ = _make_exchange()
        from unittest import mock

        # Coinbase returns newest-first; we expect oldest-first after sort.
        fake_response = mock.MagicMock(
            status_code=200,
            json=lambda: {
                "candles": [
                    {
                        "start": "1700000120",
                        "open": "102",
                        "high": "103",
                        "low": "101",
                        "close": "102.5",
                        "volume": "10",
                    },
                    {
                        "start": "1700000060",
                        "open": "101",
                        "high": "102",
                        "low": "100.5",
                        "close": "101.5",
                        "volume": "20",
                    },
                    {
                        "start": "1700000000",
                        "open": "100",
                        "high": "101",
                        "low": "99",
                        "close": "100.5",
                        "volume": "30",
                    },
                ]
            },
        )
        with mock.patch(
            "exchanges.coinbase.requests.get",
            return_value=fake_response,
        ) as patched_get:
            rows = ex.fetch_recent_candles("ETH/USD", granularity="ONE_MINUTE", limit=3)

        self.assertEqual(len(rows), 3)
        # Oldest first.
        self.assertEqual(rows[0]["open"], 100.0)
        self.assertEqual(rows[1]["open"], 101.0)
        self.assertEqual(rows[2]["open"], 102.0)
        # ISO timestamp set.
        self.assertTrue(rows[0]["timestamp"].endswith("+00:00"))
        # _unix scratch field stripped.
        self.assertNotIn("_unix", rows[0])
        # Endpoint hit with dash-style id and granularity param.
        called_url = patched_get.call_args.args[0]
        self.assertIn("/api/v3/brokerage/market/products/ETH-USD/candles", called_url)
        called_params = patched_get.call_args.kwargs.get("params", {})
        self.assertEqual(called_params.get("granularity"), "ONE_MINUTE")

    def test_get_ticker_404_emits_clear_error(self) -> None:
        ex, _ = _make_exchange()
        from unittest import mock

        fake_response = mock.MagicMock(status_code=404, text="Not Found")
        with mock.patch(
            "exchanges.coinbase.requests.get",
            return_value=fake_response,
        ):
            with self.assertRaises(ExchangeError) as cm:
                ex.get_ticker("XYZ/USD")
        self.assertIn("not listed on Coinbase", str(cm.exception))


# ---------------------------------------------------------------------------
# Kill-switch + error preservation tests
# ---------------------------------------------------------------------------


class SafetyTests(unittest.TestCase):
    def test_kill_switch_file_blocks_live_orders(self) -> None:
        ex, _ = _make_exchange(sandbox=False)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"halt")
            kill_path = tmp.name
        try:
            with mock.patch.dict(os.environ, {"KILL_SWITCH_FILE": kill_path}):
                with self.assertRaises(ExchangeError) as ctx:
                    ex.place_market_order("ETH/USDT", "buy", quote_size_usd=10.0)
                self.assertIn("Kill switch", str(ctx.exception))
                with self.assertRaises(ExchangeError):
                    ex.place_limit_order("ETH/USDT", "buy", base_size=0.1, limit_price=1.0)
        finally:
            os.unlink(kill_path)

    def test_kill_switch_does_not_block_sandbox_orders(self) -> None:
        ex, fake = _make_exchange(sandbox=True)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"halt")
            kill_path = tmp.name
        try:
            with mock.patch.dict(os.environ, {"KILL_SWITCH_FILE": kill_path}):
                result = ex.place_market_order("ETH/USDT", "buy", quote_size_usd=10.0)
                self.assertIsInstance(result, OrderResult)
                self.assertEqual(len(fake.last_client().create_order_calls), 1)
        finally:
            os.unlink(kill_path)

    def test_exchange_error_preserves_cause(self) -> None:
        ex, fake = _make_exchange()
        boom = RuntimeError("ccxt blew up")

        def raise_it(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise boom

        fake.last_client().create_order = raise_it  # type: ignore[assignment]
        with self.assertRaises(ExchangeError) as ctx:
            ex.place_market_order("ETH/USDT", "buy", quote_size_usd=10.0)
        self.assertIs(ctx.exception.__cause__, boom)


if __name__ == "__main__":
    unittest.main()
