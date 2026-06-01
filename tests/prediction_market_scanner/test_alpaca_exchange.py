"""Tests for src/exchanges/alpaca.py — Phase P3 stocks adapter (Commit 1).

Hermetic: ``requests.get`` / ``requests.post`` / ``requests.delete`` are
patched at the module level so no real HTTPS call ever leaves the box.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest import mock

from exchanges.alpaca import (
    AlpacaAccount,
    AlpacaAsset,
    AlpacaCalendarDay,
    AlpacaClock,
    AlpacaError,
    AlpacaExchange,
    Position,
    is_trading_enabled,
)
from exchanges.coinbase import OrderResult, Ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(
    json_data: Any = None,
    status_code: int = 200,
    text: str = "",
    raise_on_json: Optional[Exception] = None,
) -> mock.MagicMock:
    """Build a mocked ``requests.Response``-like object."""

    def _json() -> Any:
        if raise_on_json is not None:
            raise raise_on_json
        return json_data

    fake = mock.MagicMock()
    fake.status_code = status_code
    fake.text = text
    fake.json = _json
    return fake


def _make_exchange(*, paper: bool = True) -> AlpacaExchange:
    return AlpacaExchange(
        api_key="test-key",
        api_secret="test-secret",
        paper=paper,
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class AlpacaExchangeInitTests(unittest.TestCase):
    def test_paper_default_routes_to_paper_base_url(self) -> None:
        ex = AlpacaExchange(api_key="k", api_secret="s")
        self.assertTrue(ex.is_paper())
        self.assertIn("paper-api.alpaca.markets", ex.base_url)

    def test_live_routes_to_live_base_url(self) -> None:
        ex = AlpacaExchange(api_key="k", api_secret="s", paper=False)
        self.assertFalse(ex.is_paper())
        self.assertEqual(ex.base_url.rstrip("/"), "https://api.alpaca.markets/v2")

    def test_base_url_override_wins(self) -> None:
        ex = AlpacaExchange(
            api_key="k",
            api_secret="s",
            base_url="https://custom.alpaca.example/v2/",
        )
        self.assertEqual(ex.base_url, "https://custom.alpaca.example/v2")

    def test_data_base_url_default(self) -> None:
        ex = AlpacaExchange(api_key="k", api_secret="s")
        self.assertEqual(ex.data_base_url, "https://data.alpaca.markets/v2")


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------


class AlpacaExchangeReadTests(unittest.TestCase):
    def test_get_account_parses_response(self) -> None:
        ex = _make_exchange()
        payload = {
            "account_number": "ACCT-123",
            "equity": "100000.50",
            "cash": "25000.00",
            "buying_power": "50000.00",
            "portfolio_value": "100000.50",
            "pattern_day_trader": False,
            "multiplier": "2",
            "status": "ACTIVE",
            "currency": "USD",
        }
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ) as patched:
            account = ex.get_account()

        self.assertIsInstance(account, AlpacaAccount)
        self.assertEqual(account.account_number, "ACCT-123")
        self.assertAlmostEqual(account.equity, 100000.50)
        self.assertAlmostEqual(account.cash, 25000.00)
        self.assertAlmostEqual(account.buying_power, 50000.00)
        self.assertAlmostEqual(account.margin_multiplier, 2.0)
        self.assertEqual(account.currency, "USD")
        # Confirm headers + URL
        called_url = patched.call_args.args[0]
        self.assertIn("/v2/account", called_url)
        called_headers = patched.call_args.kwargs.get("headers", {})
        self.assertEqual(called_headers.get("APCA-API-KEY-ID"), "test-key")
        self.assertEqual(called_headers.get("APCA-API-SECRET-KEY"), "test-secret")

    def test_get_balances_returns_usd_cash(self) -> None:
        ex = _make_exchange()
        payload = {
            "account_number": "X",
            "equity": "100",
            "cash": "42.5",
            "buying_power": "84.0",
            "currency": "USD",
        }
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ):
            balances = ex.get_balances()
        self.assertEqual(balances, {"USD": 42.5})

    def test_get_clock_parses_response(self) -> None:
        ex = _make_exchange()
        payload = {
            "is_open": True,
            "next_open": "2026-05-09T13:30:00Z",
            "next_close": "2026-05-08T20:00:00Z",
            "timestamp": "2026-05-08T15:00:00Z",
        }
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ):
            clock = ex.get_clock()
        self.assertIsInstance(clock, AlpacaClock)
        self.assertTrue(clock.is_open)
        self.assertEqual(clock.next_open, "2026-05-09T13:30:00Z")
        self.assertEqual(clock.next_close, "2026-05-08T20:00:00Z")

    def test_get_calendar_parses_list(self) -> None:
        ex = _make_exchange()
        payload = [
            {"date": "2026-05-08", "open": "09:30", "close": "16:00"},
            {"date": "2026-05-09", "open": "09:30", "close": "16:00"},
        ]
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ) as patched:
            cal = ex.get_calendar(start="2026-05-08", end="2026-05-09")
        self.assertEqual(len(cal), 2)
        self.assertIsInstance(cal[0], AlpacaCalendarDay)
        self.assertEqual(cal[0].date, "2026-05-08")
        called_params = patched.call_args.kwargs.get("params", {})
        self.assertEqual(called_params.get("start"), "2026-05-08")
        self.assertEqual(called_params.get("end"), "2026-05-09")

    def test_get_asset_returns_metadata(self) -> None:
        ex = _make_exchange()
        payload = {
            "id": "asset-1",
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "class": "us_equity",
            "exchange": "NASDAQ",
            "status": "active",
            "tradable": True,
            "marginable": True,
            "shortable": True,
            "easy_to_borrow": True,
            "fractionable": True,
            "min_order_size": "0.0001",
        }
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ) as patched:
            asset = ex.get_asset("aapl")
        self.assertIsInstance(asset, AlpacaAsset)
        self.assertEqual(asset.symbol, "AAPL")
        self.assertTrue(asset.tradable)
        self.assertTrue(asset.shortable)
        self.assertTrue(asset.fractionable)
        self.assertAlmostEqual(asset.min_order_size, 0.0001)
        # URL upper-cased the symbol.
        called_url = patched.call_args.args[0]
        self.assertIn("/assets/AAPL", called_url)

    def test_get_asset_invalid_symbol_raises(self) -> None:
        ex = _make_exchange()
        with self.assertRaises(AlpacaError):
            ex.get_asset("")  # empty
        with self.assertRaises(AlpacaError):
            ex.get_asset(None)  # type: ignore[arg-type]

    def test_get_ticker_combines_quote_and_bar(self) -> None:
        ex = _make_exchange()
        quote_payload = {"quote": {"bp": "180.00", "ap": "180.05"}}
        bar_payload = {"bar": {"c": "180.02", "v": "1500"}}

        # Simulate two GETs — the data API uses /v2/stocks/.../quotes/latest
        # then /v2/stocks/.../bars/latest. Both go through requests.get.
        def _route_get(url: str, **kwargs: Any) -> Any:
            if "/quotes/latest" in url:
                return _resp(json_data=quote_payload)
            if "/bars/latest" in url:
                return _resp(json_data=bar_payload)
            return _resp(json_data={})

        with mock.patch(
            "exchanges.alpaca.requests.get",
            side_effect=_route_get,
        ):
            ticker = ex.get_ticker("AAPL")

        self.assertIsInstance(ticker, Ticker)
        self.assertEqual(ticker.symbol, "AAPL")
        self.assertAlmostEqual(ticker.bid, 180.00)
        self.assertAlmostEqual(ticker.ask, 180.05)
        self.assertAlmostEqual(ticker.last, 180.02)
        self.assertGreater(ticker.mid, 0.0)
        self.assertAlmostEqual(ticker.volume_24h_base, 1500.0)

    def test_get_ticker_falls_back_to_mid_if_bars_missing(self) -> None:
        ex = _make_exchange()

        def _route_get(url: str, **kwargs: Any) -> Any:
            if "/quotes/latest" in url:
                return _resp(json_data={"quote": {"bp": "10.0", "ap": "10.2"}})
            if "/bars/latest" in url:
                return _resp(status_code=403, text="forbidden")
            return _resp(json_data={})

        with mock.patch(
            "exchanges.alpaca.requests.get",
            side_effect=_route_get,
        ):
            ticker = ex.get_ticker("XYZ")

        # bars 403 -> last falls back to mid.
        self.assertAlmostEqual(ticker.bid, 10.0)
        self.assertAlmostEqual(ticker.ask, 10.2)
        self.assertAlmostEqual(ticker.last, 10.1)
        self.assertEqual(ticker.volume_24h_base, 0.0)

    def test_get_open_orders_parses_list(self) -> None:
        ex = _make_exchange()
        payload = [
            {
                "id": "ord-1",
                "symbol": "AAPL",
                "side": "buy",
                "type": "limit",
                "status": "new",
                "qty": "10",
                "limit_price": "175.00",
                "filled_qty": "0",
                "created_at": "2026-05-08T15:00:00Z",
            },
            {
                "id": "ord-2",
                "symbol": "MSFT",
                "side": "sell",
                "type": "market",
                "status": "accepted",
                "qty": "5",
                "filled_qty": "0",
                "created_at": "2026-05-08T15:01:00Z",
            },
        ]
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ) as patched:
            orders = ex.get_open_orders()
        self.assertEqual(len(orders), 2)
        self.assertIsInstance(orders[0], OrderResult)
        self.assertEqual(orders[0].order_id, "ord-1")
        self.assertEqual(orders[0].symbol, "AAPL")
        self.assertEqual(orders[0].side, "buy")
        self.assertEqual(orders[0].status, "open")
        self.assertEqual(orders[1].side, "sell")
        # Confirm status filter passed through.
        called_params = patched.call_args.kwargs.get("params", {})
        self.assertEqual(called_params.get("status"), "open")

    def test_get_position_returns_position(self) -> None:
        ex = _make_exchange()
        payload = {
            "symbol": "AAPL",
            "qty": "10",
            "side": "long",
            "avg_entry_price": "175.50",
            "market_value": "1800.20",
            "cost_basis": "1755.00",
            "unrealized_pl": "45.20",
            "current_price": "180.02",
        }
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(json_data=payload),
        ):
            pos = ex.get_position("AAPL")
        self.assertIsNotNone(pos)
        assert pos is not None
        self.assertIsInstance(pos, Position)
        self.assertEqual(pos.symbol, "AAPL")
        self.assertAlmostEqual(pos.qty, 10.0)
        self.assertEqual(pos.side, "long")
        self.assertAlmostEqual(pos.avg_entry_price, 175.50)

    def test_get_position_404_returns_none(self) -> None:
        ex = _make_exchange()
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(status_code=404, text="position does not exist"),
        ):
            pos = ex.get_position("AAPL")
        self.assertIsNone(pos)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class AlpacaExchangeErrorTests(unittest.TestCase):
    def test_4xx_raises_with_api_message(self) -> None:
        ex = _make_exchange()
        body = {"message": "forbidden: invalid API key"}
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(status_code=403, json_data=body),
        ):
            with self.assertRaises(AlpacaError) as cm:
                ex.get_account()
        self.assertIn("403", str(cm.exception))
        self.assertIn("forbidden", str(cm.exception))

    def test_network_error_wraps_into_alpaca_error(self) -> None:
        ex = _make_exchange()
        with mock.patch(
            "exchanges.alpaca.requests.get",
            side_effect=ConnectionError("DNS lookup failed"),
        ):
            with self.assertRaises(AlpacaError) as cm:
                ex.get_account()
        self.assertIn("DNS lookup failed", str(cm.exception))

    def test_non_json_response_wraps(self) -> None:
        ex = _make_exchange()
        with mock.patch(
            "exchanges.alpaca.requests.get",
            return_value=_resp(
                status_code=200,
                raise_on_json=ValueError("not json"),
            ),
        ):
            with self.assertRaises(AlpacaError) as cm:
                ex.get_account()
        self.assertIn("non-JSON", str(cm.exception))


# ---------------------------------------------------------------------------
# Trading flag gating
# ---------------------------------------------------------------------------


class AlpacaTradingFlagTests(unittest.TestCase):
    def test_is_trading_enabled_default_false(self) -> None:
        env = {"ALPACA_TRADING_ENABLED": ""}
        with mock.patch.dict("os.environ", env, clear=False):
            self.assertFalse(is_trading_enabled())

    def test_is_trading_enabled_truthy_values(self) -> None:
        for val in ("true", "TRUE", "1", "yes", "on"):
            with mock.patch.dict(
                "os.environ",
                {"ALPACA_TRADING_ENABLED": val},
                clear=False,
            ):
                self.assertTrue(is_trading_enabled(), f"failed for {val!r}")

    def test_place_market_order_raises_when_flag_unset(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": ""}, clear=False
        ):
            with self.assertRaises(NotImplementedError) as cm:
                ex.place_market_order("AAPL", "buy", base_size=10)
        self.assertIn("ALPACA_TRADING_ENABLED", str(cm.exception))

    def test_place_market_order_succeeds_when_flag_set(self) -> None:
        ex = _make_exchange()
        order_payload = {
            "id": "ord-99",
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "status": "accepted",
            "qty": "10",
            "filled_qty": "0",
            "created_at": "2026-05-08T15:00:00Z",
        }
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": "true"}, clear=False
        ), mock.patch(
            "exchanges.alpaca.requests.post",
            return_value=_resp(json_data=order_payload),
        ) as patched_post:
            result = ex.place_market_order("AAPL", "buy", base_size=10)
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.order_id, "ord-99")
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.side, "buy")
        self.assertEqual(result.type, "market")
        self.assertEqual(result.status, "open")
        # Confirm body shape.
        body = patched_post.call_args.kwargs.get("json", {})
        self.assertEqual(body.get("symbol"), "AAPL")
        self.assertEqual(body.get("side"), "buy")
        self.assertEqual(body.get("type"), "market")
        self.assertEqual(body.get("qty"), "10.0")

    def test_place_market_order_with_notional(self) -> None:
        ex = _make_exchange()
        order_payload = {
            "id": "ord-100",
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "status": "accepted",
            "notional": "1000",
            "filled_qty": "0",
            "created_at": "2026-05-08T15:00:00Z",
        }
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": "1"}, clear=False
        ), mock.patch(
            "exchanges.alpaca.requests.post",
            return_value=_resp(json_data=order_payload),
        ) as patched_post:
            result = ex.place_market_order("AAPL", "buy", quote_size_usd=1000.0)
        self.assertEqual(result.order_id, "ord-100")
        body = patched_post.call_args.kwargs.get("json", {})
        self.assertEqual(body.get("notional"), "1000.0")
        self.assertNotIn("qty", body)

    def test_place_limit_order_gated_and_succeeds(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": ""}, clear=False
        ):
            with self.assertRaises(NotImplementedError):
                ex.place_limit_order(
                    "AAPL", "sell", base_size=5, limit_price=200.0
                )
        order_payload = {
            "id": "ord-lim-1",
            "symbol": "AAPL",
            "side": "sell",
            "type": "limit",
            "status": "new",
            "qty": "5",
            "limit_price": "200.0",
            "filled_qty": "0",
            "created_at": "2026-05-08T15:00:00Z",
        }
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": "true"}, clear=False
        ), mock.patch(
            "exchanges.alpaca.requests.post",
            return_value=_resp(json_data=order_payload),
        ) as patched_post:
            result = ex.place_limit_order(
                "AAPL", "sell", base_size=5, limit_price=200.0
            )
        self.assertEqual(result.order_id, "ord-lim-1")
        self.assertEqual(result.type, "limit")
        body = patched_post.call_args.kwargs.get("json", {})
        self.assertEqual(body.get("limit_price"), "200.0")

    def test_cancel_order_gated_and_succeeds(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": ""}, clear=False
        ):
            with self.assertRaises(NotImplementedError):
                ex.cancel_order("ord-1")
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": "true"}, clear=False
        ), mock.patch(
            "exchanges.alpaca.requests.delete",
            return_value=_resp(status_code=204, text=""),
        ):
            result = ex.cancel_order("ord-1")
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.order_id, "ord-1")
        self.assertEqual(result.status, "cancelled")

    def test_place_market_order_validation_quote_and_base_both_unset(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(
            "os.environ", {"ALPACA_TRADING_ENABLED": "true"}, clear=False
        ):
            with self.assertRaises(ValueError):
                ex.place_market_order("AAPL", "buy")
            with self.assertRaises(ValueError):
                ex.place_market_order("AAPL", "buy", base_size=10, quote_size_usd=100)
            with self.assertRaises(ValueError):
                ex.place_market_order("AAPL", "wrong", base_size=10)  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
