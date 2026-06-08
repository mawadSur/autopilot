"""Tests for the P3 Alpaca stocks adapter.

Covers:
  * Protocol conformance (``isinstance(adapter, Tradeable)``).
  * Static metadata (symbol namespacing, asset_class, tick_size, min_size).
  * Fee model — defaults to commission-free, overridable.
  * Paper-mode safety: the underlying connector defaults to paper, and
    every write path is gated behind ``ALPACA_TRADING_ENABLED`` so a
    fresh checkout cannot place a real order.
  * Ticker fetch over a mocked ``requests`` surface — NO real HTTP.
  * Order placement happy path (with flag set + HTTP mocked).
  * Risk attributes shape.
  * Supervisor registration — ``--alpaca-symbols`` CLI parsing, the
    symbols-OR-tradeables validator, and ``_dispatch_tick`` routing.

Hermetic: every external surface is patched (``requests.get`` /
``requests.post`` / ``requests.delete`` / ``os.environ``). No real
``paper-api.alpaca.markets`` call ever leaves the box.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

# Mirror the PYTHONPATH=src convention the prediction-market suite uses.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from exchanges.adapters import AlpacaTradeable  # noqa: E402
from exchanges.adapters.alpaca_tradeable import (  # noqa: E402
    AlpacaTradeable as DirectAlpacaTradeable,
)
from exchanges.alpaca import (  # noqa: E402
    AlpacaError,
    AlpacaExchange,
    is_trading_enabled,
)
from exchanges.coinbase import OrderResult, Ticker  # noqa: E402
from protocols import AssetClass, FeeModel, RiskAttributes, Tradeable  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(
    json_data: Any = None,
    status_code: int = 200,
    text: str = "",
) -> mock.MagicMock:
    """Build a mocked ``requests.Response``-like object."""

    fake = mock.MagicMock()
    fake.status_code = status_code
    fake.text = text
    fake.json.return_value = json_data
    return fake


def _make_exchange(*, paper: bool = True) -> AlpacaExchange:
    return AlpacaExchange(
        api_key="test-key",
        api_secret="test-secret",
        paper=paper,
    )


class _FakeAlpacaExchange:
    """Records call shapes; returns canned responses. Quacks like AlpacaExchange.

    Mirrors the stub used by tests/prediction_market_scanner/test_alpaca_tradeable.py
    so the two test trees stay in sync.
    """

    def __init__(self) -> None:
        self.get_ticker_calls: List[str] = []
        self.get_balances_calls = 0
        self.get_open_orders_calls = 0
        self.place_market_calls: List[Dict[str, Any]] = []
        self.place_limit_calls: List[Dict[str, Any]] = []
        self.cancel_order_calls: List[str] = []

        self.canned_ticker = Ticker(
            symbol="AAPL",
            bid=180.00,
            ask=180.05,
            last=180.02,
            volume_24h_base=1500.0,
            as_of_utc="2026-06-08T15:00:00+00:00",
        )
        self.canned_balances: Dict[str, float] = {"USD": 25_000.0}
        self.canned_open_orders: List[OrderResult] = []
        self.canned_market_order = OrderResult(
            order_id="ord-99",
            symbol="AAPL",
            side="buy",
            type="market",
            base_size=10.0,
            status="open",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc="2026-06-08T15:00:00+00:00",
            raw_payload={},
        )
        self.canned_limit_order = OrderResult(
            order_id="ord-lim-1",
            symbol="AAPL",
            side="sell",
            type="limit",
            base_size=5.0,
            limit_price=200.0,
            status="open",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc="2026-06-08T15:00:00+00:00",
            raw_payload={},
        )
        self.flag_enabled = False

    # ---- gating mirrors AlpacaExchange's behaviour ------------------

    def _check_flag(self) -> None:
        if not self.flag_enabled:
            raise NotImplementedError(
                "Alpaca trading is feature-flag-gated; set "
                "ALPACA_TRADING_ENABLED=true to enable order placement."
            )

    # ---- read paths -------------------------------------------------

    def get_ticker(self, symbol: str) -> Ticker:
        self.get_ticker_calls.append(symbol)
        return self.canned_ticker

    def get_balances(self) -> Dict[str, float]:
        self.get_balances_calls += 1
        return dict(self.canned_balances)

    def get_open_orders(self) -> List[OrderResult]:
        self.get_open_orders_calls += 1
        return list(self.canned_open_orders)

    # ---- write paths -----------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        self._check_flag()
        self.place_market_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quote_size_usd": quote_size_usd,
                "base_size": base_size,
            }
        )
        return self.canned_market_order

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        *,
        base_size: float,
        limit_price: float,
    ) -> OrderResult:
        self._check_flag()
        self.place_limit_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "base_size": base_size,
                "limit_price": limit_price,
            }
        )
        return self.canned_limit_order

    def cancel_order(self, order_id: str) -> OrderResult:
        self._check_flag()
        self.cancel_order_calls.append(order_id)
        return OrderResult(
            order_id=order_id,
            symbol="",
            side="buy",
            type="market",
            status="cancelled",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc="2026-06-08T15:00:00+00:00",
            raw_payload={},
        )


# ---------------------------------------------------------------------------
# Protocol conformance + metadata
# ---------------------------------------------------------------------------


class AlpacaTradeableProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeAlpacaExchange()
        self.adapter = AlpacaTradeable(self.fake, "aapl")  # type: ignore[arg-type]

    def test_runtime_isinstance_check(self) -> None:
        self.assertIsInstance(self.adapter, Tradeable)

    def test_init_export_alias_matches_direct_import(self) -> None:
        # The package-level export and the direct import must be the same class.
        self.assertIs(AlpacaTradeable, DirectAlpacaTradeable)

    def test_namespaced_symbol(self) -> None:
        self.assertEqual(self.adapter.symbol, "alpaca:AAPL")
        self.assertEqual(self.adapter.raw_symbol, "AAPL")

    def test_symbol_uppercased(self) -> None:
        adapter = AlpacaTradeable(self.fake, "msft")  # type: ignore[arg-type]
        self.assertEqual(adapter.raw_symbol, "MSFT")
        self.assertEqual(adapter.symbol, "alpaca:MSFT")

    def test_asset_class_is_spot_equity(self) -> None:
        self.assertEqual(self.adapter.asset_class, AssetClass.SPOT_EQUITY)
        self.assertEqual(AssetClass.SPOT_EQUITY.value, "spot_equity")

    def test_tick_size_default_one_cent(self) -> None:
        # Reg NMS Rule 612 — US equities above $1 trade in cent ticks.
        self.assertEqual(self.adapter.tick_size, 0.01)

    def test_min_size_default_whole_share(self) -> None:
        self.assertEqual(self.adapter.min_size, 1.0)

    def test_min_size_override_for_fractional(self) -> None:
        adapter = AlpacaTradeable(
            self.fake, "AAPL", min_size=0.0001  # type: ignore[arg-type]
        )
        self.assertEqual(adapter.min_size, 0.0001)

    def test_tick_size_override(self) -> None:
        adapter = AlpacaTradeable(
            self.fake, "BRK.A", tick_size=0.001  # type: ignore[arg-type]
        )
        self.assertEqual(adapter.tick_size, 0.001)

    def test_invalid_symbol_raises(self) -> None:
        with self.assertRaises(ValueError):
            AlpacaTradeable(self.fake, "")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AlpacaTradeable(self.fake, None)  # type: ignore[arg-type]

    def test_exchange_property_returns_underlying(self) -> None:
        self.assertIs(self.adapter.exchange, self.fake)


# ---------------------------------------------------------------------------
# Fee model
# ---------------------------------------------------------------------------


class AlpacaTradeableFeeModelTests(unittest.TestCase):
    def test_default_is_commission_free(self) -> None:
        # Alpaca is commission-free for retail US equities.
        adapter = AlpacaTradeable(_FakeAlpacaExchange(), "AAPL")  # type: ignore[arg-type]
        fm = adapter.fee_model
        self.assertIsInstance(fm, FeeModel)
        self.assertEqual(fm.maker, 0.0)
        self.assertEqual(fm.taker, 0.0)
        self.assertEqual(fm.settlement_fee_bps, 0)
        self.assertEqual(fm.gas_fee_usd, 0.0)

    def test_custom_fee_model_override(self) -> None:
        # Operators on a paid SIP / non-retail tier can override.
        custom = FeeModel(maker=0.0, taker=0.0005, settlement_fee_bps=3)
        adapter = AlpacaTradeable(
            _FakeAlpacaExchange(), "AAPL", fee_model=custom  # type: ignore[arg-type]
        )
        self.assertIs(adapter.fee_model, custom)
        self.assertEqual(adapter.fee_model.taker, 0.0005)
        self.assertEqual(adapter.fee_model.settlement_fee_bps, 3)


# ---------------------------------------------------------------------------
# Paper-mode safety + flag gating
# ---------------------------------------------------------------------------


class AlpacaPaperModeSafetyTests(unittest.TestCase):
    def test_exchange_defaults_to_paper_base_url(self) -> None:
        # Default constructor MUST route to the paper API.
        ex = AlpacaExchange(api_key="k", api_secret="s")
        self.assertTrue(ex.is_paper())
        self.assertIn("paper-api.alpaca.markets", ex.base_url)

    def test_explicit_live_routes_to_live_base_url(self) -> None:
        ex = AlpacaExchange(api_key="k", api_secret="s", paper=False)
        self.assertFalse(ex.is_paper())
        self.assertIn("api.alpaca.markets", ex.base_url)
        self.assertNotIn("paper-api", ex.base_url)

    def test_trading_disabled_by_default(self) -> None:
        # Ensure a fresh process with no env override cannot trade.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPACA_TRADING_ENABLED", None)
            self.assertFalse(is_trading_enabled())

    def test_trading_flag_truthy_strings(self) -> None:
        for val in ("true", "True", "1", "yes", "on"):
            with mock.patch.dict(os.environ, {"ALPACA_TRADING_ENABLED": val}):
                self.assertTrue(is_trading_enabled(), msg=f"value={val!r}")

    def test_trading_flag_falsy_strings(self) -> None:
        for val in ("false", "False", "0", "no", "off", ""):
            with mock.patch.dict(
                os.environ, {"ALPACA_TRADING_ENABLED": val}
            ):
                self.assertFalse(is_trading_enabled(), msg=f"value={val!r}")

    def test_place_market_refuses_when_flag_unset(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPACA_TRADING_ENABLED", None)
            with self.assertRaises(NotImplementedError) as cm:
                ex.place_market_order("AAPL", "buy", base_size=1.0)
            self.assertIn("ALPACA_TRADING_ENABLED", str(cm.exception))

    def test_place_limit_refuses_when_flag_unset(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPACA_TRADING_ENABLED", None)
            with self.assertRaises(NotImplementedError):
                ex.place_limit_order(
                    "AAPL", "buy", base_size=1.0, limit_price=100.0
                )

    def test_cancel_refuses_when_flag_unset(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPACA_TRADING_ENABLED", None)
            with self.assertRaises(NotImplementedError):
                ex.cancel_order("ord-1")

    def test_adapter_propagates_flag_error(self) -> None:
        # The adapter is a pass-through — the connector is the SOLE gate
        # so a single source-of-truth governs ALPACA_TRADING_ENABLED.
        fake = _FakeAlpacaExchange()
        fake.flag_enabled = False
        adapter = AlpacaTradeable(fake, "AAPL")  # type: ignore[arg-type]
        with self.assertRaises(NotImplementedError) as cm:
            adapter.place_market_order(side="buy", base_size=10.0)
        self.assertIn("ALPACA_TRADING_ENABLED", str(cm.exception))


# ---------------------------------------------------------------------------
# Ticker / read paths (HTTP mocked)
# ---------------------------------------------------------------------------


class AlpacaExchangeReadPathsTests(unittest.TestCase):
    def test_get_ticker_combines_quote_and_bar(self) -> None:
        ex = _make_exchange()
        quote_payload = {
            "quote": {"bp": 180.10, "ap": 180.15, "bs": 100, "as": 50}
        }
        bar_payload = {"bar": {"c": 180.12, "v": 1234}}

        def _fake_get(url: str, **_: Any) -> Any:
            if "quotes/latest" in url:
                return _resp(quote_payload)
            if "bars/latest" in url:
                return _resp(bar_payload)
            return _resp({}, status_code=404)

        with mock.patch(
            "exchanges.alpaca.requests.get", side_effect=_fake_get
        ) as get_mock:
            ticker = ex.get_ticker("AAPL")
        self.assertGreaterEqual(get_mock.call_count, 1)
        self.assertEqual(ticker.symbol, "AAPL")
        self.assertAlmostEqual(ticker.bid, 180.10)
        self.assertAlmostEqual(ticker.ask, 180.15)
        self.assertAlmostEqual(ticker.last, 180.12)
        self.assertAlmostEqual(ticker.volume_24h_base, 1234.0)
        # Confirm we hit the paper market-data API.
        urls = [c.args[0] for c in get_mock.call_args_list]
        self.assertTrue(any("data.alpaca.markets" in u for u in urls))

    def test_get_ticker_falls_back_to_mid_when_bar_404s(self) -> None:
        ex = _make_exchange()
        quote_payload = {"quote": {"bp": 100.00, "ap": 100.50}}

        def _fake_get(url: str, **_: Any) -> Any:
            if "quotes/latest" in url:
                return _resp(quote_payload)
            if "bars/latest" in url:
                return _resp({"message": "no bar"}, status_code=404)
            return _resp({}, status_code=404)

        with mock.patch(
            "exchanges.alpaca.requests.get", side_effect=_fake_get
        ):
            ticker = ex.get_ticker("TSLA")
        # Mid of bid/ask = 100.25.
        self.assertAlmostEqual(ticker.last, 100.25)

    def test_adapter_get_ticker_delegates_with_raw_symbol(self) -> None:
        fake = _FakeAlpacaExchange()
        adapter = AlpacaTradeable(fake, "aapl")  # type: ignore[arg-type]
        ticker = adapter.get_ticker()
        self.assertEqual(ticker.symbol, "AAPL")
        # The adapter MUST pass the bare ticker, NOT the namespaced form.
        self.assertEqual(fake.get_ticker_calls, ["AAPL"])

    def test_adapter_get_balances_delegates(self) -> None:
        fake = _FakeAlpacaExchange()
        adapter = AlpacaTradeable(fake, "AAPL")  # type: ignore[arg-type]
        self.assertEqual(adapter.get_balances(), {"USD": 25_000.0})
        self.assertEqual(fake.get_balances_calls, 1)

    def test_adapter_get_open_orders_delegates(self) -> None:
        fake = _FakeAlpacaExchange()
        adapter = AlpacaTradeable(fake, "AAPL")  # type: ignore[arg-type]
        self.assertEqual(adapter.get_open_orders(), [])
        self.assertEqual(fake.get_open_orders_calls, 1)


# ---------------------------------------------------------------------------
# Order placement happy path (HTTP mocked, flag set)
# ---------------------------------------------------------------------------


class AlpacaOrderPlacementTests(unittest.TestCase):
    def test_place_market_buy_with_flag_set(self) -> None:
        ex = _make_exchange()
        order_payload = {
            "id": "alp-1",
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "qty": "10",
            "status": "accepted",
            "filled_qty": "0",
            "filled_avg_price": None,
            "created_at": "2026-06-08T15:00:00+00:00",
        }
        with mock.patch.dict(
            os.environ, {"ALPACA_TRADING_ENABLED": "true"}
        ), mock.patch(
            "exchanges.alpaca.requests.post",
            return_value=_resp(order_payload),
        ) as post_mock:
            result = ex.place_market_order("AAPL", "buy", base_size=10.0)
        post_mock.assert_called_once()
        # The submission URL must hit the paper trading API.
        self.assertIn("paper-api.alpaca.markets", post_mock.call_args.args[0])
        self.assertEqual(result.order_id, "alp-1")
        self.assertEqual(result.status, "open")
        self.assertEqual(result.side, "buy")

    def test_place_market_validates_exactly_one_size(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(
            os.environ, {"ALPACA_TRADING_ENABLED": "true"}
        ):
            with self.assertRaises(ValueError):
                ex.place_market_order(
                    "AAPL", "buy", base_size=10.0, quote_size_usd=100.0
                )
            with self.assertRaises(ValueError):
                ex.place_market_order("AAPL", "buy")

    def test_place_market_validates_side(self) -> None:
        ex = _make_exchange()
        with mock.patch.dict(
            os.environ, {"ALPACA_TRADING_ENABLED": "true"}
        ):
            with self.assertRaises(ValueError):
                ex.place_market_order("AAPL", "BUY!", base_size=10.0)

    def test_cancel_order_happy_path(self) -> None:
        ex = _make_exchange()
        # Alpaca returns 204 No Content on a successful cancel.
        with mock.patch.dict(
            os.environ, {"ALPACA_TRADING_ENABLED": "true"}
        ), mock.patch(
            "exchanges.alpaca.requests.delete",
            return_value=_resp({}, status_code=204, text=""),
        ) as del_mock:
            result = ex.cancel_order("ord-1")
        del_mock.assert_called_once()
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.order_id, "ord-1")


# ---------------------------------------------------------------------------
# Risk attributes
# ---------------------------------------------------------------------------


class AlpacaTradeableRiskTests(unittest.TestCase):
    def test_basic_risk_shape(self) -> None:
        adapter = AlpacaTradeable(_FakeAlpacaExchange(), "AAPL")  # type: ignore[arg-type]
        ra = adapter.risk_attributes(side="buy", size_base=10, entry_price=180)
        self.assertIsInstance(ra, RiskAttributes)
        self.assertEqual(ra.notional_exposure_usd, 1800.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        # Cash-equity V1 — no leverage / liquidation.
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_short_side_same_shape(self) -> None:
        adapter = AlpacaTradeable(_FakeAlpacaExchange(), "AAPL")  # type: ignore[arg-type]
        ra = adapter.risk_attributes(
            side="sell", size_base=4.5, entry_price=200.0
        )
        self.assertAlmostEqual(ra.notional_exposure_usd, 900.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        # Margin estimation is a documented TODO.
        self.assertIsNone(ra.margin_used_usd)


# ---------------------------------------------------------------------------
# Supervisor registration — CLI + tradeables wiring
# ---------------------------------------------------------------------------


class SupervisorAlpacaWiringTests(unittest.TestCase):
    """The CLI flag, validation, and dispatch routing.

    Imports ``live_supervisor`` lazily so a broken import is reported with
    the test that depends on it, not at module load time.
    """

    def _import(self) -> Any:
        import live_supervisor  # noqa: WPS433 - intentional late import

        return live_supervisor

    def test_cli_accepts_alpaca_symbols_flag(self) -> None:
        ls = self._import()
        ns = ls._parse_args(["--alpaca-symbols", "AAPL,MSFT"])
        self.assertEqual(ns.alpaca_symbols, "AAPL,MSFT")

    def test_cli_alpaca_symbols_default_empty(self) -> None:
        ls = self._import()
        ns = ls._parse_args([])
        self.assertEqual(ns.alpaca_symbols, "")

    def test_supervisor_config_accepts_alpaca_tradeable(self) -> None:
        ls = self._import()
        adapter = AlpacaTradeable(
            _FakeAlpacaExchange(), "AAPL"  # type: ignore[arg-type]
        )
        with tempfile.TemporaryDirectory() as td:
            cfg = ls.SupervisorConfig(
                tradeables=[adapter],
                shakedown_state_path=Path(td) / "sk.json",
            )
            self.assertEqual(cfg.symbols, [])
            self.assertEqual(len(cfg.tradeables), 1)

    def test_supervisor_config_rejects_no_sources(self) -> None:
        from pydantic import ValidationError

        ls = self._import()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValidationError):
                ls.SupervisorConfig(
                    shakedown_state_path=Path(td) / "sk.json"
                )

    def test_main_fails_without_any_source(self) -> None:
        ls = self._import()
        with mock.patch("sys.stderr"):
            rc = ls.main([])
        self.assertEqual(rc, 2)

    def test_dispatch_routes_spot_equity_to_handler(self) -> None:
        # Construct a minimal supervisor and confirm a SPOT_EQUITY tradeable
        # never reaches ``self.exchange.get_ticker`` (which would crash on
        # an alpaca-namespaced symbol). Build everything with stubs.
        ls = self._import()

        # Use the same stubs as tests/prediction_market_scanner/.
        from tests.prediction_market_scanner.test_live_supervisor import (  # noqa: WPS433
            StubCircuitBreakers,
            StubExchange,
            StubNotifier,
            StubPositionStore,
        )

        store = StubPositionStore()
        fake_exchange = _FakeAlpacaExchange()
        adapter = AlpacaTradeable(fake_exchange, "AAPL")  # type: ignore[arg-type]
        stub_coinbase = StubExchange()
        with tempfile.TemporaryDirectory() as td:
            cfg = ls.SupervisorConfig(
                tradeables=[adapter],
                tick_interval_s=0.0,
                bankroll_usd=10_000.0,
                mode="paper",
                shakedown_state_path=Path(td) / "sk.json",
                shakedown_min_days=14,
                risk_pct_per_trade=0.005,
                min_confidence_to_trade=0.6,
            )
            sup = ls.Supervisor(
                config=cfg,
                exchange=stub_coinbase,
                position_store=store,
                circuit_breakers=StubCircuitBreakers(),
                notifier=StubNotifier(),
                model_predict_fn=lambda s, t: ("buy", 0.9),
                sleep_fn=lambda s: None,
            )
            ticks = sup.run_once()
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].symbol, "alpaca:AAPL")
        self.assertEqual(ticks[0].action_taken, "allowed")
        self.assertEqual(ticks[0].notes, "spot_equity")
        # The adapter's ticker MUST have been pulled.
        self.assertEqual(fake_exchange.get_ticker_calls, ["AAPL"])
        # The Coinbase stub must NEVER have been asked to quote an
        # alpaca-namespaced symbol — that would mean we mis-routed
        # through _tick_symbol.
        self.assertEqual(stub_coinbase.ticker_calls, [])

    def test_dispatch_routes_spot_equity_kill_switch_force_flat(self) -> None:
        ls = self._import()
        adapter = AlpacaTradeable(
            _FakeAlpacaExchange(), "AAPL"  # type: ignore[arg-type]
        )

        from tests.prediction_market_scanner.test_live_supervisor import (  # noqa: WPS433
            StubCircuitBreakers,
            StubExchange,
            StubNotifier,
            StubPositionStore,
        )

        store = StubPositionStore()
        with tempfile.TemporaryDirectory() as td:
            cfg = ls.SupervisorConfig(
                tradeables=[adapter],
                tick_interval_s=0.0,
                bankroll_usd=10_000.0,
                mode="paper",
                shakedown_state_path=Path(td) / "sk.json",
                shakedown_min_days=14,
                risk_pct_per_trade=0.005,
                min_confidence_to_trade=0.6,
            )
            sup = ls.Supervisor(
                config=cfg,
                exchange=StubExchange(),
                position_store=store,
                circuit_breakers=StubCircuitBreakers(kill_switch=True),
                notifier=StubNotifier(),
                model_predict_fn=lambda s, t: ("buy", 0.9),
                sleep_fn=lambda s: None,
            )
            ticks = sup.run_once()
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].action_taken, "force_flatted")
        self.assertIn("kill_switch", ticks[0].notes or "")

    def test_dispatch_spot_equity_ticker_error_is_handled(self) -> None:
        ls = self._import()

        class _BoomFake(_FakeAlpacaExchange):
            def get_ticker(self, symbol: str) -> Ticker:
                raise AlpacaError("simulated network failure")

        adapter = AlpacaTradeable(_BoomFake(), "AAPL")  # type: ignore[arg-type]

        from tests.prediction_market_scanner.test_live_supervisor import (  # noqa: WPS433
            StubCircuitBreakers,
            StubExchange,
            StubNotifier,
            StubPositionStore,
        )

        store = StubPositionStore()
        with tempfile.TemporaryDirectory() as td:
            cfg = ls.SupervisorConfig(
                tradeables=[adapter],
                tick_interval_s=0.0,
                bankroll_usd=10_000.0,
                mode="paper",
                shakedown_state_path=Path(td) / "sk.json",
                shakedown_min_days=14,
                risk_pct_per_trade=0.005,
                min_confidence_to_trade=0.6,
            )
            sup = ls.Supervisor(
                config=cfg,
                exchange=StubExchange(),
                position_store=store,
                circuit_breakers=StubCircuitBreakers(),
                notifier=StubNotifier(),
                model_predict_fn=lambda s, t: ("buy", 0.9),
                sleep_fn=lambda s: None,
            )
            ticks = sup.run_once()
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].action_taken, "errored")
        self.assertIn("alpaca_ticker", ticks[0].notes or "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
