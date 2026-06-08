"""Tests for the ``--hyperliquid-symbols`` CLI wiring in live_supervisor.

Mirrors the polymarket coverage already in ``test_live_supervisor.py``:
  * argparse accepts the flag,
  * SupervisorConfig accepts a HyperliquidTradeable in ``tradeables``,
  * the union (symbols + polymarket + hyperliquid) is the source-set check,
  * stop-shipping when all three are empty.

The full Supervisor.run_once() flow with a real HyperliquidTradeable is
covered indirectly via the protocol-conformance tests in
``test_hyperliquid_tradeable.py`` and the polymarket tradeable wiring
tests; here we focus on the CLI surface that operators actually touch.

Run:
  env PYTHONPATH=src ./.venv/bin/python -m unittest \\
      tests.prediction_market_scanner.test_hyperliquid_supervisor_wiring -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

# Mirror the existing fee-honesty test harness: make sure both src/ and
# the repo root are importable so live_supervisor's flat-style imports
# resolve regardless of PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class HyperliquidSupervisorWiringTest(unittest.TestCase):
    def test_parse_args_accepts_hyperliquid_symbols_flag(self) -> None:
        from live_supervisor import _parse_args

        args = _parse_args(["--hyperliquid-symbols", "ETH,BTC"])
        # Default symbols/polymarket remain empty; the new flag accepted.
        self.assertEqual(args.symbols, "")
        self.assertEqual(args.polymarket_markets, "")
        self.assertEqual(args.hyperliquid_symbols, "ETH,BTC")

    def test_parse_args_default_hyperliquid_symbols_is_empty(self) -> None:
        from live_supervisor import _parse_args

        args = _parse_args(["--symbols", "BTC/USD"])
        # Back-compat: when the operator doesn't pass the new flag, the
        # field still exists and defaults to "".
        self.assertEqual(args.hyperliquid_symbols, "")

    def test_supervisor_config_accepts_hyperliquid_tradeable(self) -> None:
        # SupervisorConfig stores any object in ``tradeables``; the
        # HyperliquidTradeable instance must satisfy the Tradeable Protocol
        # so the supervisor loop can iterate without venue branching.
        from exchanges.adapters.hyperliquid_tradeable import HyperliquidTradeable
        from live_supervisor import SupervisorConfig
        from protocols import Tradeable

        adapter = HyperliquidTradeable(_FakeClient(), "ETH")
        self.assertIsInstance(adapter, Tradeable)

        cfg = SupervisorConfig(
            symbols=[],
            tradeables=[adapter],
            tick_interval_s=0.0,
            bankroll_usd=10_000.0,
            mode="paper",
            shakedown_min_days=14,
            shakedown_state_path=Path("/tmp/sk_hl_test.json"),
            risk_pct_per_trade=0.005,
            min_confidence_to_trade=0.6,
        )
        self.assertEqual(len(cfg.tradeables), 1)
        self.assertEqual(cfg.tradeables[0].symbol, "ETH")

    def test_main_rejects_empty_symbol_and_market_lists(self) -> None:
        from live_supervisor import main

        rc = main(["--mode", "paper"])
        # Same exit code as the pre-existing Polymarket-aware check: 2.
        self.assertEqual(rc, 2)

    def test_main_hyperliquid_only_invocation_does_not_error_on_arg_parse(self) -> None:
        # Verify the union check accepts hyperliquid-only (no coinbase
        # symbols, no polymarket ids). We stop short of running the full
        # supervisor loop (would touch Redis + heavy ML stack); we just
        # need ``main`` to get past the validation-and-tradeable-build
        # block, which fails closed on missing optional collaborators.
        from live_supervisor import main

        # Patch out everything past the SupervisorConfig build so the test
        # is hermetic. The substituted Supervisor's run_once returns [] so
        # main exits cleanly with rc=0.
        with mock.patch("live_supervisor.CoinbaseExchange") as exch, mock.patch(
            "live_supervisor.PositionStore"
        ) as ps, mock.patch(
            "live_supervisor.CircuitBreakerSet"
        ) as cb, mock.patch(
            "live_supervisor.Notifier"
        ) as notif, mock.patch(
            "live_supervisor.Supervisor"
        ) as sup_cls, mock.patch(
            "live_supervisor._setup_run_dir"
        ) as setup_run_dir:
            sup_instance = sup_cls.return_value
            sup_instance.run_once.return_value = []
            setup_run_dir.return_value = None

            rc = main(
                [
                    "--hyperliquid-symbols",
                    "ETH",
                    "--mode",
                    "paper",
                    "--once",
                ]
            )
            self.assertEqual(rc, 0)
            # Supervisor must be constructed exactly once, with the
            # HyperliquidTradeable wired into its config.tradeables list.
            sup_cls.assert_called_once()
            cfg = sup_cls.call_args.kwargs["config"]
            self.assertEqual(cfg.symbols, [])
            self.assertEqual(len(cfg.tradeables), 1)
            self.assertEqual(cfg.tradeables[0].symbol, "ETH")

    def test_duplicate_hyperliquid_symbols_are_deduped(self) -> None:
        # Mirror the Polymarket dedup behaviour (same flock + state-key
        # collision concern: two identical Hyperliquid tradeables would
        # otherwise both record under the same symbol-keyed state).
        from live_supervisor import main

        with mock.patch("live_supervisor.CoinbaseExchange"), mock.patch(
            "live_supervisor.PositionStore"
        ), mock.patch("live_supervisor.CircuitBreakerSet"), mock.patch(
            "live_supervisor.Notifier"
        ), mock.patch(
            "live_supervisor.Supervisor"
        ) as sup_cls, mock.patch(
            "live_supervisor._setup_run_dir"
        ) as setup_run_dir:
            sup_instance = sup_cls.return_value
            sup_instance.run_once.return_value = []
            setup_run_dir.return_value = None

            rc = main(
                [
                    "--hyperliquid-symbols",
                    "ETH,ETH,BTC",
                    "--mode",
                    "paper",
                    "--once",
                ]
            )
            self.assertEqual(rc, 0)
            cfg = sup_cls.call_args.kwargs["config"]
            self.assertEqual(
                [t.symbol for t in cfg.tradeables],
                ["ETH", "BTC"],
            )

    def test_mixed_invocation_includes_all_three_kinds(self) -> None:
        # --symbols + --polymarket-markets + --hyperliquid-symbols all
        # contribute; nothing silently drops.
        from live_supervisor import main

        with mock.patch("live_supervisor.CoinbaseExchange"), mock.patch(
            "live_supervisor.PositionStore"
        ), mock.patch("live_supervisor.CircuitBreakerSet"), mock.patch(
            "live_supervisor.Notifier"
        ), mock.patch(
            "live_supervisor.Supervisor"
        ) as sup_cls, mock.patch(
            "live_supervisor._setup_run_dir"
        ) as setup_run_dir, mock.patch(
            "fetcher.fetch_active_markets", return_value=[]
        ):
            sup_instance = sup_cls.return_value
            sup_instance.run_once.return_value = []
            setup_run_dir.return_value = None

            rc = main(
                [
                    "--symbols",
                    "BTC/USD",
                    "--polymarket-markets",
                    "mkt-1",
                    "--hyperliquid-symbols",
                    "ETH",
                    "--mode",
                    "paper",
                    "--once",
                ]
            )
            self.assertEqual(rc, 0)
            cfg = sup_cls.call_args.kwargs["config"]
            self.assertEqual(cfg.symbols, ["BTC/USD"])
            symbols = [t.symbol for t in cfg.tradeables]
            # PolymarketTradeable surfaces "polymarket:<mid>" via its
            # symbol property; HyperliquidTradeable surfaces the raw
            # perp symbol.
            self.assertEqual(len(symbols), 2)
            self.assertIn("ETH", symbols)
            self.assertTrue(any(s.startswith("polymarket:") for s in symbols))


class _FakeClient:
    """Minimal stub matching what HyperliquidTradeable touches at
    construction time. The real client is never reached because the
    Tradeable-conformance check only inspects attributes."""

    def get_ticker(self, symbol: str):  # pragma: no cover - not exercised
        return None

    def get_balances(self):  # pragma: no cover - not exercised
        return []

    def get_open_orders(self, symbol):  # pragma: no cover - not exercised
        return []

    def get_open_positions(self):  # pragma: no cover - not exercised
        return []


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
