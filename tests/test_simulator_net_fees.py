"""Regression tests for two simulator defects fixed together:

1. Per-trade ``trade_log['ret']``/``['pnl']`` (and this sim's ``report()``
   profit-factor / ``profitability.compute_profitability_metrics`` expectancy)
   are now NET of both legs' fees, not gross.
2. ``_normalize_signal`` speaks RAW (-1/0/+1) only; class-style 0/1/2 must be
   converted via ``class_to_raw`` — a class-style HOLD (1) no longer silently
   becomes a LONG (+1).

Run:
    ./.venv/bin/python -m unittest tests.test_simulator_net_fees
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trading.simulator import (  # noqa: E402
    Bar, PortfolioSimulator, SimulationConfig, class_to_raw,
)
from profitability import compute_profitability_metrics  # noqa: E402


def _one_long_roundtrip(cfg):
    """Enter a long at open=100, hit a +0.5% take-profit; return the exit row."""
    sim = PortfolioSimulator(cfg)
    bars = [
        Bar(open=100, high=100, low=100, close=100, timestamp="t0"),
        Bar(open=100, high=100, low=100, close=100, timestamp="t1"),      # entry @open
        Bar(open=100, high=100.6, low=99.9, close=100.5, timestamp="t2"),  # TP 100.5 hit
        Bar(open=100.5, high=100.5, low=100.5, close=100.5, timestamp="t3"),
    ]
    sim.step(bars[0], signal=class_to_raw(2))
    for b in bars[1:]:
        sim.step(b, signal=0)
    sim.finalize(bars[-1].close, bars[-1].timestamp)
    exits = [t for t in sim.trade_log if t["action"] == "exit"]
    return sim, exits[0]


class TestNetFeeAccounting(unittest.TestCase):
    def _cfg(self):
        return SimulationConfig.from_coinbase_fees(
            start_capital=10_000.0, use_atr_stops=False, tp_pct=0.005, sl_pct=0.05,
            use_market_depth=False, post_only_entries=False, slippage_pct=0.0,
            cooldown=0, allow_shorts=False, leverage=1.0)

    def test_winning_price_move_is_net_loss_after_fees(self):
        sim, ex = _one_long_roundtrip(self._cfg())
        # +50 bps gross price move, ~100 bps round-trip (taker in + maker-out TP).
        self.assertAlmostEqual(ex["gross_ret"], 0.005, places=6)
        self.assertLess(ex["ret"], 0.0)          # net is a LOSS
        self.assertAlmostEqual(ex["fees"], 100.0, places=2)  # $60 taker + $40 maker
        self.assertAlmostEqual(ex["pnl"], ex["gross_pnl"] - ex["fees"], places=6)
        self.assertFalse(ex["win"])              # counted as a loss, not a win

    def test_trade_log_expectancy_matches_net_equity(self):
        sim, ex = _one_long_roundtrip(self._cfg())
        m = compute_profitability_metrics(sim.report(), sim.equity_curve(), sim.trade_log)
        equity_net = sim.last_equity / sim.config.start_capital - 1.0
        # Single all-in trade => per-trade net return == total net equity return.
        self.assertAlmostEqual(m["expectancy"], equity_net, places=9)
        self.assertLess(m["profit_factor"], 1.0)  # PF now reflects the net loss


class TestSignalConvention(unittest.TestCase):
    def test_class_to_raw_mapping(self):
        self.assertEqual([class_to_raw(c) for c in (0, 1, 2)], [-1, 0, 1])
        with self.assertRaises(ValueError):
            class_to_raw(3)

    def test_normalize_rejects_class_style_two(self):
        sim = PortfolioSimulator(SimulationConfig())
        with self.assertRaises(ValueError):
            sim._normalize_signal(2)  # class-long must be converted, not guessed

    def test_normalize_passes_raw(self):
        sim = PortfolioSimulator(SimulationConfig())
        self.assertEqual([sim._normalize_signal(s) for s in (-1, 0, 1, None)], [-1, 0, 1, 0])

    def test_class_hold_no_longer_becomes_long(self):
        # A HOLD stream must leave the book flat (0 trades) — the old bug opened
        # a long on every hold bar.
        cfg = SimulationConfig(start_capital=10_000.0, allow_shorts=False)
        sim = PortfolioSimulator(cfg)
        for i in range(5):
            sim.step(Bar(open=100, high=101, low=99, close=100, timestamp=f"t{i}"),
                     signal=class_to_raw(1))  # class HOLD -> raw 0
        sim.finalize(100.0)
        self.assertEqual(sim.trades, 0)


if __name__ == "__main__":
    unittest.main()
