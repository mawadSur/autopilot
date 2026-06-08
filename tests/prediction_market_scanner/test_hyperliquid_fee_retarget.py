"""Deterministic proof that the crypto 1m directional stack flips from
unprofitable to clearable when the simulator fee schedule is retargeted
from Coinbase (60/40 bps) to Hyperliquid perps (5/2 bps).

Companion to ``test_fee_honesty_kill.py`` (which pinned the kill) and to
``docs/CRYPTO_HYPERLIQUID_RETARGET.md`` (which writes up the arithmetic).
Hermetic: no model, no data, no network — just the simulator + a perfect
+20 bps winner.

Run:
  env PYTHONPATH=src ./.venv/bin/python -m unittest \\
      tests.prediction_market_scanner.test_hyperliquid_fee_retarget -v
"""

from __future__ import annotations

import os
import sys
import unittest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRADING_DIR = os.path.join(_REPO_ROOT, "trading")
for _p in (_TRADING_DIR, os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulator import (  # noqa: E402
    Bar,
    COINBASE_MAKER_FEE_PCT,
    COINBASE_TAKER_FEE_PCT,
    HYPERLIQUID_MAKER_FEE_PCT,
    HYPERLIQUID_TAKER_FEE_PCT,
    PortfolioSimulator,
    SimulationConfig,
)


ENTRY_PRICE = 100.0
TARGET_BPS = 0.0020  # +20 bps — the rosy end of the model's 10-20 bps target
START_CAPITAL = 10_000.0


def _flat_taker_config(**overrides) -> dict:
    base = dict(
        start_capital=START_CAPITAL,
        use_market_depth=False,
        slippage_pct=0.0,
        post_only_entries=False,
        use_atr_stops=False,
        tp_pct=0.10,
        sl_pct=0.10,
        use_regime_filter=False,
        min_atr_pct=0.0,
        allow_shorts=True,
        cooldown=0,
    )
    base.update(overrides)
    return base


def _run_perfect_winner(cfg: SimulationConfig) -> PortfolioSimulator:
    sim = PortfolioSimulator(cfg)
    target_price = ENTRY_PRICE * (1.0 + TARGET_BPS)
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=1)
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=0)
    sim.finalize(target_price)
    return sim


def _round_trip_cost_bps(cfg: SimulationConfig) -> float:
    sim = PortfolioSimulator(cfg)
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=1)
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=0)
    sim.finalize(ENTRY_PRICE)
    total_fee = cfg.start_capital - sim.last_equity
    return total_fee / cfg.start_capital * 1e4


class HyperliquidFeeRetargetTest(unittest.TestCase):
    def test_hyperliquid_default_constants_match_live_adapter(self):
        # These must mirror
        # src/exchanges/adapters/hyperliquid_tradeable.py
        # (_DEFAULT_HYPERLIQUID_FEE_MODEL).
        self.assertAlmostEqual(HYPERLIQUID_TAKER_FEE_PCT, 0.0005, places=6)
        self.assertAlmostEqual(HYPERLIQUID_MAKER_FEE_PCT, 0.0002, places=6)

    def test_from_hyperliquid_fees_wires_both_taker_and_maker(self):
        cfg = SimulationConfig.from_hyperliquid_fees()
        self.assertAlmostEqual(cfg.fee_pct, 0.0005, places=6)
        self.assertIsNotNone(cfg.maker_fee_pct)
        self.assertAlmostEqual(cfg.maker_fee_pct, 0.0002, places=6)

    def test_from_fee_model_accepts_hyperliquid_fee_model(self):
        from protocols import FeeModel

        # Mirror the live HyperliquidTradeable default schedule.
        cfg = SimulationConfig.from_fee_model(
            FeeModel(maker=0.0002, taker=0.0005)
        )
        self.assertAlmostEqual(cfg.fee_pct, 0.0005, places=6)
        self.assertAlmostEqual(cfg.maker_fee_pct, 0.0002, places=6)

    def test_hyperliquid_round_trip_is_about_10bps_taker(self):
        cfg = SimulationConfig.from_hyperliquid_fees(**_flat_taker_config())
        cost_bps = _round_trip_cost_bps(cfg)
        # 2 * 5 = ~10 bps. Allow a hair of tolerance.
        self.assertGreater(cost_bps, 9.0)
        self.assertLess(cost_bps, 11.0)
        self.assertAlmostEqual(cost_bps, 10.0, places=2)

    def test_perfect_20bps_winner_nets_positive_under_hyperliquid_fees(self):
        """THE RETARGET: a +20 bps winner that LOSES at Coinbase now WINS at HL."""
        cfg = SimulationConfig.from_hyperliquid_fees(**_flat_taker_config())
        sim = _run_perfect_winner(cfg)

        net_pnl = sim.last_equity - START_CAPITAL
        # +20 bps gross on 10k = +$20; two taker fees of 5 bps on 10k = -$5
        # each => net = 20 - 10 = +$10.
        self.assertGreater(
            net_pnl,
            0.0,
            msg=(
                "A perfect +20 bps winner must net POSITIVE under "
                f"Hyperliquid taker fees (got net PnL={net_pnl:.4f}). The "
                "whole point of the retarget is that 20 bps > 10 bps."
            ),
        )
        self.assertAlmostEqual(net_pnl, 10.0, places=2)
        self.assertEqual(sim.maker_fills, 0)
        self.assertEqual(sim.taker_fills, 2)

    def test_hyperliquid_vs_coinbase_cost_gap(self):
        # The point of the retarget is the ~12x cost gap, not a tweak.
        hl_cfg = SimulationConfig.from_hyperliquid_fees(**_flat_taker_config())
        cb_cfg = SimulationConfig.from_coinbase_fees(**_flat_taker_config())
        hl_bps = _round_trip_cost_bps(hl_cfg)
        cb_bps = _round_trip_cost_bps(cb_cfg)
        self.assertAlmostEqual(hl_bps, 10.0, places=2)
        self.assertAlmostEqual(cb_bps, 120.0, places=2)
        # 120 / 10 = 12x cheaper at HL.
        self.assertAlmostEqual(cb_bps / hl_bps, 12.0, places=2)

    def test_coinbase_and_hyperliquid_constants_are_independent(self):
        # Defensive: the two constant pairs should not silently drift toward
        # each other if someone refactors the module.
        self.assertGreater(COINBASE_TAKER_FEE_PCT, HYPERLIQUID_TAKER_FEE_PCT)
        self.assertGreater(COINBASE_MAKER_FEE_PCT, HYPERLIQUID_MAKER_FEE_PCT)
        # Ratio sanity check: Coinbase is at least 5x more expensive per side.
        self.assertGreaterEqual(
            COINBASE_TAKER_FEE_PCT / HYPERLIQUID_TAKER_FEE_PCT, 5.0
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
