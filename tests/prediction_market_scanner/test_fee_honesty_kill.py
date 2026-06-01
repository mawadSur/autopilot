"""Deterministic, env-independent proof that the crypto 1m directional stack
is unprofitable by ARITHMETIC, not tuning.

The model targets a +10-20 bps move. The repo's own Coinbase adapter charges a
real fee of 60 bps taker / 40 bps maker per side (see
``src/exchanges/adapters/coinbase_tradeable.py`` ->
``FeeModel(maker=0.0040, taker=0.0060)``). A taker round-trip therefore costs
~120 bps. 20 bps gross < 120 bps cost => every "winning" trade still loses money.

Historically the simulator defaulted to ``fee_pct=0.0008`` (~16 bps round-trip)
and ``src/backtest.py`` hardcoded an even cheaper ``0.00075``, while leaving the
maker path unwired (``_maker_fee_pct`` fell back to ``fee_pct``). That ~7.5x cost
understatement made every backtest fictional. This test pins the honest behavior:

  1. A trade that PERFECTLY hits a +20 bps target, run through the simulator
     configured with the real Coinbase taker fee, nets NEGATIVE PnL.
  2. The effective round-trip cost under Coinbase-taker config is ~120 bps,
     versus ~16 bps under the old cheap default.

Run:
  env PYTHONPATH=src ./.venv/bin/python -m unittest \
      tests.prediction_market_scanner.test_fee_honesty_kill -v
"""

from __future__ import annotations

import os
import sys
import unittest


# ``trading/simulator.py`` imports flat (``from utils import ...``) like the rest
# of the legacy stack. With ``PYTHONPATH=src`` ``utils`` resolves; we just have to
# make the ``trading`` package importable too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRADING_DIR = os.path.join(_REPO_ROOT, "trading")
for _p in (_TRADING_DIR, os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulator import (  # noqa: E402
    Bar,
    COINBASE_MAKER_FEE_PCT,
    COINBASE_TAKER_FEE_PCT,
    PortfolioSimulator,
    SimulationConfig,
)


ENTRY_PRICE = 100.0
TARGET_BPS = 0.0020  # +20 bps — the rosy end of the model's 10-20 bps target
START_CAPITAL = 10_000.0


def _flat_taker_config(**overrides) -> SimulationConfig:
    """A deterministic, depth/slippage-free config with a pure taker round-trip.

    Depth and slippage are disabled so the entry fills at the bar open and the
    exit fills at the requested price — isolating the FEE arithmetic. Stops are
    set wide so neither TP nor SL fires; we exit via ``finalize`` (a taker
    ``close_end`` fill) at exactly the +20 bps target.
    """
    base = dict(
        start_capital=START_CAPITAL,
        use_market_depth=False,
        slippage_pct=0.0,
        post_only_entries=False,   # taker market entry
        use_atr_stops=False,
        tp_pct=0.10,               # wide — does not trigger
        sl_pct=0.10,               # wide — does not trigger
        use_regime_filter=False,
        min_atr_pct=0.0,
        allow_shorts=True,
        cooldown=0,
    )
    base.update(overrides)
    return base


def _run_perfect_winner(cfg: SimulationConfig) -> PortfolioSimulator:
    """Enter long (taker), then close at exactly +20 bps (taker close_end)."""
    sim = PortfolioSimulator(cfg)
    target_price = ENTRY_PRICE * (1.0 + TARGET_BPS)

    # Bar 0: emit the long signal (becomes pending), price flat at ENTRY_PRICE.
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=1)
    # Bar 1: pending long executes at this bar's open (== ENTRY_PRICE).
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=0)
    # Close the position at exactly the +20 bps target (taker close_end exit).
    sim.finalize(target_price)
    return sim


def _round_trip_cost_bps(cfg: SimulationConfig) -> float:
    """Open + close a position with ZERO gross move; return total fees in bps."""
    sim = PortfolioSimulator(cfg)
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=1)
    sim.step(Bar(open=ENTRY_PRICE, high=ENTRY_PRICE, low=ENTRY_PRICE, close=ENTRY_PRICE), signal=0)
    sim.finalize(ENTRY_PRICE)  # flat close: PnL is fees only
    total_fee = cfg.start_capital - sim.last_equity
    return total_fee / cfg.start_capital * 1e4


class FeeHonestyKillTest(unittest.TestCase):
    def test_coinbase_default_constants_match_live_adapter(self):
        # These must mirror src/exchanges/adapters/coinbase_tradeable.py.
        self.assertAlmostEqual(COINBASE_TAKER_FEE_PCT, 0.0060, places=6)
        self.assertAlmostEqual(COINBASE_MAKER_FEE_PCT, 0.0040, places=6)

    def test_from_coinbase_fees_wires_both_taker_and_maker(self):
        cfg = SimulationConfig.from_coinbase_fees()
        # ``fee_pct`` is the taker rate; ``maker_fee_pct`` is set explicitly so the
        # maker path is no longer fictional (it used to fall back to ``fee_pct``).
        self.assertAlmostEqual(cfg.fee_pct, 0.0060, places=6)
        self.assertIsNotNone(cfg.maker_fee_pct)
        self.assertAlmostEqual(cfg.maker_fee_pct, 0.0040, places=6)

    def test_from_fee_model_accepts_live_fee_model(self):
        # Duck-typed: any object exposing .maker/.taker (e.g. protocols.FeeModel).
        from protocols import FeeModel

        cfg = SimulationConfig.from_fee_model(FeeModel(maker=0.0040, taker=0.0060))
        self.assertAlmostEqual(cfg.fee_pct, 0.0060, places=6)
        self.assertAlmostEqual(cfg.maker_fee_pct, 0.0040, places=6)

    def test_perfect_20bps_winner_nets_negative_under_real_coinbase_fees(self):
        """THE KILL: a trade that perfectly hits +20 bps still LOSES money."""
        cfg = SimulationConfig.from_coinbase_fees(**_flat_taker_config())
        sim = _run_perfect_winner(cfg)

        net_pnl = sim.last_equity - START_CAPITAL
        self.assertLess(
            net_pnl,
            0.0,
            msg=(
                "A perfect +20 bps winner must net NEGATIVE under real Coinbase "
                f"taker fees (got net PnL={net_pnl:.4f}). 20 bps gross cannot "
                "clear ~120 bps round-trip cost."
            ),
        )

        # Exact arithmetic: +20 bps gross on 10k notional = +$20; two taker fees
        # of 60 bps on 10k = -$60 each => net = 20 - 120 = -$100.
        self.assertAlmostEqual(net_pnl, -100.0, places=2)

        # Both legs filled as taker (no fictional maker discount).
        self.assertEqual(sim.maker_fills, 0)
        self.assertEqual(sim.taker_fills, 2)

    def test_effective_round_trip_cost_is_about_120bps_under_coinbase(self):
        cfg = SimulationConfig.from_coinbase_fees(**_flat_taker_config())
        cost_bps = _round_trip_cost_bps(cfg)
        # 2 x 60 bps taker = ~120 bps. Allow a hair of tolerance.
        self.assertGreater(cost_bps, 115.0)
        self.assertLess(cost_bps, 125.0)
        self.assertAlmostEqual(cost_bps, 120.0, places=2)

    def test_old_cheap_default_understated_cost_at_about_16bps(self):
        # Evidence of the bug: the legacy ~8 bps default round-trips at ~16 bps,
        # which is why fictional backtests "passed". This is what we replaced.
        legacy_cfg = SimulationConfig(**_flat_taker_config())
        cost_bps = _round_trip_cost_bps(legacy_cfg)
        self.assertAlmostEqual(cost_bps, 16.0, places=2)
        # And it is ~7.5x cheaper than the honest Coinbase cost.
        coinbase_cfg = SimulationConfig.from_coinbase_fees(**_flat_taker_config())
        honest_bps = _round_trip_cost_bps(coinbase_cfg)
        self.assertAlmostEqual(honest_bps / cost_bps, 7.5, places=2)

    def test_under_old_cheap_fees_the_same_winner_is_profitable(self):
        # Control: the EXACT same +20 bps winner is profitable under the old cheap
        # fees. This proves the kill is driven by the fee correction, not by the
        # test scenario being rigged to lose.
        legacy_cfg = SimulationConfig(**_flat_taker_config())
        sim = _run_perfect_winner(legacy_cfg)
        net_pnl = sim.last_equity - START_CAPITAL
        # +20 bps gross (=$20) minus ~16 bps fees (=$16) => +$4 net.
        self.assertGreater(net_pnl, 0.0)
        self.assertAlmostEqual(net_pnl, 4.0, places=2)


if __name__ == "__main__":
    unittest.main()
