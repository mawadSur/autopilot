"""Tests for the persistent funding-carry sleeve (Rung 1; SHADOW, NO network).

Covers ``rank_persistent`` (scanner), ``build_persistent_candidates`` + the
persistence-open/exit paths of ``run_scan`` (runner), and confirms the legacy
snapshot path is byte-for-byte unchanged when no history/exit params are passed.
"""
from __future__ import annotations

import os, tempfile, types, unittest

from funding_carry_scanner import (
    rank_persistent, normalize_funding_rates, DEFAULT_CARRY_UNIVERSE,
)
from funding_carry_runner import (
    build_persistent_candidates, fold_ledger, run_scan,
)

H_MS = 3_600_000.0  # 1h in ms


def _bt(symbol, *, net_annual, favorable_pct, window_days, side="short_perp",
        gross_annual=1.0):
    """A SymbolBacktest-like stand-in (rank_persistent is duck-typed)."""
    return types.SimpleNamespace(
        symbol=symbol, side=side, net_annual=net_annual,
        favorable_pct=favorable_pct, window_days=window_days,
        gross_annual=gross_annual,
    )


def _hist(rates, start=1_000_000_000_000.0):
    """[{timestamp, fundingRate}, ...] at hourly cadence (already normalized)."""
    return [{"timestamp": start + i * H_MS, "fundingRate": r}
            for i, r in enumerate(rates)]


class RankPersistentTests(unittest.TestCase):
    def test_filters_by_window_favorable_and_net(self):
        rows = [
            _bt("KEEP", net_annual=0.30, favorable_pct=0.80, window_days=10.0),
            _bt("THIN_WINDOW", net_annual=0.30, favorable_pct=0.80, window_days=3.0),
            _bt("WHIPPY", net_annual=0.30, favorable_pct=0.40, window_days=10.0),
            _bt("NEG_NET", net_annual=-0.01, favorable_pct=0.90, window_days=10.0),
        ]
        kept = rank_persistent(rows)  # defaults: window>=7, fav>=0.60, net>0
        self.assertEqual([b.symbol for b in kept], ["KEEP"])

    def test_sorts_by_net_annual_descending(self):
        rows = [
            _bt("LOW", net_annual=0.10, favorable_pct=0.70, window_days=8.0),
            _bt("HIGH", net_annual=0.50, favorable_pct=0.70, window_days=8.0),
            _bt("MID", net_annual=0.25, favorable_pct=0.70, window_days=8.0),
        ]
        kept = rank_persistent(rows)
        self.assertEqual([b.symbol for b in kept], ["HIGH", "MID", "LOW"])

    def test_boundary_thresholds_are_inclusive_for_window_and_favorable(self):
        # window==min and favorable==min are KEPT (>=); net==min is DROPPED (>).
        rows = [
            _bt("ON_WINDOW", net_annual=0.10, favorable_pct=0.60, window_days=7.0),
            _bt("ON_NET", net_annual=0.0, favorable_pct=0.90, window_days=10.0),
        ]
        kept = rank_persistent(rows)
        self.assertEqual([b.symbol for b in kept], ["ON_WINDOW"])

    def test_custom_thresholds(self):
        rows = [_bt("X", net_annual=0.05, favorable_pct=0.55, window_days=5.0)]
        # too strict default -> dropped
        self.assertEqual(rank_persistent(rows), [])
        # relaxed thresholds -> kept
        kept = rank_persistent(rows, min_window_days=4.0, min_favorable_pct=0.50,
                               min_net_annual=0.0)
        self.assertEqual([b.symbol for b in kept], ["X"])


class BuildPersistentCandidatesTests(unittest.TestCase):
    def test_only_persistent_symbol_survives(self):
        # BTC: 8d of steady positive funding -> long window, fav=100%, net>0.
        # WHIP: 1d only -> fails the 7-day window floor and is dropped.
        histories = {
            "BTC/USDC:USDC": _hist([0.0001] * (24 * 8)),
            "WHIP/USDC:USDC": _hist([0.0002, -0.0003] * 12),  # 24 periods, ~1d
        }
        cands = build_persistent_candidates(
            histories, period_hours=1.0, round_trip_bps=20.0,
            basis_buffer_annual=0.05,
        )
        self.assertEqual([c.symbol for c in cands], ["BTC/USDC:USDC"])
        self.assertEqual(cands[0].side, "short_perp")
        self.assertGreater(cands[0].net_annual, 0.0)

    def test_accepts_raw_unsorted_history_via_normalize(self):
        # Out-of-order timestamps must still be backtested correctly.
        rates = [0.0001] * (24 * 8)
        normalized = _hist(rates)
        shuffled = list(reversed(normalized))  # raw / unsorted on the way in
        cands = build_persistent_candidates(
            {"BTC/USDC:USDC": shuffled}, period_hours=1.0,
            round_trip_bps=20.0, basis_buffer_annual=0.05,
        )
        self.assertEqual([c.symbol for c in cands], ["BTC/USDC:USDC"])

    def test_empty_history_drops_symbol(self):
        cands = build_persistent_candidates(
            {"EMPTY/USDC:USDC": []}, period_hours=1.0,
            round_trip_bps=20.0, basis_buffer_annual=0.05,
        )
        self.assertEqual(cands, [])


class RunScanPersistenceOpenTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._d.name, "carry.jsonl")
        self.t0 = 1_000_000_000_000.0
        # Current accrual rows for BOTH symbols (accrual reads these).
        self.rows = normalize_funding_rates({
            "BTC/USDC:USDC": {"fundingRate": 0.0001},
            "WHIP/USDC:USDC": {"fundingRate": 0.0002},
        }, period_hours=1.0)
        self.histories = {
            "BTC/USDC:USDC": _hist([0.0001] * (24 * 8)),      # persistent
            "WHIP/USDC:USDC": _hist([0.0002, -0.0003] * 12),  # short window
        }

    def tearDown(self):
        self._d.cleanup()

    def test_open_uses_only_persistent_symbols(self):
        res = run_scan(
            self.path, self.rows, now_ms=self.t0, period_hours=1.0,
            notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
            basis_buffer_annual=0.05, hold_days=14.0,
            funding_histories=self.histories,
        )
        self.assertEqual(res["action"], "opened")
        st = fold_ledger(self.path)
        self.assertEqual(set(st), {"BTC/USDC:USDC"})  # WHIP dropped (short window)
        self.assertEqual(st["BTC/USDC:USDC"]["side"], "short_perp")
        self.assertEqual(st["BTC/USDC:USDC"]["side_sign"], 1.0)
        self.assertFalse(st["BTC/USDC:USDC"]["closed"])  # additive flag, open=False


class RunScanExitTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._d.name, "carry.jsonl")
        self.t0 = 1_000_000_000_000.0
        self.histories = {"BTC/USDC:USDC": _hist([0.0001] * (24 * 8))}

    def tearDown(self):
        self._d.cleanup()

    def _open(self):
        run_scan(
            self.path, normalize_funding_rates(
                {"BTC/USDC:USDC": {"fundingRate": 0.0001}}, period_hours=1.0),
            now_ms=self.t0, period_hours=1.0, notional=1000.0, top_k=8,
            min_net_annual=0.10, round_trip_bps=20.0, basis_buffer_annual=0.05,
            hold_days=14.0, funding_histories=self.histories,
        )

    def test_funding_flip_closes_position_and_stops_accruing(self):
        self._open()
        st = fold_ledger(self.path)
        self.assertEqual(set(st), {"BTC/USDC:USDC"})
        self.assertFalse(st["BTC/USDC:USDC"]["closed"])

        # Funding FLIPS negative; a short-perp position now pays -> exit.
        flipped = normalize_funding_rates(
            {"BTC/USDC:USDC": {"fundingRate": -0.0002}}, period_hours=1.0)
        run_scan(
            self.path, flipped, now_ms=self.t0 + H_MS, period_hours=1.0,
            notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
            basis_buffer_annual=0.05, hold_days=14.0, exit_below_annual=0.0,
        )
        st = fold_ledger(self.path)
        pos = st["BTC/USDC:USDC"]
        self.assertTrue(pos["closed"])
        self.assertEqual(pos["exit_reason"], "funding_flip")  # favorable < 0
        # The one flipped period was still booked before closing.
        realized_after_flip = pos["realized_usd"]

        # A subsequent accrue scan must NOT add realized to a closed position.
        run_scan(
            self.path, flipped, now_ms=self.t0 + 2 * H_MS, period_hours=1.0,
            notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
            basis_buffer_annual=0.05, hold_days=14.0, exit_below_annual=0.0,
        )
        st = fold_ledger(self.path)
        self.assertAlmostEqual(st["BTC/USDC:USDC"]["realized_usd"],
                               realized_after_flip, places=9)
        self.assertTrue(st["BTC/USDC:USDC"]["closed"])  # stays closed

    def test_decay_exit_when_funding_compresses_but_stays_positive(self):
        self._open()
        # Funding stays positive but tiny: gross_annual ~ small positive.
        # 0.000001/period -> ~0.876%/yr, below an exit floor of 0.10.
        decayed = normalize_funding_rates(
            {"BTC/USDC:USDC": {"fundingRate": 0.000001}}, period_hours=1.0)
        run_scan(
            self.path, decayed, now_ms=self.t0 + H_MS, period_hours=1.0,
            notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
            basis_buffer_annual=0.05, hold_days=14.0, exit_below_annual=0.10,
        )
        st = fold_ledger(self.path)
        pos = st["BTC/USDC:USDC"]
        self.assertTrue(pos["closed"])
        self.assertEqual(pos["exit_reason"], "decay")  # favorable >= 0 but < floor

    def test_no_exit_when_funding_stays_above_floor(self):
        self._open()
        # Funding holds strong (87.6%/yr) -> well above floor; stays open.
        strong = normalize_funding_rates(
            {"BTC/USDC:USDC": {"fundingRate": 0.0001}}, period_hours=1.0)
        run_scan(
            self.path, strong, now_ms=self.t0 + H_MS, period_hours=1.0,
            notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
            basis_buffer_annual=0.05, hold_days=14.0, exit_below_annual=0.0,
        )
        st = fold_ledger(self.path)
        pos = st["BTC/USDC:USDC"]
        self.assertFalse(pos["closed"])
        self.assertAlmostEqual(pos["realized_usd"], 0.10, places=6)  # +$0.10 booked
        self.assertEqual(pos["n_accruals"], 1)


class RunScanBackCompatTests(unittest.TestCase):
    """Snapshot scan with NO history/exit params behaves EXACTLY as today."""

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._d.name, "carry.jsonl")
        self.rows = normalize_funding_rates({
            "AAA/USDC:USDC": {"fundingRate": 0.0001},    # +87.6%/yr -> short
            "BBB/USDC:USDC": {"fundingRate": -0.00008},  # -70%/yr  -> long
            "THIN/USDC:USDC": {"fundingRate": 0.000003}, # ~2.6%/yr -> filtered
        }, period_hours=1.0)

    def tearDown(self):
        self._d.cleanup()

    def test_snapshot_open_unchanged(self):
        res = run_scan(self.path, self.rows, now_ms=1_000_000_000_000.0,
                       period_hours=1.0, notional=1000.0, top_k=8,
                       min_net_annual=0.10, round_trip_bps=20.0,
                       basis_buffer_annual=0.05, hold_days=14.0)
        self.assertEqual(res["action"], "opened")
        st = fold_ledger(self.path)
        self.assertEqual(set(st), {"AAA/USDC:USDC", "BBB/USDC:USDC"})  # THIN filtered
        self.assertEqual(st["AAA/USDC:USDC"]["side"], "short_perp")
        self.assertEqual(st["BBB/USDC:USDC"]["side"], "long_perp")
        self.assertAlmostEqual(st["AAA/USDC:USDC"]["rt_cost_usd"], 2.0, places=6)

    def test_snapshot_never_closes_without_exit_param(self):
        t0 = 1_000_000_000_000.0
        run_scan(self.path, self.rows, now_ms=t0, period_hours=1.0, notional=1000.0,
                 top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
                 basis_buffer_annual=0.05, hold_days=14.0)
        # Accrue with FLIPPED funding but NO exit param -> must still accrue, never close.
        flipped = normalize_funding_rates({
            "AAA/USDC:USDC": {"fundingRate": -0.0005},   # would flip a short
            "BBB/USDC:USDC": {"fundingRate": 0.0005},    # would flip a long
        }, period_hours=1.0)
        run_scan(self.path, flipped, now_ms=t0 + H_MS, period_hours=1.0,
                 notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
                 basis_buffer_annual=0.05, hold_days=14.0)
        st = fold_ledger(self.path)
        for sym in ("AAA/USDC:USDC", "BBB/USDC:USDC"):
            self.assertFalse(st[sym]["closed"])      # additive 'closed' default stays False
            self.assertEqual(st[sym]["n_accruals"], 1)  # still booked the period

    def test_default_carry_universe_constant(self):
        self.assertEqual(DEFAULT_CARRY_UNIVERSE, ["BTC/USDC:USDC", "ETH/USDC:USDC"])


if __name__ == "__main__":
    unittest.main()
