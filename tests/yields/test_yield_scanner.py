"""Tests for the stablecoin yield scanner (READ-ONLY, hermetic — no network)."""
from __future__ import annotations

import unittest
from yield_scanner import (
    normalize_pools, filter_pools, rank_pools, tvl_tier, YieldPool,
)

def _p(project, apy, base, rwd, tvl, stable=True):
    return {"project": project, "chain": "Ethereum", "symbol": "USDC", "apy": apy,
            "apyBase": base, "apyReward": rwd, "tvlUsd": tvl, "stablecoin": stable, "ilRisk": "no"}


class NormalizeTierTests(unittest.TestCase):
    def test_tier_bands(self):
        self.assertEqual(tvl_tier(150e6), "established")
        self.assertEqual(tvl_tier(50e6), "mid")
        self.assertEqual(tvl_tier(5e6), "small")

    def test_normalize_envelope_and_trusted(self):
        raw = {"data": [_p("aave-v3", 5.0, 5.0, 0.0, 200e6), _p("rando", 20.0, 18.0, 2.0, 8e6),
                        {"project": "x", "apy": None}]}
        pools = normalize_pools(raw)
        self.assertEqual(len(pools), 2)  # None-apy dropped
        by = {p.project: p for p in pools}
        self.assertTrue(by["aave-v3"].trusted)
        self.assertEqual(by["aave-v3"].tier, "established")
        self.assertFalse(by["rando"].trusted)


class FilterRankTests(unittest.TestCase):
    def _pools(self):
        return normalize_pools({"data": [
            _p("aave-v3", 5.0, 5.0, 0.0, 200e6),      # trusted, established
            _p("morpho-blue", 9.0, 8.0, 1.0, 60e6),   # trusted, mid
            _p("rando", 25.0, 23.0, 2.0, 30e6),       # untrusted, high base
            _p("tiny", 15.0, 15.0, 0.0, 2e6),         # below TVL floor
            _p("mirage", 120.0, 119.0, 0.0, 50e6),    # above max_apy
            _p("volatile", 12.0, 12.0, 0.0, 80e6, stable=False),  # not stablecoin
        ]})

    def test_filter_defaults(self):
        kept = filter_pools(self._pools(), min_tvl=10e6, max_apy=40.0)
        projs = {p.project for p in kept}
        self.assertEqual(projs, {"aave-v3", "morpho-blue", "rando"})  # tiny/mirage/volatile cut

    def test_trusted_only(self):
        kept = filter_pools(self._pools(), min_tvl=10e6, max_apy=40.0, trusted_only=True)
        self.assertEqual({p.project for p in kept}, {"aave-v3", "morpho-blue"})

    def test_rank_by_base_apy(self):
        kept = rank_pools(filter_pools(self._pools(), min_tvl=10e6, max_apy=40.0))
        self.assertEqual([p.project for p in kept], ["rando", "morpho-blue", "aave-v3"])  # 23 > 8 > 5
