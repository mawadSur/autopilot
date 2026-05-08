"""Tests for alpha_lab.correlation_miner.

Hermetic: each test builds synthetic FeatureSource instances out of in-memory
DataFrames. No filesystem, no network.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from alpha_lab.correlation_miner import (
    CorrelationMiner,
    CorrelationResult,
    FeaturePair,
    FeatureSource,
    _forward_return,
    _spearman_rank_ic,
)


class _SyntheticSource:
    """Minimal :class:`FeatureSource` impl driven by a precomputed DataFrame."""

    def __init__(self, name: str, asset_class: str, df: pd.DataFrame) -> None:
        self._name = name
        self._asset_class = asset_class
        self._df = df

    @property
    def name(self) -> str:
        return self._name

    @property
    def asset_class(self) -> str:
        return self._asset_class

    def fetch_window(
        self, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        # The miner only requires a DataFrame back; synthetic source ignores
        # the window arguments and returns a precomputed series. That's fine
        # for the unit tests because we control the index directly.
        return self._df


def _make_index(n: int, freq_min: int = 1) -> pd.DatetimeIndex:
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    return pd.DatetimeIndex(
        [start + timedelta(minutes=i * freq_min) for i in range(n)],
        tz=timezone.utc,
    )


class FeaturePairFrozenTests(unittest.TestCase):
    def test_feature_pair_is_hashable_and_frozen(self) -> None:
        pair = FeaturePair(
            feature_a="ema_9",
            feature_b="return_1",
            horizon_bars=15,
            asset_class_a="spot_crypto",
            asset_class_b="spot_crypto",
        )
        # hashable -> can be used as a dict key (the gate stores rolling
        # rank-IC history per pair).
        d = {pair: 0.07}
        self.assertEqual(d[pair], 0.07)

        with self.assertRaises(FrozenInstanceError):
            pair.feature_a = "macd"  # type: ignore[misc]

    def test_pair_stable_id_round_trips(self) -> None:
        pair = FeaturePair(
            feature_a="ema_9",
            feature_b="return_1",
            horizon_bars=15,
            asset_class_a="spot_crypto",
            asset_class_b="prediction_binary",
        )
        sid = pair.stable_id()
        self.assertIn("ema_9", sid)
        self.assertIn("return_1", sid)
        self.assertIn("h=15", sid)
        # Distinct pairs => distinct stable IDs.
        other = FeaturePair(
            feature_a="return_1",
            feature_b="ema_9",
            horizon_bars=15,
            asset_class_a="spot_crypto",
            asset_class_b="prediction_binary",
        )
        self.assertNotEqual(sid, other.stable_id())


class HelperFunctionTests(unittest.TestCase):
    def test_forward_return_shifts_correctly(self) -> None:
        s = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        # horizon=1: (101 - 100)/100 = 0.01, etc. Final entry NaN (no future).
        fr = _forward_return(s, horizon_bars=1)
        self.assertAlmostEqual(fr.iloc[0], 0.01)
        self.assertAlmostEqual(fr.iloc[3], 1.0 / 103.0)
        self.assertTrue(np.isnan(fr.iloc[-1]))

    def test_forward_return_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            _forward_return(pd.Series([1.0, 2.0]), horizon_bars=0)

    def test_spearman_short_input_returns_zero(self) -> None:
        rank_ic, n, pvalue = _spearman_rank_ic(
            pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0])
        )
        self.assertEqual(rank_ic, 0.0)
        self.assertEqual(n, 2)
        self.assertEqual(pvalue, 1.0)

    def test_spearman_constant_series_returns_zero(self) -> None:
        # Both columns constant -> Spearman is undefined; we report 0.0.
        rank_ic, n, pvalue = _spearman_rank_ic(
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            pd.Series([2.0, 2.0, 2.0, 2.0]),
        )
        self.assertEqual(rank_ic, 0.0)
        self.assertEqual(pvalue, 1.0)


class CorrelationMinerTests(unittest.TestCase):
    """End-to-end tests over two synthetic 3-feature sources × 2 horizons."""

    def setUp(self) -> None:
        # Deterministic RNG so repeated runs produce identical results.
        self.rng = np.random.default_rng(seed=42)
        self.n_bars = 200
        self.idx = _make_index(self.n_bars)

    def _build_two_sources_pure_noise(self) -> CorrelationMiner:
        df_a = pd.DataFrame(
            {
                "feat_a1": self.rng.normal(size=self.n_bars),
                "feat_a2": self.rng.normal(size=self.n_bars),
                "feat_a3": self.rng.normal(size=self.n_bars),
            },
            index=self.idx,
        )
        df_b = pd.DataFrame(
            {
                "feat_b1": self.rng.normal(size=self.n_bars),
                "feat_b2": self.rng.normal(size=self.n_bars),
                "feat_b3": self.rng.normal(size=self.n_bars),
            },
            index=self.idx,
        )
        src_a = _SyntheticSource("crypto_a", "spot_crypto", df_a)
        src_b = _SyntheticSource("market_b", "prediction_binary", df_b)
        return CorrelationMiner([src_a, src_b], horizon_bars_options=[5, 15])

    def test_cartesian_product_count_matches_documented_math(self) -> None:
        """2 sources × 3 features each × 2 horizons.

        Per source pair:
          - same source pair (A, A) and (B, B): 3 × 3 - 3 = 6 valid feature
            tuples each (skip the 3 self-pairs feat_x == feat_x).
          - cross pair (A, B) and (B, A): 3 × 3 = 9 valid feature tuples each.
        Total feature tuples = 6 + 6 + 9 + 9 = 30.
        × 2 horizons = 60 results.
        """
        miner = self._build_two_sources_pure_noise()
        results = miner.mine(window_days=1)
        self.assertEqual(len(results), 60)
        self.assertTrue(all(isinstance(r, CorrelationResult) for r in results))

    def test_synthetic_strong_correlation_produces_high_rank_ic(self) -> None:
        # Build feat_b1 so that its forward 5-bar return tracks feat_a1
        # almost perfectly (with a tiny noise floor). The miner should
        # surface this pair near the top of the ranking.
        n = 200
        signal = self.rng.normal(size=n)
        # Construct a price series whose 5-bar forward return == 0.5 * signal[t]
        # plus a small noise floor. Concretely: shift signal back into a
        # cumulative price so that price[t+5] - price[t] ~ 0.5 * signal[t].
        future_returns = 0.5 * signal + 0.01 * self.rng.normal(size=n)
        price = np.zeros(n + 5)
        for i in range(n):
            price[i + 5] = price[i] + future_returns[i]
        price = price[:n] + 100.0  # keep non-zero baseline so pct-return is well-defined

        df_a = pd.DataFrame({"signal": signal}, index=self.idx)
        df_b = pd.DataFrame({"price": price}, index=self.idx)
        miner = CorrelationMiner(
            [
                _SyntheticSource("crypto_a", "spot_crypto", df_a),
                _SyntheticSource("market_b", "spot_crypto", df_b),
            ],
            horizon_bars_options=[5],
        )
        results = miner.mine(window_days=1)

        # Locate the (signal -> price @ h=5) result. It should have |rank_ic|
        # well above the 0.05 promotion threshold.
        target_pair = FeaturePair(
            feature_a="signal",
            feature_b="price",
            horizon_bars=5,
            asset_class_a="spot_crypto",
            asset_class_b="spot_crypto",
        )
        match = next(r for r in results if r.pair == target_pair)
        self.assertGreater(abs(match.rank_ic), 0.5)
        self.assertGreater(match.n_samples, 100)

    def test_random_noise_produces_low_rank_ic(self) -> None:
        miner = self._build_two_sources_pure_noise()
        results = miner.mine(window_days=1)
        # Most rank-ICs on pure noise should be small. The 95th percentile
        # of |rank-IC| on n=200 uniform-noise pairs is well under 0.25.
        # Allow some tolerance because the cartesian product has 60 results
        # and a few will tail out.
        median_abs = float(np.median([abs(r.rank_ic) for r in results]))
        self.assertLess(median_abs, 0.15)

    def test_results_sorted_by_abs_rank_ic_descending(self) -> None:
        miner = self._build_two_sources_pure_noise()
        results = miner.mine(window_days=1)
        if len(results) >= 2:
            for prev, nxt in zip(results, results[1:]):
                self.assertGreaterEqual(abs(prev.rank_ic), abs(nxt.rank_ic))

    def test_empty_data_returns_empty_list_no_crash(self) -> None:
        empty_df = pd.DataFrame(index=_make_index(0))
        miner = CorrelationMiner(
            [_SyntheticSource("a", "spot_crypto", empty_df)],
            horizon_bars_options=[5],
        )
        self.assertEqual(miner.mine(window_days=1), [])

    def test_all_nan_columns_are_dropped_gracefully(self) -> None:
        df = pd.DataFrame(
            {
                "all_nan": [np.nan] * self.n_bars,
                "real": self.rng.normal(size=self.n_bars),
            },
            index=self.idx,
        )
        miner = CorrelationMiner(
            [_SyntheticSource("a", "spot_crypto", df)],
            horizon_bars_options=[5],
        )
        # Single source × one valid feature -> the all-NaN column is dropped,
        # 1 feature × 1 feature × 1 horizon - 1 self-pair = 0 results.
        # (Self-pair against itself is the only candidate; gets skipped.)
        self.assertEqual(miner.mine(window_days=1), [])

    def test_source_raising_exception_is_skipped_with_warning(self) -> None:
        good_df = pd.DataFrame(
            {"f": self.rng.normal(size=self.n_bars)}, index=self.idx
        )

        class _BrokenSource:
            name = "broken"
            asset_class = "spot_crypto"

            def fetch_window(self, *_a, **_k):  # type: ignore[no-untyped-def]
                raise RuntimeError("simulated outage")

        miner = CorrelationMiner(
            [
                _BrokenSource(),
                _SyntheticSource("good", "spot_crypto", good_df),
            ],
            horizon_bars_options=[5],
        )
        # Broken source skipped, good source has 1 feature -> 0 cross pairs
        # because there's only one source after the broken one drops out
        # AND a single feature can't self-pair. Result: empty list, no raise.
        self.assertEqual(miner.mine(window_days=1), [])

    def test_rejects_empty_constructor_args(self) -> None:
        df = pd.DataFrame({"x": [1.0]}, index=_make_index(1))
        with self.assertRaises(ValueError):
            CorrelationMiner([], horizon_bars_options=[5])
        with self.assertRaises(ValueError):
            CorrelationMiner(
                [_SyntheticSource("a", "spot_crypto", df)],
                horizon_bars_options=[],
            )
        with self.assertRaises(ValueError):
            CorrelationMiner(
                [_SyntheticSource("a", "spot_crypto", df)],
                horizon_bars_options=[0, 5],  # non-positive
            )

    def test_window_days_must_be_positive(self) -> None:
        df = pd.DataFrame({"x": [1.0]}, index=_make_index(1))
        miner = CorrelationMiner(
            [_SyntheticSource("a", "spot_crypto", df)],
            horizon_bars_options=[5],
        )
        with self.assertRaises(ValueError):
            miner.mine(window_days=0)

    def test_feature_source_protocol_isinstance_check(self) -> None:
        """The FeatureSource protocol is runtime_checkable."""
        df = pd.DataFrame({"x": [1.0]}, index=_make_index(1))
        src = _SyntheticSource("a", "spot_crypto", df)
        self.assertIsInstance(src, FeatureSource)

    def test_asset_class_label_falls_back_for_enum(self) -> None:
        """Sources may emit an enum for asset_class; miner stringifies it."""

        class _EnumLike:
            value = "perp_crypto"

        class _EnumSource:
            name = "perp"
            asset_class = _EnumLike()

            def fetch_window(self, *_a, **_k):  # type: ignore[no-untyped-def]
                return pd.DataFrame(
                    {"f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
                    index=_make_index(8),
                )

        df_b = pd.DataFrame(
            {"g1": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]},
            index=_make_index(8),
        )
        miner = CorrelationMiner(
            [_EnumSource(), _SyntheticSource("b", "spot_crypto", df_b)],
            horizon_bars_options=[1],
        )
        results = miner.mine(window_days=1)
        # At least one cross-source result should label the enum source as
        # "perp_crypto" (the enum's .value), not the repr of the enum object.
        self.assertTrue(any(r.pair.asset_class_a == "perp_crypto" for r in results))


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
