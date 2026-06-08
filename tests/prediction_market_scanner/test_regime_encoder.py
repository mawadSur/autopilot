"""Unit tests for ``regime_memory.encoder``.

Covers:
* :class:`RegimeWindow` is hashable + has the expected fields
* :meth:`RegimeEncoder.encode_features` produces a deterministic, fixed-
  length vector for a given feature set
* Empty-input guard raises ``ValueError`` with a clear message
* NaN handling: column-mean fill before stats, full-NaN column → 0.0
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from regime_memory.encoder import (
    SUMMARY_STAT_NAMES,
    RegimeEncoder,
    RegimeWindow,
)


def _synthetic_window(n_rows: int = 60, n_cols: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_rows, n_cols)),
        columns=[f"f{i}" for i in range(n_cols)],
    )


class RegimeWindowTests(unittest.TestCase):
    def test_is_hashable_via_identity_triple(self) -> None:
        w = RegimeWindow(
            symbol="ETH-USD",
            window_end_utc="2026-05-01T12:00:00+00:00",
            bars=60,
            embedding=[0.1, 0.2, 0.3],
            metadata={"optimal_threshold": 0.62},
        )
        # Mutable container fields would normally make a frozen dataclass un-
        # hashable; we override __hash__ to use the identity triple instead.
        self.assertEqual(
            hash(w),
            hash(("ETH-USD", "2026-05-01T12:00:00+00:00", 60)),
        )
        # And it works inside a set
        self.assertIn(w, {w})

    def test_metadata_and_embedding_round_trip(self) -> None:
        w = RegimeWindow(
            symbol="BTC-USD",
            window_end_utc="2026-01-01T00:00:00+00:00",
            bars=30,
            embedding=[1.0, 2.0],
            metadata={"realized_sharpe": 1.4, "kelly_size_pct": 0.05},
        )
        self.assertEqual(w.embedding, [1.0, 2.0])
        self.assertEqual(w.metadata["realized_sharpe"], 1.4)

    def test_frozen_prevents_attribute_mutation(self) -> None:
        w = RegimeWindow(
            symbol="ETH-USD",
            window_end_utc="2026-05-01T12:00:00+00:00",
            bars=60,
            embedding=[0.0],
            metadata={},
        )
        with self.assertRaises(Exception):  # FrozenInstanceError is a subclass
            w.symbol = "BTC-USD"  # type: ignore[misc]


class RegimeEncoderTests(unittest.TestCase):
    def test_fixed_dim_for_pinned_feature_cols(self) -> None:
        cols = [f"f{i}" for i in range(8)]
        encoder = RegimeEncoder(feature_cols=cols)
        df = _synthetic_window(n_rows=60, n_cols=8)
        vec = encoder.encode_features(df, window_size=60)
        self.assertEqual(len(vec), len(cols) * len(SUMMARY_STAT_NAMES))
        self.assertEqual(len(vec), encoder.expected_dim())

    def test_dim_uses_inferred_columns_when_unset(self) -> None:
        encoder = RegimeEncoder()
        df = _synthetic_window(n_rows=60, n_cols=5)
        vec = encoder.encode_features(df, window_size=60)
        self.assertEqual(len(vec), 5 * len(SUMMARY_STAT_NAMES))

    def test_deterministic_for_same_input(self) -> None:
        encoder = RegimeEncoder(feature_cols=[f"f{i}" for i in range(4)])
        df = _synthetic_window(n_rows=60, n_cols=4, seed=42)
        v1 = encoder.encode_features(df)
        v2 = encoder.encode_features(df.copy())
        self.assertEqual(v1, v2)

    def test_uses_trailing_window(self) -> None:
        # Build a frame where the tail differs strongly from the head; verify
        # that increasing window_size brings the head's stats into the
        # encoding. We check this by comparing the "last" stat slot, which
        # always reflects the final bar regardless of window_size, and the
        # "mean" stat slot, which DOES change with window_size.
        cols = ["a"]
        encoder = RegimeEncoder(feature_cols=cols)
        head = pd.DataFrame({"a": [-10.0] * 60})
        tail = pd.DataFrame({"a": [+10.0] * 60})
        full = pd.concat([head, tail], ignore_index=True)

        small = encoder.encode_features(full, window_size=60)
        large = encoder.encode_features(full, window_size=120)

        # Embedding layout per feature: [mean, std, last, pct_change]
        # Both windows end with +10.0 -> "last" slot is +10.0 in both.
        self.assertAlmostEqual(small[2], 10.0, places=6)
        self.assertAlmostEqual(large[2], 10.0, places=6)
        # Mean over the trailing 60 (all +10) is +10; mean over 120 (60 of
        # -10 and 60 of +10) is 0.
        self.assertAlmostEqual(small[0], 10.0, places=6)
        self.assertAlmostEqual(large[0], 0.0, places=6)

    def test_empty_dataframe_raises_value_error(self) -> None:
        encoder = RegimeEncoder()
        with self.assertRaises(ValueError) as ctx:
            encoder.encode_features(pd.DataFrame())
        self.assertIn("non-empty", str(ctx.exception))

    def test_window_size_must_be_positive(self) -> None:
        encoder = RegimeEncoder()
        with self.assertRaises(ValueError):
            encoder.encode_features(_synthetic_window(), window_size=0)

    def test_nan_values_filled_with_column_mean(self) -> None:
        # Construct a column where all values are 5.0 except one NaN; the
        # column mean is 5.0 so the NaN should be filled to 5.0 -> mean of
        # the window stays 5.0, std stays 0.0.
        cols = ["a"]
        encoder = RegimeEncoder(feature_cols=cols)
        vals = [5.0] * 30 + [float("nan")] + [5.0] * 29
        df = pd.DataFrame({"a": vals})
        vec = encoder.encode_features(df, window_size=60)
        # Layout: [mean, std, last, pct_change]
        self.assertAlmostEqual(vec[0], 5.0, places=6)
        self.assertAlmostEqual(vec[1], 0.0, places=6)
        # last bar is 5.0
        self.assertAlmostEqual(vec[2], 5.0, places=6)
        # pct_change = (last - first)/(|first| + eps) -> 0
        self.assertAlmostEqual(vec[3], 0.0, places=6)

    def test_full_nan_column_falls_back_to_zero(self) -> None:
        # A column that's entirely NaN can't be filled with its mean — the
        # encoder must fall back to 0.0 to keep the embedding dim fixed.
        encoder = RegimeEncoder(feature_cols=["a"])
        df = pd.DataFrame({"a": [float("nan")] * 60})
        vec = encoder.encode_features(df, window_size=60)
        for x in vec:
            self.assertFalse(math.isnan(x), "no NaN should leak into vector")
        self.assertEqual(vec, [0.0, 0.0, 0.0, 0.0])

    def test_inf_values_treated_as_nan(self) -> None:
        encoder = RegimeEncoder(feature_cols=["a"])
        df = pd.DataFrame({"a": [1.0] * 30 + [float("inf"), float("-inf")] + [1.0] * 28})
        vec = encoder.encode_features(df, window_size=60)
        for x in vec:
            self.assertFalse(math.isinf(x))
            self.assertFalse(math.isnan(x))

    def test_missing_columns_filled_with_zeros(self) -> None:
        # The pinned feature_cols include "b" but the input DataFrame only
        # has "a". The encoder treats the missing column as all-NaN and
        # falls back to zeros — preserving embedding dim across schema drift.
        encoder = RegimeEncoder(feature_cols=["a", "b"])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        vec = encoder.encode_features(df, window_size=3)
        # Layout: 4 stats for "a", then 4 stats for "b".
        self.assertEqual(len(vec), 8)
        # "b" slots (indices 4..7) should be all zero.
        self.assertEqual(vec[4:8], [0.0, 0.0, 0.0, 0.0])

    def test_expected_dim_without_feature_cols_requires_n(self) -> None:
        encoder = RegimeEncoder()
        with self.assertRaises(ValueError):
            encoder.expected_dim()
        self.assertEqual(encoder.expected_dim(n_features=10), 40)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
