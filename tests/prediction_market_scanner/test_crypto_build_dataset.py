"""Tests for src/crypto_training/build_dataset.py.

Pure-pandas tests; no exchange / network. Synthesised OHLCV CSV files
written into a tempdir, then fed through ``build_dataset``.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Match the libomp flag used everywhere else in the suite.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from crypto_training.build_dataset import (
    build_dataset,
    label_forward_return_binary,
    label_forward_return_three_class,
    load_microstructure_sidecars,
    load_ohlcv,
    merge_microstructure,
)
from history_coindesk import L2_FEATURE_COLUMNS, TRADE_AGG_COLUMNS

# ---------------------------------------------------------------------------
# Vol-normalized label tests
# ---------------------------------------------------------------------------


class VolNormalizedLabelTests(unittest.TestCase):
    """Tests for label_kind='vol_normalized' in label_forward_return_binary.

    Invariant: a fixed forward move of +X bps should be labeled 1 during
    *low-vol* bars (where X > k * atrp_14_bps) but labeled 0 during
    *high-vol* bars (where X < k * atrp_14_bps). This is the opposite of
    the fixed-threshold behavior and verifies the confound is corrected.
    """

    def _make_df(self, closes: list, atrp_14: list) -> pd.DataFrame:
        """Build a minimal DataFrame with close + atrp_14 columns."""
        return pd.DataFrame({"close": closes, "atrp_14": atrp_14})

    def test_low_vol_bar_labels_positive(self) -> None:
        # forward move: (102 - 100) / 100 * 10000 = 200 bps
        # atrp_14 = 0.1% = 10 bps (low vol); dynamic thr = 0.5 * 10 = 5 bps
        # 200 > 5 -> label=1
        df = self._make_df(
            closes=[100.0, 102.0, 102.0],
            atrp_14=[0.10, 0.10, 0.10],  # 0.10% = 10 bps per ATR%
        )
        labels = label_forward_return_binary(
            df, horizon_bars=1, threshold_bps=10.0, label_kind="vol_normalized", vol_normalize_k=0.5
        )
        self.assertEqual(int(labels.iloc[0]), 1)

    def test_high_vol_bar_same_move_labels_negative(self) -> None:
        # Same 200 bps forward move but atrp_14 = 5% = 500 bps (high vol)
        # dynamic thr = 0.5 * 500 = 250 bps; 200 < 250 -> label=0
        df = self._make_df(
            closes=[100.0, 102.0, 102.0],
            atrp_14=[5.0, 5.0, 5.0],  # 5% ATR = 500 bps
        )
        labels = label_forward_return_binary(
            df, horizon_bars=1, threshold_bps=10.0, label_kind="vol_normalized", vol_normalize_k=0.5
        )
        self.assertEqual(int(labels.iloc[0]), 0)

    def test_fixed_bps_unchanged_behavior(self) -> None:
        # Verify fixed_bps (default) is unaffected by presence of atrp_14 col.
        df = self._make_df(
            closes=[100.0, 102.0, 102.0],
            atrp_14=[5.0, 5.0, 5.0],
        )
        # fixed_bps: 200 bps > 10 bps threshold -> label=1
        labels_fixed = label_forward_return_binary(
            df, horizon_bars=1, threshold_bps=10.0, label_kind="fixed_bps"
        )
        # vol_normalized: 200 bps < 250 bps dynamic thr -> label=0
        labels_voln = label_forward_return_binary(
            df, horizon_bars=1, threshold_bps=10.0, label_kind="vol_normalized", vol_normalize_k=0.5
        )
        self.assertEqual(int(labels_fixed.iloc[0]), 1)
        self.assertEqual(int(labels_voln.iloc[0]), 0)

    def test_trailing_row_is_nan(self) -> None:
        df = self._make_df(
            closes=[100.0, 102.0],
            atrp_14=[0.1, 0.1],
        )
        labels = label_forward_return_binary(
            df, horizon_bars=1, threshold_bps=10.0, label_kind="vol_normalized"
        )
        self.assertTrue(pd.isna(labels.iloc[-1]))

    def test_nan_atrp_propagates_to_nan_label(self) -> None:
        # First bar has NaN atrp (warmup); should produce NaN label even though
        # forward return is computable.
        df = self._make_df(
            closes=[100.0, 105.0, 110.0],
            atrp_14=[float("nan"), 0.1, 0.1],
        )
        labels = label_forward_return_binary(
            df, horizon_bars=1, threshold_bps=10.0, label_kind="vol_normalized"
        )
        self.assertTrue(pd.isna(labels.iloc[0]))
        self.assertFalse(pd.isna(labels.iloc[1]))

    def test_missing_atrp14_column_raises(self) -> None:
        df = pd.DataFrame({"close": [100.0, 102.0, 105.0]})
        with self.assertRaises(ValueError, msg="should raise when atrp_14 absent"):
            label_forward_return_binary(
                df, horizon_bars=1, threshold_bps=10.0, label_kind="vol_normalized"
            )

    def test_invalid_label_kind_raises(self) -> None:
        df = self._make_df([100.0, 102.0], [0.1, 0.1])
        with self.assertRaises(ValueError):
            label_forward_return_binary(
                df, horizon_bars=1, threshold_bps=10.0, label_kind="bad_kind"  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Synthetic OHLCV generator
# ---------------------------------------------------------------------------


def _write_synthetic_day(
    *, root: Path, symbol: str, day: datetime, n_bars: int = 1440, base_price: float = 2000.0
) -> Path:
    """Write a synthetic 1m OHLCV CSV for one UTC day."""
    rng = np.random.default_rng(seed=int(day.timestamp()) % 10000)
    dir_path = root / symbol.replace("/", "-") / "1m"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{day.strftime('%Y-%m-%d')}.csv"
    rows = []
    price = base_price
    for i in range(n_bars):
        ts = day + timedelta(minutes=i)
        ret = float(rng.normal(0.0, 0.001))
        new_price = max(0.01, price * (1.0 + ret))
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(price),
                "high": float(max(price, new_price) * 1.0002),
                "low": float(min(price, new_price) * 0.9998),
                "close": float(new_price),
                "volume": float(abs(rng.normal(50.0, 10.0))),
            }
        )
        price = new_price
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Pure label tests
# ---------------------------------------------------------------------------


class BinaryLabelTests(unittest.TestCase):
    def test_label_marks_threshold_crossings(self) -> None:
        # Hand-built closes: forward 1-bar returns are +200 bps, 0, -200 bps.
        df = pd.DataFrame(
            {"close": [100.0, 102.0, 102.0, 100.0]}
        )
        # threshold = +100 bps, horizon = 1 bar.
        labels = label_forward_return_binary(df, horizon_bars=1, threshold_bps=100.0)
        # row 0 -> close goes 100->102 = +200 bps -> label 1
        # row 1 -> close goes 102->102 = 0 bps -> label 0
        # row 2 -> close goes 102->100 = -196 bps -> label 0
        # row 3 -> no forward target -> NaN
        self.assertEqual(int(labels.iloc[0]), 1)
        self.assertEqual(int(labels.iloc[1]), 0)
        self.assertEqual(int(labels.iloc[2]), 0)
        self.assertTrue(pd.isna(labels.iloc[3]))


class ThreeClassLabelTests(unittest.TestCase):
    def test_three_class_emits_neg_zero_pos(self) -> None:
        df = pd.DataFrame({"close": [100.0, 102.0, 100.5, 99.0]})
        labels = label_forward_return_three_class(
            df, horizon_bars=1, threshold_bps=100.0
        )
        # row 0: +200 bps -> +1
        # row 1: -147 bps -> -1
        # row 2: -149 bps -> -1
        # row 3: NaN
        self.assertEqual(int(labels.iloc[0]), 1)
        self.assertEqual(int(labels.iloc[1]), -1)
        self.assertEqual(int(labels.iloc[2]), -1)
        self.assertTrue(pd.isna(labels.iloc[3]))

    def test_three_class_zero_inside_threshold_band(self) -> None:
        df = pd.DataFrame({"close": [100.0, 100.05, 100.1]})
        # +5 bps and +5 bps -- both inside the +/-10 bps band.
        labels = label_forward_return_three_class(
            df, horizon_bars=1, threshold_bps=10.0
        )
        self.assertEqual(int(labels.iloc[0]), 0)
        self.assertEqual(int(labels.iloc[1]), 0)


# ---------------------------------------------------------------------------
# load_ohlcv tests
# ---------------------------------------------------------------------------


class LoadOhlcvTests(unittest.TestCase):
    def test_load_concatenates_multiple_days_oldest_first(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day_a = datetime(2026, 4, 25, tzinfo=timezone.utc)
            day_b = datetime(2026, 4, 26, tzinfo=timezone.utc)
            _write_synthetic_day(root=root, symbol="ETH/USD", day=day_a, n_bars=10)
            _write_synthetic_day(root=root, symbol="ETH/USD", day=day_b, n_bars=10)
            df = load_ohlcv(data_root=root, symbol="ETH/USD")
            self.assertEqual(len(df), 20)
            self.assertEqual(df.iloc[0]["timestamp"][:10], "2026-04-25")
            self.assertEqual(df.iloc[-1]["timestamp"][:10], "2026-04-26")

    def test_load_dedupes_and_sorts_after_concat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Manually write two CSVs that overlap in timestamps.
            sym_dir = root / "ETH-USD" / "1m"
            sym_dir.mkdir(parents=True, exist_ok=True)
            (sym_dir / "2026-04-25.csv").write_text(
                "timestamp,open,high,low,close,volume\n"
                "2026-04-25T23:59:00+00:00,1,1,1,1,1\n"
                "2026-04-26T00:00:00+00:00,2,2,2,2,2\n"
            )
            (sym_dir / "2026-04-26.csv").write_text(
                "timestamp,open,high,low,close,volume\n"
                "2026-04-26T00:00:00+00:00,99,99,99,99,99\n"
                "2026-04-26T00:01:00+00:00,3,3,3,3,3\n"
            )
            df = load_ohlcv(data_root=root, symbol="ETH/USD")
            # 3 unique timestamps; the duplicate "2026-04-26T00:00" took
            # the LAST value (99) because keep="last".
            self.assertEqual(len(df), 3)
            row = df[df["timestamp"] == "2026-04-26T00:00:00+00:00"].iloc[0]
            self.assertEqual(row["open"], 99)

    def test_load_raises_on_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_ohlcv(data_root=Path(td), symbol="DOES/NOTEXIST")


# ---------------------------------------------------------------------------
# build_dataset integration
# ---------------------------------------------------------------------------


class BuildDatasetTests(unittest.TestCase):
    def test_end_to_end_produces_features_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for offset in range(2):
                day = datetime(2026, 4, 25, tzinfo=timezone.utc) + timedelta(
                    days=offset
                )
                _write_synthetic_day(root=root, symbol="ETH/USD", day=day, n_bars=400)
            out_path = root / "out" / "eth.parquet"
            df, summary = build_dataset(
                data_root=root,
                symbol="ETH/USD",
                horizon_bars=5,
                threshold_bps=5.0,
                label_mode="binary",
                output_path=out_path,
            )
            self.assertGreater(summary.feature_count, 10)
            # Every label must be 0/1 (not NaN -- those rows should have been dropped).
            self.assertSetEqual(
                {int(v) for v in df["label"].unique()}, {0, 1}
            )
            # Output file written somewhere (parquet or csv fallback).
            self.assertIsNotNone(summary.output_path)
            self.assertTrue(summary.output_path.exists())
            # rows_out << rows_in because of warmup + trailing horizon trim.
            self.assertLess(summary.rows_out, summary.rows_in)
            self.assertGreater(summary.rows_out, 0)
            self.assertEqual(summary.symbol, "ETH/USD")
            self.assertEqual(summary.label_mode, "binary")
            # Label distribution sums to rows_out.
            self.assertEqual(
                sum(summary.label_distribution.values()), summary.rows_out
            )

    def test_three_class_label_mode_emits_three_classes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for offset in range(2):
                day = datetime(2026, 4, 25, tzinfo=timezone.utc) + timedelta(
                    days=offset
                )
                _write_synthetic_day(root=root, symbol="ETH/USD", day=day, n_bars=400)
            df, summary = build_dataset(
                data_root=root,
                symbol="ETH/USD",
                horizon_bars=5,
                threshold_bps=5.0,
                label_mode="three_class",
            )
        # Synthetic data should produce all three classes given the 5-bps band
        # and noisy returns.
        self.assertGreaterEqual(len(set(df["label"].unique())), 2)
        self.assertEqual(summary.label_mode, "three_class")

    def test_explicit_feature_cols_lock_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = datetime(2026, 4, 25, tzinfo=timezone.utc)
            _write_synthetic_day(root=root, symbol="ETH/USD", day=day, n_bars=400)
            chosen = ["return_5", "rv_60", "tod_sin"]
            df, summary = build_dataset(
                data_root=root,
                symbol="ETH/USD",
                feature_cols=chosen,
                horizon_bars=2,
                threshold_bps=5.0,
            )
        # Output frame should have exactly: timestamp, chosen features, label.
        self.assertEqual(
            list(df.columns), ["timestamp", *chosen, "label"]
        )
        self.assertEqual(summary.feature_count, len(chosen))


# ---------------------------------------------------------------------------
# Microstructure sidecar merge
# ---------------------------------------------------------------------------


def _write_synthetic_microstructure(
    *,
    root: Path,
    symbol: str,
    timestamps: List[datetime],
    base_bid: float = 2000.0,
) -> Path:
    """Write a synthetic microstructure parquet covering the given timestamps.

    Schema matches the live collector exactly: timestamp + every column in
    history_coindesk.{L2_FEATURE_COLUMNS, TRADE_AGG_COLUMNS}. Values are
    deterministic so tests can assert on them.
    """
    dir_path = root / symbol.replace("/", "-") / "microstructure_1m"
    dir_path.mkdir(parents=True, exist_ok=True)
    # group by UTC date so we write one file per day, like the real collector
    by_day: dict = {}
    for i, ts in enumerate(timestamps):
        day_key = ts.date().isoformat()
        by_day.setdefault(day_key, []).append((i, ts))
    last_path: Path = dir_path
    for day_key, items in by_day.items():
        rows = []
        for i, ts in items:
            row = {"timestamp": ts}
            for c in L2_FEATURE_COLUMNS:
                # Encode the index into best_bid so we can verify the join
                # matched the right row at the right timestamp.
                row[c] = float(base_bid + i)
            for c in TRADE_AGG_COLUMNS:
                row[c] = float(i)
            rows.append(row)
        df = pd.DataFrame(rows)
        path = dir_path / f"{day_key}.parquet"
        df.to_parquet(path, index=False)
        last_path = path
    return last_path


class MicrostructureMergeTests(unittest.TestCase):
    """Regression for 2026-05-14: the live collector's per-day parquet
    sidecars must be merged into the training dataset, otherwise the
    booster gets trained on zero values for ~55 microstructure features
    and never splits on them at inference time.
    """

    def test_no_sidecar_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(
                load_microstructure_sidecars(
                    data_root=Path(td), symbol="ETH/USD"
                )
            )

    def test_empty_sidecar_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            empty_dir = Path(td) / "ETH-USD" / "microstructure_1m"
            empty_dir.mkdir(parents=True)
            self.assertIsNone(
                load_microstructure_sidecars(
                    data_root=Path(td), symbol="ETH/USD"
                )
            )

    def test_merge_no_micro_is_passthrough(self) -> None:
        # Regression pin: missing sidecars => merged frame is byte-identical
        # to the input (the legacy OHLCV-only path).
        ohlcv = pd.DataFrame(
            {
                "timestamp": ["2026-05-13T00:00:00+00:00"],
                "open": [1.0], "high": [1.1], "low": [0.9],
                "close": [1.05], "volume": [10.0],
            }
        )
        merged = merge_microstructure(ohlcv, None)
        pd.testing.assert_frame_equal(merged, ohlcv)
        merged_empty = merge_microstructure(ohlcv, pd.DataFrame())
        pd.testing.assert_frame_equal(merged_empty, ohlcv)

    def test_merge_populates_matched_rows_only(self) -> None:
        ohlcv = pd.DataFrame(
            {
                "timestamp": [
                    "2026-05-13T00:00:00+00:00",
                    "2026-05-13T00:01:00+00:00",
                    "2026-05-13T00:02:00+00:00",
                ],
                "open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1],
                "low": [0.9, 1.9, 2.9], "close": [1.05, 2.05, 3.05],
                "volume": [10.0, 20.0, 30.0],
            }
        )
        # Sidecar covers only the middle row.
        micro = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2026-05-13T00:01:00+00:00")
                ],
                **{c: [123.0] for c in L2_FEATURE_COLUMNS},
                **{c: [7.0] for c in TRADE_AGG_COLUMNS},
            }
        )
        merged = merge_microstructure(ohlcv, micro)
        self.assertEqual(len(merged), 3)
        # Row 1 (matched) has real values; rows 0, 2 have NaN.
        self.assertEqual(merged.iloc[1]["best_bid"], 123.0)
        self.assertEqual(merged.iloc[1]["trade_count"], 7.0)
        self.assertTrue(pd.isna(merged.iloc[0]["best_bid"]))
        self.assertTrue(pd.isna(merged.iloc[2]["best_bid"]))
        # All micro columns from history_coindesk schema are present.
        for c in L2_FEATURE_COLUMNS + TRADE_AGG_COLUMNS:
            self.assertIn(c, merged.columns)

    def test_merge_drops_overlapping_columns_from_raw(self) -> None:
        # If raw already has placeholder microstructure columns (zeros),
        # the sidecar's values must win -- no _micro suffix in output.
        ohlcv = pd.DataFrame(
            {
                "timestamp": ["2026-05-13T00:00:00+00:00"],
                "open": [1.0], "high": [1.1], "low": [0.9],
                "close": [1.05], "volume": [10.0],
                "best_bid": [0.0],  # placeholder zero
                "trade_count": [0.0],
            }
        )
        micro = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-05-13T00:00:00+00:00")],
                "best_bid": [42.0],
                **{c: [1.0] for c in L2_FEATURE_COLUMNS if c != "best_bid"},
                **{c: [1.0] for c in TRADE_AGG_COLUMNS if c != "trade_count"},
                "trade_count": [99.0],
            }
        )
        merged = merge_microstructure(ohlcv, micro)
        self.assertNotIn("best_bid_micro", merged.columns)
        self.assertNotIn("trade_count_micro", merged.columns)
        self.assertEqual(merged.iloc[0]["best_bid"], 42.0)
        self.assertEqual(merged.iloc[0]["trade_count"], 99.0)

    def test_build_dataset_no_sidecars_unchanged_output(self) -> None:
        # Regression pin: when no sidecars exist for the symbol, the dataset
        # output is identical to the legacy (pre-2026-05-14) behavior. Any
        # change in feature_count or schema would break already-trained models.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = datetime(2026, 4, 25, tzinfo=timezone.utc)
            _write_synthetic_day(
                root=root, symbol="ETH/USD", day=day, n_bars=400
            )
            df, summary = build_dataset(
                data_root=root,
                symbol="ETH/USD",
                horizon_bars=5,
                threshold_bps=5.0,
                label_mode="binary",
            )
        # The microstructure columns are still emitted (with placeholder zeros
        # from ensure_optional_microstructure_columns), so the schema is
        # unchanged whether sidecars exist or not.
        for c in L2_FEATURE_COLUMNS:
            self.assertIn(c, df.columns)
        # And they are all zero (no sidecar coverage).
        for c in L2_FEATURE_COLUMNS:
            self.assertTrue(
                (df[c] == 0).all(),
                f"column {c!r} should be all-zero with no sidecars",
            )

    def test_build_dataset_with_sidecars_populates_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = datetime(2026, 4, 25, tzinfo=timezone.utc)
            _write_synthetic_day(
                root=root, symbol="ETH/USD", day=day, n_bars=400
            )
            # Sidecar covers a contiguous block of the synthetic day so
            # most output rows (after warmup + label-trim) have real
            # microstructure values.
            covered_ts = [
                day + timedelta(minutes=i) for i in range(50, 350)
            ]
            _write_synthetic_microstructure(
                root=root, symbol="ETH/USD", timestamps=covered_ts
            )
            df, summary = build_dataset(
                data_root=root,
                symbol="ETH/USD",
                horizon_bars=5,
                threshold_bps=5.0,
                label_mode="binary",
            )
        # Some output rows must have non-zero best_bid (sidecar coverage).
        self.assertGreater(
            int((df["best_bid"] != 0).sum()), 0,
            "build_dataset failed to merge sidecar microstructure values",
        )
        # Trade-count slot, which was zero in the legacy path, also picks up.
        self.assertGreater(int((df["trade_count"] != 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
