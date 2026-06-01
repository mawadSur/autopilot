"""Tests for src/confluence_filters.py.

Hermetic, no model load needed -- gates are pure functions over a row
dict / pandas Series. We exercise each gate with a positive case
(should pass), a negative case (should reject), the gate_all
composition, and the vectorised helpers.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from confluence_filters import (
    ATR_PERCENTILE_CAPS,
    gate_all,
    gate_atr_not_extreme,
    gate_spread_ok,
    gate_trend_align,
    gate_volume_above_ma,
    gate_volume_above_ma_proxy,
    vectorised_atr_not_extreme,
    vectorised_spread_ok,
    vectorised_trend_align,
    vectorised_volume_above_ma,
    vectorised_volume_above_ma_proxy,
)


def _row(**overrides) -> dict:
    """Build a fully-specified 'row' (dict subscriptable). Defaults are
    neutral / passable so a test only needs to override what it cares
    about."""
    base = {
        "volume_quote": 100.0,
        "vol_ma_20": 50.0,
        "vol_log": math.log1p(100.0),
        "atrp_14": 0.0005,
        "close_over_ema_50": 0.001,
        "spread_pct": 0.0001,  # 1 bp
    }
    base.update(overrides)
    return base


class TestVolumeAboveMa(unittest.TestCase):
    def test_passes_when_volume_quote_above_multiplier_x_ma(self) -> None:
        # vq=100, ma=50, multiplier=1.5 -> threshold=75, 100>=75 -> True
        self.assertTrue(gate_volume_above_ma(_row(volume_quote=100, vol_ma_20=50), multiplier=1.5))

    def test_rejects_when_volume_quote_below_threshold(self) -> None:
        self.assertFalse(gate_volume_above_ma(_row(volume_quote=70, vol_ma_20=50), multiplier=1.5))

    def test_rejects_when_vol_ma_20_is_zero(self) -> None:
        # Division-by-zero guard: if rolling MA is 0 we can't compare meaningfully.
        self.assertFalse(gate_volume_above_ma(_row(volume_quote=100, vol_ma_20=0)))

    def test_rejects_when_volume_quote_is_zero_real_world_case(self) -> None:
        # The current dataset has volume_quote == 0 everywhere; gate must
        # reject so the probe correctly surfaces the degeneracy.
        self.assertFalse(gate_volume_above_ma(_row(volume_quote=0, vol_ma_20=50)))


class TestVolumeAboveMaProxy(unittest.TestCase):
    def test_passes_when_expm1_vol_log_above_threshold(self) -> None:
        # vol_log=log1p(100)=ln(101), so expm1 -> 100. ma=50, mult=1.5 -> threshold=75.
        self.assertTrue(
            gate_volume_above_ma_proxy(
                _row(vol_log=math.log1p(100.0), vol_ma_20=50.0), multiplier=1.5
            )
        )

    def test_rejects_when_proxy_volume_below_threshold(self) -> None:
        # vol_log=log1p(40)=ln(41), expm1 -> 40. ma=50, mult=1.5 -> threshold=75. 40<75.
        self.assertFalse(
            gate_volume_above_ma_proxy(
                _row(vol_log=math.log1p(40.0), vol_ma_20=50.0), multiplier=1.5
            )
        )


class TestAtrNotExtreme(unittest.TestCase):
    def test_passes_under_cap(self) -> None:
        self.assertTrue(gate_atr_not_extreme(_row(atrp_14=0.0005), cap=0.0010))

    def test_rejects_over_cap(self) -> None:
        self.assertFalse(gate_atr_not_extreme(_row(atrp_14=0.0020), cap=0.0010))

    def test_fail_open_when_no_cap_configured(self) -> None:
        # No cap arg, no symbol-cap configured -> True (so we can tell it's
        # uncalibrated, rather than silently rejecting everything).
        self.assertTrue(gate_atr_not_extreme(_row(atrp_14=0.05)))

    def test_uses_per_symbol_cap_when_present(self) -> None:
        ATR_PERCENTILE_CAPS["TEST/USD"] = {80: 0.0007}
        try:
            self.assertTrue(
                gate_atr_not_extreme(_row(atrp_14=0.0005), symbol="TEST/USD", percentile=80)
            )
            self.assertFalse(
                gate_atr_not_extreme(_row(atrp_14=0.0010), symbol="TEST/USD", percentile=80)
            )
        finally:
            ATR_PERCENTILE_CAPS.pop("TEST/USD", None)


class TestTrendAlign(unittest.TestCase):
    def test_passes_when_price_above_ema50(self) -> None:
        self.assertTrue(gate_trend_align(_row(close_over_ema_50=0.001)))

    def test_rejects_when_price_below_ema50(self) -> None:
        self.assertFalse(gate_trend_align(_row(close_over_ema_50=-0.001)))

    def test_rejects_when_exactly_at_ema50(self) -> None:
        # Strict > 0 to bias slightly with-trend rather than dead-flat.
        self.assertFalse(gate_trend_align(_row(close_over_ema_50=0.0)))


class TestSpreadOk(unittest.TestCase):
    def test_passes_when_spread_under_2bps(self) -> None:
        # spread_pct=0.0001 -> 1 bp -> <= 2bps -> True
        self.assertTrue(gate_spread_ok(_row(spread_pct=0.0001), max_spread_bps=2.0))

    def test_rejects_when_spread_above_2bps(self) -> None:
        # spread_pct=0.0005 -> 5 bp -> > 2bps -> False
        self.assertFalse(gate_spread_ok(_row(spread_pct=0.0005), max_spread_bps=2.0))

    def test_passes_when_spread_is_zero_dataset_realistic(self) -> None:
        # Current dataset has spread_pct=0 everywhere; trivially passes.
        self.assertTrue(gate_spread_ok(_row(spread_pct=0.0)))


class TestGateAll(unittest.TestCase):
    def test_all_pass_returns_true(self) -> None:
        row = _row()  # all defaults pass
        filters = [
            lambda r: gate_volume_above_ma_proxy(r, multiplier=1.5),
            lambda r: gate_atr_not_extreme(r, cap=0.001),
            gate_trend_align,
            gate_spread_ok,
        ]
        self.assertTrue(gate_all(row, filters))

    def test_one_fails_returns_false(self) -> None:
        row = _row(close_over_ema_50=-0.005)  # trend gate will reject
        filters = [
            lambda r: gate_volume_above_ma_proxy(r, multiplier=1.5),
            lambda r: gate_atr_not_extreme(r, cap=0.001),
            gate_trend_align,
            gate_spread_ok,
        ]
        self.assertFalse(gate_all(row, filters))

    def test_empty_filters_returns_true(self) -> None:
        self.assertTrue(gate_all(_row(), []))

    def test_short_circuits_on_raising_filter(self) -> None:
        def bad(_r):
            raise RuntimeError("kaboom")

        # A raising gate should be treated as a REJECT, not propagate, so
        # the supervisor can never crash because one bad gate.
        self.assertFalse(gate_all(_row(), [bad, gate_trend_align]))


class TestPandasSeriesCompat(unittest.TestCase):
    """Gates must accept pandas.Series (the real production row type)."""

    def test_pandas_series_works(self) -> None:
        s = pd.Series(_row())
        self.assertTrue(gate_trend_align(s))
        self.assertTrue(gate_atr_not_extreme(s, cap=0.001))

    def test_missing_column_rejects_safely(self) -> None:
        s = pd.Series({"atrp_14": 0.0005})  # no close_over_ema_50
        self.assertFalse(gate_trend_align(s))


class TestVectorisedHelpers(unittest.TestCase):
    """Vectorised versions must agree with scalar gates row-by-row."""

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # passes everything (proxy variant)
                _row(),
                # too-quiet bar: proxy vol fails
                _row(vol_log=math.log1p(10.0)),
                # extreme ATR: atr gate fails
                _row(atrp_14=0.005),
                # downtrend: trend gate fails
                _row(close_over_ema_50=-0.002),
                # wide spread: spread gate fails
                _row(spread_pct=0.0005),
            ]
        )

    def test_vectorised_volume_proxy_matches_scalar(self) -> None:
        df = self._df()
        vec = vectorised_volume_above_ma_proxy(df, multiplier=1.5)
        scalar = np.array([gate_volume_above_ma_proxy(r, 1.5) for _, r in df.iterrows()])
        np.testing.assert_array_equal(vec, scalar)

    def test_vectorised_atr_matches_scalar(self) -> None:
        df = self._df()
        vec = vectorised_atr_not_extreme(df, cap=0.001)
        scalar = np.array([gate_atr_not_extreme(r, cap=0.001) for _, r in df.iterrows()])
        np.testing.assert_array_equal(vec, scalar)

    def test_vectorised_trend_matches_scalar(self) -> None:
        df = self._df()
        vec = vectorised_trend_align(df)
        scalar = np.array([gate_trend_align(r) for _, r in df.iterrows()])
        np.testing.assert_array_equal(vec, scalar)

    def test_vectorised_spread_matches_scalar(self) -> None:
        df = self._df()
        vec = vectorised_spread_ok(df, max_spread_bps=2.0)
        scalar = np.array([gate_spread_ok(r, max_spread_bps=2.0) for _, r in df.iterrows()])
        np.testing.assert_array_equal(vec, scalar)

    def test_vectorised_volume_spec_literal_all_false_on_zero_quote(self) -> None:
        # Real-world: dataset has volume_quote=0. Spec-literal gate must
        # reject every row.
        df = pd.DataFrame([_row(volume_quote=0.0), _row(volume_quote=0.0)])
        vec = vectorised_volume_above_ma(df, multiplier=1.5)
        self.assertFalse(vec.any())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
