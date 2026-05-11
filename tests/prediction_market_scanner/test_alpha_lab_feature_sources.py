"""Tests for alpha_lab.feature_sources.

Hermetic: the crypto source reads from a synthetic parquet in tmpdir; the
polymarket source uses an injected fetcher callable. No filesystem state
outside tmpdir, no network.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from unittest.mock import MagicMock, patch

from alpha_lab.correlation_miner import FeatureSource
from alpha_lab.feature_sources import (
    CryptoFeatureSource,
    PolymarketFeatureSource,
    build_default_feature_sources,
    make_clob_history_fetcher,
)


def _make_synthetic_crypto_parquet(path: Path, n_rows: int = 60) -> None:
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    timestamps = [
        (start + timedelta(minutes=i)).isoformat() for i in range(n_rows)
    ]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "return_1": np.linspace(-0.01, 0.01, n_rows),
            "return_5": np.linspace(-0.05, 0.05, n_rows),
            "ema_9": np.linspace(100.0, 110.0, n_rows),
            "label": np.zeros(n_rows),
        }
    )
    df.to_parquet(path)


class CryptoFeatureSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.parquet = Path(self.tmp.name) / "btc_usd_v1.parquet"
        _make_synthetic_crypto_parquet(self.parquet)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_construction_stores_parquet_path(self) -> None:
        src = CryptoFeatureSource("BTC-USD", parquet_path=self.parquet)
        self.assertEqual(src.symbol, "BTC-USD")
        self.assertEqual(src.parquet_path, self.parquet)
        self.assertEqual(src.name, "crypto:BTC-USD")

    def test_default_parquet_path_uses_snake_case(self) -> None:
        # Construct without a path to verify the default-path derivation.
        # The file isn't there (under cwd), so a warning fires but no exception.
        src = CryptoFeatureSource("BTC-USD")
        self.assertEqual(
            src.parquet_path, Path("data/crypto/datasets/btc_usd_v1.parquet")
        )

    def test_asset_class_is_spot_crypto(self) -> None:
        src = CryptoFeatureSource("BTC-USD", parquet_path=self.parquet)
        ac = src.asset_class
        # Either the enum value or the string fallback; both compare to the
        # snake_case label used by the miner.
        ac_str = ac.value if hasattr(ac, "value") else str(ac)
        self.assertEqual(ac_str, "spot_crypto")

    def test_implements_feature_source_protocol(self) -> None:
        src = CryptoFeatureSource("BTC-USD", parquet_path=self.parquet)
        self.assertIsInstance(src, FeatureSource)

    def test_fetch_window_returns_filtered_numeric_features(self) -> None:
        src = CryptoFeatureSource("BTC-USD", parquet_path=self.parquet)
        start = datetime(2026, 5, 1, 0, 10, tzinfo=timezone.utc)
        end = datetime(2026, 5, 1, 0, 30, tzinfo=timezone.utc)
        df = src.fetch_window(start, end)
        self.assertFalse(df.empty)
        # Window-filter retains only timestamps within [start, end].
        self.assertGreaterEqual(df.index.min(), pd.Timestamp(start))
        self.assertLessEqual(df.index.max(), pd.Timestamp(end))
        # Label column is dropped (it's in the blacklist).
        self.assertNotIn("label", df.columns)
        # timestamp column is dropped (it became the index).
        self.assertNotIn("timestamp", df.columns)
        # Real features survive.
        self.assertIn("return_1", df.columns)
        self.assertIn("ema_9", df.columns)

    def test_fetch_window_with_explicit_columns_filter(self) -> None:
        src = CryptoFeatureSource(
            "BTC-USD",
            parquet_path=self.parquet,
            feature_columns=["ema_9"],
        )
        df = src.fetch_window(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(list(df.columns), ["ema_9"])

    def test_missing_parquet_returns_empty(self) -> None:
        src = CryptoFeatureSource(
            "MISSING", parquet_path=Path(self.tmp.name) / "nope.parquet"
        )
        df = src.fetch_window(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        self.assertTrue(df.empty)

    def test_window_outside_data_returns_empty(self) -> None:
        src = CryptoFeatureSource("BTC-USD", parquet_path=self.parquet)
        df = src.fetch_window(
            datetime(2030, 1, 1, tzinfo=timezone.utc),
            datetime(2030, 1, 2, tzinfo=timezone.utc),
        )
        self.assertTrue(df.empty)

    def test_empty_symbol_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CryptoFeatureSource("")


class PolymarketFeatureSourceTests(unittest.TestCase):
    def test_construction_stores_market_id(self) -> None:
        src = PolymarketFeatureSource("0xabc")
        self.assertEqual(src.market_id, "0xabc")
        self.assertEqual(src.name, "polymarket:0xabc")

    def test_name_includes_question_when_provided(self) -> None:
        src = PolymarketFeatureSource(
            "0xabc",
            question="Will the Fed cut rates in June?",
        )
        self.assertIn("Will the Fed", src.name)

    def test_asset_class_is_prediction_binary(self) -> None:
        src = PolymarketFeatureSource("0xabc")
        ac = src.asset_class
        ac_str = ac.value if hasattr(ac, "value") else str(ac)
        self.assertEqual(ac_str, "prediction_binary")

    def test_implements_feature_source_protocol(self) -> None:
        src = PolymarketFeatureSource("0xabc")
        self.assertIsInstance(src, FeatureSource)

    def test_no_fetcher_returns_empty(self) -> None:
        src = PolymarketFeatureSource("0xabc")
        df = src.fetch_window(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        self.assertTrue(df.empty)

    def test_injected_fetcher_called_with_window(self) -> None:
        captured: List[tuple] = []

        def stub(market_id: str, start: datetime, end: datetime) -> pd.DataFrame:
            captured.append((market_id, start, end))
            idx = pd.DatetimeIndex(
                [start + timedelta(hours=i) for i in range(3)],
                tz=timezone.utc,
            )
            return pd.DataFrame({"midpoint": [0.4, 0.45, 0.5]}, index=idx)

        src = PolymarketFeatureSource("0xabc", fetcher=stub)
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = datetime(2026, 5, 2, tzinfo=timezone.utc)
        df = src.fetch_window(start, end)
        self.assertEqual(captured, [("0xabc", start, end)])
        self.assertEqual(list(df.columns), ["midpoint"])
        self.assertEqual(len(df), 3)

    def test_fetcher_raising_returns_empty(self) -> None:
        def boom(*_a, **_k):  # type: ignore[no-untyped-def]
            raise RuntimeError("polymarket down")

        src = PolymarketFeatureSource("0xabc", fetcher=boom)
        df = src.fetch_window(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        self.assertTrue(df.empty)

    def test_fetcher_returning_naive_index_is_localized_to_utc(self) -> None:
        def naive_fetcher(_market_id, start, _end):  # type: ignore[no-untyped-def]
            idx = pd.DatetimeIndex(
                [start.replace(tzinfo=None) + timedelta(hours=i) for i in range(2)]
            )
            return pd.DataFrame({"midpoint": [0.5, 0.55]}, index=idx)

        src = PolymarketFeatureSource("0xabc", fetcher=naive_fetcher)
        df = src.fetch_window(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        self.assertIsNotNone(df.index.tz)

    def test_fetcher_returning_none_returns_empty(self) -> None:
        src = PolymarketFeatureSource("0xabc", fetcher=lambda *a, **k: None)
        df = src.fetch_window(
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        self.assertTrue(df.empty)

    def test_empty_market_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolymarketFeatureSource("")


class FactoryTests(unittest.TestCase):
    def test_default_factory_returns_empty_in_skeleton(self) -> None:
        # Skeleton state: no production sources wired. Operators override
        # build_default_feature_sources via a deploy-only module.
        self.assertEqual(build_default_feature_sources(), [])


class ClobHistoryFetcherTests(unittest.TestCase):
    """Tests for ``make_clob_history_fetcher`` — the production wiring of
    ``PolymarketFeatureSource`` against the CLOB ``prices-history`` endpoint."""

    def _samples(self, start: datetime, n: int = 3):
        return [
            (start + timedelta(hours=i), 0.40 + 0.05 * i) for i in range(n)
        ]

    def test_fetcher_calls_upstream_and_returns_dataframe(self) -> None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        upstream = MagicMock(return_value=self._samples(start))
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            fetch = make_clob_history_fetcher()
            df = fetch("0xmarket", start, end)
        upstream.assert_called_once()
        self.assertEqual(list(df.columns), ["midpoint"])
        self.assertEqual(len(df), 3)
        self.assertIsNotNone(df.index.tz)

    def test_inmemory_cache_short_circuits_second_call(self) -> None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        upstream = MagicMock(return_value=self._samples(start))
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            fetch = make_clob_history_fetcher(cache_ttl_s=300)
            fetch("0xmarket", start, end)
            fetch("0xmarket", start, end)
        # Second call must hit the in-memory cache, not the upstream.
        self.assertEqual(upstream.call_count, 1)

    def test_distinct_windows_each_hit_upstream(self) -> None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        upstream = MagicMock(return_value=self._samples(start))
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            fetch = make_clob_history_fetcher()
            fetch("0xmarket", start, start + timedelta(days=1))
            fetch("0xmarket", start, start + timedelta(days=2))
        self.assertEqual(upstream.call_count, 2)

    def test_redis_cache_read_short_circuits_upstream(self) -> None:
        # When Redis has a cached payload, the fetcher should NOT call
        # upstream and must reconstruct the DataFrame from the cache.
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        cached_payload = {
            "samples": [
                [int((start + timedelta(hours=i)).timestamp()), 0.5 + 0.01 * i]
                for i in range(2)
            ]
        }
        fake_redis = MagicMock()
        import json as _json

        fake_redis.get.return_value = _json.dumps(cached_payload).encode("utf-8")
        upstream = MagicMock()
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            fetch = make_clob_history_fetcher(redis_client=fake_redis)
            df = fetch("0xmarket", start, end)
        upstream.assert_not_called()
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df["midpoint"].iloc[0]), 0.5)

    def test_redis_cache_write_after_upstream_fetch(self) -> None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        fake_redis = MagicMock()
        fake_redis.get.return_value = None  # cache miss
        upstream = MagicMock(return_value=self._samples(start, n=2))
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            fetch = make_clob_history_fetcher(redis_client=fake_redis)
            fetch("0xmarket", start, end)
        # Upstream called once; setex called with our key prefix.
        upstream.assert_called_once()
        fake_redis.setex.assert_called_once()
        key_arg = fake_redis.setex.call_args[0][0]
        self.assertTrue(key_arg.startswith("alpha_lab:polymarket_history:"))

    def test_redis_read_failure_falls_through(self) -> None:
        # If Redis is down on read, the fetcher must still produce results
        # via the upstream call (in-memory cache also picks up the result).
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        fake_redis = MagicMock()
        fake_redis.get.side_effect = RuntimeError("redis down")
        upstream = MagicMock(return_value=self._samples(start, n=1))
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            fetch = make_clob_history_fetcher(redis_client=fake_redis)
            df = fetch("0xmarket", start, end)
        upstream.assert_called_once()
        self.assertEqual(len(df), 1)

    def test_empty_upstream_yields_empty_dataframe(self) -> None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=MagicMock(return_value=[]),
        ):
            fetch = make_clob_history_fetcher()
            df = fetch("0xmarket", start, end)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["midpoint"])

    def test_integrates_with_polymarket_source(self) -> None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        upstream = MagicMock(return_value=self._samples(start))
        with patch(
            "alpha_lab.feature_sources.fetch_market_price_history",
            new=upstream,
        ):
            src = PolymarketFeatureSource(
                "0xabc", fetcher=make_clob_history_fetcher()
            )
            df = src.fetch_window(start, end)
        self.assertEqual(list(df.columns), ["midpoint"])
        self.assertEqual(len(df), 3)
        self.assertIsNotNone(df.index.tz)


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
