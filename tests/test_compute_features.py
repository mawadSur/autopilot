import unittest
import numpy as np
import pandas as pd

from utils import compute_features, FEATURE_COLUMNS


class FeatureColumnsTest(unittest.TestCase):
    def test_compute_features_matches_feature_columns(self):
        n = 300
        rng = np.random.default_rng(0)
        base = np.linspace(100.0, 101.0, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="T", tz="UTC"),
            "open": base,
            "high": base * 1.001,
            "low": base * 0.999,
            "close": base * 1.0005,
            "volume": rng.lognormal(mean=2.0, sigma=0.3, size=n),
        })

        engineered = compute_features(df)

        self.assertEqual(
            FEATURE_COLUMNS,
            list(engineered.columns),
            "compute_features must return columns exactly matching FEATURE_COLUMNS",
        )


if __name__ == "__main__":
    unittest.main()
