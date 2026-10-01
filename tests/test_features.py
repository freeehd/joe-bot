import unittest

import pandas as pd

from features.feature_engine import create_features


class FeatureEngineTests(unittest.TestCase):
    def test_session_aware_returns_do_not_bridge_overnight(self):
        index = pd.to_datetime([
            "2026-01-02 15:58",
            "2026-01-02 15:59",
            "2026-01-05 09:30",
            "2026-01-05 09:31",
        ])
        df = pd.DataFrame(
            {
                "open": [100, 100, 110, 110],
                "high": [101, 101, 111, 111],
                "low": [99, 99, 109, 109],
                "close": [100, 100.1, 110, 110.1],
                "volume": [1000, 1200, 900, 1000],
                "vwap": [100, 100, 110, 110],
            },
            index=index,
        )
        features = create_features(df, session_aware=True)
        self.assertTrue(pd.isna(features.loc[index[2], "return_1m"]))


if __name__ == "__main__":
    unittest.main()
