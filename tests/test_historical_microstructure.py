import unittest
import pandas as pd
from market.historical import normalize_quote_frame, normalize_trade_frame


class HistoricalMicrostructureTests(unittest.TestCase):
    def test_quote_and_trade_normalizers_support_multiindex_provider_shape(self):
        idx = pd.MultiIndex.from_arrays([
            ["AAPL", "AAPL"],
            pd.to_datetime(["2026-01-05T14:30:00Z", "2026-01-05T14:30:01Z"]),
        ], names=["symbol", "timestamp"])
        quotes = pd.DataFrame({"bid_price":[100,100.01],"ask_price":[100.02,100.03],"bid_size":[10,12],"ask_size":[11,9]}, index=idx)
        trades = pd.DataFrame({"price":[100.01,100.02],"size":[5,8]}, index=idx)
        q=normalize_quote_frame(quotes); t=normalize_trade_frame(trades)
        self.assertEqual(list(q.columns[:5]), ["symbol","bid_price","ask_price","bid_size","ask_size"])
        self.assertEqual(list(t.columns[:3]), ["symbol","price","size"])
        self.assertIsNotNone(q.index.tz)
        self.assertEqual(t.iloc[0]["symbol"],"AAPL")


if __name__ == "__main__":
    unittest.main()
