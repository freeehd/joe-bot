import unittest
import pandas as pd

from market.dynamic_universe import UniverseConfig, apply_intraday_updates, select_morning_universe


class DynamicUniverseTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for i in range(60):
            rows.append({
                "symbol": f"S{i}", "price": 20+i, "avg_dollar_volume": 30_000_000+i*1_000_000,
                "avg_volume": 1_000_000+i*10_000, "spread_bps": 5+i*0.1,
                "premarket_activity": i/60, "relative_volume": 1+i/100,
                "volatility": .01+i/10000, "mover_score": i/60, "event_score": (i%10)/10,
            })
        return pd.DataFrame(rows)

    def test_morning_selector_is_ranked_and_tradeable(self):
        result=select_morning_universe(self.frame(), UniverseConfig(target_size=50,min_size=50))
        self.assertEqual(len(result),50)
        self.assertTrue((result["spread_bps"] <= 30).all())
        self.assertEqual(result.iloc[0]["universe_rank"],1)

    def test_intraday_add_remove(self):
        stats=pd.DataFrame([
            {"symbol":"NEW","relative_volume":3,"volatility":.01,"event_score":0,"spread_bps":10,"intraday_dollar_volume":3_000_000},
            {"symbol":"OLD","relative_volume":1,"volatility":.01,"event_score":0,"spread_bps":90,"intraday_dollar_volume":3_000_000},
        ])
        result=apply_intraday_updates(["OLD"],stats)
        self.assertEqual(result["add"][0]["symbol"],"NEW")
        self.assertEqual(result["remove"][0]["symbol"],"OLD")


if __name__ == "__main__":
    unittest.main()
