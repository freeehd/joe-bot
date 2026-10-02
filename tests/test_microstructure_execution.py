import tempfile
import unittest
import importlib.util
from pathlib import Path

import pandas as pd

from backtest.micro_execution import ExecutionOrder, ExecutionPolicy, simulate_execution
from features.microstructure import MICROSTRUCTURE_FEATURES, build_microstructure_features
from research.execution_policy_benchmark import build_execution_policy_gate
from research.microstructure_storage import MicrostructureStore


class MicrostructureExecutionTests(unittest.TestCase):
    def test_feature_builder_is_minute_causal_shape(self):
        idx = pd.date_range("2026-01-05 14:30:00", periods=120, freq="s", tz="UTC")
        quotes = pd.DataFrame({
            "bid_price": 100.0, "ask_price": 100.04,
            "bid_size": 100 + (pd.Series(range(120), index=idx) % 10),
            "ask_size": 90 + (pd.Series(range(120), index=idx) % 8),
        }, index=idx)
        trades = pd.DataFrame({"price": [100.01 + (i%3)*.01 for i in range(120)], "size": 20, "side": ["BUY","SELL"]*60}, index=idx)
        features = build_microstructure_features(quotes, trades)
        self.assertEqual(list(features.columns), MICROSTRUCTURE_FEATURES)
        self.assertGreaterEqual(len(features), 2)

    def test_marketable_limit_can_partial_fill_and_passive_can_miss(self):
        idx = pd.date_range("2026-01-05 14:30:00", periods=3, freq="s", tz="UTC")
        quotes = pd.DataFrame({"bid_price":[99.98]*3,"ask_price":[100.02,100.03,100.04],"bid_size":[10]*3,"ask_size":[10]*3},index=idx)
        order=ExecutionOrder("BUY",10,100.0,ExecutionPolicy.MARKETABLE_LIMIT,limit_price=100.03,max_participation=.2)
        fill=simulate_execution(order,quotes)
        self.assertTrue(fill.partial)
        self.assertEqual(fill.filled_quantity,4)
        trades=pd.DataFrame({"price":[100.05,100.06],"size":[10,10]},index=idx[:2])
        passive=simulate_execution(ExecutionOrder("BUY",5,100.0,ExecutionPolicy.PASSIVE_LIMIT,limit_price=99.99),quotes,trades)
        self.assertTrue(passive.missed)

    def test_execution_gate_requires_cost_and_fill_quality(self):
        market={"avg_cost_bps":4.0,"p95_cost_bps":7.0,"fill_rate":1.0}
        challenger={"avg_cost_bps":3.0,"p95_cost_bps":7.2,"fill_rate":.95}
        self.assertTrue(build_execution_policy_gate(market=market,challenger=challenger)["passed"])
        challenger["fill_rate"]=.5
        self.assertFalse(build_execution_policy_gate(market=market,challenger=challenger)["passed"])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow not installed in this sandbox")
    def test_microstructure_store_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MicrostructureStore(tmp)
            idx=pd.date_range("2026-01-05",periods=2,freq="s",tz="UTC")
            frame=pd.DataFrame({"bid_price":[1,1],"ask_price":[1.01,1.01],"bid_size":[10,10],"ask_size":[10,10]},index=idx)
            path=store.write_partition("v1","quotes","AAA","2026-01-05",frame)
            self.assertTrue(path.exists())
            with self.assertRaises(FileExistsError):
                store.write_partition("v1","quotes","AAA","2026-01-05",frame)


if __name__ == "__main__":
    unittest.main()
