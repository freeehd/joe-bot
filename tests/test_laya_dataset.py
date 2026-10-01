import tempfile
import unittest

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig
from research.laya_dataset import build_laya_examples, split_chronologically, write_laya_dataset
from strategy.ev_model import EmpiricalEVModel
from strategy.portfolio_allocator import EVRanker


class LayaDatasetTests(unittest.TestCase):
    def test_ground_truth_comes_from_realized_outcome_not_quant_direction(self):
        index = pd.date_range("2026-01-05 14:30Z", periods=3, freq="min")
        raw = pd.DataFrame(
            [
                [100.0, 100.1, 99.9, 100.0],
                [100.0, 100.2, 98.5, 99.0],
                [99.0, 99.1, 98.9, 99.0],
            ],
            columns=["open", "high", "low", "close"],
            index=index,
        )
        signals = pd.DataFrame(
            {
                "training_symbol": ["AAA"],
                "p_wait": [0.1],
                "p_long": [0.8],
                "p_short": [0.1],
                "direction": ["LONG"],
                "confidence": [0.8],
                "return_1m": [0.002],
            },
            index=[index[0]],
        )
        trade = TradeConfig(target_pct=0.01, stop_pct=0.01, max_holding_bars=2)
        execution = ExecutionConfig(spread_bps=0, slippage_bps=0, fee_bps=0)
        ev_model = EmpiricalEVModel(trade_config=trade, execution_config=execution)
        examples = build_laya_examples(
            signals,
            raw_loader=lambda symbol: raw,
            ranker=EVRanker(ev_model),
            trade_config=trade,
            execution_config=execution,
            top_candidates=1,
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["metadata"]["quant_direction"], "LONG")
        self.assertEqual(examples[0]["answers"]["action"], "SHORT")
        self.assertTrue(examples[0]["answers"]["risk_concern"])

    def test_chronological_split_and_writer(self):
        examples = []
        for i in range(20):
            examples.append({
                "state": {"symbol": "AAA"},
                "questions": {},
                "answers": {"action": "WAIT", "quality": "POOR", "risk_concern": True},
                "metadata": {"timestamp": f"2026-01-{i+1:02d}T00:00:00+00:00", "symbol": "AAA", "quant_direction": "LONG"},
            })
        splits = split_chronologically(examples)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 14, "validation": 3, "test": 3})
        self.assertLess(splits["train"][-1]["metadata"]["timestamp"], splits["validation"][0]["metadata"]["timestamp"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_laya_dataset(examples, directory)
            self.assertEqual(manifest["records"], 20)
            self.assertTrue((__import__("pathlib").Path(directory) / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
