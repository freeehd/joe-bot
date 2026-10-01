import unittest

from backtest.engine import ExecutionConfig
from backtest.stress import execution_stress_scenarios


class ExecutionStressTests(unittest.TestCase):
    def test_stress_scenarios_are_never_more_optimistic_than_baseline_inputs(self):
        base = ExecutionConfig(spread_bps=4, slippage_bps=2, entry_delay_bars=1)
        scenarios = execution_stress_scenarios(base)
        self.assertIn("combined_adverse", scenarios)
        self.assertEqual(scenarios["baseline"], base)
        self.assertGreaterEqual(scenarios["double_spread"].spread_bps, base.spread_bps)
        self.assertGreaterEqual(scenarios["double_slippage"].slippage_bps, base.slippage_bps)
        self.assertGreater(scenarios["extra_delay"].entry_delay_bars, base.entry_delay_bars)
        combined = scenarios["combined_adverse"]
        self.assertGreaterEqual(combined.spread_bps, base.spread_bps)
        self.assertGreaterEqual(combined.slippage_bps, base.slippage_bps)
        self.assertGreater(combined.entry_delay_bars, base.entry_delay_bars)


if __name__ == "__main__":
    unittest.main()
