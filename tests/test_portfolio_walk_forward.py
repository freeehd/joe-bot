import unittest

import numpy as np
import pandas as pd

from backtest.engine import ExecutionConfig
from research.portfolio_walk_forward import run_portfolio_walk_forward
from research.walk_forward import WalkForwardConfig
from risk.portfolio_risk import CorrelationConfig
from strategy.ev_model import EVConfig
from strategy.portfolio_allocator import PortfolioConstraints


class _FakeLake:
    def __init__(self, raw_by_symbol):
        self.raw_by_symbol = raw_by_symbol

    def load_raw_bars(self, version, symbol):
        return self.raw_by_symbol[symbol]


class PortfolioWalkForwardTests(unittest.TestCase):
    def test_end_to_end_ev_fit_ranking_allocation_and_execution(self):
        symbols = ["AAA", "BBB", "CCC"]
        rows = []
        raw_by_symbol = {}
        business_days = pd.bdate_range("2025-01-01", "2025-06-30", tz="UTC")

        for symbol_index, symbol in enumerate(symbols):
            raw_rows = []
            raw_index = []
            for day_index, day in enumerate(business_days):
                signal_time = day + pd.Timedelta(hours=14, minutes=30)
                entry_time = signal_time + pd.Timedelta(minutes=1)
                feature = float(symbol_index) + np.sin(day_index / 20.0) * 0.01
                rows.append(
                    {
                        "timestamp": signal_time,
                        "training_symbol": symbol,
                        "feature_1": feature,
                        "return_1m": (symbol_index - 1) * 0.001 + np.sin(day_index / 10.0) * 0.0001,
                        "trade_label": symbol_index,
                    }
                )
                price = 100.0 + symbol_index
                if symbol == "BBB":  # LONG target only
                    second = [price, price + 0.50, price - 0.05, price + 0.30]
                elif symbol == "CCC":  # SHORT target only
                    second = [price, price + 0.05, price - 0.50, price - 0.30]
                else:
                    second = [price, price + 0.05, price - 0.05, price]
                raw_index.extend([signal_time, entry_time])
                raw_rows.extend(
                    [
                        [price, price + 0.05, price - 0.05, price],
                        second,
                    ]
                )
            raw_by_symbol[symbol] = pd.DataFrame(
                raw_rows,
                columns=["open", "high", "low", "close"],
                index=pd.DatetimeIndex(raw_index),
            )

        dataset = pd.DataFrame(rows).set_index("timestamp").sort_index()
        manifest = {
            "dataset_version": "synthetic-v07",
            "raw_snapshot_version": "synthetic-v07",
            "feature_engine": "v2",
            "label_parameters": {
                "horizon_bars": 1,
                "target_pct": 0.003,
                "stop_pct": 0.003,
                "use_atr": False,
                "atr_period": 14,
                "atr_target_multiplier": 1.0,
                "atr_stop_multiplier": 0.5,
            },
        }
        result = run_portfolio_walk_forward(
            dataset,
            manifest,
            ["feature_1"],
            lake=_FakeLake(raw_by_symbol),
            model_name="hist_gradient_boosting",
            walk_forward_config=WalkForwardConfig(
                train_months=2,
                calibration_months=1,
                test_months=1,
                step_months=1,
                confidence_threshold=0.0,
            ),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
            portfolio_constraints=PortfolioConstraints(
                max_positions=2,
                max_sector_deployed_fraction=0.8,
                max_correlated_risk_fraction=0.01,
                min_position_dollars=1.0,
            ),
            ev_config=EVConfig(min_samples_per_bucket=5, shrinkage_samples=10),
            correlation_config=CorrelationConfig(minimum_periods=10),
            initial_equity=10_000,
        )
        self.assertGreaterEqual(result["windows_completed"], 2)
        self.assertIn("portfolio_trade_metrics", result)
        self.assertIn("baseline_trade_metrics", result)
        self.assertIn("expectancy_improvement_bps", result)
        self.assertGreater(result["portfolio_trade_metrics"]["total_trades"], 0)
        self.assertGreaterEqual(result["portfolio_positive_return_window_rate"], 0.0)
        self.assertLessEqual(result["portfolio_positive_return_window_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
