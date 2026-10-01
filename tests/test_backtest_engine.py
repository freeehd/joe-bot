import unittest

import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig, run_signal_backtest, simulate_trade
from backtest.metrics import summarize_trades


def bars(rows, start="2026-01-05 14:30Z"):
    index = pd.date_range(start, periods=len(rows), freq="min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


class BacktestEngineTests(unittest.TestCase):
    def test_signal_enters_next_bar_and_hits_long_target(self):
        raw = bars([
            [100.0, 100.1, 99.9, 100.0],
            [100.0, 100.4, 99.95, 100.3],
            [100.3, 100.5, 100.2, 100.4],
        ])
        result = simulate_trade(
            raw,
            symbol="AAA",
            signal_time=raw.index[0],
            side="LONG",
            confidence=0.8,
            p_wait=0.1,
            p_long=0.8,
            p_short=0.1,
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.002, max_holding_bars=2),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0, fee_bps=0),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.entry_time, raw.index[1])
        self.assertEqual(result.exit_reason, "TARGET")
        self.assertGreater(result.net_return, 0)

    def test_same_bar_ambiguity_resolves_to_stop(self):
        raw = bars([
            [100.0, 100.1, 99.9, 100.0],
            [100.0, 100.5, 99.5, 100.1],
        ])
        result = simulate_trade(
            raw,
            symbol="AAA",
            signal_time=raw.index[0],
            side="LONG",
            confidence=0.8,
            p_wait=0.1,
            p_long=0.8,
            p_short=0.1,
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.003, max_holding_bars=1),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
        )
        self.assertEqual(result.exit_reason, "STOP")
        self.assertLess(result.net_return, 0)

    def test_near_close_signal_does_not_enter_next_session(self):
        index = pd.to_datetime(["2026-01-05 20:59Z", "2026-01-06 14:30Z"])
        raw = pd.DataFrame(
            [[100, 100.1, 99.9, 100], [101, 101.2, 100.8, 101]],
            columns=["open", "high", "low", "close"],
            index=index,
        )
        result = simulate_trade(
            raw,
            symbol="AAA",
            signal_time=index[0],
            side="LONG",
            confidence=0.8,
            p_wait=0.1,
            p_long=0.8,
            p_short=0.1,
            trade_config=TradeConfig(),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
        )
        self.assertIsNone(result)

    def test_costs_reduce_returns(self):
        raw = bars([
            [100.0, 100.1, 99.9, 100.0],
            [100.0, 100.4, 99.95, 100.3],
        ])
        free = simulate_trade(
            raw,
            symbol="AAA",
            signal_time=raw.index[0],
            side="LONG",
            confidence=0.8,
            p_wait=0.1,
            p_long=0.8,
            p_short=0.1,
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.002, max_holding_bars=1),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
        )
        costly = simulate_trade(
            raw,
            symbol="AAA",
            signal_time=raw.index[0],
            side="LONG",
            confidence=0.8,
            p_wait=0.1,
            p_long=0.8,
            p_short=0.1,
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.002, max_holding_bars=1),
            execution_config=ExecutionConfig(spread_bps=10, slippage_bps=5, fee_bps=1),
        )
        self.assertLess(costly.net_return, free.net_return)

    def test_backtest_enforces_one_position_per_symbol(self):
        raw = bars([
            [100, 100.1, 99.9, 100],
            [100, 100.05, 99.95, 100],
            [100, 100.05, 99.95, 100],
            [100, 100.4, 99.95, 100.3],
            [100.3, 100.5, 100.2, 100.4],
        ])
        signals = pd.DataFrame(
            {
                "training_symbol": ["AAA", "AAA"],
                "p_wait": [0.1, 0.1],
                "p_long": [0.8, 0.8],
                "p_short": [0.1, 0.1],
                "direction": ["LONG", "LONG"],
                "confidence": [0.8, 0.8],
            },
            index=[raw.index[0], raw.index[1]],
        )
        trades = run_signal_backtest(
            signals,
            raw_loader=lambda symbol: raw,
            trade_config=TradeConfig(target_pct=0.003, stop_pct=0.01, max_holding_bars=3),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
            confidence_threshold=0.6,
        )
        self.assertEqual(len(trades), 1)
        summary = summarize_trades(trades)
        self.assertEqual(summary["total_trades"], 1)


if __name__ == "__main__":
    unittest.main()
