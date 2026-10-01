"""Offline event-driven backtesting primitives."""

from backtest.engine import ExecutionConfig, TradeConfig, TradeResult, run_signal_backtest, simulate_trade
from backtest.metrics import confidence_ev_buckets, summarize_trades, trades_to_frame
from backtest.signals import probability_frame
from backtest.stress import execution_stress_scenarios

__all__ = [
    "ExecutionConfig",
    "TradeConfig",
    "TradeResult",
    "run_signal_backtest",
    "simulate_trade",
    "confidence_ev_buckets",
    "summarize_trades",
    "trades_to_frame",
    "probability_frame",
    "execution_stress_scenarios",
]
