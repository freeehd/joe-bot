import unittest

import numpy as np
import pandas as pd

from backtest.engine import ExecutionConfig, TradeConfig, simulate_trade
from backtest.metrics import trades_to_frame
from models.exit_model import ExitContinuationModel
from research.alpha_decay import excursion_report
from research.exit_intelligence import build_exit_intelligence_gate, build_exit_state_dataset


class AlphaDecayExitTests(unittest.TestCase):
    def _trade_and_bars(self):
        idx = pd.date_range("2026-01-05 14:30", periods=15, freq="min", tz="UTC")
        close = np.array([100,100.1,100.2,100.3,100.4,100.5,100.45,100.4,100.35,100.3,100.25,100.2,100.15,100.1,100.05])
        bars = pd.DataFrame({"open": close, "high": close+.12, "low": close-.12, "close": close}, index=idx)
        trade = simulate_trade(
            bars, symbol="AAA", signal_time=idx[0], side="LONG", confidence=.8,
            p_wait=.1,p_long=.8,p_short=.1,
            trade_config=TradeConfig(target_pct=.05, stop_pct=.05, max_holding_bars=10),
            execution_config=ExecutionConfig(spread_bps=0, slippage_bps=0),
        )
        return trade, bars

    def test_excursion_report_uses_new_mfe_mae_telemetry(self):
        trade, _ = self._trade_and_bars()
        report = excursion_report(trades_to_frame([trade]))
        self.assertGreater(report["avg_mfe_bps"], 0)
        self.assertLess(report["avg_mae_bps"], 0)

    def test_exit_state_dataset_and_model_contract(self):
        # Build multiple shifted trades so chronological fitting sees both labels.
        frames=[]; trades=[]
        for day in range(1,13):
            idx = pd.date_range(f"2026-01-{day:02d} 14:30", periods=15, freq="min", tz="UTC")
            direction = 1 if day % 2 else -1
            shape = np.r_[np.linspace(0,.6,6), np.linspace(.5,-.2,9)] * direction
            close=100+shape
            bars=pd.DataFrame({"open":close,"high":close+.12,"low":close-.12,"close":close},index=idx)
            trade=simulate_trade(bars,symbol=f"S{day}",signal_time=idx[0],side="LONG" if direction==1 else "SHORT",confidence=.75,p_wait=.15,p_long=.75 if direction==1 else .1,p_short=.1 if direction==1 else .75,trade_config=TradeConfig(target_pct=.05,stop_pct=.05,max_holding_bars=10),execution_config=ExecutionConfig(spread_bps=0,slippage_bps=0))
            trades.append(trade); frames.append((f"S{day}",bars))
        lookup=dict(frames)
        ds=build_exit_state_dataset(trades_to_frame(trades),raw_loader=lambda s:lookup[s],execution_config=ExecutionConfig(spread_bps=0,slippage_bps=0),max_holding_bars=10)
        self.assertFalse(ds.empty)
        # Synthetic labels can be imbalanced; flip a few to assert model plumbing rather than economics.
        ds.loc[ds.index[::5],"continue_label"]=1-ds.loc[ds.index[::5],"continue_label"]
        n=len(ds); train=ds.iloc[:int(n*.65)]; cal=ds.iloc[int(n*.65):int(n*.82)]; test=ds.iloc[int(n*.82):]
        model=ExitContinuationModel().fit(train,cal)
        metrics=model.evaluate(test)
        self.assertIn("brier",metrics)

    def test_exit_gate_refuses_non_additive_model(self):
        gate=build_exit_intelligence_gate(baseline_metrics={"expectancy_bps":5,"max_drawdown":-.05},adaptive_metrics={"expectancy_bps":4,"max_drawdown":-.04},classifier_metrics={"roc_auc":.7})
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
