"""Exit-state dataset construction and Sprint 6 exit-policy gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from backtest.engine import ExecutionConfig, _execution_price


def build_exit_state_dataset(
    trades: pd.DataFrame,
    *,
    raw_loader: Callable[[str], pd.DataFrame],
    execution_config: ExecutionConfig | None = None,
    max_holding_bars: int = 20,
    min_incremental_edge_bps: float = 1.0,
) -> pd.DataFrame:
    execution = execution_config or ExecutionConfig()
    rows = []
    for trade in trades.itertuples():
        bars = raw_loader(str(trade.symbol)).sort_index().copy()
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")
        entry_time = pd.Timestamp(trade.entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        matches = np.flatnonzero(bars.index == entry_time)
        if len(matches) != 1:
            continue
        entry_pos = int(matches[0])
        entry = float(trade.entry_price)
        side = str(trade.side)
        action = "SELL" if side == "LONG" else "BUY"
        target = float(trade.target_price)
        stop = float(trade.stop_price)
        highs, lows = [], []
        path_returns = []
        path_times = []
        for holding in range(1, max_holding_bars + 1):
            pos = entry_pos + holding - 1
            if pos >= len(bars):
                break
            bar = bars.iloc[pos]
            highs.append(float(bar.high)); lows.append(float(bar.low))
            exit_price = _execution_price(float(bar.close), action=action, config=execution)
            current = exit_price / entry - 1.0 if side == "LONG" else (entry - exit_price) / entry
            path_returns.append(current)
            path_times.append((holding, pos, current))
        if len(path_times) < 2:
            continue
        future_best = np.maximum.accumulate(np.asarray(path_returns[::-1]))[::-1]
        for j, (holding, pos, current) in enumerate(path_times[:-1]):
            if side == "LONG":
                mfe = max(highs[: j + 1]) / entry - 1.0
                mae = min(lows[: j + 1]) / entry - 1.0
                dist_target = (target - float(bars.iloc[pos].close)) / entry
                dist_stop = (float(bars.iloc[pos].close) - stop) / entry
                directional_probability = float(trade.p_long)
            else:
                mfe = (entry - min(lows[: j + 1])) / entry
                mae = (entry - max(highs[: j + 1])) / entry
                dist_target = (float(bars.iloc[pos].close) - target) / entry
                dist_stop = (stop - float(bars.iloc[pos].close)) / entry
                directional_probability = float(trade.p_short)
            incremental = float(future_best[j + 1] - current)
            rows.append({
                "symbol": str(trade.symbol),
                "timestamp": bars.index[pos],
                "current_return": float(current),
                "mfe_return": float(mfe),
                "mae_return": float(mae),
                "giveback_from_mfe": float(mfe - current),
                "holding_fraction": float(holding / max_holding_bars),
                "distance_to_target": float(dist_target),
                "distance_to_stop": float(dist_stop),
                "signal_confidence": float(trade.confidence),
                "directional_probability": directional_probability,
                "p_wait": float(trade.p_wait),
                "incremental_future_edge_bps": incremental * 10_000.0,
                "continue_label": int(incremental * 10_000.0 > min_incremental_edge_bps),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


@dataclass(frozen=True)
class ExitIntelligenceThresholds:
    min_expectancy_improvement_bps: float = 0.0
    max_drawdown_degradation: float = 0.0
    min_classifier_auc: float = 0.55


def build_exit_intelligence_gate(
    *,
    baseline_metrics: dict,
    adaptive_metrics: dict,
    classifier_metrics: dict,
    thresholds: ExitIntelligenceThresholds | None = None,
) -> dict:
    thresholds = thresholds or ExitIntelligenceThresholds()
    ev_delta = float(adaptive_metrics.get("expectancy_bps", 0.0) - baseline_metrics.get("expectancy_bps", 0.0))
    dd_delta = float(adaptive_metrics.get("max_drawdown", 0.0) - baseline_metrics.get("max_drawdown", 0.0))
    auc = float(classifier_metrics.get("roc_auc", float("nan")))
    gates = [
        {"name": "adaptive_exit_adds_expectancy", "passed": ev_delta > thresholds.min_expectancy_improvement_bps, "value": ev_delta, "threshold": f">{thresholds.min_expectancy_improvement_bps}"},
        {"name": "adaptive_exit_drawdown_not_worse", "passed": dd_delta >= -thresholds.max_drawdown_degradation, "value": dd_delta, "threshold": f">=-{thresholds.max_drawdown_degradation}"},
        {"name": "exit_model_predictive_signal", "passed": np.isfinite(auc) and auc >= thresholds.min_classifier_auc, "value": auc, "threshold": f">={thresholds.min_classifier_auc}"},
    ]
    passed = all(g["passed"] for g in gates)
    return {
        "gate": "EXIT INTELLIGENCE",
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": asdict(thresholds),
        "gates": gates,
        "expectancy_improvement_bps": ev_delta,
        "promotion_instruction": (
            "Adaptive exit policy may challenge deterministic exits in the next governed stage."
            if passed else
            "Keep deterministic target/stop/time exits authoritative."
        ),
    }
