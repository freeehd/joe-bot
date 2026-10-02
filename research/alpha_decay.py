"""Alpha-decay and excursion diagnostics for Victory Sprint 6."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from backtest.engine import ExecutionConfig, _execution_price


def alpha_decay_curve(
    trades: pd.DataFrame,
    *,
    raw_loader: Callable[[str], pd.DataFrame],
    execution_config: ExecutionConfig | None = None,
    max_bars: int = 20,
) -> dict:
    execution = execution_config or ExecutionConfig()
    if trades.empty:
        return {"trades": 0, "curve": [], "peak_bar": None, "half_life_bar": None}
    observations: dict[int, list[float]] = {bar: [] for bar in range(1, max_bars + 1)}
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
        entry_price = float(trade.entry_price)
        side = str(trade.side)
        action = "SELL" if side == "LONG" else "BUY"
        session_date = bars.index[entry_pos].tz_convert("America/New_York").date()
        for holding in range(1, max_bars + 1):
            pos = entry_pos + holding - 1
            if pos >= len(bars) or bars.index[pos].tz_convert("America/New_York").date() != session_date:
                break
            exit_price = _execution_price(float(bars.iloc[pos]["close"]), action=action, config=execution)
            gross = exit_price / entry_price - 1.0 if side == "LONG" else (entry_price - exit_price) / entry_price
            net = gross - 2.0 * execution.fee_bps / 10_000.0
            observations[holding].append(float(net))
    curve = []
    for holding, values in observations.items():
        if not values:
            continue
        curve.append({
            "holding_bar": holding,
            "observations": len(values),
            "avg_net_return": float(np.mean(values)),
            "expectancy_bps": float(np.mean(values) * 10_000.0),
            "median_net_return": float(np.median(values)),
            "positive_rate": float(np.mean(np.asarray(values) > 0)),
        })
    if not curve:
        return {"trades": int(len(trades)), "curve": [], "peak_bar": None, "half_life_bar": None}
    peak = max(curve, key=lambda row: row["expectancy_bps"])
    peak_ev = peak["expectancy_bps"]
    half_life = None
    if peak_ev > 0:
        for row in curve:
            if row["holding_bar"] > peak["holding_bar"] and row["expectancy_bps"] <= peak_ev * 0.5:
                half_life = row["holding_bar"]
                break
    return {
        "trades": int(len(trades)),
        "peak_bar": int(peak["holding_bar"]),
        "peak_expectancy_bps": float(peak_ev),
        "half_life_bar": half_life,
        "curve": curve,
    }


def excursion_report(trades: pd.DataFrame) -> dict:
    if trades.empty or not {"mfe_return", "mae_return", "net_return"}.issubset(trades.columns):
        return {"trades": 0}
    frame = trades[["mfe_return", "mae_return", "net_return"]].astype(float)
    capture = frame["net_return"] / frame["mfe_return"].replace(0, np.nan)
    giveback = frame["mfe_return"] - frame["net_return"]
    return {
        "trades": int(len(frame)),
        "avg_mfe_bps": float(frame["mfe_return"].mean() * 10_000.0),
        "avg_mae_bps": float(frame["mae_return"].mean() * 10_000.0),
        "avg_realized_bps": float(frame["net_return"].mean() * 10_000.0),
        "median_capture_ratio": float(capture.replace([np.inf, -np.inf], np.nan).dropna().median()),
        "avg_giveback_bps": float(giveback.mean() * 10_000.0),
    }
