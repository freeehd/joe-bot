"""Trade-level net-expectancy metrics for V0.6 research backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.engine import TradeResult


def trades_to_frame(trades: list[TradeResult]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol", "side", "signal_time", "entry_time", "exit_time", "confidence",
                "p_wait", "p_long", "p_short", "entry_price", "exit_price",
                "gross_return", "net_return", "exit_reason", "holding_bars",
                "target_price", "stop_price", "mfe_return", "mae_return",
            ]
        )
    return pd.DataFrame([trade.to_dict() for trade in trades])


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns.loc[returns > 0].sum())
    losses = float(-returns.loc[returns < 0].sum())
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def confidence_ev_buckets(
    trades: pd.DataFrame,
    *,
    edges=(0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.000001),
) -> list[dict]:
    if trades.empty:
        return []
    rows: list[dict] = []
    for low, high in zip(edges[:-1], edges[1:]):
        subset = trades.loc[(trades["confidence"] >= low) & (trades["confidence"] < high)]
        if subset.empty:
            continue
        returns = subset["net_return"].astype(float)
        rows.append(
            {
                "low": float(low),
                "high": float(min(high, 1.0)),
                "trades": int(len(subset)),
                "avg_net_return": float(returns.mean()),
                "median_net_return": float(returns.median()),
                "win_rate": float((returns > 0).mean()),
                "expectancy_bps": float(returns.mean() * 10_000.0),
            }
        )
    return rows


def summarize_trades(trades: list[TradeResult] | pd.DataFrame) -> dict:
    frame = trades if isinstance(trades, pd.DataFrame) else trades_to_frame(trades)
    if frame.empty:
        return {
            "total_trades": 0,
            "avg_net_return": 0.0,
            "expectancy_bps": 0.0,
            "win_rate": 0.0,
            "confidence_ev_buckets": [],
        }

    frame = frame.copy()
    returns = frame["net_return"].astype(float)
    positive = returns.loc[returns > 0]
    negative = returns.loc[returns < 0]
    side_stats = {}
    for side, group in frame.groupby("side"):
        side_returns = group["net_return"].astype(float)
        side_stats[str(side)] = {
            "trades": int(len(group)),
            "avg_net_return": float(side_returns.mean()),
            "expectancy_bps": float(side_returns.mean() * 10_000.0),
            "win_rate": float((side_returns > 0).mean()),
        }

    exit_counts = frame["exit_reason"].value_counts().to_dict()
    return {
        "total_trades": int(len(frame)),
        "long_trades": int((frame["side"] == "LONG").sum()),
        "short_trades": int((frame["side"] == "SHORT").sum()),
        "win_rate": float((returns > 0).mean()),
        "avg_net_return": float(returns.mean()),
        "median_net_return": float(returns.median()),
        "expectancy_bps": float(returns.mean() * 10_000.0),
        "avg_win": float(positive.mean()) if len(positive) else 0.0,
        "avg_loss": float(negative.mean()) if len(negative) else 0.0,
        "profit_factor": float(_profit_factor(returns)),
        "avg_holding_bars": float(frame["holding_bars"].astype(float).mean()),
        "avg_mfe_bps": float(frame["mfe_return"].astype(float).mean() * 10_000.0) if "mfe_return" in frame else 0.0,
        "avg_mae_bps": float(frame["mae_return"].astype(float).mean() * 10_000.0) if "mae_return" in frame else 0.0,
        "avg_edge_capture_ratio": float((frame["net_return"].astype(float) / frame["mfe_return"].replace(0, np.nan).astype(float)).replace([np.inf, -np.inf], np.nan).dropna().mean()) if "mfe_return" in frame and (frame["mfe_return"].astype(float) > 0).any() else 0.0,
        "target_exits": int(exit_counts.get("TARGET", 0)),
        "stop_exits": int(exit_counts.get("STOP", 0)),
        "time_exits": int(exit_counts.get("TIME", 0)),
        "session_close_exits": int(exit_counts.get("SESSION_CLOSE", 0)),
        "data_end_exits": int(exit_counts.get("DATA_END", 0)),
        "by_side": side_stats,
        "confidence_ev_buckets": confidence_ev_buckets(frame),
    }
