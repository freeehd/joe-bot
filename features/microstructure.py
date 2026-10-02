"""Causal quote/trade microstructure features for Victory Sprint 7 research."""
from __future__ import annotations

import numpy as np
import pandas as pd

MICROSTRUCTURE_FEATURES = [
    "quoted_spread_bps_mean",
    "quoted_spread_bps_p90",
    "depth_imbalance_mean",
    "microprice_dislocation_bps_mean",
    "log_quote_depth_mean",
    "trade_intensity",
    "signed_trade_imbalance",
    "effective_spread_bps_mean",
    "short_horizon_price_impact_bps",
]


def _utc_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} requires a DatetimeIndex")
    result = frame.sort_index().copy()
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    return result


def build_microstructure_features(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tick quote/trade data into causal one-minute features.

    Quotes require bid/ask price+size. Trades require price+size. Trade direction
    is inferred with a causal tick rule when an explicit `side` column is absent.
    """
    q = _utc_index(quotes, "quotes")
    t = _utc_index(trades, "trades")
    required_q = {"bid_price", "ask_price", "bid_size", "ask_size"}
    required_t = {"price", "size"}
    if required_q.difference(q.columns):
        raise ValueError(f"quotes missing columns: {sorted(required_q.difference(q.columns))}")
    if required_t.difference(t.columns):
        raise ValueError(f"trades missing columns: {sorted(required_t.difference(t.columns))}")

    mid = (q["bid_price"].astype(float) + q["ask_price"].astype(float)) / 2.0
    spread = (q["ask_price"].astype(float) - q["bid_price"].astype(float)).clip(lower=0)
    q["quoted_spread_bps"] = spread / mid.replace(0, np.nan) * 10_000.0
    bid_size = q["bid_size"].astype(float).clip(lower=0)
    ask_size = q["ask_size"].astype(float).clip(lower=0)
    depth = bid_size + ask_size
    q["depth_imbalance"] = (bid_size - ask_size) / depth.replace(0, np.nan)
    microprice = (q["ask_price"].astype(float) * bid_size + q["bid_price"].astype(float) * ask_size) / depth.replace(0, np.nan)
    q["microprice_dislocation_bps"] = (microprice - mid) / mid.replace(0, np.nan) * 10_000.0
    q["log_quote_depth"] = np.log1p(depth)

    q_min = pd.DataFrame({
        "quoted_spread_bps_mean": q["quoted_spread_bps"].resample("1min").mean(),
        "quoted_spread_bps_p90": q["quoted_spread_bps"].resample("1min").quantile(.90),
        "depth_imbalance_mean": q["depth_imbalance"].resample("1min").mean(),
        "microprice_dislocation_bps_mean": q["microprice_dislocation_bps"].resample("1min").mean(),
        "log_quote_depth_mean": q["log_quote_depth"].resample("1min").mean(),
        "mid_close": mid.resample("1min").last(),
    })

    prices = t["price"].astype(float)
    if "side" in t.columns:
        side = t["side"].astype(str).str.upper().map({"BUY": 1.0, "B": 1.0, "SELL": -1.0, "S": -1.0}).fillna(0.0)
    else:
        tick = np.sign(prices.diff())
        side = tick.replace(0, np.nan).ffill().fillna(0.0)
    size = t["size"].astype(float).clip(lower=0)
    signed_volume = size * side
    t_work = pd.DataFrame({"price": prices, "size": size, "signed_volume": signed_volume}, index=t.index)
    total_volume = t_work["size"].resample("1min").sum()
    t_min = pd.DataFrame({
        "trade_intensity": t_work["size"].resample("1min").count().astype(float),
        "signed_trade_imbalance": t_work["signed_volume"].resample("1min").sum() / total_volume.replace(0, np.nan),
        "trade_vwap": (t_work["price"] * t_work["size"]).resample("1min").sum() / total_volume.replace(0, np.nan),
    })

    result = q_min.join(t_min, how="outer")
    result["effective_spread_bps_mean"] = (result["trade_vwap"] - result["mid_close"]).abs() / result["mid_close"].replace(0, np.nan) * 20_000.0
    result["short_horizon_price_impact_bps"] = result["mid_close"].pct_change().shift(-1) * 10_000.0 * np.sign(result["signed_trade_imbalance"].fillna(0.0))
    # Shift future-looking impact so row t only contains impact realized from the
    # previous minute's order flow; no row sees future market movement.
    result["short_horizon_price_impact_bps"] = result["short_horizon_price_impact_bps"].shift(1)
    return result[MICROSTRUCTURE_FEATURES].replace([np.inf, -np.inf], np.nan)
