"""Dynamic tradeable-universe selection for GOLD 23 / Victory Sprint 11.

Selection is intentionally deterministic and data-source agnostic.  Upstream jobs
provide liquidity/spread/premarket/event statistics; this module only filters,
scores, and records why a symbol entered or left the active universe.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseConfig:
    min_price: float = 3.0
    max_price: float = 1000.0
    min_avg_dollar_volume: float = 20_000_000.0
    min_avg_volume: float = 500_000.0
    max_spread_bps: float = 30.0
    target_size: int = 500
    min_size: int = 50

    def __post_init__(self) -> None:
        if self.min_price <= 0 or self.max_price <= self.min_price:
            raise ValueError("invalid price bounds")
        if self.target_size < 1 or self.min_size < 1 or self.min_size > self.target_size:
            raise ValueError("invalid universe size bounds")


_REQUIRED = {
    "symbol", "price", "avg_dollar_volume", "avg_volume", "spread_bps",
    "premarket_activity", "relative_volume", "volatility", "mover_score", "event_score",
}


def _percentile(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    return series.rank(pct=True, method="average").astype(float)


def select_morning_universe(stats: pd.DataFrame, config: UniverseConfig | None = None) -> pd.DataFrame:
    """Return a ranked, filtered morning universe with transparent score components."""
    config = config or UniverseConfig()
    missing = _REQUIRED.difference(stats.columns)
    if missing:
        raise ValueError(f"universe stats missing columns: {sorted(missing)}")
    frame = stats.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.drop_duplicates("symbol", keep="last")
    numeric = [c for c in _REQUIRED if c != "symbol"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric)
    eligible = frame.loc[
        frame["price"].between(config.min_price, config.max_price)
        & (frame["avg_dollar_volume"] >= config.min_avg_dollar_volume)
        & (frame["avg_volume"] >= config.min_avg_volume)
        & (frame["spread_bps"] <= config.max_spread_bps)
    ].copy()
    if len(eligible) < config.min_size:
        raise RuntimeError(f"only {len(eligible)} symbols passed mandatory universe filters; need >= {config.min_size}")

    # Liquidity dominates. Activity/movers/events can promote an already-tradeable name,
    # but cannot rescue an illiquid or prohibitively wide-spread symbol.
    eligible["liquidity_score"] = _percentile(np.log1p(eligible["avg_dollar_volume"]))
    eligible["spread_score"] = 1.0 - _percentile(eligible["spread_bps"])
    eligible["activity_score"] = (
        _percentile(eligible["premarket_activity"]) + _percentile(eligible["relative_volume"])
    ) / 2.0
    eligible["movement_score"] = (
        _percentile(eligible["volatility"]) + _percentile(eligible["mover_score"])
    ) / 2.0
    eligible["event_rank"] = _percentile(eligible["event_score"])
    eligible["universe_score"] = (
        0.35 * eligible["liquidity_score"]
        + 0.20 * eligible["spread_score"]
        + 0.20 * eligible["activity_score"]
        + 0.15 * eligible["movement_score"]
        + 0.10 * eligible["event_rank"]
    )
    eligible = eligible.sort_values(["universe_score", "avg_dollar_volume"], ascending=False)
    selected = eligible.head(config.target_size).copy()
    selected["universe_rank"] = range(1, len(selected) + 1)
    selected["selection_reason"] = "morning_rank"
    return selected.reset_index(drop=True)


def apply_intraday_updates(
    active: Iterable[str],
    stats: pd.DataFrame,
    *,
    add_rvol: float = 2.0,
    add_volatility: float = 0.02,
    add_event_score: float = 0.75,
    remove_spread_bps: float = 60.0,
    remove_min_dollar_volume: float = 2_000_000.0,
) -> dict:
    """Add/remove symbols using only explicit real-time tradeability rules."""
    required = {"symbol", "relative_volume", "volatility", "event_score", "spread_bps", "intraday_dollar_volume"}
    missing = required.difference(stats.columns)
    if missing:
        raise ValueError(f"intraday stats missing columns: {sorted(missing)}")
    active_set = {str(symbol).upper() for symbol in active}
    rows = stats.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.upper()
    adds: list[dict] = []
    removes: list[dict] = []
    for _, row in rows.iterrows():
        symbol = row["symbol"]
        tradeable = float(row["spread_bps"]) <= remove_spread_bps and float(row["intraday_dollar_volume"]) >= remove_min_dollar_volume
        trigger = (
            float(row["relative_volume"]) >= add_rvol
            or float(row["volatility"]) >= add_volatility
            or float(row["event_score"]) >= add_event_score
        )
        if symbol not in active_set and tradeable and trigger:
            adds.append({"symbol": symbol, "reason": "intraday_activity"})
        elif symbol in active_set and not tradeable:
            removes.append({"symbol": symbol, "reason": "tradeability_degraded"})
    return {"add": adds, "remove": removes, "active_before": len(active_set)}


def config_dict(config: UniverseConfig) -> dict:
    return asdict(config)
