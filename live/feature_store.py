"""Synchronized live Feature Engine V2 state built from historical seed bars."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from features.v2 import FEATURE_COLUMNS_V2, add_market_context, create_features_v2
from market.stream import BarEvent

NY = ZoneInfo("America/New_York")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_index()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("history requires DatetimeIndex")
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    return result


def _tail_sessions(frame: pd.DataFrame, sessions: int) -> pd.DataFrame:
    frame = _normalize(frame)
    local_dates = pd.Series(frame.index.tz_convert(NY).date, index=frame.index)
    unique = list(dict.fromkeys(local_dates.tolist()))
    keep = set(unique[-sessions:])
    return frame.loc[local_dates.isin(keep)].copy()


@dataclass(frozen=True)
class FeatureBatch:
    timestamp: pd.Timestamp
    frame: pd.DataFrame
    coverage: float


class LiveFeatureStore:
    def __init__(
        self,
        symbols: Sequence[str],
        historical_bars: Mapping[str, pd.DataFrame],
        *,
        warmup_sessions: int = 30,
        minimum_coverage: float = 1.0,
    ) -> None:
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        if "SPY" not in self.symbols or "QQQ" not in self.symbols:
            raise ValueError("live feature store requires SPY and QQQ")
        if warmup_sessions < 2:
            raise ValueError("warmup_sessions must be >= 2")
        if not 0 < minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1]")
        self.minimum_coverage = minimum_coverage
        self._bars = {symbol: _tail_sessions(historical_bars[symbol], warmup_sessions) for symbol in self.symbols}
        self._pending: dict[pd.Timestamp, set[str]] = {}
        self._emitted: set[pd.Timestamp] = set()

    def add_bar(self, event: BarEvent) -> FeatureBatch | None:
        if event.symbol not in self._bars:
            return None
        ts = pd.Timestamp(event.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        row = pd.DataFrame([{
            "open": event.open, "high": event.high, "low": event.low, "close": event.close,
            "volume": event.volume, "trade_count": event.trade_count, "vwap": event.vwap if event.vwap is not None else event.close,
        }], index=pd.DatetimeIndex([ts]))
        bars = self._bars[event.symbol]
        bars = pd.concat([bars.loc[bars.index != ts], row]).sort_index()
        self._bars[event.symbol] = bars
        self._pending.setdefault(ts, set()).add(event.symbol)
        if ts in self._emitted:
            return None
        coverage = len(self._pending[ts]) / len(self.symbols)
        if coverage < self.minimum_coverage or not {"SPY", "QQQ"}.issubset(self._pending[ts]):
            return None

        rows = []
        for symbol in self.symbols:
            bars = self._bars[symbol]
            if ts not in bars.index:
                continue
            enriched = create_features_v2(bars)
            latest = enriched.loc[[ts]].copy()
            latest["training_symbol"] = symbol
            rows.append(latest)
        if not rows:
            return None
        batch = pd.concat(rows).sort_index()
        batch = add_market_context(batch, symbol_column="training_symbol")
        self._emitted.add(ts)
        # Only retain recent pending timestamps; historical bars stay in _bars.
        for old in list(self._pending):
            if old < ts:
                self._pending.pop(old, None)
        missing = set(FEATURE_COLUMNS_V2).difference(batch.columns)
        if missing:
            raise ValueError(f"feature batch missing V2 columns: {sorted(missing)}")
        return FeatureBatch(ts, batch, coverage)
