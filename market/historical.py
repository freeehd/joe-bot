"""Historical minute-bar acquisition and normalization.

Broker SDK imports are deferred so the research/storage/test stack remains
usable offline. Alpaca is the initial provider, but callers depend on the small
``HistoricalBarsSource`` protocol rather than on Alpaca directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REQUIRED_BAR_COLUMNS = ("open", "high", "low", "close", "volume", "vwap")


class HistoricalBarsSource(Protocol):
    source_name: str

    def fetch_minute_bars(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        adjustment: str = "all",
        feed: str | None = None,
        asof: str | None = None,
    ) -> pd.DataFrame:
        ...


def normalize_bar_frame(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Normalize provider bars into a UTC DatetimeIndex + symbol column."""

    if df is None or len(df) == 0:
        empty = pd.DataFrame(columns=["symbol", *REQUIRED_BAR_COLUMNS])
        empty.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return empty

    frame = df.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()

    if "timestamp" not in frame.columns:
        datetime_candidates = [
            col for col in frame.columns
            if pd.api.types.is_datetime64_any_dtype(frame[col])
        ]
        if len(datetime_candidates) != 1:
            raise ValueError("Unable to identify historical-bar timestamp column")
        frame = frame.rename(columns={datetime_candidates[0]: "timestamp"})

    if "symbol" not in frame.columns:
        if symbol is None:
            raise ValueError("Historical bars do not contain a symbol column")
        frame["symbol"] = symbol

    missing = set(REQUIRED_BAR_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Historical bars missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(frame.pop("timestamp"), utc=True)
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[["symbol", *REQUIRED_BAR_COLUMNS, *[
        col for col in ("trade_count",) if col in frame.columns
    ]]]
    return frame.sort_index()


def filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep US regular session bars: 09:30 <= ET < 16:00."""

    if df.empty:
        return df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")

    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return df.loc[mask].copy()


@dataclass
class AlpacaHistoricalBarsSource:
    """Alpaca-backed implementation of ``HistoricalBarsSource``."""

    api_key: str | None = None
    secret_key: str | None = None
    source_name: str = "alpaca"

    def __post_init__(self) -> None:
        if self.api_key is None or self.secret_key is None:
            from dotenv import load_dotenv

            load_dotenv()
            self.api_key = self.api_key or os.getenv("ALPACA_API_KEY")
            self.secret_key = self.secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

    def fetch_minute_bars(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        adjustment: str = "all",
        feed: str | None = None,
        asof: str | None = None,
    ) -> pd.DataFrame:
        if not symbols:
            return normalize_bar_frame(pd.DataFrame())

        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        request_kwargs = {
            "symbol_or_symbols": list(symbols),
            "timeframe": TimeFrame.Minute,
            "start": start,
            "end": end,
            "adjustment": Adjustment(adjustment),
        }
        if feed:
            request_kwargs["feed"] = DataFeed(feed)
        if asof:
            request_kwargs["asof"] = asof

        request = StockBarsRequest(**request_kwargs)
        bars = client.get_stock_bars(request)
        return normalize_bar_frame(bars.df)

QUOTE_COLUMNS = ("bid_price", "ask_price", "bid_size", "ask_size")
TRADE_COLUMNS = ("price", "size")


def _normalize_tick_frame(df: pd.DataFrame, *, symbol: str | None, required: tuple[str, ...]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        empty = pd.DataFrame(columns=["symbol", *required])
        empty.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return empty
    frame = df.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
    if "timestamp" not in frame.columns:
        candidates = [col for col in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[col])]
        if len(candidates) != 1:
            raise ValueError("Unable to identify tick timestamp column")
        frame = frame.rename(columns={candidates[0]: "timestamp"})
    if "symbol" not in frame.columns:
        if symbol is None:
            raise ValueError("Tick data do not contain a symbol column")
        frame["symbol"] = symbol
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Tick data missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame.pop("timestamp"), utc=True)
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    keep = ["symbol", *required, *[c for c in ("side", "exchange", "conditions") if c in frame.columns]]
    return frame[keep].sort_index()


def normalize_quote_frame(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    return _normalize_tick_frame(df, symbol=symbol, required=QUOTE_COLUMNS)


def normalize_trade_frame(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    return _normalize_tick_frame(df, symbol=symbol, required=TRADE_COLUMNS)


class HistoricalMicrostructureSource(Protocol):
    source_name: str
    def fetch_quotes(self, symbols: Sequence[str], *, start: datetime, end: datetime, feed: str | None = None) -> pd.DataFrame: ...
    def fetch_trades(self, symbols: Sequence[str], *, start: datetime, end: datetime, feed: str | None = None) -> pd.DataFrame: ...


@dataclass
class AlpacaHistoricalMicrostructureSource:
    """Historical quote/trade source using alpaca-py's stock data client."""
    api_key: str | None = None
    secret_key: str | None = None
    source_name: str = "alpaca"

    def __post_init__(self) -> None:
        if self.api_key is None or self.secret_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            self.api_key = self.api_key or os.getenv("ALPACA_API_KEY")
            self.secret_key = self.secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

    def _client(self):
        from alpaca.data.historical.stock import StockHistoricalDataClient
        return StockHistoricalDataClient(self.api_key, self.secret_key)

    def fetch_quotes(self, symbols: Sequence[str], *, start: datetime, end: datetime, feed: str | None = None) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockQuotesRequest
        kwargs = {"symbol_or_symbols": list(symbols), "start": start, "end": end}
        if feed:
            kwargs["feed"] = DataFeed(feed)
        response = self._client().get_stock_quotes(StockQuotesRequest(**kwargs))
        return normalize_quote_frame(response.df)

    def fetch_trades(self, symbols: Sequence[str], *, start: datetime, end: datetime, feed: str | None = None) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockTradesRequest
        kwargs = {"symbol_or_symbols": list(symbols), "start": start, "end": end}
        if feed:
            kwargs["feed"] = DataFeed(feed)
        response = self._client().get_stock_trades(StockTradesRequest(**kwargs))
        return normalize_trade_frame(response.df)
