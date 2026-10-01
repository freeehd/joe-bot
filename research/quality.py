"""Explicit historical minute-bar cleaning and quality diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from market.historical import REQUIRED_BAR_COLUMNS

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DataQualityReport:
    symbol: str
    input_rows: int
    output_rows: int
    duplicate_rows: int
    invalid_numeric_rows: int
    invalid_ohlc_rows: int
    missing_minute_intervals: int
    sessions: int
    first_timestamp: str | None
    last_timestamp: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _count_missing_intervals(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert(NY)
    sessions = pd.Series(local.date, index=df.index)

    missing = 0
    for _, positions in sessions.groupby(sessions).groups.items():
        session_index = pd.DatetimeIndex(positions).sort_values()
        if len(session_index) < 2:
            continue
        diffs = session_index.to_series().diff().dropna()
        for gap in diffs:
            minutes = int(gap.total_seconds() // 60)
            if minutes > 1:
                missing += minutes - 1
    return int(missing)


def clean_and_validate_bars(df: pd.DataFrame, *, symbol: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean deterministic errors without fabricating missing market data.

    Policy:
    - duplicates are deduplicated deterministically (last observation wins),
    - malformed/non-positive OHLC values and negative volumes are dropped,
    - impossible OHLC relationships are dropped,
    - missing minutes are measured but never forward-filled.
    """

    missing_columns = set(REQUIRED_BAR_COLUMNS).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Historical bars must use a DatetimeIndex")

    frame = df.copy().sort_index()
    input_rows = len(frame)

    duplicate_rows = int(frame.index.duplicated(keep="last").sum())
    frame = frame[~frame.index.duplicated(keep="last")]

    numeric = frame[list(REQUIRED_BAR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = (
        numeric[["open", "high", "low", "close", "vwap"]].isna().any(axis=1)
        | (numeric[["open", "high", "low", "close", "vwap"]] <= 0).any(axis=1)
        | numeric["volume"].isna()
        | (numeric["volume"] < 0)
    )
    invalid_numeric_rows = int(invalid_numeric.sum())
    frame = frame.loc[~invalid_numeric].copy()

    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    invalid_ohlc_rows = int(invalid_ohlc.sum())
    frame = frame.loc[~invalid_ohlc].copy().sort_index()

    missing_intervals = _count_missing_intervals(frame)
    if frame.empty:
        sessions = 0
        first = last = None
    else:
        index = frame.index if frame.index.tz is not None else frame.index.tz_localize("UTC")
        sessions = int(len(set(index.tz_convert(NY).date)))
        first = frame.index.min().isoformat()
        last = frame.index.max().isoformat()

    report = DataQualityReport(
        symbol=symbol,
        input_rows=input_rows,
        output_rows=len(frame),
        duplicate_rows=duplicate_rows,
        invalid_numeric_rows=invalid_numeric_rows,
        invalid_ohlc_rows=invalid_ohlc_rows,
        missing_minute_intervals=missing_intervals,
        sessions=sessions,
        first_timestamp=first,
        last_timestamp=last,
    )
    return frame, report
