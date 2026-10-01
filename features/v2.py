"""Leakage-aware Feature Engine V2 for short-horizon intraday research.

All single-symbol features are causal and reset at US regular-session boundaries.
Cross-sectional market context is added separately after all symbols have been
processed for a timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

MOMENTUM_FEATURES = [
    "return_1m", "return_2m", "return_3m", "return_5m", "return_10m", "return_15m",
    "return_acceleration_1m", "momentum_vs_mean_5", "momentum_vs_mean_15",
    "distance_from_high_20", "distance_from_low_20", "breakout_strength_20_atr",
    "pullback_from_session_high_atr", "bounce_from_session_low_atr",
]

TREND_FEATURES = [
    "price_to_ema_5", "price_to_ema_10", "price_to_ema_20", "price_to_ema_50",
    "ema_5_20_separation", "ema_10_50_separation",
    "ema_5_slope_3", "ema_20_slope_5", "ema_50_slope_10",
    "trend_persistence_10", "vwap_distance", "vwap_slope_5", "vwap_cross_age_signed",
]

VOLATILITY_FEATURES = [
    "atr_14_norm", "realized_vol_1m", "realized_vol_5m", "realized_vol_15m",
    "range_norm", "range_expansion_20", "bollinger_width_20", "vol_acceleration_5_15",
]

VOLUME_FEATURES = [
    "log_volume", "volume_acceleration_5", "log_cumulative_session_volume",
    "tod_relative_volume", "log_trade_count", "trade_count_acceleration_5",
    "tod_relative_trade_count",
]

TIME_FEATURES = [
    "minutes_since_open_norm", "minutes_until_close_norm", "opening_15m_flag",
    "opening_hour_flag", "lunch_flag", "power_hour_flag", "day_of_week",
]

MARKET_CONTEXT_FEATURES = [
    "spy_return_1m", "spy_return_5m", "spy_return_15m",
    "qqq_return_1m", "qqq_return_5m", "qqq_return_15m",
    "relative_strength_spy_5m", "relative_strength_qqq_5m", "breadth_up_1m",
]

FEATURE_COLUMNS_V2 = (
    MOMENTUM_FEATURES
    + TREND_FEATURES
    + VOLATILITY_FEATURES
    + VOLUME_FEATURES
    + TIME_FEATURES
    + MARKET_CONTEXT_FEATURES
)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Feature Engine V2 requires a DatetimeIndex")
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _wilder_atr(session: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = session["close"].shift(1)
    true_range = pd.concat(
        [
            session["high"] - session["low"],
            (session["high"] - previous_close).abs(),
            (session["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _signed_cross_age(distance: pd.Series) -> pd.Series:
    """Signed bars since the last price/VWAP side change, reset per session."""

    signs = np.sign(distance.fillna(0.0).to_numpy())
    ages = np.zeros(len(signs), dtype=float)
    last_sign = 0.0
    age = 0
    for i, sign in enumerate(signs):
        if sign == 0:
            ages[i] = 0.0
            continue
        if sign != last_sign:
            age = 1
            last_sign = sign
        else:
            age += 1
        ages[i] = sign * age
    return pd.Series(ages, index=distance.index, dtype=float)


def _causal_time_of_day_baseline(values: pd.Series, minute_of_day: pd.Series) -> pd.Series:
    """Expected value for a clock minute using only earlier sessions.

    Each regular-session clock minute occurs once per session for a symbol, so
    grouping by minute-of-day and applying an expanding mean shifted by one row
    yields a historical-only baseline for every observation.
    """

    work = pd.DataFrame({"value": values.astype(float), "minute": minute_of_day}, index=values.index)
    baseline = work.groupby("minute", sort=False)["value"].transform(
        lambda series: series.expanding(min_periods=1).mean().shift(1)
    )
    return baseline


def _single_session_features(session: pd.DataFrame) -> pd.DataFrame:
    frame = session.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)

    # Price / momentum.
    for horizon in (1, 2, 3, 5, 10, 15):
        frame[f"return_{horizon}m"] = close.pct_change(horizon)
    frame["return_acceleration_1m"] = frame["return_1m"] - frame["return_1m"].shift(1)
    frame["momentum_vs_mean_5"] = close / close.rolling(5).mean() - 1.0
    frame["momentum_vs_mean_15"] = close / close.rolling(15).mean() - 1.0

    prior_high_20 = high.rolling(20).max().shift(1)
    prior_low_20 = low.rolling(20).min().shift(1)
    frame["distance_from_high_20"] = close / prior_high_20 - 1.0
    frame["distance_from_low_20"] = close / prior_low_20 - 1.0

    atr = _wilder_atr(frame, 14)
    safe_atr = atr.replace(0, np.nan)
    frame["breakout_strength_20_atr"] = (close - prior_high_20) / safe_atr
    session_high = high.cummax()
    session_low = low.cummin()
    frame["pullback_from_session_high_atr"] = (session_high - close) / safe_atr
    frame["bounce_from_session_low_atr"] = (close - session_low) / safe_atr

    # Trend.
    emas: dict[int, pd.Series] = {}
    for span in (5, 10, 20, 50):
        ema = close.ewm(span=span, adjust=False).mean()
        emas[span] = ema
        frame[f"price_to_ema_{span}"] = close / ema - 1.0
    frame["ema_5_20_separation"] = (emas[5] - emas[20]) / close
    frame["ema_10_50_separation"] = (emas[10] - emas[50]) / close
    frame["ema_5_slope_3"] = emas[5].pct_change(3) / 3.0
    frame["ema_20_slope_5"] = emas[20].pct_change(5) / 5.0
    frame["ema_50_slope_10"] = emas[50].pct_change(10) / 10.0
    frame["trend_persistence_10"] = np.sign(frame["return_1m"]).rolling(10).mean()
    frame["vwap_distance"] = (close - frame["vwap"].astype(float)) / frame["vwap"].astype(float)
    frame["vwap_slope_5"] = frame["vwap"].astype(float).pct_change(5) / 5.0
    frame["vwap_cross_age_signed"] = _signed_cross_age(frame["vwap_distance"])

    # Volatility.
    log_return = np.log(close).diff()
    frame["atr_14_norm"] = atr / close
    frame["realized_vol_1m"] = log_return.abs()
    frame["realized_vol_5m"] = log_return.rolling(5).std(ddof=0) * np.sqrt(5)
    frame["realized_vol_15m"] = log_return.rolling(15).std(ddof=0) * np.sqrt(15)
    frame["range_norm"] = (high - low) / close
    prior_range_median = frame["range_norm"].rolling(20).median().shift(1)
    frame["range_expansion_20"] = frame["range_norm"] / prior_range_median.replace(0, np.nan)
    rolling_mean_20 = close.rolling(20).mean()
    rolling_std_20 = close.rolling(20).std(ddof=0)
    frame["bollinger_width_20"] = (4.0 * rolling_std_20) / rolling_mean_20
    frame["vol_acceleration_5_15"] = (
        frame["realized_vol_5m"] / frame["realized_vol_15m"].replace(0, np.nan) - 1.0
    )

    # Volume / activity. Exact-time expected values are injected later because
    # they need observations from prior sessions, not only this session.
    volume = frame["volume"].astype(float)
    frame["log_volume"] = np.log1p(volume)
    previous_volume_mean = volume.rolling(5).mean().shift(1)
    frame["volume_acceleration_5"] = volume / previous_volume_mean.replace(0, np.nan) - 1.0
    frame["log_cumulative_session_volume"] = np.log1p(volume.cumsum())

    trade_count = (
        frame["trade_count"].astype(float)
        if "trade_count" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    frame["log_trade_count"] = np.log1p(trade_count)
    previous_trade_mean = trade_count.rolling(5).mean().shift(1)
    frame["trade_count_acceleration_5"] = trade_count / previous_trade_mean.replace(0, np.nan) - 1.0

    # Time features.
    local = _ensure_utc_index(frame).tz_convert(NY)
    minutes_since_open = (local.hour * 60 + local.minute) - (9 * 60 + 30)
    minutes_until_close = (16 * 60) - (local.hour * 60 + local.minute)
    frame["minutes_since_open_norm"] = minutes_since_open / 390.0
    frame["minutes_until_close_norm"] = minutes_until_close / 390.0
    frame["opening_15m_flag"] = (minutes_since_open < 15).astype(int)
    frame["opening_hour_flag"] = (minutes_since_open < 60).astype(int)
    frame["lunch_flag"] = ((local.hour >= 12) & (local.hour < 14)).astype(int)
    frame["power_hour_flag"] = (minutes_until_close <= 60).astype(int)
    frame["day_of_week"] = local.dayofweek.astype(int)
    return frame


def create_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal, session-aware V2 features for one symbol.

    The input should already be filtered to regular hours and sorted. Features
    never cross a New York trading-session boundary. Exact-time relative volume
    and trade-count baselines use only *earlier* sessions for the same symbol.
    """

    frame = df.copy().sort_index()
    utc = _ensure_utc_index(frame)
    local = utc.tz_convert(NY)
    session_key = pd.Series(local.date, index=frame.index)

    pieces = []
    for _, positions in session_key.groupby(session_key).groups.items():
        pieces.append(_single_session_features(frame.loc[positions]))
    enriched = pd.concat(pieces).sort_index() if pieces else frame.copy()

    local = _ensure_utc_index(enriched).tz_convert(NY)
    minute_of_day = pd.Series(local.hour * 60 + local.minute, index=enriched.index)
    volume_baseline = _causal_time_of_day_baseline(enriched["volume"], minute_of_day)
    enriched["tod_relative_volume"] = enriched["volume"].astype(float) / volume_baseline.replace(0, np.nan)

    if "trade_count" in enriched.columns:
        trade_baseline = _causal_time_of_day_baseline(enriched["trade_count"], minute_of_day)
        enriched["tod_relative_trade_count"] = (
            enriched["trade_count"].astype(float) / trade_baseline.replace(0, np.nan)
        )
    else:
        enriched["tod_relative_trade_count"] = np.nan

    return enriched


def add_market_context(
    dataset: pd.DataFrame,
    *,
    symbol_column: str = "training_symbol",
) -> pd.DataFrame:
    """Add SPY/QQQ relative context and cross-sectional breadth.

    ``dataset`` must contain all symbols on a common timestamp index after their
    single-symbol V2 features have been calculated. No future timestamps are used.
    """

    if symbol_column not in dataset.columns:
        raise ValueError(f"Missing symbol column {symbol_column!r}")
    frame = dataset.copy().sort_index()

    benchmark_spec = {
        "SPY": "spy",
        "QQQ": "qqq",
    }
    for symbol, prefix in benchmark_spec.items():
        benchmark = frame.loc[frame[symbol_column] == symbol]
        if benchmark.empty:
            raise ValueError(f"Market context requires {symbol} in the research dataset")
        for horizon in (1, 5, 15):
            values = benchmark[f"return_{horizon}m"]
            # One benchmark row per timestamp; duplicated indices would make map ambiguous.
            values = values[~values.index.duplicated(keep="last")]
            frame[f"{prefix}_return_{horizon}m"] = frame.index.map(values)

    frame["relative_strength_spy_5m"] = frame["return_5m"] - frame["spy_return_5m"]
    frame["relative_strength_qqq_5m"] = frame["return_5m"] - frame["qqq_return_5m"]

    breadth = frame.groupby(level=0)["return_1m"].transform(lambda s: (s > 0).mean())
    frame["breadth_up_1m"] = breadth.astype(float)
    return frame
