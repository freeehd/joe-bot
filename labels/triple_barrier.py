"""Triple-barrier labels for short-horizon intraday research.

The labeler answers the question the trading system actually cares about:
starting from a completed bar, does a profit target or stop get hit first
within a bounded future horizon?

Only future OHLC values are used to determine the outcome. Distances can be
fixed percentages or volatility-aware ATR multiples computed from information
available at the entry bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal

import numpy as np
import pandas as pd


class TradeLabel(IntEnum):
    """Single multiclass ground-truth target."""

    WAIT = 0
    LONG = 1
    SHORT = 2


class BarrierOutcome(StrEnum):
    """Outcome of one hypothetical directional trade."""

    WIN = "WIN"
    LOSS = "LOSS"
    NO_RESOLUTION = "NO_RESOLUTION"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_FUTURE = "INSUFFICIENT_FUTURE"


@dataclass(frozen=True)
class BarrierConfig:
    """Configuration for target/stop/time barriers.

    Fixed mode uses ``target_pct`` and ``stop_pct`` as fractions of entry
    price. ATR mode uses ``ATR * multiplier`` as an absolute price distance.
    """

    horizon_bars: int = 10
    target_pct: float = 0.003
    stop_pct: float = 0.0015
    use_atr: bool = False
    atr_period: int = 14
    atr_target_multiplier: float = 1.0
    atr_stop_multiplier: float = 0.5

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.target_pct <= 0:
            raise ValueError("target_pct must be > 0")
        if self.stop_pct <= 0:
            raise ValueError("stop_pct must be > 0")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.atr_target_multiplier <= 0:
            raise ValueError("atr_target_multiplier must be > 0")
        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be > 0")


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate causal Wilder-style ATR from completed bars only."""

    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _barrier_prices(
    *,
    entry_price: float,
    side: Literal["LONG", "SHORT"],
    target_distance: float,
    stop_distance: float,
) -> tuple[float, float]:
    if side == "LONG":
        return entry_price + target_distance, entry_price - stop_distance
    if side == "SHORT":
        return entry_price - target_distance, entry_price + stop_distance
    raise ValueError(f"Unsupported side: {side}")


def evaluate_barriers(
    df: pd.DataFrame,
    entry_position: int,
    *,
    side: Literal["LONG", "SHORT"],
    horizon_bars: int,
    target_distance: float,
    stop_distance: float,
) -> tuple[BarrierOutcome, int | None]:
    """Evaluate one directional trade in chronological order.

    Returns ``(outcome, bars_to_event)``. If target and stop are both touched
    by the same OHLC bar, intrabar ordering cannot be known from minute bars,
    so the result is deliberately ``AMBIGUOUS`` rather than optimistically
    choosing a winner.
    """

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if target_distance <= 0 or stop_distance <= 0:
        raise ValueError("barrier distances must be > 0")
    if entry_position < 0 or entry_position >= len(df):
        raise IndexError("entry_position is outside the dataframe")

    final_position = entry_position + horizon_bars
    if final_position >= len(df):
        return BarrierOutcome.INSUFFICIENT_FUTURE, None

    entry_price = float(df.iloc[entry_position]["close"])
    target_price, stop_price = _barrier_prices(
        entry_price=entry_price,
        side=side,
        target_distance=target_distance,
        stop_distance=stop_distance,
    )

    for future_position in range(entry_position + 1, final_position + 1):
        bar = df.iloc[future_position]
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "LONG":
            target_hit = high >= target_price
            stop_hit = low <= stop_price
        else:
            target_hit = low <= target_price
            stop_hit = high >= stop_price

        bars_to_event = future_position - entry_position

        if target_hit and stop_hit:
            return BarrierOutcome.AMBIGUOUS, bars_to_event
        if target_hit:
            return BarrierOutcome.WIN, bars_to_event
        if stop_hit:
            return BarrierOutcome.LOSS, bars_to_event

    return BarrierOutcome.NO_RESOLUTION, horizon_bars


def _distances_for_row(
    row: pd.Series,
    config: BarrierConfig,
) -> tuple[float, float] | None:
    entry_price = float(row["close"])

    if config.use_atr:
        atr = row.get("label_atr", np.nan)
        if pd.isna(atr) or float(atr) <= 0:
            return None
        return (
            float(atr) * config.atr_target_multiplier,
            float(atr) * config.atr_stop_multiplier,
        )

    return (
        entry_price * config.target_pct,
        entry_price * config.stop_pct,
    )


def _resolve_multiclass_label(
    long_outcome: BarrierOutcome,
    short_outcome: BarrierOutcome,
) -> TradeLabel:
    """Collapse two directional simulations into one unambiguous action."""

    if long_outcome == BarrierOutcome.WIN and short_outcome != BarrierOutcome.WIN:
        return TradeLabel.LONG
    if short_outcome == BarrierOutcome.WIN and long_outcome != BarrierOutcome.WIN:
        return TradeLabel.SHORT
    return TradeLabel.WAIT


def create_multiclass_labels(
    df: pd.DataFrame,
    config: BarrierConfig | None = None,
    *,
    respect_sessions: bool = True,
) -> pd.DataFrame:
    """Attach reproducible LONG/WAIT/SHORT triple-barrier labels.

    This implementation preserves the original row-by-row semantics while
    evaluating each future horizon step with NumPy arrays. That matters for
    production datasets: millions of ``DataFrame.iloc`` calls can turn label
    generation into an hours-long preprocessing step.

    Rows without a complete future horizon remain in the returned dataframe but
    are marked ``label_valid=False`` and have nullable label fields. This avoids
    silently treating unknown future outcomes as WAIT.
    """

    config = config or BarrierConfig()
    required = {"high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labeled = df.copy()
    if config.use_atr:
        labeled["label_atr"] = calculate_atr(labeled, config.atr_period)

    n = len(labeled)
    close = labeled["close"].to_numpy(dtype=float, copy=False)
    high = labeled["high"].to_numpy(dtype=float, copy=False)
    low = labeled["low"].to_numpy(dtype=float, copy=False)

    if config.use_atr:
        atr = labeled["label_atr"].to_numpy(dtype=float, copy=False)
        target_distances = atr * config.atr_target_multiplier
        stop_distances = atr * config.atr_stop_multiplier
        distance_valid = np.isfinite(atr) & (atr > 0)
        target_distances = np.where(distance_valid, target_distances, np.nan)
        stop_distances = np.where(distance_valid, stop_distances, np.nan)
    else:
        target_distances = close * config.target_pct
        stop_distances = close * config.stop_pct
        distance_valid = (
            np.isfinite(target_distances)
            & np.isfinite(stop_distances)
            & (target_distances > 0)
            & (stop_distances > 0)
        )

    insufficient = str(BarrierOutcome.INSUFFICIENT_FUTURE)
    no_resolution = str(BarrierOutcome.NO_RESOLUTION)
    win = str(BarrierOutcome.WIN)
    loss = str(BarrierOutcome.LOSS)
    ambiguous = str(BarrierOutcome.AMBIGUOUS)

    long_outcomes = np.full(n, insufficient, dtype=object)
    short_outcomes = np.full(n, insufficient, dtype=object)
    long_event_bars = np.full(n, -1, dtype=np.int16)
    short_event_bars = np.full(n, -1, dtype=np.int16)
    valid_flags = np.zeros(n, dtype=bool)

    if respect_sessions and isinstance(labeled.index, pd.DatetimeIndex):
        session_values = labeled.index.normalize().to_numpy()
        if n:
            boundaries = np.flatnonzero(session_values[1:] != session_values[:-1]) + 1
            session_slices = np.split(np.arange(n, dtype=np.int64), boundaries)
        else:
            session_slices = []
    else:
        session_slices = [np.arange(n, dtype=np.int64)] if n else []

    horizon = config.horizon_bars

    for positions in session_slices:
        m = len(positions)
        if m <= horizon:
            continue

        local_entries = np.arange(0, m - horizon, dtype=np.int64)
        absolute_entries = positions[local_entries]
        usable = distance_valid[absolute_entries]
        if not np.any(usable):
            continue

        active_abs = absolute_entries[usable]
        active_local = local_entries[usable]
        valid_flags[active_abs] = True
        long_outcomes[active_abs] = no_resolution
        short_outcomes[active_abs] = no_resolution
        long_event_bars[active_abs] = horizon
        short_event_bars[active_abs] = horizon

        entry_close = close[active_abs]
        td = target_distances[active_abs]
        sd = stop_distances[active_abs]

        long_target = entry_close + td
        long_stop = entry_close - sd
        short_target = entry_close - td
        short_stop = entry_close + sd

        long_open = np.ones(len(active_abs), dtype=bool)
        short_open = np.ones(len(active_abs), dtype=bool)

        for step in range(1, horizon + 1):
            future_abs = positions[active_local + step]
            future_high = high[future_abs]
            future_low = low[future_abs]

            if np.any(long_open):
                lt = future_high >= long_target
                ls = future_low <= long_stop
                both = long_open & lt & ls
                target_only = long_open & lt & ~ls
                stop_only = long_open & ls & ~lt
                resolved = both | target_only | stop_only
                if np.any(both):
                    idx = active_abs[both]
                    long_outcomes[idx] = ambiguous
                    long_event_bars[idx] = step
                if np.any(target_only):
                    idx = active_abs[target_only]
                    long_outcomes[idx] = win
                    long_event_bars[idx] = step
                if np.any(stop_only):
                    idx = active_abs[stop_only]
                    long_outcomes[idx] = loss
                    long_event_bars[idx] = step
                long_open &= ~resolved

            if np.any(short_open):
                st = future_low <= short_target
                ss = future_high >= short_stop
                both = short_open & st & ss
                target_only = short_open & st & ~ss
                stop_only = short_open & ss & ~st
                resolved = both | target_only | stop_only
                if np.any(both):
                    idx = active_abs[both]
                    short_outcomes[idx] = ambiguous
                    short_event_bars[idx] = step
                if np.any(target_only):
                    idx = active_abs[target_only]
                    short_outcomes[idx] = win
                    short_event_bars[idx] = step
                if np.any(stop_only):
                    idx = active_abs[stop_only]
                    short_outcomes[idx] = loss
                    short_event_bars[idx] = step
                short_open &= ~resolved

            if not np.any(long_open) and not np.any(short_open):
                break

    labels = np.full(n, -1, dtype=np.int8)
    label_names = np.full(n, None, dtype=object)

    valid_idx = np.flatnonzero(valid_flags)
    if len(valid_idx):
        long_wins = long_outcomes[valid_idx] == win
        short_wins = short_outcomes[valid_idx] == win
        resolved_labels = np.full(len(valid_idx), int(TradeLabel.WAIT), dtype=np.int8)
        resolved_labels[long_wins & ~short_wins] = int(TradeLabel.LONG)
        resolved_labels[short_wins & ~long_wins] = int(TradeLabel.SHORT)
        labels[valid_idx] = resolved_labels

        names = np.full(len(valid_idx), TradeLabel.WAIT.name, dtype=object)
        names[resolved_labels == int(TradeLabel.LONG)] = TradeLabel.LONG.name
        names[resolved_labels == int(TradeLabel.SHORT)] = TradeLabel.SHORT.name
        label_names[valid_idx] = names

    label_values = [pd.NA if value < 0 else int(value) for value in labels]
    long_events = [pd.NA if value < 0 else int(value) for value in long_event_bars]
    short_events = [pd.NA if value < 0 else int(value) for value in short_event_bars]

    labeled["trade_label"] = pd.array(label_values, dtype="Int8")
    labeled["trade_label_name"] = [pd.NA if value is None else value for value in label_names]
    labeled["long_outcome"] = long_outcomes.tolist()
    labeled["short_outcome"] = short_outcomes.tolist()
    labeled["long_event_bars"] = pd.array(long_events, dtype="Int16")
    labeled["short_event_bars"] = pd.array(short_events, dtype="Int16")
    labeled["target_distance"] = target_distances
    labeled["stop_distance"] = stop_distances
    labeled["label_horizon_bars"] = config.horizon_bars
    labeled["label_valid"] = valid_flags

    return labeled
