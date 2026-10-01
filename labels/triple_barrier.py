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

    labels: list[object] = []
    label_names: list[object] = []
    long_outcomes: list[str] = []
    short_outcomes: list[str] = []
    long_event_bars: list[object] = []
    short_event_bars: list[object] = []
    target_distances: list[float | np.floating] = []
    stop_distances: list[float | np.floating] = []
    valid_flags: list[bool] = []

    if respect_sessions and isinstance(labeled.index, pd.DatetimeIndex):
        session_keys = pd.Series(labeled.index.normalize(), index=labeled.index)
    else:
        session_keys = pd.Series(0, index=labeled.index)

    session_positions: dict[object, list[int]] = {}
    for absolute_position, session_key in enumerate(session_keys.to_numpy()):
        session_positions.setdefault(session_key, []).append(absolute_position)

    position_lookup: dict[int, tuple[pd.DataFrame, int]] = {}
    for positions in session_positions.values():
        session_frame = labeled.iloc[positions]
        for local_position, absolute_position in enumerate(positions):
            position_lookup[absolute_position] = (session_frame, local_position)

    for position in range(len(labeled)):
        distances = _distances_for_row(labeled.iloc[position], config)

        if distances is None:
            long_outcome = BarrierOutcome.INSUFFICIENT_FUTURE
            short_outcome = BarrierOutcome.INSUFFICIENT_FUTURE
            long_event = None
            short_event = None
            target_distance = np.nan
            stop_distance = np.nan
        else:
            target_distance, stop_distance = distances
            evaluation_frame, local_position = position_lookup[position]
            long_outcome, long_event = evaluate_barriers(
                evaluation_frame,
                local_position,
                side="LONG",
                horizon_bars=config.horizon_bars,
                target_distance=target_distance,
                stop_distance=stop_distance,
            )
            short_outcome, short_event = evaluate_barriers(
                evaluation_frame,
                local_position,
                side="SHORT",
                horizon_bars=config.horizon_bars,
                target_distance=target_distance,
                stop_distance=stop_distance,
            )

        valid = (
            long_outcome != BarrierOutcome.INSUFFICIENT_FUTURE
            and short_outcome != BarrierOutcome.INSUFFICIENT_FUTURE
        )

        if valid:
            label = _resolve_multiclass_label(long_outcome, short_outcome)
            labels.append(int(label))
            label_names.append(label.name)
        else:
            labels.append(pd.NA)
            label_names.append(pd.NA)

        long_outcomes.append(str(long_outcome))
        short_outcomes.append(str(short_outcome))
        long_event_bars.append(long_event if long_event is not None else pd.NA)
        short_event_bars.append(short_event if short_event is not None else pd.NA)
        target_distances.append(target_distance)
        stop_distances.append(stop_distance)
        valid_flags.append(valid)

    labeled["trade_label"] = pd.array(labels, dtype="Int8")
    labeled["trade_label_name"] = label_names
    labeled["long_outcome"] = long_outcomes
    labeled["short_outcome"] = short_outcomes
    labeled["long_event_bars"] = pd.array(long_event_bars, dtype="Int16")
    labeled["short_event_bars"] = pd.array(short_event_bars, dtype="Int16")
    labeled["target_distance"] = target_distances
    labeled["stop_distance"] = stop_distances
    labeled["label_horizon_bars"] = config.horizon_bars
    labeled["label_valid"] = valid_flags

    return labeled
