"""Event-driven minute-bar trade simulator for V0.6 research.

Signals are assumed to be known only after the signal bar closes. The default
execution policy therefore enters on the next bar open and applies explicit
spread/slippage/fees. This module intentionally models single-trade execution
and same-symbol overlap; portfolio capital constraints belong to V0.7.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class ExecutionConfig:
    """Conservative execution assumptions for minute-bar simulation.

    ``spread_bps`` is the full quoted spread; half is paid on each execution.
    ``slippage_bps`` is additional adverse movement per execution.
    ``fee_bps`` is charged on entry and exit notional separately.
    """

    spread_bps: float = 4.0
    slippage_bps: float = 2.0
    fee_bps: float = 0.0
    entry_delay_bars: int = 1
    minimum_tick: float = 0.01
    same_bar_policy: Literal["stop", "skip"] = "stop"

    def __post_init__(self) -> None:
        if self.spread_bps < 0 or self.slippage_bps < 0 or self.fee_bps < 0:
            raise ValueError("execution costs must be non-negative")
        if self.entry_delay_bars < 1:
            raise ValueError("entry_delay_bars must be >= 1")
        if self.minimum_tick <= 0:
            raise ValueError("minimum_tick must be > 0")


@dataclass(frozen=True)
class TradeConfig:
    target_pct: float = 0.003
    stop_pct: float = 0.0015
    max_holding_bars: int = 10

    def __post_init__(self) -> None:
        if self.target_pct <= 0 or self.stop_pct <= 0:
            raise ValueError("target_pct and stop_pct must be > 0")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    side: Side
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    confidence: float
    p_wait: float
    p_long: float
    p_short: float
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    exit_reason: str
    holding_bars: int
    target_price: float
    stop_price: float
    mfe_return: float = 0.0
    mae_return: float = 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("signal_time", "entry_time", "exit_time"):
            payload[key] = payload[key].isoformat()
        return payload


def _ensure_index(raw_bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw_bars.index, pd.DatetimeIndex):
        raise ValueError("raw bars must use a DatetimeIndex")
    bars = raw_bars.sort_index().copy()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    required = {"open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"raw bars missing columns: {sorted(missing)}")
    return bars


def _adverse_tick(price: float, *, action: Literal["BUY", "SELL"], tick: float) -> float:
    units = price / tick
    if action == "BUY":
        return ceil(units - 1e-12) * tick
    return floor(units + 1e-12) * tick


def _execution_price(
    reference_price: float,
    *,
    action: Literal["BUY", "SELL"],
    config: ExecutionConfig,
) -> float:
    half_spread = config.spread_bps / 2.0 / 10_000.0
    slippage = config.slippage_bps / 10_000.0
    adverse = half_spread + slippage
    if action == "BUY":
        price = reference_price * (1.0 + adverse)
    else:
        price = reference_price * (1.0 - adverse)
    return float(_adverse_tick(price, action=action, tick=config.minimum_tick))


def _session_date(timestamp: pd.Timestamp):
    ts = timestamp
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(NY).date()


def _barrier_levels(entry_price: float, side: Side, trade: TradeConfig) -> tuple[float, float]:
    if side == "LONG":
        return entry_price * (1.0 + trade.target_pct), entry_price * (1.0 - trade.stop_pct)
    return entry_price * (1.0 - trade.target_pct), entry_price * (1.0 + trade.stop_pct)


def _raw_exit_reference(
    bar: pd.Series,
    *,
    side: Side,
    target_price: float,
    stop_price: float,
    same_bar_policy: str,
) -> tuple[str | None, float | None]:
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])

    if side == "LONG":
        # Gap-through logic is evaluated before intrabar high/low touches.
        if open_price <= stop_price:
            return "STOP", open_price
        if open_price >= target_price:
            return "TARGET", open_price
        target_hit = high >= target_price
        stop_hit = low <= stop_price
    else:
        if open_price >= stop_price:
            return "STOP", open_price
        if open_price <= target_price:
            return "TARGET", open_price
        target_hit = low <= target_price
        stop_hit = high >= stop_price

    if target_hit and stop_hit:
        if same_bar_policy == "skip":
            return "AMBIGUOUS", None
        return "STOP", stop_price
    if target_hit:
        return "TARGET", target_price
    if stop_hit:
        return "STOP", stop_price
    return None, None


def simulate_trade(
    raw_bars: pd.DataFrame,
    *,
    symbol: str,
    signal_time: pd.Timestamp,
    side: Side,
    confidence: float,
    p_wait: float,
    p_long: float,
    p_short: float,
    trade_config: TradeConfig,
    execution_config: ExecutionConfig,
) -> TradeResult | None:
    """Simulate one signal using only bars available after signal creation."""

    bars = _ensure_index(raw_bars)
    signal = pd.Timestamp(signal_time)
    if signal.tzinfo is None:
        signal = signal.tz_localize("UTC")
    else:
        signal = signal.tz_convert("UTC")

    matches = np.flatnonzero(bars.index == signal)
    if len(matches) != 1:
        return None
    signal_pos = int(matches[0])
    entry_pos = signal_pos + execution_config.entry_delay_bars
    if entry_pos >= len(bars):
        return None

    entry_time = bars.index[entry_pos]
    if _session_date(entry_time) != _session_date(signal):
        # Never turn a near-close signal into an unplanned overnight entry.
        return None

    entry_reference = float(bars.iloc[entry_pos]["open"])
    entry_action = "BUY" if side == "LONG" else "SELL"
    exit_action = "SELL" if side == "LONG" else "BUY"
    entry_price = _execution_price(entry_reference, action=entry_action, config=execution_config)
    target_price, stop_price = _barrier_levels(entry_price, side, trade_config)

    exit_reason = "TIME"
    exit_reference: float | None = None
    exit_time = entry_time
    holding_bars = 0

    session = _session_date(entry_time)
    last_position = min(entry_pos + trade_config.max_holding_bars - 1, len(bars) - 1)

    for position in range(entry_pos, last_position + 1):
        timestamp = bars.index[position]
        if _session_date(timestamp) != session:
            break
        holding_bars = position - entry_pos + 1
        reason, reference = _raw_exit_reference(
            bars.iloc[position],
            side=side,
            target_price=target_price,
            stop_price=stop_price,
            same_bar_policy=execution_config.same_bar_policy,
        )
        if reason == "AMBIGUOUS":
            return None
        if reason is not None:
            exit_reason = reason
            exit_reference = reference
            exit_time = timestamp
            break

        exit_time = timestamp
        exit_reference = float(bars.iloc[position]["close"])

    if holding_bars == 0 or exit_reference is None:
        return None

    # If the configured horizon ran past the session, the latest available
    # same-session close is a SESSION_CLOSE exit rather than an overnight hold.
    planned_end_pos = entry_pos + trade_config.max_holding_bars - 1
    if exit_reason == "TIME" and planned_end_pos < len(bars):
        if _session_date(bars.index[planned_end_pos]) != session:
            exit_reason = "SESSION_CLOSE"
    elif exit_reason == "TIME" and planned_end_pos >= len(bars) - 1:
        exit_reason = "DATA_END"

    excursion_slice = bars.iloc[entry_pos : entry_pos + holding_bars]
    if side == "LONG":
        mfe_return = float(excursion_slice["high"].astype(float).max() / entry_price - 1.0)
        mae_return = float(excursion_slice["low"].astype(float).min() / entry_price - 1.0)
    else:
        mfe_return = float((entry_price - excursion_slice["low"].astype(float).min()) / entry_price)
        mae_return = float((entry_price - excursion_slice["high"].astype(float).max()) / entry_price)

    exit_price = _execution_price(float(exit_reference), action=exit_action, config=execution_config)
    if side == "LONG":
        gross_return = exit_price / entry_price - 1.0
    else:
        gross_return = (entry_price - exit_price) / entry_price

    fee_return = 2.0 * execution_config.fee_bps / 10_000.0
    net_return = gross_return - fee_return

    return TradeResult(
        symbol=symbol,
        side=side,
        signal_time=signal,
        entry_time=entry_time,
        exit_time=exit_time,
        confidence=float(confidence),
        p_wait=float(p_wait),
        p_long=float(p_long),
        p_short=float(p_short),
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        gross_return=float(gross_return),
        net_return=float(net_return),
        exit_reason=exit_reason,
        holding_bars=int(holding_bars),
        target_price=float(target_price),
        stop_price=float(stop_price),
        mfe_return=float(mfe_return),
        mae_return=float(mae_return),
    )


def run_signal_backtest(
    signals: pd.DataFrame,
    *,
    raw_loader: Callable[[str], pd.DataFrame],
    trade_config: TradeConfig,
    execution_config: ExecutionConfig | None = None,
    confidence_threshold: float = 0.60,
    symbol_column: str = "training_symbol",
    enforce_one_position_per_symbol: bool = True,
) -> list[TradeResult]:
    """Run directional model signals through the minute-bar execution engine."""

    execution_config = execution_config or ExecutionConfig()
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    required = {symbol_column, "p_wait", "p_long", "p_short", "direction", "confidence"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {sorted(missing)}")

    results: list[TradeResult] = []
    for symbol, symbol_signals in signals.sort_index().groupby(symbol_column, sort=True):
        raw = raw_loader(str(symbol))
        last_exit: pd.Timestamp | None = None
        for timestamp, row in symbol_signals.sort_index().iterrows():
            direction = str(row["direction"])
            confidence = float(row["confidence"])
            if direction not in {"LONG", "SHORT"} or confidence < confidence_threshold:
                continue
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            if enforce_one_position_per_symbol and last_exit is not None and ts < last_exit:
                continue

            result = simulate_trade(
                raw,
                symbol=str(symbol),
                signal_time=ts,
                side=direction,  # type: ignore[arg-type]
                confidence=confidence,
                p_wait=float(row["p_wait"]),
                p_long=float(row["p_long"]),
                p_short=float(row["p_short"]),
                trade_config=trade_config,
                execution_config=execution_config,
            )
            if result is not None:
                results.append(result)
                last_exit = result.exit_time
    return sorted(results, key=lambda trade: (trade.signal_time, trade.symbol))
