"""Correlation and exposure utilities for V0.7 portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationConfig:
    lookback_rows: int = 20_000
    minimum_periods: int = 100
    correlation_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.lookback_rows < 2:
            raise ValueError("lookback_rows must be >= 2")
        if self.minimum_periods < 2:
            raise ValueError("minimum_periods must be >= 2")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")


def correlation_matrix_from_feature_rows(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "training_symbol",
    return_column: str = "return_1m",
    config: CorrelationConfig | None = None,
) -> pd.DataFrame:
    """Build a leakage-safe cross-symbol correlation matrix from prior rows."""

    config = config or CorrelationConfig()
    required = {symbol_column, return_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must use a DatetimeIndex")

    work = frame[[symbol_column, return_column]].copy().sort_index()
    if len(work) > config.lookback_rows:
        # Preserve all symbols for the latest timestamps rather than slicing a
        # symbol-grouped frame independently.
        latest_times = pd.DatetimeIndex(work.index.unique()).sort_values()
        keep_times = latest_times[-config.lookback_rows :]
        work = work.loc[work.index.isin(keep_times)]

    pivot = work.pivot_table(
        index=work.index,
        columns=symbol_column,
        values=return_column,
        aggfunc="last",
    )
    return pivot.corr(min_periods=config.minimum_periods).fillna(0.0)


def directional_correlation(
    correlation: float,
    candidate_side: str,
    existing_side: str,
) -> float:
    """Translate price correlation into P&L-factor correlation.

    A LONG/LONG pair in positively correlated names is concentrated. A
    LONG/SHORT pair in those same names is closer to a hedge. Negative price
    correlation reverses those relationships.
    """

    if candidate_side not in {"LONG", "SHORT"} or existing_side not in {"LONG", "SHORT"}:
        raise ValueError("sides must be LONG or SHORT")
    candidate_sign = 1.0 if candidate_side == "LONG" else -1.0
    existing_sign = 1.0 if existing_side == "LONG" else -1.0
    return float(correlation * candidate_sign * existing_sign)


def candidate_correlation_penalty(
    *,
    symbol: str,
    side: str,
    positions: list[dict],
    correlations: pd.DataFrame | None,
) -> tuple[float, float]:
    """Return (diversification multiplier, max directional correlation)."""

    if not positions or correlations is None or correlations.empty:
        return 1.0, 0.0

    max_directional = 0.0
    for position in positions:
        other = str(position["symbol"])
        if symbol == other:
            max_directional = 1.0
            continue
        if symbol not in correlations.index or other not in correlations.columns:
            corr = 0.0
        else:
            corr = float(correlations.loc[symbol, other])
            if not np.isfinite(corr):
                corr = 0.0
        directional = directional_correlation(corr, side, str(position["direction"]))
        max_directional = max(max_directional, directional)

    # Only concentration-like positive P&L correlation is penalized. Hedges do
    # not receive a bonus; their multiplier simply remains 1.
    penalty = max(0.0, min(max_directional, 1.0))
    multiplier = 1.0 - 0.65 * penalty
    return float(max(multiplier, 0.20)), float(max_directional)
