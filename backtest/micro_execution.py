"""Quote-aware partial/missed-fill simulator for Sprint 7 execution research."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["BUY", "SELL"]


class ExecutionPolicy(str, Enum):
    MARKET = "MARKET"
    MARKETABLE_LIMIT = "MARKETABLE_LIMIT"
    PASSIVE_LIMIT = "PASSIVE_LIMIT"


@dataclass(frozen=True)
class ExecutionOrder:
    side: Side
    quantity: int
    decision_price: float
    policy: ExecutionPolicy
    limit_price: float | None = None
    max_participation: float = 0.25

    def __post_init__(self) -> None:
        if self.quantity < 1 or self.decision_price <= 0:
            raise ValueError("quantity and decision_price must be positive")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0,1]")
        if self.policy != ExecutionPolicy.MARKET and self.limit_price is None:
            raise ValueError("limit policy requires limit_price")


@dataclass(frozen=True)
class FillSimulation:
    requested_quantity: int
    filled_quantity: int
    average_fill_price: float | None
    fill_rate: float
    cost_bps: float | None
    missed: bool
    partial: bool
    events_consumed: int


def _cost_bps(side: Side, decision: float, fill: float) -> float:
    signed = (fill / decision - 1.0) if side == "BUY" else (decision / fill - 1.0)
    return float(signed * 10_000.0)


def simulate_execution(order: ExecutionOrder, quotes: pd.DataFrame, trades: pd.DataFrame | None = None) -> FillSimulation:
    q = quotes.sort_index().copy()
    required = {"bid_price", "ask_price", "bid_size", "ask_size"}
    if required.difference(q.columns):
        raise ValueError(f"quotes missing columns: {sorted(required.difference(q.columns))}")
    remaining = order.quantity
    fills: list[tuple[int, float]] = []
    consumed = 0

    if order.policy in {ExecutionPolicy.MARKET, ExecutionPolicy.MARKETABLE_LIMIT}:
        for row in q.itertuples():
            consumed += 1
            price = float(row.ask_price if order.side == "BUY" else row.bid_price)
            available = int(max(0.0, float(row.ask_size if order.side == "BUY" else row.bid_size)) * order.max_participation)
            if available <= 0:
                continue
            if order.policy == ExecutionPolicy.MARKETABLE_LIMIT:
                limit = float(order.limit_price)
                if order.side == "BUY" and price > limit:
                    continue
                if order.side == "SELL" and price < limit:
                    continue
            qty = min(remaining, max(1, available))
            fills.append((qty, price))
            remaining -= qty
            if remaining <= 0:
                break
    else:
        if trades is None:
            raise ValueError("PASSIVE_LIMIT simulation requires trades")
        t = trades.sort_index()
        limit = float(order.limit_price)
        for row in t.itertuples():
            consumed += 1
            price = float(row.price)
            touched = price <= limit if order.side == "BUY" else price >= limit
            if not touched:
                continue
            available = int(max(0.0, float(row.size)) * order.max_participation)
            if available <= 0:
                continue
            qty = min(remaining, max(1, available))
            fills.append((qty, limit))
            remaining -= qty
            if remaining <= 0:
                break

    filled = order.quantity - remaining
    if filled <= 0:
        return FillSimulation(order.quantity, 0, None, 0.0, None, True, False, consumed)
    average = float(sum(qty * price for qty, price in fills) / filled)
    return FillSimulation(
        requested_quantity=order.quantity,
        filled_quantity=filled,
        average_fill_price=average,
        fill_rate=float(filled / order.quantity),
        cost_bps=_cost_bps(order.side, order.decision_price, average),
        missed=False,
        partial=filled < order.quantity,
        events_consumed=consumed,
    )
