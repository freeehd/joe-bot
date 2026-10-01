"""Deterministic live/paper exit policy used by the V0.9 state engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from risk.position_manager import ManagedPosition


UTC = timezone.utc


class ExitReason(str, Enum):
    TARGET = "TARGET"
    STOP = "STOP"
    MAX_HOLDING_TIME = "MAX_HOLDING_TIME"
    SESSION_FLATTEN = "SESSION_FLATTEN"


@dataclass(frozen=True)
class ExitPolicy:
    target_pct: float = 0.003
    stop_pct: float = 0.0015
    max_holding_minutes: int = 20
    flatten_before_close_minutes: int = 5

    def __post_init__(self) -> None:
        if self.target_pct <= 0 or self.stop_pct <= 0:
            raise ValueError("target_pct and stop_pct must be > 0")
        if self.max_holding_minutes < 1:
            raise ValueError("max_holding_minutes must be >= 1")
        if self.flatten_before_close_minutes < 0:
            raise ValueError("flatten_before_close_minutes must be non-negative")


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: ExitReason | None = None
    trigger_price: float | None = None


class ExitEngine:
    def __init__(self, policy: ExitPolicy | None = None) -> None:
        self.policy = policy or ExitPolicy()

    def levels(self, direction: str, entry_price: float) -> tuple[float, float]:
        if direction == "LONG":
            return entry_price * (1 + self.policy.target_pct), entry_price * (1 - self.policy.stop_pct)
        if direction == "SHORT":
            return entry_price * (1 - self.policy.target_pct), entry_price * (1 + self.policy.stop_pct)
        raise ValueError("direction must be LONG or SHORT")

    def evaluate(
        self,
        position: ManagedPosition,
        *,
        price: float,
        now: datetime,
        session_close: datetime | None = None,
    ) -> ExitDecision:
        if price <= 0:
            return ExitDecision(False)
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
        target = position.target_price
        stop = position.stop_price
        if target is None or stop is None:
            target, stop = self.levels(position.direction, position.average_entry_price)

        if position.direction == "LONG":
            if price <= stop:
                return ExitDecision(True, ExitReason.STOP, price)
            if price >= target:
                return ExitDecision(True, ExitReason.TARGET, price)
        else:
            if price >= stop:
                return ExitDecision(True, ExitReason.STOP, price)
            if price <= target:
                return ExitDecision(True, ExitReason.TARGET, price)

        if now - position.opened_at >= timedelta(minutes=self.policy.max_holding_minutes):
            return ExitDecision(True, ExitReason.MAX_HOLDING_TIME, price)

        if session_close is not None:
            close = session_close.replace(tzinfo=UTC) if session_close.tzinfo is None else session_close.astimezone(UTC)
            if now >= close - timedelta(minutes=self.policy.flatten_before_close_minutes):
                return ExitDecision(True, ExitReason.SESSION_FLATTEN, price)
        return ExitDecision(False)
