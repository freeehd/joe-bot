"""Independent production-style risk controls and V0.9 entry kill switches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


UTC = timezone.utc


class KillSwitchReason(str, Enum):
    MANUAL_HALT = "manual_halt"
    MARKET_DATA_STALE = "market_data_stale"
    BROKER_DISCONNECTED = "broker_disconnected"
    CLOCK_DRIFT = "clock_drift"
    WEBSOCKET_UNSTABLE = "websocket_unstable"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_FEATURES = "invalid_features"
    POSITION_STATE_INCONSISTENT = "position_state_inconsistent"
    DAILY_LOSS = "daily_loss"
    DAILY_DRAWDOWN = "daily_drawdown"
    DAILY_TRADE_LIMIT = "daily_trade_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    UNEXPECTED_VOLATILITY = "unexpected_volatility"
    BROKER_TRADING_BLOCKED = "broker_trading_blocked"


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade_fraction: float = 0.0025
    max_position_fraction: float = 0.30
    max_positions: int = 3
    max_gross_exposure_fraction: float = 0.70
    max_net_exposure_fraction: float = 0.70
    max_sector_exposure_fraction: float = 0.40
    max_daily_loss_fraction: float = 0.01
    max_daily_drawdown_fraction: float = 0.015
    max_daily_trades: int = 30
    max_consecutive_losses: int = 5
    max_spread_bps: float = 25.0
    min_liquidity_dollars: float = 0.0
    max_volatility_fraction: float = 0.10
    max_clock_drift_seconds: float = 2.0
    stale_market_data_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.max_positions < 1 or self.max_daily_trades < 1 or self.max_consecutive_losses < 1:
            raise ValueError("integer risk limits must be positive")
        for name in (
            "risk_per_trade_fraction", "max_position_fraction", "max_gross_exposure_fraction",
            "max_net_exposure_fraction", "max_sector_exposure_fraction", "max_daily_loss_fraction",
            "max_daily_drawdown_fraction", "max_volatility_fraction",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class SystemHealth:
    broker_connected: bool = False
    websocket_stable: bool = True
    model_ready: bool = True
    features_valid: bool = True
    position_state_consistent: bool = True
    unexpected_volatility: bool = False
    broker_trading_blocked: bool = False
    clock_drift_seconds: float = 0.0
    last_market_event_at: datetime | None = None


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    kill_switches: tuple[str, ...] = ()


class ProductionRiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.health = SystemHealth()
        self.session_start_equity: float | None = None
        self.peak_equity: float | None = None
        self.current_equity: float | None = None
        self.daily_trades = 0
        self.consecutive_losses = 0
        self._manual_halt_reason: str | None = None

    def reset_session(self, starting_equity: float) -> None:
        if starting_equity <= 0:
            raise ValueError("starting_equity must be > 0")
        self.session_start_equity = starting_equity
        self.peak_equity = starting_equity
        self.current_equity = starting_equity
        self.daily_trades = 0
        self.consecutive_losses = 0
        self._manual_halt_reason = None

    def update_equity(self, equity: float) -> None:
        if equity <= 0:
            raise ValueError("equity must be > 0")
        self.current_equity = equity
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)

    def record_closed_trade(self, pnl_dollars: float, *, equity: float | None = None) -> None:
        self.daily_trades += 1
        self.consecutive_losses = self.consecutive_losses + 1 if pnl_dollars < 0 else 0
        if equity is not None:
            self.update_equity(equity)

    def manual_halt(self, reason: str = "operator requested halt") -> None:
        self._manual_halt_reason = reason

    def clear_manual_halt(self) -> None:
        self._manual_halt_reason = None

    def mark_market_event(self, timestamp: datetime) -> None:
        self.health.last_market_event_at = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)

    def update_health(self, **values: Any) -> None:
        for name, value in values.items():
            if not hasattr(self.health, name):
                raise AttributeError(f"unknown health field {name!r}")
            setattr(self.health, name, value)

    def active_kill_switches(self, now: datetime | None = None) -> list[KillSwitchReason]:
        now = now or datetime.now(tz=UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        else:
            now = now.astimezone(UTC)
        reasons: list[KillSwitchReason] = []
        h = self.health
        if self._manual_halt_reason is not None:
            reasons.append(KillSwitchReason.MANUAL_HALT)
        if not h.broker_connected:
            reasons.append(KillSwitchReason.BROKER_DISCONNECTED)
        if not h.websocket_stable:
            reasons.append(KillSwitchReason.WEBSOCKET_UNSTABLE)
        if not h.model_ready:
            reasons.append(KillSwitchReason.MODEL_UNAVAILABLE)
        if not h.features_valid:
            reasons.append(KillSwitchReason.INVALID_FEATURES)
        if not h.position_state_consistent:
            reasons.append(KillSwitchReason.POSITION_STATE_INCONSISTENT)
        if h.unexpected_volatility:
            reasons.append(KillSwitchReason.UNEXPECTED_VOLATILITY)
        if h.broker_trading_blocked:
            reasons.append(KillSwitchReason.BROKER_TRADING_BLOCKED)
        if abs(h.clock_drift_seconds) > self.limits.max_clock_drift_seconds:
            reasons.append(KillSwitchReason.CLOCK_DRIFT)
        if h.last_market_event_at is None or (now - h.last_market_event_at).total_seconds() > self.limits.stale_market_data_seconds:
            reasons.append(KillSwitchReason.MARKET_DATA_STALE)

        if self.session_start_equity and self.current_equity is not None:
            if self.current_equity <= self.session_start_equity * (1 - self.limits.max_daily_loss_fraction):
                reasons.append(KillSwitchReason.DAILY_LOSS)
        if self.peak_equity and self.current_equity is not None:
            if self.current_equity <= self.peak_equity * (1 - self.limits.max_daily_drawdown_fraction):
                reasons.append(KillSwitchReason.DAILY_DRAWDOWN)
        if self.daily_trades >= self.limits.max_daily_trades:
            reasons.append(KillSwitchReason.DAILY_TRADE_LIMIT)
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            reasons.append(KillSwitchReason.CONSECUTIVE_LOSSES)
        return list(dict.fromkeys(reasons))

    def approve_entry(
        self,
        *,
        account_equity: float,
        proposed_notional: float,
        open_positions: int,
        gross_exposure: float,
        net_exposure: float,
        sector_exposure: float,
        direction: str,
        spread_bps: float = 0.0,
        liquidity_dollars: float = float("inf"),
        volatility_fraction: float = 0.0,
        now: datetime | None = None,
    ) -> RiskDecision:
        self.update_equity(account_equity)
        kills = self.active_kill_switches(now)
        if kills:
            return RiskDecision(False, "entry kill switch active", tuple(reason.value for reason in kills))
        if open_positions >= self.limits.max_positions:
            return RiskDecision(False, "maximum open positions reached")
        if proposed_notional > account_equity * self.limits.max_position_fraction:
            return RiskDecision(False, "position notional limit exceeded")
        if gross_exposure + proposed_notional > account_equity * self.limits.max_gross_exposure_fraction:
            return RiskDecision(False, "gross exposure limit exceeded")
        signed = proposed_notional if direction == "LONG" else -proposed_notional
        if abs(net_exposure + signed) > account_equity * self.limits.max_net_exposure_fraction:
            return RiskDecision(False, "net exposure limit exceeded")
        if sector_exposure + proposed_notional > account_equity * self.limits.max_sector_exposure_fraction:
            return RiskDecision(False, "sector exposure limit exceeded")
        if spread_bps > self.limits.max_spread_bps:
            return RiskDecision(False, "spread limit exceeded")
        if liquidity_dollars < self.limits.min_liquidity_dollars:
            return RiskDecision(False, "liquidity below minimum")
        if volatility_fraction > self.limits.max_volatility_fraction:
            return RiskDecision(False, "volatility limit exceeded")
        return RiskDecision(True, "risk checks passed")

    def approve_exit(self) -> RiskDecision:
        # Critical invariant: entry kill switches must never block risk-reducing exits.
        return RiskDecision(True, "exits remain enabled")

    def calculate_position_size(self, account_equity: float, entry_price: float, stop_price: float) -> int:
        risk_amount = account_equity * self.limits.risk_per_trade_fraction
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        risk_qty = int(risk_amount / risk_per_share)
        notional_qty = int((account_equity * self.limits.max_position_fraction) / entry_price)
        return max(0, min(risk_qty, notional_qty))

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        return {
            "limits": asdict(self.limits),
            "health": {
                **asdict(self.health),
                "last_market_event_at": self.health.last_market_event_at.isoformat() if self.health.last_market_event_at else None,
            },
            "session_start_equity": self.session_start_equity,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "manual_halt_reason": self._manual_halt_reason,
            "active_kill_switches": [reason.value for reason in self.active_kill_switches(now)],
        }


class RiskEngine:
    """Compatibility wrapper for the original v0.3 risk API."""

    def __init__(self, max_positions: int = 1, max_daily_loss_percent: float = 1.0, risk_per_trade_percent: float = 0.25) -> None:
        self.max_positions = max_positions
        self.max_daily_loss_percent = max_daily_loss_percent
        self.risk_per_trade_percent = risk_per_trade_percent

    def approve_trade(self, decision, account_equity, daily_pnl, open_positions):
        if not decision["approved"]:
            return {"approved": False, "reason": "Strategy did not approve trade."}
        if open_positions >= self.max_positions:
            return {"approved": False, "reason": "Maximum open positions reached."}
        max_daily_loss = account_equity * (self.max_daily_loss_percent / 100)
        if daily_pnl <= -max_daily_loss:
            return {"approved": False, "reason": "Daily loss limit reached."}
        return {"approved": True, "reason": "Risk checks passed."}

    def calculate_position_size(self, account_equity, entry_price, stop_price):
        risk_amount = account_equity * (self.risk_per_trade_percent / 100)
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        return int(risk_amount / risk_per_share)
