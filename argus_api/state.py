"""Thread-safe operator state projection used by REST and WebSocket surfaces."""
from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass
class OperatorState:
    mode: str = "OFFLINE"
    account: dict[str, Any] = field(default_factory=lambda: {"equity": 0.0, "cash": 0.0, "buying_power": 0.0})
    performance: dict[str, Any] = field(default_factory=lambda: {"daily_pnl": 0.0, "drawdown": 0.0})
    regime: dict[str, Any] = field(default_factory=lambda: {"label": "UNKNOWN", "confidence": 0.0, "spy_return": 0.0, "qqq_return": 0.0, "breadth": 0.0})
    scanner: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    portfolio: dict[str, Any] = field(default_factory=lambda: {
        "gross_exposure": 0.0, "net_exposure": 0.0, "long_exposure": 0.0,
        "short_exposure": 0.0, "risk_used": 0.0, "risk_remaining": 1.0,
        "sector_exposure": {}, "correlation_clusters": [],
    })
    risk: dict[str, Any] = field(default_factory=lambda: {"entries_enabled": False, "kill_switches": ["offline"], "daily_loss_limit": 0.0})
    models: list[dict[str, Any]] = field(default_factory=list)
    system: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=_now)


class OperatorStateStore:
    def __init__(self, initial: OperatorState | None = None) -> None:
        self._lock = threading.RLock()
        self._state = initial or OperatorState()
        self._version = 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = deepcopy(self._state.__dict__)
            payload["version"] = self._version
            return payload

    def update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            for key, value in changes.items():
                if not hasattr(self._state, key):
                    raise AttributeError(f"unknown operator-state field {key!r}")
                setattr(self._state, key, deepcopy(value))
            self._state.updated_at = _now()
            self._version += 1
            return self.snapshot()
