"""Small deterministic supervisor for runtime service-health decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

UTC = timezone.utc


@dataclass
class ServiceState:
    name: str
    healthy: bool = True
    restart_count: int = 0
    last_heartbeat: datetime | None = None
    last_error: str | None = None
    restart_times: list[datetime] = field(default_factory=list)


class RuntimeSupervisor:
    def __init__(self, *, heartbeat_timeout_seconds: float = 15.0, max_restarts_per_hour: int = 3) -> None:
        if heartbeat_timeout_seconds <= 0 or max_restarts_per_hour < 0:
            raise ValueError("invalid supervisor limits")
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.max_restarts_per_hour = max_restarts_per_hour
        self.services: dict[str, ServiceState] = {}

    def heartbeat(self, name: str, *, now: datetime | None = None) -> None:
        now = now or datetime.now(tz=UTC)
        state = self.services.setdefault(name, ServiceState(name))
        state.healthy = True
        state.last_heartbeat = now
        state.last_error = None

    def fail(self, name: str, error: str) -> None:
        state = self.services.setdefault(name, ServiceState(name))
        state.healthy = False
        state.last_error = error

    def can_restart(self, name: str, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(tz=UTC)
        state = self.services.setdefault(name, ServiceState(name))
        cutoff = now - timedelta(hours=1)
        state.restart_times[:] = [item for item in state.restart_times if item >= cutoff]
        return len(state.restart_times) < self.max_restarts_per_hour

    def record_restart(self, name: str, *, now: datetime | None = None) -> None:
        now = now or datetime.now(tz=UTC)
        if not self.can_restart(name, now=now):
            raise RuntimeError(f"restart budget exhausted for {name}")
        state = self.services.setdefault(name, ServiceState(name))
        state.restart_count += 1
        state.restart_times.append(now)
        state.healthy = False

    def evaluate(self, *, now: datetime | None = None) -> dict:
        now = now or datetime.now(tz=UTC)
        rows = []
        all_healthy = True
        for state in self.services.values():
            stale = state.last_heartbeat is None or (now - state.last_heartbeat).total_seconds() > self.heartbeat_timeout_seconds
            healthy = bool(state.healthy and not stale)
            all_healthy &= healthy
            rows.append({**asdict(state), "last_heartbeat": state.last_heartbeat.isoformat() if state.last_heartbeat else None,
                         "restart_times": [t.isoformat() for t in state.restart_times], "stale": stale, "healthy": healthy})
        return {"healthy": all_healthy, "services": rows}
