"""Deterministic market-data source failover policy."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSourceState:
    name: str
    connected: bool
    stale: bool
    latency_ms: float


class MarketDataFailover:
    def __init__(self, preferred_order: list[str], *, max_latency_ms: float = 2000.0) -> None:
        if not preferred_order:
            raise ValueError("at least one market-data source is required")
        self.preferred_order = list(dict.fromkeys(preferred_order))
        self.max_latency_ms = max_latency_ms

    def choose(self, states: list[DataSourceState]) -> str | None:
        by_name = {item.name: item for item in states}
        for name in self.preferred_order:
            state = by_name.get(name)
            if state and state.connected and not state.stale and state.latency_ms <= self.max_latency_ms:
                return name
        return None
