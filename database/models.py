"""Persistence models for append-only paper-trading audit events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


UTC = timezone.utc


@dataclass(frozen=True)
class AuditEvent:
    event_id: int | None
    timestamp: datetime
    event_type: str
    payload: dict[str, Any]
    decision_id: str | None = None
    symbol: str | None = None

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        else:
            ts = ts.astimezone(UTC)
        object.__setattr__(self, "timestamp", ts)
        if self.symbol is not None:
            object.__setattr__(self, "symbol", self.symbol.upper())
