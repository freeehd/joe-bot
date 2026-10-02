"""Structured alerting that is safe to test without an external provider."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

UTC = timezone.utc


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    code: str
    message: str
    severity: AlertSeverity
    created_at_utc: str
    details: dict

    @classmethod
    def create(cls, code: str, message: str, severity: AlertSeverity, **details) -> "Alert":
        return cls(code, message, severity, datetime.now(tz=UTC).isoformat(), details)


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class InMemoryAlertSink:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def export(self) -> list[dict]:
        return [asdict(item) for item in self.alerts]
