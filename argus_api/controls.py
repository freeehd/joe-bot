"""Operator-control boundary. Default adapter denies all mutations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from database.db import AuditStore
from risk.risk_engine import ProductionRiskEngine

UTC = timezone.utc


class OperatorControls(Protocol):
    def disable_entries(self, reason: str) -> dict: ...
    def enable_entries(self, reason: str) -> dict: ...


class ReadOnlyControls:
    def _denied(self, action: str) -> dict:
        return {"accepted": False, "action": action, "reason": "operator API is read-only; no runtime control adapter attached"}

    def disable_entries(self, reason: str) -> dict:
        return self._denied("disable_entries")

    def enable_entries(self, reason: str) -> dict:
        return self._denied("enable_entries")


@dataclass
class PaperRiskControls:
    risk_engine: ProductionRiskEngine
    audit: AuditStore

    def disable_entries(self, reason: str) -> dict:
        self.risk_engine.manual_halt(reason)
        self.audit.append("operator_control", {"action": "disable_entries", "reason": reason, "accepted": True}, timestamp=datetime.now(tz=UTC))
        return {"accepted": True, "action": "disable_entries", "reason": reason}

    def enable_entries(self, reason: str) -> dict:
        self.risk_engine.clear_manual_halt()
        self.audit.append("operator_control", {"action": "enable_entries", "reason": reason, "accepted": True}, timestamp=datetime.now(tz=UTC))
        return {"accepted": True, "action": "enable_entries", "reason": reason}
