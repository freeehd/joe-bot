"""Project append-only Joe Bot audit events into GUI-friendly state."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Sequence

from database.db import AuditStore
from database.models import AuditEvent
from argus_api.state import OperatorStateStore


class AuditProjector:
    def __init__(self, store: OperatorStateStore, *, max_orders: int = 100, max_alerts: int = 100) -> None:
        self.store = store
        self.max_orders = max_orders
        self.max_alerts = max_alerts

    def project_event(self, event: AuditEvent) -> None:
        state = self.store.snapshot()
        orders = deque(state["orders"], maxlen=self.max_orders)
        alerts = deque(state["alerts"], maxlen=self.max_alerts)
        positions = state["positions"]
        risk = state["risk"]
        system = state["system"]
        mode = state["mode"]
        account = state["account"]

        if event.event_type in {"engine_started", "shadow_mode_started"}:
            mode = "SHADOW" if event.event_type == "shadow_mode_started" else "PAPER"
            account_payload = event.payload.get("account") if isinstance(event.payload, dict) else None
            if isinstance(account_payload, dict):
                account = account_payload
        elif event.event_type == "account_reconciliation":
            account = dict(event.payload)
        elif event.event_type in {"entry_order_submitted", "exit_order_submitted", "order_update"}:
            orders.append({
                "timestamp": event.timestamp.isoformat(), "event_type": event.event_type,
                "symbol": event.symbol, "decision_id": event.decision_id, "payload": event.payload,
            })
        elif event.event_type == "position_reconciliation":
            if not event.payload.get("consistent", False):
                alerts.append({"severity": "CRITICAL", "type": "position_mismatch", "timestamp": event.timestamp.isoformat(), "payload": event.payload})
        elif event.event_type in {"broker_disconnected", "market_stream_unstable", "market_stream_crashed", "trade_stream_crashed", "shadow_market_stream_crashed", "order_reconciliation_mismatch"}:
            alerts.append({"severity": "CRITICAL", "type": event.event_type, "timestamp": event.timestamp.isoformat(), "payload": event.payload})
        elif event.event_type == "runtime_health":
            kill_switches = list(event.payload.get("kill_switches", []))
            risk = {**risk, "kill_switches": kill_switches, "entries_enabled": not kill_switches}
            market = event.payload.get("market_stream", {})
            system = [
                {"service": "market stream", "status": "UP" if market.get("connected") and not market.get("stale") else "DEGRADED", "last_heartbeat": market.get("last_event_at"), "error": market.get("last_error")},
                {"service": "broker stream", "status": "UP" if event.payload.get("trade_stream_connected") else "DEGRADED", "last_heartbeat": event.timestamp.isoformat(), "error": event.payload.get("trade_stream_error")},
            ]

        self.store.update(mode=mode, account=account, orders=list(orders), positions=positions, risk=risk, system=system, alerts=list(alerts))

    def load_audits(self, paths: Sequence[str | Path]) -> None:
        events: list[AuditEvent] = []
        for path in paths:
            with AuditStore(path) as audit:
                events.extend(list(audit.iter_events()))
        for event in sorted(events, key=lambda item: (item.timestamp, item.event_id or 0)):
            self.project_event(event)
