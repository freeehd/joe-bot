"""Idempotent order lifecycle management for the V0.9 paper engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from execution.broker import BrokerProtocol, OrderIntent, OrderSnapshot, OrderStatus


_ALLOWED_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED: {OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.UNKNOWN},
    OrderStatus.SUBMITTED: {OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN},
    OrderStatus.ACCEPTED: {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.UNKNOWN},
    OrderStatus.UNKNOWN: set(OrderStatus),
    OrderStatus.FILLED: {OrderStatus.FILLED},
    OrderStatus.CANCELED: {OrderStatus.CANCELED},
    OrderStatus.REJECTED: {OrderStatus.REJECTED},
    OrderStatus.EXPIRED: {OrderStatus.EXPIRED},
}


class OrderLifecycleError(RuntimeError):
    pass


@dataclass
class ManagedOrder:
    intent: OrderIntent
    snapshot: OrderSnapshot | None = None

    @property
    def status(self) -> OrderStatus:
        return OrderStatus.CREATED if self.snapshot is None else self.snapshot.status


class OrderManager:
    def __init__(self, broker: BrokerProtocol) -> None:
        if not getattr(broker, "paper", False):
            raise ValueError("V0.9 OrderManager requires a paper broker")
        self.broker = broker
        self._by_client_id: dict[str, ManagedOrder] = {}
        self._by_broker_id: dict[str, str] = {}

    def submit(self, intent: OrderIntent) -> OrderSnapshot:
        existing = self._by_client_id.get(intent.client_order_id)
        if existing is not None:
            if existing.intent != intent:
                raise OrderLifecycleError("client_order_id reused for a different intent")
            if existing.snapshot is None:
                raise OrderLifecycleError("duplicate submission encountered before broker acknowledgement")
            return existing.snapshot

        managed = ManagedOrder(intent=intent)
        self._by_client_id[intent.client_order_id] = managed
        try:
            snapshot = self.broker.submit_market_order(intent)
        except Exception:
            # Retain the intent for audit/idempotency. A retry must use a new
            # reconciliation path rather than blindly submitting a duplicate.
            raise
        self.apply_update(snapshot)
        return snapshot

    def apply_update(self, snapshot: OrderSnapshot) -> ManagedOrder:
        managed = self._by_client_id.get(snapshot.client_order_id)
        if managed is None:
            raise OrderLifecycleError(f"unknown client_order_id {snapshot.client_order_id!r}")
        previous = managed.status
        if snapshot.status not in _ALLOWED_TRANSITIONS.get(previous, set()):
            raise OrderLifecycleError(f"invalid order transition {previous.value} -> {snapshot.status.value}")
        if snapshot.symbol != managed.intent.symbol or snapshot.side != managed.intent.side:
            raise OrderLifecycleError("broker update does not match original order intent")
        if snapshot.quantity != managed.intent.quantity:
            raise OrderLifecycleError("broker quantity does not match original order intent")
        if managed.snapshot is not None and snapshot.filled_quantity < managed.snapshot.filled_quantity:
            raise OrderLifecycleError("filled quantity moved backwards")
        managed.snapshot = snapshot
        if snapshot.broker_order_id:
            self._by_broker_id[snapshot.broker_order_id] = snapshot.client_order_id
        return managed

    def reconcile_open_orders(self) -> dict[str, Any]:
        broker_orders = self.broker.get_open_orders()
        seen: set[str] = set()
        unknown: list[dict[str, Any]] = []
        for snapshot in broker_orders:
            seen.add(snapshot.client_order_id)
            if snapshot.client_order_id in self._by_client_id:
                self.apply_update(snapshot)
            else:
                unknown.append(snapshot.to_dict())
        local_live = [
            client_id
            for client_id, managed in self._by_client_id.items()
            if not managed.status.terminal and managed.status != OrderStatus.CREATED
        ]
        missing = sorted(client_id for client_id in local_live if client_id not in seen)
        return {"consistent": not unknown and not missing, "unknown_broker_orders": unknown, "missing_local_orders": missing}

    def get(self, client_order_id: str) -> ManagedOrder | None:
        return self._by_client_id.get(client_order_id)

    def live_orders(self) -> list[ManagedOrder]:
        return [order for order in self._by_client_id.values() if not order.status.terminal]

    def has_live_entry(self, symbol: str) -> bool:
        symbol = symbol.upper()
        return any(
            order.intent.symbol == symbol
            and order.intent.intent_type.value == "ENTRY"
            and not order.status.terminal
            for order in self._by_client_id.values()
        )
