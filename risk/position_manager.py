"""Local live-position state and broker reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from execution.broker import OrderIntent, OrderIntentType, OrderSide, OrderSnapshot, PositionSnapshot


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class ManagedPosition:
    symbol: str
    quantity: float
    average_entry_price: float
    opened_at: datetime
    current_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    source_client_order_id: str | None = None
    sector: str = "UNKNOWN"

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        if self.quantity == 0:
            raise ValueError("position quantity cannot be zero")
        if self.average_entry_price <= 0:
            raise ValueError("average_entry_price must be positive")
        if self.opened_at.tzinfo is None:
            self.opened_at = self.opened_at.replace(tzinfo=UTC)
        else:
            self.opened_at = self.opened_at.astimezone(UTC)

    @property
    def direction(self) -> str:
        return "LONG" if self.quantity > 0 else "SHORT"

    @property
    def absolute_quantity(self) -> float:
        return abs(self.quantity)

    @property
    def mark_price(self) -> float:
        return self.current_price if self.current_price and self.current_price > 0 else self.average_entry_price

    @property
    def notional(self) -> float:
        return self.absolute_quantity * self.mark_price

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["opened_at"] = self.opened_at.isoformat()
        result["direction"] = self.direction
        result["notional"] = self.notional
        return result


class LivePositionManager:
    def __init__(self) -> None:
        self._positions: dict[str, ManagedPosition] = {}

    def positions(self) -> list[ManagedPosition]:
        return list(self._positions.values())

    def get(self, symbol: str) -> ManagedPosition | None:
        return self._positions.get(symbol.upper())

    def mark_price(self, symbol: str, price: float) -> None:
        position = self.get(symbol)
        if position is not None and price > 0:
            position.current_price = float(price)

    def adopt_broker_positions(self, snapshots: list[PositionSnapshot], *, opened_at: datetime | None = None) -> None:
        opened_at = opened_at or utc_now()
        existing = self._positions
        rebuilt: dict[str, ManagedPosition] = {}
        for snapshot in snapshots:
            if snapshot.quantity == 0:
                continue
            previous = existing.get(snapshot.symbol)
            rebuilt[snapshot.symbol] = ManagedPosition(
                symbol=snapshot.symbol,
                quantity=snapshot.quantity,
                average_entry_price=snapshot.average_entry_price,
                opened_at=previous.opened_at if previous else opened_at,
                current_price=snapshot.market_price,
                stop_price=previous.stop_price if previous else None,
                target_price=previous.target_price if previous else None,
                source_client_order_id=previous.source_client_order_id if previous else None,
                sector=previous.sector if previous else "UNKNOWN",
            )
        self._positions = rebuilt

    def apply_filled_order(self, intent: OrderIntent, snapshot: OrderSnapshot) -> None:
        if snapshot.status.value != "FILLED" or snapshot.filled_quantity <= 0 or snapshot.average_fill_price is None:
            return
        symbol = intent.symbol
        signed_fill = float(snapshot.filled_quantity) * (1.0 if intent.side == OrderSide.BUY else -1.0)
        existing = self.get(symbol)
        if intent.intent_type == OrderIntentType.ENTRY:
            if existing is not None:
                new_qty = existing.quantity + signed_fill
                if new_qty == 0 or (existing.quantity > 0) != (new_qty > 0):
                    raise RuntimeError("entry fill would flatten/flip an existing position")
                total_cost = existing.average_entry_price * abs(existing.quantity) + snapshot.average_fill_price * abs(signed_fill)
                existing.quantity = new_qty
                existing.average_entry_price = total_cost / abs(new_qty)
                return
            metadata = intent.metadata
            self._positions[symbol] = ManagedPosition(
                symbol=symbol,
                quantity=signed_fill,
                average_entry_price=float(snapshot.average_fill_price),
                opened_at=snapshot.updated_at or snapshot.submitted_at or utc_now(),
                current_price=float(snapshot.average_fill_price),
                stop_price=metadata.get("stop_price"),
                target_price=metadata.get("target_price"),
                source_client_order_id=intent.client_order_id,
                sector=str(metadata.get("sector", "UNKNOWN")),
            )
            return

        if existing is None:
            raise RuntimeError("exit fill received for a position that is not locally open")
        new_qty = existing.quantity + signed_fill
        if existing.quantity > 0 and new_qty < -1e-9 or existing.quantity < 0 and new_qty > 1e-9:
            raise RuntimeError("exit fill would reverse the position")
        if abs(new_qty) < 1e-9:
            self._positions.pop(symbol, None)
        else:
            existing.quantity = new_qty

    def reconcile(self, snapshots: list[PositionSnapshot], *, quantity_tolerance: float = 1e-6) -> dict[str, Any]:
        broker = {item.symbol: item for item in snapshots if abs(item.quantity) > quantity_tolerance}
        local = self._positions
        mismatches: list[dict[str, Any]] = []
        for symbol in sorted(set(local) | set(broker)):
            lp = local.get(symbol)
            bp = broker.get(symbol)
            if lp is None:
                mismatches.append({"symbol": symbol, "reason": "broker_only", "broker_quantity": bp.quantity if bp else None})
                continue
            if bp is None:
                mismatches.append({"symbol": symbol, "reason": "local_only", "local_quantity": lp.quantity})
                continue
            if abs(lp.quantity - bp.quantity) > quantity_tolerance:
                mismatches.append({
                    "symbol": symbol,
                    "reason": "quantity_mismatch",
                    "local_quantity": lp.quantity,
                    "broker_quantity": bp.quantity,
                })
        return {"consistent": not mismatches, "mismatches": mismatches}

    def gross_exposure(self) -> float:
        return sum(position.notional for position in self._positions.values())

    def net_exposure(self) -> float:
        return sum(position.notional if position.direction == "LONG" else -position.notional for position in self._positions.values())

    def sector_exposure(self, sector: str) -> float:
        return sum(position.notional for position in self._positions.values() if position.sector == sector)
