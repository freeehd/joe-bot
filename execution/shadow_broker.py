"""Network-isolated virtual broker for V0.95 shadow production.

The broker consumes normalized market events and simulates orders locally.  It
contains no broker SDK imports, credentials, endpoints, or network methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from backtest.engine import ExecutionConfig, _execution_price
from execution.broker import (
    AccountSnapshot,
    OrderIntent,
    OrderSide,
    OrderSnapshot,
    OrderStatus,
    PositionSnapshot,
)
from market.stream import BarEvent, MarketEvent

UTC = timezone.utc


@dataclass
class _PendingOrder:
    intent: OrderIntent
    submitted_at: datetime
    broker_order_id: str


class ShadowBroker:
    """Local-only execution simulator implementing the broker protocol.

    Market orders are acknowledged immediately and filled on the next eligible
    bar open, using the same adverse spread/slippage/tick convention as V0.6.
    """

    paper = True
    shadow = True
    external_orders = False

    def __init__(
        self,
        *,
        initial_equity: float = 10_000.0,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        self.execution_config = execution_config or ExecutionConfig()
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self._positions: dict[str, PositionSnapshot] = {}
        self._last_prices: dict[str, float] = {}
        self._pending: dict[str, _PendingOrder] = {}
        self._orders: dict[str, OrderSnapshot] = {}
        self._sequence = 0

    def submit_market_order(self, intent: OrderIntent) -> OrderSnapshot:
        if intent.client_order_id in self._pending or intent.client_order_id in self._orders:
            existing = self._orders.get(intent.client_order_id)
            if existing is not None:
                return existing
            raise RuntimeError("shadow client_order_id already pending")
        self._sequence += 1
        raw_signal_time = intent.metadata.get("signal_time") if isinstance(intent.metadata, dict) else None
        now = datetime.fromisoformat(raw_signal_time) if raw_signal_time else datetime.now(tz=UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        else:
            now = now.astimezone(UTC)
        broker_id = f"shadow-{self._sequence}"
        self._pending[intent.client_order_id] = _PendingOrder(intent, now, broker_id)
        snapshot = OrderSnapshot(
            broker_order_id=broker_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            filled_quantity=0,
            status=OrderStatus.ACCEPTED,
            submitted_at=now,
            updated_at=now,
            raw_status="shadow_accepted",
        )
        self._orders[intent.client_order_id] = snapshot
        return snapshot

    def cancel_order(self, broker_order_id: str) -> None:
        for client_id, pending in list(self._pending.items()):
            if pending.broker_order_id != broker_order_id:
                continue
            previous = self._orders[client_id]
            self._orders[client_id] = OrderSnapshot(
                broker_order_id=previous.broker_order_id,
                client_order_id=client_id,
                symbol=previous.symbol,
                side=previous.side,
                quantity=previous.quantity,
                filled_quantity=previous.filled_quantity,
                status=OrderStatus.CANCELED,
                submitted_at=previous.submitted_at,
                updated_at=datetime.now(tz=UTC),
                raw_status="shadow_canceled",
            )
            self._pending.pop(client_id, None)
            return

    def get_open_orders(self) -> list[OrderSnapshot]:
        return [self._orders[cid] for cid in self._pending]

    def get_positions(self) -> list[PositionSnapshot]:
        result: list[PositionSnapshot] = []
        for symbol, position in self._positions.items():
            mark = self._last_prices.get(symbol, position.average_entry_price)
            signed_pl = (mark - position.average_entry_price) * position.quantity
            result.append(
                PositionSnapshot(
                    symbol=symbol,
                    quantity=position.quantity,
                    average_entry_price=position.average_entry_price,
                    market_price=mark,
                    unrealized_pl=signed_pl,
                )
            )
        return result

    def get_account(self) -> AccountSnapshot:
        marked_value = sum(position.quantity * self._last_prices.get(symbol, position.average_entry_price)
                           for symbol, position in self._positions.items())
        equity = self.cash + marked_value
        return AccountSnapshot(equity=float(equity), buying_power=float(max(self.cash, 0.0)), cash=float(self.cash))

    def _apply_fill(self, intent: OrderIntent, fill_price: float) -> None:
        signed_qty = float(intent.quantity) * (1.0 if intent.side == OrderSide.BUY else -1.0)
        notional = abs(signed_qty * fill_price)
        fee = notional * self.execution_config.fee_bps / 10_000.0
        self.cash -= signed_qty * fill_price
        self.cash -= fee
        previous = self._positions.get(intent.symbol)
        if previous is None:
            if abs(signed_qty) > 0:
                self._positions[intent.symbol] = PositionSnapshot(intent.symbol, signed_qty, fill_price, market_price=fill_price)
            return
        new_qty = previous.quantity + signed_qty
        if abs(new_qty) < 1e-9:
            self._positions.pop(intent.symbol, None)
            return
        if (previous.quantity > 0) == (signed_qty > 0):
            avg = (
                previous.average_entry_price * abs(previous.quantity)
                + fill_price * abs(signed_qty)
            ) / abs(new_qty)
        else:
            avg = previous.average_entry_price
        self._positions[intent.symbol] = PositionSnapshot(intent.symbol, new_qty, avg, market_price=fill_price)

    def on_market_event(self, event: MarketEvent) -> list[OrderSnapshot]:
        if isinstance(event, BarEvent):
            self._last_prices[event.symbol] = event.close
        else:
            mid = event.mid
            if mid is not None:
                self._last_prices[event.symbol] = mid
            return []

        updates: list[OrderSnapshot] = []
        for client_id, pending in list(self._pending.items()):
            intent = pending.intent
            if intent.symbol != event.symbol or event.timestamp <= pending.submitted_at:
                continue
            action = "BUY" if intent.side == OrderSide.BUY else "SELL"
            fill_price = _execution_price(float(event.open), action=action, config=self.execution_config)
            self._apply_fill(intent, fill_price)
            snapshot = OrderSnapshot(
                broker_order_id=pending.broker_order_id,
                client_order_id=client_id,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                filled_quantity=intent.quantity,
                status=OrderStatus.FILLED,
                average_fill_price=fill_price,
                submitted_at=pending.submitted_at,
                updated_at=event.timestamp,
                raw_status="shadow_filled",
            )
            self._orders[client_id] = snapshot
            self._pending.pop(client_id, None)
            updates.append(snapshot)
        return updates

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "shadow",
            "external_orders": False,
            "account": self.get_account().__dict__,
            "positions": [item.__dict__ for item in self.get_positions()],
            "open_orders": [item.to_dict() for item in self.get_open_orders()],
        }
