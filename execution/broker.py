"""Broker-neutral paper execution contracts and an Alpaca paper adapter."""

from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _utc(value: Any | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderIntentType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        return self in {self.FILLED, self.CANCELED, self.REJECTED, self.EXPIRED}


_STATUS_MAP = {
    "new": OrderStatus.ACCEPTED,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.ACCEPTED,
    "pending_replace": OrderStatus.ACCEPTED,
    "stopped": OrderStatus.CANCELED,
    "suspended": OrderStatus.CANCELED,
    "calculated": OrderStatus.ACCEPTED,
}


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    quantity: int
    intent_type: OrderIntentType
    client_order_id: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        if self.quantity < 1:
            raise ValueError("quantity must be >= 1")
        if not self.client_order_id.strip():
            raise ValueError("client_order_id must not be empty")


@dataclass(frozen=True)
class OrderSnapshot:
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    status: OrderStatus
    average_fill_price: float | None = None
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    raw_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        if self.quantity < 0 or self.filled_quantity < 0:
            raise ValueError("order quantities must be non-negative")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled quantity cannot exceed order quantity")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["side"] = self.side.value
        result["status"] = self.status.value
        for key in ("submitted_at", "updated_at"):
            if result[key] is not None:
                result[key] = result[key].isoformat()
        return result


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    quantity: float
    average_entry_price: float
    market_price: float | None = None
    unrealized_pl: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    buying_power: float
    cash: float
    trading_blocked: bool = False


class BrokerProtocol(Protocol):
    paper: bool

    def submit_market_order(self, intent: OrderIntent) -> OrderSnapshot: ...
    def cancel_order(self, broker_order_id: str) -> None: ...
    def get_open_orders(self) -> list[OrderSnapshot]: ...
    def get_positions(self) -> list[PositionSnapshot]: ...
    def get_account(self) -> AccountSnapshot: ...


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def order_snapshot_from_alpaca(order: Any) -> OrderSnapshot:
    raw_status = str(_get(order, "status", "unknown"))
    if "." in raw_status:
        raw_status = raw_status.rsplit(".", 1)[-1]
    raw_status = raw_status.lower()
    raw_side = str(_get(order, "side", "buy"))
    if "." in raw_side:
        raw_side = raw_side.rsplit(".", 1)[-1]
    side = OrderSide.BUY if raw_side.lower() == "buy" else OrderSide.SELL
    qty = int(float(_get(order, "qty", 0) or 0))
    filled = int(float(_get(order, "filled_qty", 0) or 0))
    avg = _get(order, "filled_avg_price")
    return OrderSnapshot(
        broker_order_id=str(_get(order, "id", "")),
        client_order_id=str(_get(order, "client_order_id", "")),
        symbol=str(_get(order, "symbol", "")),
        side=side,
        quantity=qty,
        filled_quantity=filled,
        status=_STATUS_MAP.get(raw_status, OrderStatus.UNKNOWN),
        average_fill_price=None if avg in (None, "") else float(avg),
        submitted_at=_utc(_get(order, "submitted_at")),
        updated_at=_utc(_get(order, "updated_at")),
        raw_status=raw_status,
    )


class AlpacaPaperBroker:
    """Alpaca broker adapter that is *hard-wired* to paper mode.

    There is intentionally no ``paper=False`` option in this V0.9 adapter.
    Live-capital support belongs to a later gated phase and a separate class.
    """

    paper = True

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self._client = client

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca paper credentials are missing")
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("alpaca-py is required for paper execution") from exc
        self._client = TradingClient(self.api_key, self.secret_key, paper=True)
        return self._client

    def submit_market_order(self, intent: OrderIntent) -> OrderSnapshot:
        try:
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("alpaca-py is required for paper execution") from exc
        request = MarketOrderRequest(
            symbol=intent.symbol,
            qty=intent.quantity,
            side=AlpacaOrderSide.BUY if intent.side == OrderSide.BUY else AlpacaOrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=intent.client_order_id,
        )
        return order_snapshot_from_alpaca(self._load_client().submit_order(order_data=request))

    def cancel_order(self, broker_order_id: str) -> None:
        self._load_client().cancel_order_by_id(broker_order_id)

    def get_open_orders(self) -> list[OrderSnapshot]:
        client = self._load_client()
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except (ImportError, TypeError):  # pragma: no cover - compatibility fallback
            orders = client.get_orders()
        return [order_snapshot_from_alpaca(order) for order in orders]

    def get_positions(self) -> list[PositionSnapshot]:
        return [
            PositionSnapshot(
                symbol=str(position.symbol),
                quantity=float(position.qty),
                average_entry_price=float(position.avg_entry_price),
                market_price=None if getattr(position, "current_price", None) is None else float(position.current_price),
                unrealized_pl=None if getattr(position, "unrealized_pl", None) is None else float(position.unrealized_pl),
            )
            for position in self._load_client().get_all_positions()
        ]

    def get_account(self) -> AccountSnapshot:
        account = self._load_client().get_account()
        return AccountSnapshot(
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            cash=float(account.cash),
            trading_blocked=bool(getattr(account, "trading_blocked", False)),
        )


TradeUpdateHandler = Callable[[OrderSnapshot], None | Awaitable[None]]


class AlpacaPaperTradeStream:
    """Alpaca paper order-update WebSocket, separated from market data."""

    def __init__(
        self,
        handler: TradeUpdateHandler,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.handler = handler
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.stream_factory = stream_factory
        self._stream: Any | None = None
        self.connected = False
        self.last_error: str | None = None

    async def _on_update(self, update: Any) -> None:
        order = _get(update, "order", update)
        snapshot = order_snapshot_from_alpaca(order)
        result = self.handler(snapshot)
        if inspect.isawaitable(result):
            await result

    def run(self) -> None:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca paper credentials are missing")
        if self.stream_factory is not None:
            self._stream = self.stream_factory(self.api_key, self.secret_key)
        else:
            try:
                from alpaca.trading.stream import TradingStream
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("alpaca-py is required for paper trade updates") from exc
            self._stream = TradingStream(self.api_key, self.secret_key, paper=True)
        self._stream.subscribe_trade_updates(self._on_update)
        self.connected = True
        try:
            self._stream.run()
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)
            raise
        finally:
            self.connected = False

    def stop(self) -> None:
        if self._stream is not None:
            stop = getattr(self._stream, "stop", None)
            if callable(stop):
                stop()
        self.connected = False
