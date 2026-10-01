"""Live market-data event types, health tracking, and Alpaca WebSocket adapter.

The core engine depends on the small protocol in this module rather than on
Alpaca directly.  The optional Alpaca runtime is imported lazily so unit tests
and historical research stay broker/network independent.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol, Sequence


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class BarEvent:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: float | None = None
    vwap: float | None = None
    received_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "timestamp", _utc(self.timestamp))
        object.__setattr__(self, "received_at", _utc(self.received_at or utc_now()))
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("bar prices must be positive")
        if self.volume < 0:
            raise ValueError("bar volume must be non-negative")
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError("bar OHLC values are inconsistent")


@dataclass(frozen=True)
class QuoteEvent:
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    received_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "timestamp", _utc(self.timestamp))
        object.__setattr__(self, "received_at", _utc(self.received_at or utc_now()))
        if self.bid_price < 0 or self.ask_price < 0:
            raise ValueError("quote prices must be non-negative")
        if self.ask_price and self.bid_price and self.ask_price < self.bid_price:
            raise ValueError("ask_price must be >= bid_price")

    @property
    def mid(self) -> float | None:
        if self.bid_price <= 0 or self.ask_price <= 0:
            return None
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0:
            return None
        return (self.ask_price - self.bid_price) / mid * 10_000.0


MarketEvent = BarEvent | QuoteEvent
MarketEventHandler = Callable[[MarketEvent], None | Awaitable[None]]


@dataclass
class StreamHealth:
    connected: bool = False
    connected_at: datetime | None = None
    disconnected_at: datetime | None = None
    last_event_at: datetime | None = None
    last_error: str | None = None
    reconnect_count: int = 0
    event_count: int = 0

    def mark_connected(self, now: datetime | None = None) -> None:
        now = _utc(now or utc_now())
        if self.connected_at is not None and not self.connected:
            self.reconnect_count += 1
        self.connected = True
        self.connected_at = now
        self.last_error = None

    def mark_disconnected(self, error: Exception | str | None = None, now: datetime | None = None) -> None:
        self.connected = False
        self.disconnected_at = _utc(now or utc_now())
        if error is not None:
            self.last_error = str(error)

    def mark_event(self, event: MarketEvent) -> None:
        self.last_event_at = _utc(event.received_at or event.timestamp)
        self.event_count += 1

    def is_stale(self, max_age_seconds: float, now: datetime | None = None) -> bool:
        if not self.connected or self.last_event_at is None:
            return True
        age = (_utc(now or utc_now()) - self.last_event_at).total_seconds()
        return age > max_age_seconds

    def snapshot(self, max_age_seconds: float | None = None, now: datetime | None = None) -> dict[str, Any]:
        result = asdict(self)
        for key in ("connected_at", "disconnected_at", "last_event_at"):
            if result[key] is not None:
                result[key] = result[key].isoformat()
        if max_age_seconds is not None:
            result["stale"] = self.is_stale(max_age_seconds, now=now)
        return result


class MarketStreamProtocol(Protocol):
    health: StreamHealth

    def run(self) -> None: ...

    def stop(self) -> None: ...


async def _dispatch(handler: MarketEventHandler, event: MarketEvent) -> None:
    result = handler(event)
    if inspect.isawaitable(result):
        await result


class AlpacaStockStream:
    """Paper/research market-data WebSocket adapter.

    This class only carries market data; it never has trading credentials or an
    order method.  The actual Alpaca SDK is imported on first ``run()``.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        handler: MarketEventHandler,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: str = "iex",
        subscribe_quotes: bool = True,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        self.handler = handler
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.feed = feed.lower()
        self.subscribe_quotes_enabled = subscribe_quotes
        self.stream_factory = stream_factory
        self.health = StreamHealth()
        self._stream: Any | None = None

    def _build_stream(self) -> Any:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca market-data credentials are missing")
        if self.stream_factory is not None:
            return self.stream_factory(self.api_key, self.secret_key, self.feed)
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.live import StockDataStream
        except ImportError as exc:  # pragma: no cover - optional network runtime
            raise RuntimeError("alpaca-py is required for the live market stream") from exc
        feed_enum = DataFeed.SIP if self.feed == "sip" else DataFeed.IEX
        return StockDataStream(self.api_key, self.secret_key, feed=feed_enum)

    async def _on_bar(self, bar: Any) -> None:
        event = BarEvent(
            symbol=str(bar.symbol),
            timestamp=_utc(bar.timestamp),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            trade_count=None if getattr(bar, "trade_count", None) is None else float(bar.trade_count),
            vwap=None if getattr(bar, "vwap", None) is None else float(bar.vwap),
        )
        self.health.mark_event(event)
        await _dispatch(self.handler, event)

    async def _on_quote(self, quote: Any) -> None:
        event = QuoteEvent(
            symbol=str(quote.symbol),
            timestamp=_utc(quote.timestamp),
            bid_price=float(quote.bid_price),
            ask_price=float(quote.ask_price),
            bid_size=float(getattr(quote, "bid_size", 0.0) or 0.0),
            ask_size=float(getattr(quote, "ask_size", 0.0) or 0.0),
        )
        self.health.mark_event(event)
        await _dispatch(self.handler, event)

    def run(self) -> None:
        self._stream = self._build_stream()
        self._stream.subscribe_bars(self._on_bar, *self.symbols)
        if self.subscribe_quotes_enabled:
            self._stream.subscribe_quotes(self._on_quote, *self.symbols)
        self.health.mark_connected()
        try:
            self._stream.run()
        except Exception as exc:  # pragma: no cover - network behavior
            self.health.mark_disconnected(exc)
            raise
        finally:
            if self.health.connected:
                self.health.mark_disconnected()

    def stop(self) -> None:
        if self._stream is None:
            return
        stop = getattr(self._stream, "stop", None)
        if callable(stop):
            result = stop()
            if inspect.isawaitable(result):  # pragma: no cover - runtime-dependent
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(result)
                else:
                    loop.create_task(result)
        self.health.mark_disconnected()
