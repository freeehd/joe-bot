"""Threaded V0.9 paper runtime wiring market/order streams to the state engine."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from live.paper_engine import PaperTradingEngine
from market.stream import MarketStreamProtocol


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


class TradeStreamProtocol(Protocol):
    connected: bool
    last_error: str | None

    def run(self) -> None: ...
    def stop(self) -> None: ...


@dataclass(frozen=True)
class RuntimeConfig:
    watchdog_interval_seconds: float = 1.0
    reconciliation_interval_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.watchdog_interval_seconds <= 0 or self.reconciliation_interval_seconds <= 0:
            raise ValueError("runtime intervals must be > 0")


class PaperRuntime:
    def __init__(
        self,
        engine: PaperTradingEngine,
        market_stream: MarketStreamProtocol,
        trade_stream: TradeStreamProtocol,
        *,
        config: RuntimeConfig | None = None,
        clock: Callable[[], datetime] = utc_now,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.engine = engine
        self.market_stream = market_stream
        self.trade_stream = trade_stream
        self.config = config or RuntimeConfig()
        self.clock = clock
        self.sleeper = sleeper
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_reconciliation: datetime | None = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _stream_target(self, name: str, run: Callable[[], None], disconnect: Callable[[str | None], None]) -> None:
        try:
            run()
        except Exception as exc:  # pragma: no cover - live runtime behavior
            disconnect(str(exc))
            self.engine.audit.append(f"{name}_crashed", {"error": str(exc)}, timestamp=self.clock())
            self._stop.set()

    def start(self) -> None:
        self.engine.start(now=self.clock())
        self._last_reconciliation = self.clock()
        self._threads = [
            threading.Thread(
                target=self._stream_target,
                args=("market_stream", self.market_stream.run, self.engine.market_stream_unstable),
                name="joe-market-stream",
                daemon=True,
            ),
            threading.Thread(
                target=self._stream_target,
                args=("trade_stream", self.trade_stream.run, self.engine.broker_disconnected),
                name="joe-trade-stream",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def watchdog_once(self) -> None:
        now = self.clock()
        market_connected = self.market_stream.health.connected
        stale = self.market_stream.health.is_stale(
            self.engine.risk_engine.limits.stale_market_data_seconds,
            now=now,
        )
        self.engine.risk_engine.update_health(
            websocket_stable=market_connected and not stale,
            broker_connected=bool(self.trade_stream.connected),
        )
        self.engine.audit.append(
            "runtime_health",
            {
                "market_stream": self.market_stream.health.snapshot(
                    self.engine.risk_engine.limits.stale_market_data_seconds,
                    now=now,
                ),
                "trade_stream_connected": bool(self.trade_stream.connected),
                "trade_stream_error": self.trade_stream.last_error,
                "kill_switches": [reason.value for reason in self.engine.risk_engine.active_kill_switches(now)],
            },
            timestamp=now,
        )

        if self._last_reconciliation is None or (
            now - self._last_reconciliation
        ).total_seconds() >= self.config.reconciliation_interval_seconds:
            try:
                self.engine.reconcile_account(timestamp=now)
                self.engine.reconcile_positions(timestamp=now)
                order_result = self.engine.orders.reconcile_open_orders()
                if not order_result["consistent"]:
                    self.engine.risk_engine.update_health(position_state_consistent=False)
                    self.engine.audit.append("order_reconciliation_mismatch", order_result, timestamp=now)
            except Exception as exc:  # pragma: no cover - broker/network runtime
                self.engine.broker_disconnected(str(exc))
            self._last_reconciliation = now

    def run_forever(self) -> None:
        self.start()
        previous_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda *_: self.stop("signal"))
            except (ValueError, OSError):  # not main thread / unsupported platform
                pass
        try:
            while not self._stop.is_set():
                self.watchdog_once()
                self.sleeper(self.config.watchdog_interval_seconds)
        finally:
            self.stop("runtime shutdown")
            for sig, handler in previous_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def stop(self, reason: str = "operator shutdown") -> None:
        if self._stop.is_set() and not self.engine.started:
            return
        self._stop.set()
        try:
            self.market_stream.stop()
        finally:
            self.trade_stream.stop()
            if self.engine.started:
                self.engine.stop(reason, now=self.clock())
