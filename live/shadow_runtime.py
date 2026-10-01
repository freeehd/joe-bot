"""Single-stream runtime for V0.95 shadow production."""
from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from live.shadow_engine import ShadowTradingEngine
from market.stream import MarketStreamProtocol

UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class ShadowRuntimeConfig:
    watchdog_interval_seconds: float = 1.0
    checkpoint_interval_seconds: float = 15.0


class ShadowRuntime:
    def __init__(self, engine: ShadowTradingEngine, market_stream: MarketStreamProtocol, *, config: ShadowRuntimeConfig | None = None,
                 clock: Callable[[], datetime] = utc_now, sleeper: Callable[[float], None] = time.sleep) -> None:
        self.engine = engine
        self.market_stream = market_stream
        self.config = config or ShadowRuntimeConfig()
        self.clock = clock
        self.sleeper = sleeper
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_checkpoint: datetime | None = None

    def _stream_target(self) -> None:
        try:
            self.market_stream.run()
        except Exception as exc:  # pragma: no cover
            self.engine.market_stream_unstable(str(exc))
            self.engine.audit.append("shadow_market_stream_crashed", {"error": str(exc)}, timestamp=self.clock())
            self._stop.set()

    def start(self) -> None:
        now = self.clock()
        self.engine.start(now=now)
        self.engine.risk_engine.update_health(broker_connected=True)
        self._last_checkpoint = now
        self._thread = threading.Thread(target=self._stream_target, name="joe-shadow-market-stream", daemon=True)
        self._thread.start()

    def watchdog_once(self) -> None:
        now = self.clock()
        stale = self.market_stream.health.is_stale(self.engine.risk_engine.limits.stale_market_data_seconds, now=now)
        self.engine.risk_engine.update_health(
            broker_connected=True,
            websocket_stable=self.market_stream.health.connected and not stale,
            position_state_consistent=True,
        )
        if self._last_checkpoint is None or (now - self._last_checkpoint).total_seconds() >= self.config.checkpoint_interval_seconds:
            self.engine.audit.append(
                "shadow_checkpoint",
                {
                    "market_stream": self.market_stream.health.snapshot(self.engine.risk_engine.limits.stale_market_data_seconds, now=now),
                    "broker": self.engine.broker.snapshot(),
                    "kill_switches": [reason.value for reason in self.engine.risk_engine.active_kill_switches(now)],
                },
                timestamp=now,
            )
            self._last_checkpoint = now

    def run_forever(self) -> None:
        self.start()
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda *_: self.stop("signal"))
            except (ValueError, OSError):
                pass
        try:
            while not self._stop.is_set():
                self.watchdog_once()
                self.sleeper(self.config.watchdog_interval_seconds)
        finally:
            self.stop("runtime shutdown")
            for sig, handler in previous.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def stop(self, reason: str = "operator shutdown") -> None:
        self._stop.set()
        self.market_stream.stop()
        if self.engine.started:
            self.engine.stop(reason, now=self.clock())
