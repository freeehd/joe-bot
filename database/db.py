"""SQLite append-only audit/replay store for V0.9 paper execution."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from database.models import AuditEvent


UTC = timezone.utc

def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class AuditStore:
    def __init__(self, path: str | Path = "data/paper/audit.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._closed = False
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    decision_id TEXT,
                    symbol TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_events(decision_id, event_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_symbol ON audit_events(symbol, event_id)")

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        decision_id: str | None = None,
        symbol: str | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEvent:
        ts = timestamp or datetime.now(tz=UTC)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        else:
            ts = ts.astimezone(UTC)
        encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO audit_events(timestamp,event_type,decision_id,symbol,payload_json) VALUES(?,?,?,?,?)",
                (ts.isoformat(), event_type, decision_id, symbol.upper() if symbol else None, encoded),
            )
            event_id = int(cursor.lastrowid)
        return AuditEvent(event_id, ts, event_type, payload, decision_id=decision_id, symbol=symbol)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> AuditEvent:
        return AuditEvent(
            event_id=int(row["event_id"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=str(row["event_type"]),
            payload=json.loads(row["payload_json"]),
            decision_id=row["decision_id"],
            symbol=row["symbol"],
        )

    def get(self, event_id: int) -> AuditEvent | None:
        row = self._conn.execute("SELECT * FROM audit_events WHERE event_id=?", (event_id,)).fetchone()
        return None if row is None else self._row_to_event(row)

    def events_for_decision(self, decision_id: str) -> list[AuditEvent]:
        rows = self._conn.execute(
            "SELECT * FROM audit_events WHERE decision_id=? ORDER BY event_id", (decision_id,)
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def replay(self, decision_id: str) -> dict[str, Any]:
        events = self.events_for_decision(decision_id)
        return {
            "decision_id": decision_id,
            "event_count": len(events),
            "events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "symbol": event.symbol,
                    "payload": event.payload,
                }
                for event in events
            ],
        }

    def iter_events(self, *, event_type: str | None = None) -> Iterable[AuditEvent]:
        if event_type is None:
            rows = self._conn.execute("SELECT * FROM audit_events ORDER BY event_id")
        else:
            rows = self._conn.execute("SELECT * FROM audit_events WHERE event_type=? ORDER BY event_id", (event_type,))
        for row in rows:
            yield self._row_to_event(row)

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "AuditStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
