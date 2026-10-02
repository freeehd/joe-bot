"""Atomic runtime state snapshots with integrity checks for crash recovery."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class RecoverySnapshot:
    created_at_utc: str
    sequence: int
    state: dict[str, Any]
    checksum_sha256: str


class RecoveryStore:
    def __init__(self, path: str | Path = "data/runtime/recovery.json") -> None:
        self.path = Path(path)

    def save(self, state: dict[str, Any], *, sequence: int) -> RecoverySnapshot:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        created = datetime.now(tz=UTC).isoformat()
        payload = {"created_at_utc": created, "sequence": int(sequence), "state": state}
        checksum = hashlib.sha256(_canonical(payload)).hexdigest()
        record = {**payload, "checksum_sha256": checksum}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        return RecoverySnapshot(created, sequence, state, checksum)

    def load(self) -> RecoverySnapshot | None:
        if not self.path.exists():
            return None
        record = json.loads(self.path.read_text(encoding="utf-8"))
        checksum = record.pop("checksum_sha256", None)
        actual = hashlib.sha256(_canonical(record)).hexdigest()
        if checksum != actual:
            raise RuntimeError("recovery snapshot failed checksum verification")
        return RecoverySnapshot(
            created_at_utc=str(record["created_at_utc"]),
            sequence=int(record["sequence"]),
            state=dict(record["state"]),
            checksum_sha256=actual,
        )
