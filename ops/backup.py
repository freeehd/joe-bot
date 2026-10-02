"""Verified SQLite backup/restore helpers for runtime audit/state databases."""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class BackupResult:
    source: str
    backup: str
    sha256: str
    created_at_utc: str


def backup_sqlite(source: str | Path, destination: str | Path) -> BackupResult:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with sqlite3.connect(source) as src, sqlite3.connect(tmp) as dst:
        src.backup(dst)
    tmp.replace(destination)
    return BackupResult(str(source), str(destination), _sha256(destination), datetime.now(tz=UTC).isoformat())


def restore_sqlite(backup: str | Path, destination: str | Path, *, expected_sha256: str | None = None) -> Path:
    backup = Path(backup)
    destination = Path(destination)
    if not backup.is_file():
        raise FileNotFoundError(backup)
    if expected_sha256 is not None and _sha256(backup) != expected_sha256:
        raise RuntimeError("backup checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(backup, tmp)
    # Open and run a cheap integrity check before atomic replacement.
    with sqlite3.connect(tmp) as conn:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"restored SQLite integrity check failed: {result}")
    tmp.replace(destination)
    return destination
