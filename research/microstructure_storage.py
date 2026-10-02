"""Immutable quote/trade partition storage for Sprint 7 microstructure research."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.artifact_registry import sha256_file


class MicrostructureStore:
    def __init__(self, root: str | Path = "data/microstructure") -> None:
        self.root = Path(root)

    def _path(self, version: str, kind: str, symbol: str, date: str) -> Path:
        if kind not in {"quotes", "trades"}:
            raise ValueError("kind must be quotes or trades")
        return self.root / version / kind / f"symbol={symbol.upper()}" / f"date={date}" / f"{kind}.parquet"

    def write_partition(self, version: str, kind: str, symbol: str, date: str, frame: pd.DataFrame) -> Path:
        path = self._path(version, kind, symbol, date)
        if path.exists():
            raise FileExistsError(f"microstructure partition already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_index().to_parquet(path)
        return path

    def write_manifest(self, version: str, files: list[Path], metadata: dict) -> Path:
        path = self.root / version / "manifest.json"
        if path.exists():
            raise FileExistsError(f"microstructure version {version!r} is immutable")
        payload = {
            "version": version,
            "files": [{"path": str(file), "sha256": sha256_file(file), "size_bytes": file.stat().st_size} for file in sorted(files)],
            "metadata": metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
