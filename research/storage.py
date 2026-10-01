"""Parquet-backed historical data lake for research datasets."""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


class ParquetDataLake:
    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    def version_exists(self, version: str) -> bool:
        return (
            (self.manifest_dir / f"{version}.json").exists()
            or (self.root / "raw" / "datasets" / version).exists()
            or (self.root / "processed" / "datasets" / version).exists()
        )

    def assert_new_version(self, version: str) -> None:
        if self.version_exists(version):
            raise FileExistsError(
                f"Dataset version {version!r} already exists. "
                "Dataset versions are immutable; choose a new version name."
            )

    def raw_partition_path(self, version: str, symbol: str, year: int, month: int) -> Path:
        return (
            self.root
            / "raw"
            / "datasets"
            / version
            / "bars"
            / f"symbol={symbol.upper()}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "bars.parquet"
        )

    def processed_partition_path(self, version: str, symbol: str, year: int, month: int) -> Path:
        return (
            self.root
            / "processed"
            / "datasets"
            / version
            / f"symbol={symbol.upper()}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "training.parquet"
        )

    @staticmethod
    def _session_parts(df: pd.DataFrame):
        index = df.index
        if index.tz is None:
            index = index.tz_localize("UTC")
        local = index.tz_convert(NY)
        keys = pd.DataFrame({"year": local.year, "month": local.month}, index=df.index)
        for (year, month), positions in keys.groupby(["year", "month"]).groups.items():
            yield int(year), int(month), df.loc[positions].sort_index()

    def write_raw_bars(self, version: str, symbol: str, df: pd.DataFrame) -> list[str]:
        paths: list[str] = []
        for year, month, partition in self._session_parts(df):
            path = self.raw_partition_path(version, symbol, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                partition.to_parquet(path, index=True)
            except ImportError as exc:
                raise RuntimeError("Parquet support requires pyarrow. Install requirements.txt.") from exc
            paths.append(str(path))
        return paths


    def load_raw_bars(self, version: str, symbol: str) -> pd.DataFrame:
        base = self.root / "raw" / "datasets" / version / "bars" / f"symbol={symbol.upper()}"
        paths = sorted(base.glob("year=*/month=*/bars.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No raw bars found for dataset {version!r}, symbol {symbol!r}"
            )
        try:
            frames = [pd.read_parquet(path) for path in paths]
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow. Install requirements.txt.") from exc
        return pd.concat(frames).sort_index()

    def write_processed_training(self, version: str, symbol: str, df: pd.DataFrame) -> list[str]:
        paths: list[str] = []
        for year, month, partition in self._session_parts(df):
            path = self.processed_partition_path(version, symbol, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                partition.to_parquet(path, index=True)
            except ImportError as exc:
                raise RuntimeError("Parquet support requires pyarrow. Install requirements.txt.") from exc
            paths.append(str(path))
        return paths

    def load_processed_dataset(self, version: str) -> pd.DataFrame:
        base = self.root / "processed" / "datasets" / version
        paths = sorted(base.glob("symbol=*/year=*/month=*/training.parquet"))
        if not paths:
            raise FileNotFoundError(f"No processed dataset found for version {version!r}")
        try:
            frames = [pd.read_parquet(path) for path in paths]
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow. Install requirements.txt.") from exc
        return pd.concat(frames).sort_index()

    def write_manifest(self, version: str, manifest: dict) -> Path:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_dir / f"{version}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return path

    def read_manifest(self, version: str) -> dict:
        path = self.manifest_dir / f"{version}.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
