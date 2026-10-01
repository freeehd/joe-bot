"""Rebuild processed features/labels from an immutable raw dataset snapshot."""

from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from config.settings import (
    LABEL_HORIZON_BARS,
    V04_CALIBRATION_FRACTION,
    V04_TRAIN_FRACTION,
)
from features.schema import FEATURE_COLUMNS
from models.train_multiclass import barrier_config_from_settings, build_training_frame
from research.build_dataset import _split_periods
from research.diagnostics import (
    class_distribution,
    hourly_label_distribution,
    outcome_distribution,
    per_symbol_distribution,
)
from research.storage import ParquetDataLake


def rebuild_from_raw(
    *,
    source_version: str,
    new_version: str,
    root: str = "data",
    use_source_label_config: bool = False,
) -> dict:
    lake = ParquetDataLake(root)
    lake.assert_new_version(new_version)
    source_manifest = lake.read_manifest(source_version)
    raw_snapshot_version = source_manifest.get("raw_snapshot_version", source_version)

    if use_source_label_config:
        from labels.triple_barrier import BarrierConfig

        barrier_config = BarrierConfig(**source_manifest["label_parameters"])
    else:
        barrier_config = barrier_config_from_settings()

    frames: list[pd.DataFrame] = []
    processed_files: set[str] = set()
    processed_rows_by_symbol: dict[str, int] = {}
    failed_symbols: dict[str, str] = {}

    symbols = list(source_manifest["symbols_succeeded"])
    print("\n" + "=" * 78)
    print(f"REBUILDING {new_version} FROM RAW SNAPSHOT {raw_snapshot_version}")
    print("=" * 78)

    for symbol in symbols:
        try:
            raw = lake.load_raw_bars(raw_snapshot_version, symbol)
            training = build_training_frame(
                raw,
                symbol=symbol,
                barrier_config=barrier_config,
            )
            if training.empty:
                raise RuntimeError("no usable labeled rows")
            processed_rows_by_symbol[symbol] = len(training)
            processed_files.update(
                lake.write_processed_training(new_version, symbol, training)
            )
            frames.append(training)
            print(f"  {symbol}: {len(training):,} rows")
        except Exception as exc:
            failed_symbols[symbol] = str(exc)
            print(f"  {symbol}: ERROR {exc}")

    if not frames:
        raise RuntimeError("Rebuild produced no usable rows")

    dataset = pd.concat(frames).sort_index()
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        code_commit = None

    manifest = {
        "schema_version": 1,
        "dataset_version": new_version,
        "raw_snapshot_version": raw_snapshot_version,
        "derived_from_dataset": source_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "git_commit": code_commit,
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "source": source_manifest["source"],
        "date_range": source_manifest["date_range"],
        "symbols_requested": source_manifest["symbols_requested"],
        "symbols_succeeded": sorted(processed_rows_by_symbol),
        "failed_symbols": failed_symbols,
        "features": FEATURE_COLUMNS,
        "label_parameters": asdict(barrier_config),
        "missing_data_policy": source_manifest["missing_data_policy"],
        "row_counts": {
            "raw_total": int(source_manifest["row_counts"]["raw_total"]),
            "processed_total": int(len(dataset)),
            "raw_by_symbol": source_manifest["row_counts"]["raw_by_symbol"],
            "processed_by_symbol": processed_rows_by_symbol,
        },
        "class_distribution": class_distribution(dataset),
        "long_outcomes": outcome_distribution(dataset, "long_outcome"),
        "short_outcomes": outcome_distribution(dataset, "short_outcome"),
        "class_distribution_by_symbol": per_symbol_distribution(dataset),
        "class_distribution_by_hour_et": hourly_label_distribution(dataset),
        "quality": source_manifest["quality"],
        "chronological_periods": _split_periods(dataset.index, barrier_config.horizon_bars),
        "split_policy": {
            "train_fraction": V04_TRAIN_FRACTION,
            "calibration_fraction": V04_CALIBRATION_FRACTION,
            "test_fraction": 1.0 - V04_TRAIN_FRACTION - V04_CALIBRATION_FRACTION,
            "purge_bars": barrier_config.horizon_bars,
        },
        "files": {
            "raw_partitions": source_manifest["files"]["raw_partitions"],
            "processed_partitions": sorted(processed_files),
        },
    }
    path = lake.write_manifest(new_version, manifest)
    print(f"\nRebuild complete: {len(dataset):,} rows")
    print(f"Manifest: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild labels/features from stored raw bars")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--reuse-source-label-config",
        action="store_true",
        help="Reuse the source manifest's label parameters instead of current settings",
    )
    args = parser.parse_args()
    rebuild_from_raw(
        source_version=args.source_version,
        new_version=args.new_version,
        root=args.data_root,
        use_source_label_config=args.reuse_source_label_config,
    )


if __name__ == "__main__":
    main()
