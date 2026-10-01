"""Rebuild V2 processed features/labels from an immutable raw snapshot."""

from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from config.settings import V04_CALIBRATION_FRACTION, V04_TRAIN_FRACTION
from features.v2 import FEATURE_COLUMNS_V2
from labels.triple_barrier import BarrierConfig
from market.universe import MARKET_CONTEXT_SYMBOLS
from models.train_multiclass import barrier_config_from_settings
from research.build_dataset import _split_periods
from research.diagnostics import (
    class_distribution,
    hourly_label_distribution,
    outcome_distribution,
    per_symbol_distribution,
)
from research.preprocessing import add_context_and_finalize_v2, build_symbol_frame_v2
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

    barrier_config = (
        BarrierConfig(**source_manifest["label_parameters"])
        if use_source_label_config
        else barrier_config_from_settings()
    )

    training_symbols = list(source_manifest["symbols_requested"])
    context_symbols = list(source_manifest.get("context_symbols", MARKET_CONTEXT_SYMBOLS))
    acquisition_symbols = list(
        dict.fromkeys(source_manifest.get("acquisition_symbols", training_symbols + context_symbols))
    )

    print("\n" + "=" * 78)
    print(f"REBUILDING {new_version} FROM RAW SNAPSHOT {raw_snapshot_version}")
    print("=" * 78)

    frames: list[pd.DataFrame] = []
    failed_symbols: dict[str, str] = {}
    available_training: list[str] = []

    for symbol in acquisition_symbols:
        try:
            raw = lake.load_raw_bars(raw_snapshot_version, symbol)
            base = build_symbol_frame_v2(
                raw,
                symbol=symbol,
                barrier_config=barrier_config,
            )
            if base.empty:
                raise RuntimeError("no usable V2 feature/label rows")
            frames.append(base)
            if symbol in training_symbols:
                available_training.append(symbol)
            print(f"  {symbol}: {len(base):,} pre-context rows")
        except Exception as exc:
            failed_symbols[symbol] = str(exc)
            print(f"  {symbol}: ERROR {exc}")

    missing_context = [symbol for symbol in context_symbols if symbol in failed_symbols]
    if missing_context:
        raise RuntimeError(f"Context rebuild failed for {missing_context}")
    if not available_training:
        raise RuntimeError("No training symbols survived rebuild preprocessing")

    dataset = add_context_and_finalize_v2(frames, training_symbols=available_training)
    if dataset.empty:
        raise RuntimeError("Rebuild produced no final V2 rows")

    processed_files: set[str] = set()
    processed_rows_by_symbol: dict[str, int] = {}
    for symbol, symbol_frame in dataset.groupby("training_symbol", sort=True):
        processed_rows_by_symbol[str(symbol)] = len(symbol_frame)
        processed_files.update(
            lake.write_processed_training(new_version, str(symbol), symbol_frame)
        )

    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        code_commit = None

    manifest = {
        "schema_version": 2,
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
        "symbols_requested": training_symbols,
        "context_symbols": context_symbols,
        "acquisition_symbols": acquisition_symbols,
        "symbols_succeeded": sorted(processed_rows_by_symbol),
        "failed_symbols": failed_symbols,
        "feature_engine": "v2",
        "features": FEATURE_COLUMNS_V2,
        "label_parameters": asdict(barrier_config),
        "missing_data_policy": {
            **source_manifest["missing_data_policy"],
            "time_of_day_baseline": "causal expanding mean using earlier sessions only",
        },
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
    parser = argparse.ArgumentParser(description="Rebuild V2 labels/features from stored raw bars")
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
