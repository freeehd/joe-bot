"""Build a versioned Phase B historical research dataset.

Example:
    python -m research.build_dataset --version v04-r50-2y --years 2 --feed iex

The builder stores corporate-action-adjusted raw minute bars as Parquet,
creates V0.4 session-safe features + triple-barrier labels, stores processed
training partitions, and writes a manifest describing exactly what was built.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import (
    LABEL_HORIZON_BARS,
    RESEARCH_BATCH_SIZE,
    RESEARCH_DATA_ADJUSTMENT,
    RESEARCH_DATA_FEED,
    RESEARCH_REGULAR_HOURS_ONLY,
    V04_CALIBRATION_FRACTION,
    V04_TRAIN_FRACTION,
)
from features.schema import FEATURE_COLUMNS
from market.historical import AlpacaHistoricalBarsSource, filter_regular_hours
from market.universe import RESEARCH_UNIVERSE_50
from models.train_multiclass import barrier_config_from_settings, build_training_frame
from research.diagnostics import (
    class_distribution,
    hourly_label_distribution,
    outcome_distribution,
    per_symbol_distribution,
)
from research.quality import clean_and_validate_bars
from research.storage import ParquetDataLake


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _split_periods(timestamps: pd.DatetimeIndex, purge_bars: int) -> dict:
    unique = pd.DatetimeIndex(timestamps.unique()).sort_values()
    if len(unique) < 10:
        return {}

    train_boundary = int(len(unique) * V04_TRAIN_FRACTION)
    calibration_boundary = int(
        len(unique) * (V04_TRAIN_FRACTION + V04_CALIBRATION_FRACTION)
    )
    train_end = max(0, train_boundary - purge_bars)
    calibration_end = max(train_boundary, calibration_boundary - purge_bars)

    def iso(position: int | None):
        if position is None or position < 0 or position >= len(unique):
            return None
        return unique[position].isoformat()

    return {
        "train": {
            "start": iso(0),
            "end_inclusive": iso(train_end - 1),
        },
        "purge_before_calibration": {
            "start": iso(train_end),
            "end_inclusive": iso(train_boundary - 1),
            "bars": purge_bars,
        },
        "calibration": {
            "start": iso(train_boundary),
            "end_inclusive": iso(calibration_end - 1),
        },
        "purge_before_test": {
            "start": iso(calibration_end),
            "end_inclusive": iso(calibration_boundary - 1),
            "bars": purge_bars,
        },
        "test": {
            "start": iso(calibration_boundary),
            "end_inclusive": iso(len(unique) - 1),
        },
    }


def build_dataset(
    *,
    version: str,
    start: datetime,
    end: datetime,
    symbols: list[str] | None = None,
    feed: str | None = RESEARCH_DATA_FEED,
    adjustment: str = RESEARCH_DATA_ADJUSTMENT,
    batch_size: int = RESEARCH_BATCH_SIZE,
    regular_hours_only: bool = RESEARCH_REGULAR_HOURS_ONLY,
    root: str | Path = "data",
    source=None,
    store=None,
) -> dict:
    if start >= end:
        raise ValueError("start must be before end")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    symbols = list(symbols or RESEARCH_UNIVERSE_50)
    if len(symbols) != len(set(symbols)):
        raise ValueError("Universe contains duplicate symbols")

    source = source or AlpacaHistoricalBarsSource()
    lake = store or ParquetDataLake(root)
    if hasattr(lake, "assert_new_version"):
        lake.assert_new_version(version)
    barrier_config = barrier_config_from_settings()

    quality_reports: dict[str, dict] = {}
    raw_rows_by_symbol: dict[str, int] = {}
    processed_rows_by_symbol: dict[str, int] = {}
    raw_files: set[str] = set()
    processed_files: set[str] = set()
    processed_frames: list[pd.DataFrame] = []
    failed_symbols: dict[str, str] = {}

    asof = end.date().isoformat()
    print("\n" + "=" * 78)
    print(f"BUILDING DATASET {version}")
    print(f"{start.isoformat()} -> {end.isoformat()} | {len(symbols)} symbols")
    print("=" * 78)

    for batch_number, batch in enumerate(_chunks(symbols, batch_size), start=1):
        print(f"\nBatch {batch_number}: {', '.join(batch)}")
        try:
            combined = source.fetch_minute_bars(
                batch,
                start=start,
                end=end,
                adjustment=adjustment,
                feed=feed,
                asof=asof,
            )
        except Exception as exc:
            for symbol in batch:
                failed_symbols[symbol] = f"download error: {exc}"
            print(f"  DOWNLOAD ERROR: {exc}")
            continue

        for symbol in batch:
            symbol_frame = combined.loc[combined["symbol"] == symbol].copy()
            if symbol_frame.empty:
                failed_symbols[symbol] = "no bars returned"
                print(f"  {symbol}: NO DATA")
                continue

            try:
                if regular_hours_only:
                    symbol_frame = filter_regular_hours(symbol_frame)

                cleaned, quality = clean_and_validate_bars(symbol_frame, symbol=symbol)
                if cleaned.empty:
                    raise RuntimeError("no valid bars after quality filtering")

                raw_rows_by_symbol[symbol] = len(cleaned)
                quality_reports[symbol] = quality.to_dict()
                raw_files.update(lake.write_raw_bars(version, symbol, cleaned))

                training = build_training_frame(
                    cleaned,
                    symbol=symbol,
                    barrier_config=barrier_config,
                )
                if training.empty:
                    raise RuntimeError("no usable labeled rows")

                processed_rows_by_symbol[symbol] = len(training)
                processed_files.update(
                    lake.write_processed_training(version, symbol, training)
                )
                processed_frames.append(training)
                print(
                    f"  {symbol}: raw={len(cleaned):,} "
                    f"training={len(training):,} gaps={quality.missing_minute_intervals:,}"
                )
            except Exception as exc:
                failed_symbols[symbol] = str(exc)
                print(f"  {symbol}: ERROR {exc}")

    if not processed_frames:
        raise RuntimeError("Dataset build produced no usable processed rows")

    dataset = pd.concat(processed_frames).sort_index()
    split_periods = _split_periods(dataset.index, barrier_config.horizon_bars)

    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        code_commit = None

    manifest = {
        "schema_version": 1,
        "dataset_version": version,
        "raw_snapshot_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "git_commit": code_commit,
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "source": {
            "provider": getattr(source, "source_name", source.__class__.__name__),
            "feed": feed,
            "timeframe": "1Min",
            "adjustment": adjustment,
            "asof": asof,
            "regular_hours_only": regular_hours_only,
            "regular_hours": "09:30-16:00 America/New_York" if regular_hours_only else None,
        },
        "date_range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "symbols_requested": symbols,
        "symbols_succeeded": sorted(processed_rows_by_symbol),
        "failed_symbols": failed_symbols,
        "features": FEATURE_COLUMNS,
        "label_parameters": asdict(barrier_config),
        "missing_data_policy": {
            "forward_fill": False,
            "synthetic_bars": False,
            "duplicate_policy": "last observation wins",
            "invalid_rows": "drop and count",
            "missing_minutes": "retain natural gaps and report count",
        },
        "row_counts": {
            "raw_total": int(sum(raw_rows_by_symbol.values())),
            "processed_total": int(len(dataset)),
            "raw_by_symbol": raw_rows_by_symbol,
            "processed_by_symbol": processed_rows_by_symbol,
        },
        "class_distribution": class_distribution(dataset),
        "long_outcomes": outcome_distribution(dataset, "long_outcome"),
        "short_outcomes": outcome_distribution(dataset, "short_outcome"),
        "class_distribution_by_symbol": per_symbol_distribution(dataset),
        "class_distribution_by_hour_et": hourly_label_distribution(dataset),
        "quality": quality_reports,
        "chronological_periods": split_periods,
        "split_policy": {
            "train_fraction": V04_TRAIN_FRACTION,
            "calibration_fraction": V04_CALIBRATION_FRACTION,
            "test_fraction": 1.0 - V04_TRAIN_FRACTION - V04_CALIBRATION_FRACTION,
            "purge_bars": barrier_config.horizon_bars,
        },
        "files": {
            "raw_partitions": sorted(raw_files),
            "processed_partitions": sorted(processed_files),
        },
    }

    manifest_path = lake.write_manifest(version, manifest)
    print("\n" + "=" * 78)
    print("DATASET COMPLETE")
    print("=" * 78)
    print(f"Processed rows: {len(dataset):,}")
    print(f"Symbols:        {len(processed_rows_by_symbol)}/{len(symbols)}")
    print(f"Manifest:       {manifest_path}")
    print("Class distribution:")
    for label, stats in manifest["class_distribution"].items():
        print(f"  {label:<5} {stats['count']:>10,}  {stats['percent']:>7.3f}%")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase B historical dataset")
    parser.add_argument("--version", required=True, help="Immutable dataset version name")
    parser.add_argument("--years", type=float, default=2.0, help="Calendar years of history")
    parser.add_argument("--start", help="Explicit UTC/date start; overrides --years")
    parser.add_argument("--end", help="Explicit UTC/date end; default now")
    parser.add_argument("--feed", default=RESEARCH_DATA_FEED)
    parser.add_argument("--batch-size", type=int, default=RESEARCH_BATCH_SIZE)
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional explicit symbols instead of the Phase B 50-symbol universe",
    )
    return parser.parse_args()


def _parse_datetime(value: str) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def main() -> None:
    args = _parse_args()
    end = _parse_datetime(args.end) if args.end else datetime.now(timezone.utc)
    start = _parse_datetime(args.start) if args.start else end - timedelta(days=365.25 * args.years)
    build_dataset(
        version=args.version,
        start=start,
        end=end,
        symbols=args.symbols or None,
        feed=args.feed or None,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
