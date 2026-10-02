"""Compare triple-barrier label regimes on an immutable raw snapshot.

This deliberately evaluates label behavior before model training. It answers:
How selective is each regime? How often are outcomes unresolved or ambiguous?
Does LONG/SHORT balance collapse for particular target/stop/horizon choices?
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from labels.triple_barrier import BarrierConfig, create_multiclass_labels
from research.storage import ParquetDataLake


def summarize_labels(labeled: pd.DataFrame) -> dict:
    valid = labeled.loc[labeled["label_valid"]].copy()
    counts = valid["trade_label_name"].value_counts()
    total = len(valid)

    long_outcomes = valid["long_outcome"].value_counts()
    short_outcomes = valid["short_outcome"].value_counts()

    def pct(count: int) -> float:
        return float(count / total) if total else 0.0

    ambiguous = int(long_outcomes.get("AMBIGUOUS", 0) + short_outcomes.get("AMBIGUOUS", 0))
    unresolved = int(
        long_outcomes.get("NO_RESOLUTION", 0) + short_outcomes.get("NO_RESOLUTION", 0)
    )
    return {
        "valid_rows": int(total),
        "wait_count": int(counts.get("WAIT", 0)),
        "long_count": int(counts.get("LONG", 0)),
        "short_count": int(counts.get("SHORT", 0)),
        "wait_rate": pct(int(counts.get("WAIT", 0))),
        "long_rate": pct(int(counts.get("LONG", 0))),
        "short_rate": pct(int(counts.get("SHORT", 0))),
        # Two directional simulations are performed per row, hence denominator 2N.
        "directional_ambiguous_rate": float(ambiguous / (2 * total)) if total else 0.0,
        "directional_no_resolution_rate": float(unresolved / (2 * total)) if total else 0.0,
    }


def evaluate_config(
    lake: ParquetDataLake,
    *,
    raw_snapshot_version: str,
    symbols: list[str],
    config: BarrierConfig,
) -> dict:
    aggregates = {
        "valid_rows": 0,
        "wait_count": 0,
        "long_count": 0,
        "short_count": 0,
        "directional_ambiguous_events": 0,
        "directional_no_resolution_events": 0,
    }
    per_symbol: dict[str, dict] = {}

    for symbol in symbols:
        raw = lake.load_raw_bars(raw_snapshot_version, symbol)
        labeled = create_multiclass_labels(raw, config, respect_sessions=True)
        summary = summarize_labels(labeled)
        per_symbol[symbol] = summary
        aggregates["valid_rows"] += summary["valid_rows"]
        aggregates["wait_count"] += summary["wait_count"]
        aggregates["long_count"] += summary["long_count"]
        aggregates["short_count"] += summary["short_count"]
        aggregates["directional_ambiguous_events"] += round(
            summary["directional_ambiguous_rate"] * 2 * summary["valid_rows"]
        )
        aggregates["directional_no_resolution_events"] += round(
            summary["directional_no_resolution_rate"] * 2 * summary["valid_rows"]
        )

    total = aggregates["valid_rows"]
    result = {
        **asdict(config),
        "valid_rows": int(total),
        "wait_count": int(aggregates["wait_count"]),
        "long_count": int(aggregates["long_count"]),
        "short_count": int(aggregates["short_count"]),
        "wait_rate": aggregates["wait_count"] / total if total else 0.0,
        "long_rate": aggregates["long_count"] / total if total else 0.0,
        "short_rate": aggregates["short_count"] / total if total else 0.0,
        "directional_ambiguous_rate": (
            aggregates["directional_ambiguous_events"] / (2 * total) if total else 0.0
        ),
        "directional_no_resolution_rate": (
            aggregates["directional_no_resolution_events"] / (2 * total) if total else 0.0
        ),
        "per_symbol": per_symbol,
    }
    return result


def run_atr_grid(
    *,
    source_version: str,
    target_multipliers: list[float],
    stop_multipliers: list[float],
    horizons: list[int],
    atr_periods: list[int],
    root: str = "data",
    symbols: list[str] | None = None,
) -> list[dict]:
    lake = ParquetDataLake(root)
    manifest = lake.read_manifest(source_version)
    raw_snapshot = manifest.get("raw_snapshot_version", source_version)
    training_symbols = symbols or list(manifest["symbols_requested"])

    results: list[dict] = []
    for target, stop, horizon, atr_period in itertools.product(
        target_multipliers, stop_multipliers, horizons, atr_periods
    ):
        config = BarrierConfig(
            horizon_bars=horizon,
            use_atr=True,
            atr_period=atr_period,
            atr_target_multiplier=target,
            atr_stop_multiplier=stop,
        )
        print(
            f"ATR target={target:.2f}x stop={stop:.2f}x horizon={horizon} period={atr_period}...",
            end=" ", flush=True,
        )
        result = evaluate_config(
            lake, raw_snapshot_version=raw_snapshot, symbols=training_symbols, config=config
        )
        results.append(result)
        print(
            f"WAIT={result['wait_rate']:.1%} "
            f"LONG={result['long_rate']:.1%} SHORT={result['short_rate']:.1%}"
        )
    return results


def run_fixed_grid(
    *,
    source_version: str,
    targets: list[float],
    stops: list[float],
    horizons: list[int],
    root: str = "data",
    symbols: list[str] | None = None,
) -> list[dict]:
    lake = ParquetDataLake(root)
    manifest = lake.read_manifest(source_version)
    raw_snapshot = manifest.get("raw_snapshot_version", source_version)
    training_symbols = symbols or list(manifest["symbols_requested"])

    results: list[dict] = []
    for target, stop, horizon in itertools.product(targets, stops, horizons):
        config = BarrierConfig(
            horizon_bars=horizon,
            target_pct=target,
            stop_pct=stop,
            use_atr=False,
        )
        print(
            f"target={target:.3%} stop={stop:.3%} horizon={horizon}...",
            end=" ",
            flush=True,
        )
        result = evaluate_config(
            lake,
            raw_snapshot_version=raw_snapshot,
            symbols=training_symbols,
            config=config,
        )
        results.append(result)
        print(
            f"WAIT={result['wait_rate']:.1%} "
            f"LONG={result['long_rate']:.1%} SHORT={result['short_rate']:.1%}"
        )
    return results


def write_results(results: list[dict], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        return

    flat = [{key: value for key, value in row.items() if key != "per_symbol"} for row in results]
    pd.DataFrame(flat).to_csv(path, index=False)


def _parse_percent_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep fixed and ATR-aware triple-barrier label regimes")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--targets", default="0.002,0.003,0.004")
    parser.add_argument("--stops", default="0.001,0.0015,0.002")
    parser.add_argument("--horizons", default="5,10,15,20")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--atr", action="store_true", help="Sweep ATR-multiple barriers instead of fixed percentages")
    parser.add_argument("--atr-targets", default="0.75,1.0,1.5")
    parser.add_argument("--atr-stops", default="0.5,0.75,1.0")
    parser.add_argument("--atr-periods", default="14")
    parser.add_argument("--output", default="data/experiments/barrier_sweep.csv")
    args = parser.parse_args()

    if args.atr:
        results = run_atr_grid(
            source_version=args.source_version,
            target_multipliers=_parse_percent_list(args.atr_targets),
            stop_multipliers=_parse_percent_list(args.atr_stops),
            horizons=_parse_int_list(args.horizons),
            atr_periods=_parse_int_list(args.atr_periods),
            root=args.data_root,
            symbols=args.symbols or None,
        )
    else:
        results = run_fixed_grid(
            source_version=args.source_version,
            targets=_parse_percent_list(args.targets),
            stops=_parse_percent_list(args.stops),
            horizons=_parse_int_list(args.horizons),
            root=args.data_root,
            symbols=args.symbols or None,
        )
    write_results(results, args.output)
    print(f"\nSaved {len(results)} configurations to {args.output}")


if __name__ == "__main__":
    main()
