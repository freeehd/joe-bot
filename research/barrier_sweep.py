"""Compare triple-barrier label regimes on an immutable raw snapshot.

This deliberately evaluates label behavior before model training. It answers:
How selective is each regime? How often are outcomes unresolved or ambiguous?
Does LONG/SHORT balance collapse for particular target/stop/horizon choices?
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
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


def _new_aggregate() -> dict:
    return {
        "valid_rows": 0,
        "wait_count": 0,
        "long_count": 0,
        "short_count": 0,
        "directional_ambiguous_events": 0,
        "directional_no_resolution_events": 0,
        "per_symbol": {},
    }


def _accumulate_summary(aggregate: dict, symbol: str, summary: dict) -> None:
    aggregate["per_symbol"][symbol] = summary
    aggregate["valid_rows"] += summary["valid_rows"]
    aggregate["wait_count"] += summary["wait_count"]
    aggregate["long_count"] += summary["long_count"]
    aggregate["short_count"] += summary["short_count"]
    aggregate["directional_ambiguous_events"] += round(
        summary["directional_ambiguous_rate"] * 2 * summary["valid_rows"]
    )
    aggregate["directional_no_resolution_events"] += round(
        summary["directional_no_resolution_rate"] * 2 * summary["valid_rows"]
    )


def _finalize_config(config: BarrierConfig, aggregate: dict) -> dict:
    total = aggregate["valid_rows"]
    return {
        **asdict(config),
        "valid_rows": int(total),
        "wait_count": int(aggregate["wait_count"]),
        "long_count": int(aggregate["long_count"]),
        "short_count": int(aggregate["short_count"]),
        "wait_rate": aggregate["wait_count"] / total if total else 0.0,
        "long_rate": aggregate["long_count"] / total if total else 0.0,
        "short_rate": aggregate["short_count"] / total if total else 0.0,
        "directional_ambiguous_rate": (
            aggregate["directional_ambiguous_events"] / (2 * total) if total else 0.0
        ),
        "directional_no_resolution_rate": (
            aggregate["directional_no_resolution_events"] / (2 * total) if total else 0.0
        ),
        "per_symbol": aggregate["per_symbol"],
    }


def evaluate_configs_cached(
    lake: ParquetDataLake,
    *,
    raw_snapshot_version: str,
    symbols: list[str],
    configs: list[BarrierConfig],
) -> list[dict]:
    """Evaluate many configs while loading each symbol exactly once."""

    aggregates = [_new_aggregate() for _ in configs]
    total_symbols = len(symbols)
    total_configs = len(configs)
    sweep_started = time.perf_counter()

    for symbol_index, symbol in enumerate(symbols, start=1):
        symbol_started = time.perf_counter()
        print(
            f"  [sweep] symbol {symbol_index}/{total_symbols} {symbol}: loading raw bars...",
            flush=True,
        )
        load_started = time.perf_counter()
        raw = lake.load_raw_bars(raw_snapshot_version, symbol)
        load_seconds = time.perf_counter() - load_started
        print(
            f"  [sweep] {symbol}: loaded {len(raw):,} rows in {load_seconds:.1f}s; "
            f"evaluating {total_configs} configs...",
            flush=True,
        )

        for config_index, config in enumerate(configs, start=1):
            config_started = time.perf_counter()
            summary = summarize_labels(
                create_multiclass_labels(raw, config, respect_sessions=True)
            )
            _accumulate_summary(aggregates[config_index - 1], symbol, summary)
            config_seconds = time.perf_counter() - config_started

            if config_index == 1 or config_index % 5 == 0 or config_index == total_configs:
                symbol_elapsed = time.perf_counter() - symbol_started
                avg_config = symbol_elapsed / config_index
                config_eta = avg_config * (total_configs - config_index)
                print(
                    f"    [sweep] {symbol}: config {config_index}/{total_configs} "
                    f"done in {config_seconds:.2f}s | symbol elapsed {symbol_elapsed:.1f}s "
                    f"| config ETA ~{config_eta:.1f}s",
                    flush=True,
                )

        symbol_seconds = time.perf_counter() - symbol_started
        sweep_elapsed = time.perf_counter() - sweep_started
        avg_symbol = sweep_elapsed / symbol_index
        sweep_eta = avg_symbol * (total_symbols - symbol_index)
        print(
            f"  [sweep] symbol {symbol_index}/{total_symbols} {symbol} complete in "
            f"{symbol_seconds:.1f}s | total elapsed {sweep_elapsed/60:.1f}m "
            f"| ETA ~{sweep_eta/60:.1f}m",
            flush=True,
        )

    return [
        _finalize_config(config, aggregate)
        for config, aggregate in zip(configs, aggregates, strict=True)
    ]


def evaluate_config(
    lake: ParquetDataLake,
    *,
    raw_snapshot_version: str,
    symbols: list[str],
    config: BarrierConfig,
) -> dict:
    return evaluate_configs_cached(
        lake,
        raw_snapshot_version=raw_snapshot_version,
        symbols=symbols,
        configs=[config],
    )[0]


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
    configs = [
        BarrierConfig(
            horizon_bars=horizon,
            use_atr=True,
            atr_period=atr_period,
            atr_target_multiplier=target,
            atr_stop_multiplier=stop,
        )
        for target, stop, horizon, atr_period in itertools.product(
            target_multipliers, stop_multipliers, horizons, atr_periods
        )
    ]
    results = evaluate_configs_cached(
        lake,
        raw_snapshot_version=raw_snapshot,
        symbols=training_symbols,
        configs=configs,
    )
    for result in results:
        print(
            f"ATR target={result['atr_target_multiplier']:.2f}x "
            f"stop={result['atr_stop_multiplier']:.2f}x "
            f"horizon={result['horizon_bars']} period={result['atr_period']}... "
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
    configs = [
        BarrierConfig(
            horizon_bars=horizon,
            target_pct=target,
            stop_pct=stop,
            use_atr=False,
        )
        for target, stop, horizon in itertools.product(targets, stops, horizons)
    ]
    results = evaluate_configs_cached(
        lake,
        raw_snapshot_version=raw_snapshot,
        symbols=training_symbols,
        configs=configs,
    )
    for result in results:
        print(
            f"target={result['target_pct']:.3%} stop={result['stop_pct']:.3%} "
            f"horizon={result['horizon_bars']}... "
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
