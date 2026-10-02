"""Leakage-safe barrier tournament that never inspects the final test period."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from labels.triple_barrier import BarrierConfig, create_multiclass_labels
from research.barrier_sweep import summarize_labels
from research.storage import ParquetDataLake


def _selection_cutoff(manifest: dict) -> pd.Timestamp:
    periods = manifest.get("chronological_periods", {})
    raw = periods.get("calibration", {}).get("end_inclusive")
    if not raw:
        raise ValueError("manifest has no calibration end; cannot guarantee test isolation")
    cutoff = pd.Timestamp(raw)
    return cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")


def evaluate_pretest_config(
    lake: ParquetDataLake,
    *,
    raw_snapshot_version: str,
    symbols: list[str],
    config: BarrierConfig,
    cutoff: pd.Timestamp,
) -> dict:
    totals = {"valid_rows": 0, "wait": 0, "long": 0, "short": 0, "ambiguous": 0.0, "unresolved": 0.0}
    for symbol in symbols:
        raw = lake.load_raw_bars(raw_snapshot_version, symbol)
        index = raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index.tz_convert("UTC")
        pretest = raw.loc[index <= cutoff].copy()
        if pretest.empty:
            continue
        summary = summarize_labels(create_multiclass_labels(pretest, config, respect_sessions=True))
        n = summary["valid_rows"]
        totals["valid_rows"] += n
        totals["wait"] += summary["wait_count"]
        totals["long"] += summary["long_count"]
        totals["short"] += summary["short_count"]
        totals["ambiguous"] += summary["directional_ambiguous_rate"] * 2 * n
        totals["unresolved"] += summary["directional_no_resolution_rate"] * 2 * n
    n = totals["valid_rows"]
    directional = 2 * n
    result = {
        **asdict(config),
        "selection_window_end": cutoff.isoformat(),
        "valid_rows": n,
        "wait_rate": totals["wait"] / n if n else 0.0,
        "long_rate": totals["long"] / n if n else 0.0,
        "short_rate": totals["short"] / n if n else 0.0,
        "directional_ambiguous_rate": totals["ambiguous"] / directional if directional else 0.0,
        "directional_no_resolution_rate": totals["unresolved"] / directional if directional else 0.0,
    }
    directional_rate = result["long_rate"] + result["short_rate"]
    balance_penalty = abs(result["long_rate"] - result["short_rate"])
    pathological_density_penalty = abs(directional_rate - 0.30)
    result["label_behavior_score"] = (
        1.0
        - balance_penalty
        - pathological_density_penalty
        - result["directional_ambiguous_rate"]
        - 0.5 * result["directional_no_resolution_rate"]
    )
    return result


def run_tournament(
    source_version: str,
    configs: list[BarrierConfig],
    *,
    root: str = "data",
    symbols: list[str] | None = None,
) -> list[dict]:
    lake = ParquetDataLake(root)
    manifest = lake.read_manifest(source_version)
    cutoff = _selection_cutoff(manifest)
    raw_snapshot = manifest.get("raw_snapshot_version", source_version)
    training_symbols = symbols or list(manifest["symbols_requested"])
    results = [
        evaluate_pretest_config(
            lake,
            raw_snapshot_version=raw_snapshot,
            symbols=training_symbols,
            config=config,
            cutoff=cutoff,
        )
        for config in configs
    ]
    return sorted(results, key=lambda row: row["label_behavior_score"], reverse=True)


def default_configs() -> list[BarrierConfig]:
    fixed = [
        BarrierConfig(horizon_bars=h, target_pct=t, stop_pct=s)
        for t in (0.002, 0.003, 0.004)
        for s in (0.001, 0.0015, 0.002)
        for h in (5, 10, 15, 20)
    ]
    atr = [
        BarrierConfig(
            horizon_bars=h,
            use_atr=True,
            atr_period=14,
            atr_target_multiplier=t,
            atr_stop_multiplier=s,
        )
        for t in (0.75, 1.0, 1.5)
        for s in (0.5, 0.75, 1.0)
        for h in (5, 10, 15, 20)
    ]
    return fixed + atr


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank label regimes using train+calibration only")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", default="data/experiments/label_tournament.json")
    args = parser.parse_args()
    results = run_tournament(args.source_version, default_configs(), root=args.data_root, symbols=args.symbols or None)
    payload = {
        "source_version": args.source_version,
        "test_period_used_for_selection": False,
        "selection_principle": "label behavior only; final choice still requires downstream held-out economic validation",
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for i, row in enumerate(results[: args.top], start=1):
        kind = "ATR" if row["use_atr"] else "FIXED"
        print(f"{i:>2}. {kind} h={row['horizon_bars']} score={row['label_behavior_score']:.4f} WAIT={row['wait_rate']:.1%} L={row['long_rate']:.1%} S={row['short_rate']:.1%}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
