"""Build a leakage-safe historical Laya specialist dataset from V0.7 windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight

from backtest.engine import ExecutionConfig, TradeConfig, run_signal_backtest
from backtest.metrics import trades_to_frame
from backtest.signals import probability_frame
from labels.triple_barrier import BarrierConfig
from models.train_v2 import load_dataset
from research.benchmark_models import available_candidates
from research.laya_dataset import LayaLabelConfig, build_laya_examples, write_laya_dataset
from research.storage import ParquetDataLake
from research.walk_forward import WalkForwardConfig, generate_windows, slice_window
from risk.portfolio_risk import CorrelationConfig, correlation_matrix_from_feature_rows
from strategy.ev_model import EVConfig, EmpiricalEVModel
from strategy.portfolio_allocator import EVRanker


def _trade_config(manifest: dict) -> TradeConfig:
    barrier = BarrierConfig(**manifest["label_parameters"])
    if barrier.use_atr:
        raise NotImplementedError("V0.8 dataset builder currently requires fixed-percent barriers")
    return TradeConfig(
        target_pct=barrier.target_pct,
        stop_pct=barrier.stop_pct,
        max_holding_bars=barrier.horizon_bars,
    )


def _classes_ok(frame: pd.DataFrame) -> bool:
    return {0, 1, 2}.issubset(set(int(value) for value in frame["trade_label"].unique()))


def build_walk_forward_laya_dataset(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    lake: ParquetDataLake,
    model_name: str = "xgboost",
    walk_forward_config: WalkForwardConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    ev_config: EVConfig | None = None,
    correlation_config: CorrelationConfig | None = None,
    label_config: LayaLabelConfig | None = None,
    top_candidates: int = 5,
    calibration_method: str = "sigmoid",
) -> tuple[list[dict], dict]:
    wf = walk_forward_config or WalkForwardConfig()
    execution = execution_config or ExecutionConfig()
    ev_config = ev_config or EVConfig()
    correlation_config = correlation_config or CorrelationConfig()
    label_config = label_config or LayaLabelConfig()
    trade = _trade_config(manifest)

    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"Unknown/unavailable model {model_name!r}; available={sorted(factories)}")

    windows = generate_windows(pd.DatetimeIndex(dataset.index.unique()), wf)
    raw_snapshot = manifest.get("raw_snapshot_version", manifest["dataset_version"])
    raw_cache: dict[str, pd.DataFrame] = {}

    def raw_loader(symbol: str) -> pd.DataFrame:
        if symbol not in raw_cache:
            raw_cache[symbol] = lake.load_raw_bars(raw_snapshot, symbol)
        return raw_cache[symbol]

    examples: list[dict] = []
    completed = 0
    for window in windows:
        barrier = BarrierConfig(**manifest["label_parameters"])
        train, calibration, test = slice_window(dataset, window, purge_bars=barrier.horizon_bars)
        if train.empty or calibration.empty or test.empty:
            continue
        if not (_classes_ok(train) and _classes_ok(calibration) and _classes_ok(test)):
            continue

        base = factories[model_name]()
        weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
        base.fit(train[feature_columns], train["trade_label"].astype(int), sample_weight=weights)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=calibration_method)
        calibrated.fit(calibration[feature_columns], calibration["trade_label"].astype(int))

        calibration_probs = calibrated.predict_proba(calibration[feature_columns])
        calibration_signals = probability_frame(calibration, calibration_probs, classes=calibrated.classes_)
        calibration_trades = run_signal_backtest(
            calibration_signals,
            raw_loader=raw_loader,
            trade_config=trade,
            execution_config=execution,
            confidence_threshold=0.0,
        )
        ev_model = EmpiricalEVModel(
            trade_config=trade,
            execution_config=execution,
            config=ev_config,
        ).fit(trades_to_frame(calibration_trades))

        correlations = correlation_matrix_from_feature_rows(
            pd.concat([train, calibration]).sort_index(),
            config=correlation_config,
        )
        test_probs = calibrated.predict_proba(test[feature_columns])
        test_signals = probability_frame(test, test_probs, classes=calibrated.classes_)
        window_examples = build_laya_examples(
            test_signals,
            raw_loader=raw_loader,
            ranker=EVRanker(ev_model),
            trade_config=trade,
            execution_config=execution,
            correlations=correlations,
            top_candidates=top_candidates,
            label_config=label_config,
            source_window=window.number,
        )
        examples.extend(window_examples)
        completed += 1
        print(f"Laya window {window.number}: {len(window_examples):,} examples")

    # Overlapping window configurations can repeat the same held-out state.
    deduplicated: dict[tuple[str, str, str], dict] = {}
    for record in examples:
        key = (
            record["metadata"]["timestamp"],
            record["metadata"]["symbol"],
            record["metadata"]["quant_direction"],
        )
        deduplicated[key] = record
    examples = sorted(deduplicated.values(), key=lambda item: item["metadata"]["timestamp"])
    if not examples:
        raise RuntimeError("No Laya examples were produced; inspect model EV filters and source data")

    build_manifest = {
        "source_dataset_version": manifest["dataset_version"],
        "raw_snapshot_version": raw_snapshot,
        "model": model_name,
        "walk_forward_config": asdict(wf),
        "execution_config": asdict(execution),
        "ev_config": asdict(ev_config),
        "correlation_config": asdict(correlation_config),
        "label_config": asdict(label_config),
        "top_candidates": top_candidates,
        "windows_requested": len(windows),
        "windows_completed": completed,
    }
    return examples, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe V0.8 Laya training data")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--calibration-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--output-dir", default="data/laya/v08")
    args = parser.parse_args()

    dataset, manifest, features = load_dataset(args.dataset_version, root=args.data_root)
    examples, build_manifest = build_walk_forward_laya_dataset(
        dataset,
        manifest,
        features,
        lake=ParquetDataLake(args.data_root),
        model_name=args.model,
        walk_forward_config=WalkForwardConfig(
            train_months=args.train_months,
            calibration_months=args.calibration_months,
            test_months=args.test_months,
            step_months=args.step_months,
            confidence_threshold=0.0,
        ),
        execution_config=ExecutionConfig(
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            fee_bps=args.fee_bps,
            entry_delay_bars=args.entry_delay_bars,
        ),
        top_candidates=args.top_candidates,
    )
    output = Path(args.output_dir)
    final_manifest = write_laya_dataset(
        examples,
        output,
        manifest_extra=build_manifest,
    )
    print("\n" + "=" * 78)
    print("V0.8 LAYA DATASET")
    print("=" * 78)
    print(f"Records: {final_manifest['records']:,}")
    print(f"Splits: {final_manifest['split_counts']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
