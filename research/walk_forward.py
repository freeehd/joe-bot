"""Rolling walk-forward validation for Feature Engine V2 models.

Every window follows the same chronology:

    train -> purge -> calibration -> purge -> untouched test

The model is fitted and calibrated independently per window. Test probabilities
are converted into executable signals and passed through the V0.6 minute-bar
execution simulator. No threshold is selected from the test period.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight

from backtest.engine import ExecutionConfig, TradeConfig, run_signal_backtest
from backtest.metrics import summarize_trades, trades_to_frame
from backtest.signals import probability_frame
from backtest.stress import execution_stress_scenarios
from labels.triple_barrier import BarrierConfig
from models.train_v2 import evaluate_probabilities, load_dataset
from research.benchmark_models import available_candidates
from research.storage import ParquetDataLake


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 6
    calibration_months: int = 1
    test_months: int = 1
    step_months: int = 1
    confidence_threshold: float = 0.60

    def __post_init__(self) -> None:
        for name in ("train_months", "calibration_months", "test_months", "step_months"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


@dataclass(frozen=True)
class WalkForwardWindow:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, pd.Timestamp):
                payload[key] = value.isoformat()
        return payload


def _month_floor(timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def generate_windows(
    timestamps: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    if len(timestamps) == 0:
        return []
    index = timestamps
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    first = _month_floor(index.min())
    last = index.max()
    windows: list[WalkForwardWindow] = []
    cursor = first
    number = 1

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=config.train_months)
        calibration_start = train_end
        calibration_end = calibration_start + pd.DateOffset(months=config.calibration_months)
        test_start = calibration_end
        test_end = test_start + pd.DateOffset(months=config.test_months)

        if test_start > last:
            break
        # Require the full requested test window. Partial end-of-dataset windows
        # otherwise look artificially good/bad depending on the final few days.
        # Market data naturally ends on the final trading day rather than at
        # the calendar-month boundary. Allow a small weekend/holiday gap, but
        # reject genuinely partial final test months.
        if test_end > last + pd.Timedelta(days=7):
            break

        windows.append(
            WalkForwardWindow(
                number=number,
                train_start=train_start,
                train_end=train_end,
                calibration_start=calibration_start,
                calibration_end=calibration_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        number += 1
        cursor = cursor + pd.DateOffset(months=config.step_months)

    return windows


def _purge_tail(frame: pd.DataFrame, bars: int) -> pd.DataFrame:
    if bars <= 0 or frame.empty:
        return frame.copy()
    unique = pd.DatetimeIndex(frame.index.unique()).sort_values()
    if len(unique) <= bars:
        return frame.iloc[0:0].copy()
    keep = unique[:-bars]
    return frame.loc[frame.index.isin(keep)].copy()


def slice_window(
    dataset: pd.DataFrame,
    window: WalkForwardWindow,
    *,
    purge_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = dataset.loc[
        (dataset.index >= window.train_start) & (dataset.index < window.train_end)
    ].copy()
    calibration = dataset.loc[
        (dataset.index >= window.calibration_start)
        & (dataset.index < window.calibration_end)
    ].copy()
    test = dataset.loc[
        (dataset.index >= window.test_start) & (dataset.index < window.test_end)
    ].copy()

    train = _purge_tail(train, purge_bars)
    calibration = _purge_tail(calibration, purge_bars)
    return train, calibration, test


def _validate_partition_classes(frame: pd.DataFrame, *, name: str) -> None:
    classes = set(int(value) for value in frame["trade_label"].unique())
    missing = {0, 1, 2}.difference(classes)
    if missing:
        raise ValueError(f"{name} partition missing classes {sorted(missing)}")


def _trade_config_from_manifest(manifest: dict) -> TradeConfig:
    barrier = BarrierConfig(**manifest["label_parameters"])
    if barrier.use_atr:
        raise NotImplementedError(
            "V0.6 execution backtesting currently requires fixed-percent barriers; "
            "ATR execution distances will be added as a separate experiment."
        )
    return TradeConfig(
        target_pct=barrier.target_pct,
        stop_pct=barrier.stop_pct,
        max_holding_bars=barrier.horizon_bars,
    )


def run_walk_forward(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    lake: ParquetDataLake,
    model_name: str = "xgboost",
    config: WalkForwardConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    calibration_method: str = "sigmoid",
    stress_execution: bool = False,
) -> dict:
    config = config or WalkForwardConfig()
    execution_config = execution_config or ExecutionConfig()
    barrier = BarrierConfig(**manifest["label_parameters"])
    trade_config = _trade_config_from_manifest(manifest)

    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"Unknown/unavailable model {model_name!r}; available={sorted(factories)}")

    windows = generate_windows(pd.DatetimeIndex(dataset.index.unique()), config)
    if not windows:
        raise ValueError("Dataset is too short for the requested walk-forward schedule")

    raw_snapshot = manifest.get("raw_snapshot_version", manifest["dataset_version"])
    results: list[dict] = []
    all_trade_frames: list[pd.DataFrame] = []
    stress_trade_frames: dict[str, list[pd.DataFrame]] = {}

    for window in windows:
        print(
            f"\nWindow {window.number}: train {window.train_start.date()}->{window.train_end.date()} "
            f"cal {window.calibration_start.date()}->{window.calibration_end.date()} "
            f"test {window.test_start.date()}->{window.test_end.date()}"
        )
        train, calibration, test = slice_window(
            dataset,
            window,
            purge_bars=barrier.horizon_bars,
        )
        if train.empty or calibration.empty or test.empty:
            print("  skipped: empty partition")
            continue
        try:
            _validate_partition_classes(train, name="train")
            _validate_partition_classes(calibration, name="calibration")
            _validate_partition_classes(test, name="test")
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue

        base = factories[model_name]()
        weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
        base.fit(
            train[feature_columns],
            train["trade_label"].astype(int),
            sample_weight=weights,
        )
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=calibration_method)
        calibrated.fit(
            calibration[feature_columns],
            calibration["trade_label"].astype(int),
        )

        probabilities = calibrated.predict_proba(test[feature_columns])
        model_metrics = evaluate_probabilities(test["trade_label"], probabilities)
        signals = probability_frame(test, probabilities, classes=calibrated.classes_)

        def raw_loader(symbol: str) -> pd.DataFrame:
            return lake.load_raw_bars(raw_snapshot, symbol)

        trades = run_signal_backtest(
            signals,
            raw_loader=raw_loader,
            trade_config=trade_config,
            execution_config=execution_config,
            confidence_threshold=config.confidence_threshold,
        )
        trade_metrics = summarize_trades(trades)
        trade_frame = trades_to_frame(trades)
        if not trade_frame.empty:
            trade_frame["walk_forward_window"] = window.number
            all_trade_frames.append(trade_frame)

        window_stress = {}
        if stress_execution:
            for scenario_name, scenario_config in execution_stress_scenarios(execution_config).items():
                scenario_trades = (
                    trades
                    if scenario_name == "baseline"
                    else run_signal_backtest(
                        signals,
                        raw_loader=raw_loader,
                        trade_config=trade_config,
                        execution_config=scenario_config,
                        confidence_threshold=config.confidence_threshold,
                    )
                )
                scenario_metrics = summarize_trades(scenario_trades)
                window_stress[scenario_name] = scenario_metrics
                scenario_frame = trades_to_frame(scenario_trades)
                if not scenario_frame.empty:
                    scenario_frame["walk_forward_window"] = window.number
                    stress_trade_frames.setdefault(scenario_name, []).append(scenario_frame)

        print(
            f"  test={len(test):,} trades={trade_metrics['total_trades']:,} "
            f"EV={trade_metrics['expectancy_bps']:.2f}bps "
            f"win={trade_metrics['win_rate']:.1%}"
        )
        results.append(
            {
                "window": window.to_dict(),
                "rows": {
                    "train": int(len(train)),
                    "calibration": int(len(calibration)),
                    "test": int(len(test)),
                },
                "model_metrics": model_metrics,
                "trade_metrics": trade_metrics,
                "execution_stress": window_stress,
            }
        )

    if not results:
        raise RuntimeError("All walk-forward windows were skipped")

    combined_trades = (
        pd.concat(all_trade_frames, ignore_index=True)
        if all_trade_frames
        else pd.DataFrame()
    )
    combined_metrics = summarize_trades(combined_trades)
    combined_stress = {}
    if stress_execution:
        for scenario_name in execution_stress_scenarios(execution_config):
            frames = stress_trade_frames.get(scenario_name, [])
            combined_stress[scenario_name] = summarize_trades(
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )

    positive_windows = sum(
        1 for result in results if result["trade_metrics"].get("avg_net_return", 0.0) > 0
    )

    return {
        "dataset_version": manifest["dataset_version"],
        "raw_snapshot_version": raw_snapshot,
        "model": model_name,
        "feature_engine": manifest.get("feature_engine"),
        "feature_count": len(feature_columns),
        "walk_forward_config": asdict(config),
        "execution_config": asdict(execution_config),
        "trade_config": asdict(trade_config),
        "windows_requested": len(windows),
        "windows_completed": len(results),
        "positive_expectancy_windows": positive_windows,
        "positive_expectancy_window_rate": positive_windows / len(results),
        "combined_trade_metrics": combined_metrics,
        "execution_stress": combined_stress,
        "windows": results,
    }


def write_result(result: dict, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling V0.6 walk-forward validation")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--calibration-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--stress-execution", action="store_true")
    parser.add_argument("--output", default="data/experiments/walk_forward.json")
    args = parser.parse_args()

    dataset, manifest, features = load_dataset(args.dataset_version, root=args.data_root)
    lake = ParquetDataLake(args.data_root)
    result = run_walk_forward(
        dataset,
        manifest,
        features,
        lake=lake,
        model_name=args.model,
        config=WalkForwardConfig(
            train_months=args.train_months,
            calibration_months=args.calibration_months,
            test_months=args.test_months,
            step_months=args.step_months,
            confidence_threshold=args.confidence,
        ),
        execution_config=ExecutionConfig(
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            fee_bps=args.fee_bps,
            entry_delay_bars=args.entry_delay_bars,
        ),
        calibration_method=args.calibration,
        stress_execution=args.stress_execution,
    )
    write_result(result, args.output)
    summary = result["combined_trade_metrics"]
    print("\n" + "=" * 78)
    print("WALK-FORWARD SUMMARY")
    print("=" * 78)
    print(f"Windows: {result['windows_completed']}/{result['windows_requested']}")
    print(f"Positive-EV windows: {result['positive_expectancy_windows']}")
    print(f"Trades: {summary['total_trades']:,}")
    print(f"Net expectancy: {summary['expectancy_bps']:.2f} bps/trade")
    print(f"Win rate: {summary['win_rate']:.2%}")
    if args.stress_execution and result["execution_stress"]:
        print("Execution stress:")
        for name, metrics in result["execution_stress"].items():
            print(f"  {name:<18} {metrics['expectancy_bps']:>8.2f} bps/trade  n={metrics['total_trades']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
