"""V0.7 walk-forward validation for EV ranking and portfolio construction.

For each chronological window:

1. fit the alpha model on the training partition;
2. calibrate probabilities on the calibration partition;
3. backtest calibration signals to fit an empirical EV model;
4. estimate a correlation matrix from train+calibration rows only;
5. rank untouched test signals by blended net EV;
6. allocate positions with risk/correlation/exposure constraints; and
7. compare portfolio-selected trades with the V0.6 all-signal baseline.

No test-period outcome is used to estimate EV, correlations, thresholds, or size.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight

from backtest.engine import ExecutionConfig, TradeConfig, run_signal_backtest
from backtest.metrics import summarize_trades, trades_to_frame
from backtest.portfolio import CandidateGateProtocol, PortfolioBacktestConfig, run_portfolio_backtest
from backtest.signals import probability_frame
from labels.triple_barrier import BarrierConfig
from models.train_v2 import evaluate_probabilities, load_dataset
from research.benchmark_models import available_candidates
from research.walk_forward import WalkForwardConfig, generate_windows, slice_window
from research.storage import ParquetDataLake
from risk.portfolio_risk import CorrelationConfig, correlation_matrix_from_feature_rows
from strategy.ev_model import EVConfig, EmpiricalEVModel
from strategy.portfolio_allocator import EVRanker, PortfolioAllocatorV2, PortfolioConstraints


def _trade_config(manifest: dict) -> TradeConfig:
    barrier = BarrierConfig(**manifest["label_parameters"])
    if barrier.use_atr:
        raise NotImplementedError("V0.7 portfolio simulator currently requires fixed-percent barriers")
    return TradeConfig(
        target_pct=barrier.target_pct,
        stop_pct=barrier.stop_pct,
        max_holding_bars=barrier.horizon_bars,
    )


def _validate_classes(frame: pd.DataFrame, name: str) -> None:
    present = set(int(value) for value in frame["trade_label"].unique())
    missing = {0, 1, 2}.difference(present)
    if missing:
        raise ValueError(f"{name} partition missing classes {sorted(missing)}")


def _serializable_trade_summary(frame: pd.DataFrame) -> dict:
    return summarize_trades(frame if not frame.empty else pd.DataFrame())


def run_portfolio_walk_forward(
    dataset: pd.DataFrame,
    manifest: dict,
    feature_columns: list[str],
    *,
    lake: ParquetDataLake,
    model_name: str = "xgboost",
    walk_forward_config: WalkForwardConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    portfolio_constraints: PortfolioConstraints | None = None,
    ev_config: EVConfig | None = None,
    correlation_config: CorrelationConfig | None = None,
    initial_equity: float = 10_000.0,
    calibration_method: str = "sigmoid",
    candidate_gate: CandidateGateProtocol | None = None,
) -> dict:
    wf = walk_forward_config or WalkForwardConfig()
    execution = execution_config or ExecutionConfig()
    constraints = portfolio_constraints or PortfolioConstraints()
    ev_config = ev_config or EVConfig()
    correlation_config = correlation_config or CorrelationConfig()
    trade = _trade_config(manifest)

    factories = available_candidates()
    if model_name not in factories:
        raise ValueError(f"Unknown/unavailable model {model_name!r}; available={sorted(factories)}")

    windows = generate_windows(pd.DatetimeIndex(dataset.index.unique()), wf)
    if not windows:
        raise ValueError("Dataset is too short for the requested walk-forward schedule")

    raw_snapshot = manifest.get("raw_snapshot_version", manifest["dataset_version"])
    window_results: list[dict] = []
    combined_portfolio_trades: list[pd.DataFrame] = []
    combined_baseline_trades: list[pd.DataFrame] = []
    combined_gated_portfolio_trades: list[pd.DataFrame] = []

    def raw_loader(symbol: str) -> pd.DataFrame:
        return lake.load_raw_bars(raw_snapshot, symbol)

    for window in windows:
        print(
            f"\nV0.7 Window {window.number}: "
            f"train {window.train_start.date()}->{window.train_end.date()} "
            f"cal {window.calibration_start.date()}->{window.calibration_end.date()} "
            f"test {window.test_start.date()}->{window.test_end.date()}"
        )
        barrier = BarrierConfig(**manifest["label_parameters"])
        train, calibration, test = slice_window(
            dataset,
            window,
            purge_bars=barrier.horizon_bars,
        )
        if train.empty or calibration.empty or test.empty:
            print("  skipped: empty partition")
            continue
        try:
            _validate_classes(train, "train")
            _validate_classes(calibration, "calibration")
            _validate_classes(test, "test")
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue

        base = factories[model_name]()
        weights = compute_sample_weight("balanced", train["trade_label"].astype(int))
        base.fit(train[feature_columns], train["trade_label"].astype(int), sample_weight=weights)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=calibration_method)
        calibrated.fit(calibration[feature_columns], calibration["trade_label"].astype(int))

        # Calibration outcomes are earlier than test and may therefore be used
        # to estimate realized EV by side/confidence bucket.
        calibration_probabilities = calibrated.predict_proba(calibration[feature_columns])
        calibration_signals = probability_frame(
            calibration,
            calibration_probabilities,
            classes=calibrated.classes_,
        )
        calibration_trades = run_signal_backtest(
            calibration_signals,
            raw_loader=raw_loader,
            trade_config=trade,
            execution_config=execution,
            confidence_threshold=0.0,
        )
        calibration_trade_frame = trades_to_frame(calibration_trades)
        ev_model = EmpiricalEVModel(
            trade_config=trade,
            execution_config=execution,
            config=ev_config,
        ).fit(calibration_trade_frame)

        # Correlations are estimated entirely before the test boundary.
        history = pd.concat([train, calibration]).sort_index()
        correlations = correlation_matrix_from_feature_rows(
            history,
            config=correlation_config,
        )

        test_probabilities = calibrated.predict_proba(test[feature_columns])
        model_metrics = evaluate_probabilities(test["trade_label"], test_probabilities)
        test_signals = probability_frame(test, test_probabilities, classes=calibrated.classes_)

        baseline_trades = run_signal_backtest(
            test_signals,
            raw_loader=raw_loader,
            trade_config=trade,
            execution_config=execution,
            confidence_threshold=wf.confidence_threshold,
        )
        baseline_frame = trades_to_frame(baseline_trades)
        if not baseline_frame.empty:
            baseline_frame["walk_forward_window"] = window.number
            combined_baseline_trades.append(baseline_frame)
        baseline_metrics = _serializable_trade_summary(baseline_frame)

        portfolio = run_portfolio_backtest(
            test_signals,
            raw_loader=raw_loader,
            ranker=EVRanker(ev_model),
            allocator=PortfolioAllocatorV2(
                constraints=constraints,
                trade_config=trade,
            ),
            trade_config=trade,
            execution_config=execution,
            correlations=correlations,
            config=PortfolioBacktestConfig(
                initial_equity=initial_equity,
                min_confidence=0.0,
            ),
        )
        portfolio_frame = portfolio["trade_records"].copy()
        if not portfolio_frame.empty:
            portfolio_frame["walk_forward_window"] = window.number
            combined_portfolio_trades.append(portfolio_frame)

        portfolio_summary = {
            "ending_equity": portfolio["ending_equity"],
            "net_pnl_dollars": portfolio["net_pnl_dollars"],
            "total_return": portfolio["total_return"],
            "max_drawdown": portfolio["max_drawdown"],
            "max_drawdown_dollars": portfolio["max_drawdown_dollars"],
            "profit_factor_dollars": portfolio["profit_factor_dollars"],
            "trade_metrics": portfolio["trade_metrics"],
            "allocation_decisions": len(portfolio["allocation_snapshots"]),
        }

        gated_summary = None
        if candidate_gate is not None:
            gated = run_portfolio_backtest(
                test_signals,
                raw_loader=raw_loader,
                ranker=EVRanker(ev_model),
                allocator=PortfolioAllocatorV2(
                    constraints=constraints,
                    trade_config=trade,
                ),
                trade_config=trade,
                execution_config=execution,
                correlations=correlations,
                config=PortfolioBacktestConfig(
                    initial_equity=initial_equity,
                    min_confidence=0.0,
                ),
                candidate_gate=candidate_gate,
            )
            gated_frame = gated["trade_records"].copy()
            if not gated_frame.empty:
                gated_frame["walk_forward_window"] = window.number
                combined_gated_portfolio_trades.append(gated_frame)
            gated_summary = {
                "ending_equity": gated["ending_equity"],
                "net_pnl_dollars": gated["net_pnl_dollars"],
                "total_return": gated["total_return"],
                "max_drawdown": gated["max_drawdown"],
                "max_drawdown_dollars": gated["max_drawdown_dollars"],
                "profit_factor_dollars": gated["profit_factor_dollars"],
                "trade_metrics": gated["trade_metrics"],
                "allocation_decisions": len(gated["allocation_snapshots"]),
                "gate_decisions": len(gated["gate_snapshots"]),
                "veto_count": int(sum(item["laya_vetoed"] for item in gated["gate_snapshots"])),
            }

        print(
            f"  baseline n={baseline_metrics['total_trades']} "
            f"EV={baseline_metrics['expectancy_bps']:.2f}bps | "
            f"portfolio n={portfolio_summary['trade_metrics']['total_trades']} "
            f"EV={portfolio_summary['trade_metrics']['expectancy_bps']:.2f}bps "
            f"return={portfolio_summary['total_return']:.2%} "
            f"maxDD={portfolio_summary['max_drawdown']:.2%}"
        )
        window_results.append(
            {
                "window": window.to_dict(),
                "rows": {
                    "train": int(len(train)),
                    "calibration": int(len(calibration)),
                    "test": int(len(test)),
                },
                "model_metrics": model_metrics,
                "calibration_trade_count": int(len(calibration_trade_frame)),
                "ev_buckets": ev_model.stats_frame().to_dict(orient="records"),
                "baseline_trade_metrics": baseline_metrics,
                "portfolio": portfolio_summary,
                **({"gated_portfolio": gated_summary} if gated_summary is not None else {}),
            }
        )

    if not window_results:
        raise RuntimeError("All V0.7 walk-forward windows were skipped")

    portfolio_all = (
        pd.concat(combined_portfolio_trades, ignore_index=True)
        if combined_portfolio_trades
        else pd.DataFrame()
    )
    baseline_all = (
        pd.concat(combined_baseline_trades, ignore_index=True)
        if combined_baseline_trades
        else pd.DataFrame()
    )
    portfolio_trade_metrics = _serializable_trade_summary(portfolio_all)
    baseline_trade_metrics = _serializable_trade_summary(baseline_all)

    positive_portfolio_windows = sum(
        1 for result in window_results if result["portfolio"]["total_return"] > 0
    )
    positive_baseline_ev_windows = sum(
        1
        for result in window_results
        if result["baseline_trade_metrics"].get("avg_net_return", 0.0) > 0
    )
    worst_drawdown = min(result["portfolio"]["max_drawdown"] for result in window_results)
    average_return = sum(result["portfolio"]["total_return"] for result in window_results) / len(window_results)

    gated_summary_all = None
    if candidate_gate is not None:
        gated_all = (
            pd.concat(combined_gated_portfolio_trades, ignore_index=True)
            if combined_gated_portfolio_trades
            else pd.DataFrame()
        )
        gated_trade_metrics = _serializable_trade_summary(gated_all)
        gated_positive_windows = sum(
            1 for result in window_results if result.get("gated_portfolio", {}).get("total_return", 0.0) > 0
        )
        gated_worst_drawdown = min(
            result.get("gated_portfolio", {}).get("max_drawdown", 0.0) for result in window_results
        )
        gated_average_return = sum(
            result.get("gated_portfolio", {}).get("total_return", 0.0) for result in window_results
        ) / len(window_results)
        gated_summary_all = {
            "trade_metrics": gated_trade_metrics,
            "positive_return_windows": gated_positive_windows,
            "positive_return_window_rate": gated_positive_windows / len(window_results),
            "average_window_return": gated_average_return,
            "worst_window_drawdown": gated_worst_drawdown,
            "expectancy_delta_vs_ungated_bps": (
                gated_trade_metrics.get("expectancy_bps", 0.0)
                - portfolio_trade_metrics.get("expectancy_bps", 0.0)
            ),
            "average_return_delta_vs_ungated": gated_average_return - average_return,
            "worst_drawdown_delta_vs_ungated": gated_worst_drawdown - worst_drawdown,
        }

    return {
        "dataset_version": manifest["dataset_version"],
        "raw_snapshot_version": raw_snapshot,
        "model": model_name,
        "walk_forward_config": asdict(wf),
        "execution_config": asdict(execution),
        "portfolio_constraints": asdict(constraints),
        "ev_config": asdict(ev_config),
        "correlation_config": asdict(correlation_config),
        "initial_equity_per_window": initial_equity,
        "windows_requested": len(windows),
        "windows_completed": len(window_results),
        "portfolio_positive_return_windows": positive_portfolio_windows,
        "portfolio_positive_return_window_rate": positive_portfolio_windows / len(window_results),
        "baseline_positive_ev_windows": positive_baseline_ev_windows,
        "baseline_positive_ev_window_rate": positive_baseline_ev_windows / len(window_results),
        "portfolio_average_window_return": average_return,
        "portfolio_worst_window_drawdown": worst_drawdown,
        "portfolio_trade_metrics": portfolio_trade_metrics,
        "baseline_trade_metrics": baseline_trade_metrics,
        "expectancy_improvement_bps": (
            portfolio_trade_metrics.get("expectancy_bps", 0.0)
            - baseline_trade_metrics.get("expectancy_bps", 0.0)
        ),
        **({"gated_portfolio": gated_summary_all} if gated_summary_all is not None else {}),
        "windows": window_results,
    }


def write_result(result: dict, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V0.7 EV/portfolio walk-forward validation")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--calibration-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--baseline-confidence", type=float, default=0.60)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--risk-per-trade", type=float, default=0.0025)
    parser.add_argument("--max-total-risk", type=float, default=0.01)
    parser.add_argument("--max-position", type=float, default=0.30)
    parser.add_argument("--max-deployed", type=float, default=0.70)
    parser.add_argument("--max-sector", type=float, default=0.40)
    parser.add_argument("--min-net-ev-bps", type=float, default=0.0)
    parser.add_argument("--output", default="data/experiments/portfolio_walk_forward.json")
    args = parser.parse_args()

    dataset, manifest, features = load_dataset(args.dataset_version, root=args.data_root)
    result = run_portfolio_walk_forward(
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
            confidence_threshold=args.baseline_confidence,
        ),
        execution_config=ExecutionConfig(
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            fee_bps=args.fee_bps,
            entry_delay_bars=args.entry_delay_bars,
        ),
        portfolio_constraints=PortfolioConstraints(
            max_positions=args.max_positions,
            risk_per_trade_fraction=args.risk_per_trade,
            max_total_risk_fraction=args.max_total_risk,
            max_position_fraction=args.max_position,
            max_deployed_fraction=args.max_deployed,
            max_sector_deployed_fraction=args.max_sector,
            min_net_ev_bps=args.min_net_ev_bps,
        ),
        initial_equity=args.initial_equity,
    )
    write_result(result, args.output)

    print("\n" + "=" * 78)
    print("V0.7 PORTFOLIO WALK-FORWARD SUMMARY")
    print("=" * 78)
    print(f"Windows: {result['windows_completed']}/{result['windows_requested']}")
    print(f"Portfolio positive-return windows: {result['portfolio_positive_return_windows']}")
    print(f"Average window return: {result['portfolio_average_window_return']:.2%}")
    print(f"Worst window drawdown: {result['portfolio_worst_window_drawdown']:.2%}")
    print(f"Portfolio EV: {result['portfolio_trade_metrics']['expectancy_bps']:.2f} bps/trade")
    print(f"Baseline EV:  {result['baseline_trade_metrics']['expectancy_bps']:.2f} bps/trade")
    print(f"EV improvement: {result['expectancy_improvement_bps']:.2f} bps/trade")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
