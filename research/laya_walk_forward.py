"""Held-out V0.8 comparison: identical V0.7 portfolio with and without Laya."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.engine import ExecutionConfig
from models.laya_calibration import LayaCalibration
from models.laya_engine import LayaEngine
from models.train_v2 import load_dataset
from research.portfolio_walk_forward import run_portfolio_walk_forward
from research.storage import ParquetDataLake
from research.walk_forward import WalkForwardConfig
from strategy.laya_gate import LayaGateConfig, LayaVetoGate
from strategy.portfolio_allocator import PortfolioConstraints


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Laya as a V0.8 veto gate")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--laya-model", required=True, help="Fine-tuned Laya checkpoint/repository id")
    parser.add_argument("--laya-calibration", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--calibration-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
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
    parser.add_argument("--top-laya-candidates", type=int, default=5)
    parser.add_argument("--min-laya-confidence", type=float, default=0.0)
    parser.add_argument("--risk-veto-probability", type=float, default=0.70)
    parser.add_argument("--max-drawdown-degradation", type=float, default=0.005)
    parser.add_argument("--output", default="data/experiments/laya_walk_forward.json")
    args = parser.parse_args()

    calibration = LayaCalibration.load(args.laya_calibration) if args.laya_calibration else None
    gate = LayaVetoGate(
        LayaEngine(args.laya_model, device=args.device),
        config=LayaGateConfig(
            top_candidates=args.top_laya_candidates,
            min_action_answer_confidence=args.min_laya_confidence,
            risk_veto_probability=args.risk_veto_probability,
            fail_closed=True,
        ),
        calibration=calibration,
    )

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
            confidence_threshold=0.60,
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
        ),
        initial_equity=args.initial_equity,
        candidate_gate=gate,
    )

    gated = result["gated_portfolio"]
    passed = (
        gated["expectancy_delta_vs_ungated_bps"] > 0.0
        and gated["worst_drawdown_delta_vs_ungated"] >= -args.max_drawdown_degradation
    )
    result["laya_pass_condition"] = {
        "passed": passed,
        "requires_positive_expectancy_delta": True,
        "max_drawdown_degradation": args.max_drawdown_degradation,
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")

    print("\n" + "=" * 78)
    print("V0.8 LAYA HELD-OUT ECONOMIC COMPARISON")
    print("=" * 78)
    print(f"Ungated EV: {result['portfolio_trade_metrics']['expectancy_bps']:.2f} bps/trade")
    print(f"Laya-gated EV: {gated['trade_metrics']['expectancy_bps']:.2f} bps/trade")
    print(f"EV delta: {gated['expectancy_delta_vs_ungated_bps']:.2f} bps/trade")
    print(f"Average return delta: {gated['average_return_delta_vs_ungated']:.2%}")
    print(f"Worst drawdown delta: {gated['worst_drawdown_delta_vs_ungated']:.2%}")
    print(f"PASS: {passed}")
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
