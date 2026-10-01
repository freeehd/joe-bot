"""V0.95 live-market shadow runner: real data, local-only hypothetical execution."""
from __future__ import annotations

import argparse
import importlib

import pandas as pd
from typing import Callable

from backtest.engine import ExecutionConfig, TradeConfig
from database.db import AuditStore
from execution.exit_engine import ExitEngine, ExitPolicy
from execution.shadow_broker import ShadowBroker
from live.paper_engine import CandidateProvider
from live.shadow_engine import ShadowTradingEngine
from live.shadow_runtime import ShadowRuntime, ShadowRuntimeConfig
from market.stream import AlpacaStockStream
from market.universe import MARKET_CONTEXT_SYMBOLS, RESEARCH_UNIVERSE_50
from risk.risk_engine import ProductionRiskEngine, RiskLimits
from research.storage import ParquetDataLake
from live.feature_store import LiveFeatureStore
from live.candidate_provider import LiveCandidateProvider, LiveProviderConfig, build_ev_model_from_trades
from models.live_alpha import LiveAlphaModel
from models.laya_engine import LayaEngine
from models.laya_calibration import LayaCalibration
from strategy.laya_gate import LayaVetoGate, LayaGateConfig
from strategy.portfolio_allocator import PortfolioAllocatorV2, PortfolioConstraints


def load_provider(spec: str | None) -> CandidateProvider | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--provider must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    provider = getattr(importlib.import_module(module_name), attribute)
    if not callable(provider):
        raise TypeError("candidate provider must be callable")
    return provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Joe Bot V0.95 shadow production")
    parser.add_argument("--symbols", default=",".join(RESEARCH_UNIVERSE_50 + MARKET_CONTEXT_SYMBOLS))
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--provider", default=None, help="Custom validated live candidate provider as module:function")
    parser.add_argument("--model-bundle", default=None, help="Frozen V0.5 Feature Engine V2 calibrated bundle")
    parser.add_argument("--ev-trades", default=None, help="Earlier realized calibration/paper trades CSV or Parquet for V0.7 EV")
    parser.add_argument("--raw-version", default=None, help="Immutable raw dataset version used to seed live Feature Engine V2")
    parser.add_argument("--correlations", default=None, help="Frozen pre-live symbol correlation matrix CSV")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--warmup-sessions", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-net-ev-bps", type=float, default=0.0)
    parser.add_argument("--laya-model", default=None)
    parser.add_argument("--laya-calibration", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--audit-db", default="data/shadow/audit.sqlite3")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--target-pct", type=float, default=0.003)
    parser.add_argument("--stop-pct", type=float, default=0.0015)
    parser.add_argument("--max-hold-minutes", type=int, default=20)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    provider = load_provider(args.provider)
    builtin_requested = any([args.model_bundle, args.ev_trades, args.raw_version, args.correlations])
    if args.provider and builtin_requested:
        raise ValueError("Use either --provider or the built-in artifact arguments, not both")
    if builtin_requested:
        required = {
            "--model-bundle": args.model_bundle,
            "--ev-trades": args.ev_trades,
            "--raw-version": args.raw_version,
            "--correlations": args.correlations,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Built-in live provider missing required artifacts: {missing}")
        lake = ParquetDataLake(args.data_root)
        history = {symbol: lake.load_raw_bars(args.raw_version, symbol) for symbol in symbols}
        feature_store = LiveFeatureStore(symbols, history, warmup_sessions=args.warmup_sessions)
        trade_config = TradeConfig(
            target_pct=args.target_pct, stop_pct=args.stop_pct, max_holding_bars=args.max_hold_minutes
        )
        execution_config = ExecutionConfig(spread_bps=args.spread_bps, slippage_bps=args.slippage_bps)
        ev_model = build_ev_model_from_trades(args.ev_trades, trade_config=trade_config, execution_config=execution_config)
        correlations = pd.read_csv(args.correlations, index_col=0)
        laya_gate = None
        if args.laya_model:
            calibration = LayaCalibration.load(args.laya_calibration) if args.laya_calibration else None
            laya_gate = LayaVetoGate(
                LayaEngine(args.laya_model, device=args.device),
                config=LayaGateConfig(fail_closed=True),
                calibration=calibration,
            )
        provider = LiveCandidateProvider(
            feature_store, LiveAlphaModel(args.model_bundle), ev_model,
            allocator=PortfolioAllocatorV2(constraints=PortfolioConstraints(min_net_ev_bps=args.min_net_ev_bps), trade_config=trade_config),
            correlations=correlations, laya_gate=laya_gate,
            config=LiveProviderConfig(minimum_direction_confidence=args.min_confidence, min_net_ev_bps=args.min_net_ev_bps),
        )
    broker = ShadowBroker(
        initial_equity=args.initial_equity,
        execution_config=ExecutionConfig(spread_bps=args.spread_bps, slippage_bps=args.slippage_bps),
    )
    audit = AuditStore(args.audit_db)
    engine = ShadowTradingEngine(
        broker=broker,
        risk_engine=ProductionRiskEngine(RiskLimits()),
        exit_engine=ExitEngine(ExitPolicy(target_pct=args.target_pct, stop_pct=args.stop_pct, max_holding_minutes=args.max_hold_minutes)),
        audit_store=audit,
        candidate_provider=provider,
    )
    market = AlpacaStockStream(symbols, engine.on_market_event, feed=args.feed)
    runtime = ShadowRuntime(engine, market, config=ShadowRuntimeConfig())

    print("=" * 78)
    print("JOE BOT V0.95 — SHADOW PRODUCTION")
    print("=" * 78)
    print("Live market data: ON")
    print("External broker orders: IMPOSSIBLE")
    print(f"Candidate provider: {'BUILT-IN V0.5→V0.7' + ('→V0.8' if args.laya_model else '') if builtin_requested else (args.provider or 'NONE (monitor-only)')}")
    print(f"Symbols: {len(symbols)} | Feed: {args.feed} | Audit DB: {args.audit_db}")
    runtime.run_forever()


if __name__ == "__main__":
    main()
