"""V0.95 live-market shadow runner: real data, local-only hypothetical execution."""
from __future__ import annotations

import argparse
import importlib
from typing import Callable

from backtest.engine import ExecutionConfig
from database.db import AuditStore
from execution.exit_engine import ExitEngine, ExitPolicy
from execution.shadow_broker import ShadowBroker
from live.paper_engine import CandidateProvider
from live.shadow_engine import ShadowTradingEngine
from live.shadow_runtime import ShadowRuntime, ShadowRuntimeConfig
from market.stream import AlpacaStockStream
from market.universe import RESEARCH_SYMBOLS
from risk.risk_engine import ProductionRiskEngine, RiskLimits


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
    parser.add_argument("--symbols", default=",".join(RESEARCH_SYMBOLS))
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--provider", default=None, help="Validated live candidate provider as module:function")
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
    print(f"Candidate provider: {args.provider or 'NONE (monitor-only)'}")
    print(f"Symbols: {len(symbols)} | Feed: {args.feed} | Audit DB: {args.audit_db}")
    runtime.run_forever()


if __name__ == "__main__":
    main()
