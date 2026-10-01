"""Connectivity/reconciliation runner for the V0.9 Alpaca paper engine.

New entries are intentionally disabled here until V0.5-V0.8 pass their real
validation gates and a validated live feature/model provider is plugged into
``PaperTradingEngine``. Existing paper positions can still be monitored/exited.
"""

from __future__ import annotations

import argparse

from database.db import AuditStore
from execution.broker import AlpacaPaperBroker, AlpacaPaperTradeStream
from execution.exit_engine import ExitEngine, ExitPolicy
from live.paper_engine import PaperTradingEngine
from live.runtime import PaperRuntime, RuntimeConfig
from market.stream import AlpacaStockStream
from market.universe import RESEARCH_SYMBOLS
from risk.risk_engine import ProductionRiskEngine, RiskLimits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V0.9 Alpaca PAPER connectivity/position engine")
    parser.add_argument("--symbols", default=",".join(RESEARCH_SYMBOLS))
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--audit-db", default="data/paper/audit.sqlite3")
    parser.add_argument("--stale-seconds", type=float, default=10.0)
    parser.add_argument("--reconcile-seconds", type=float, default=15.0)
    parser.add_argument("--target-pct", type=float, default=0.003)
    parser.add_argument("--stop-pct", type=float, default=0.0015)
    parser.add_argument("--max-hold-minutes", type=int, default=20)
    args = parser.parse_args()

    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    broker = AlpacaPaperBroker()
    audit = AuditStore(args.audit_db)
    risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=args.stale_seconds))
    engine = PaperTradingEngine(
        broker=broker,
        risk_engine=risk,
        exit_engine=ExitEngine(
            ExitPolicy(
                target_pct=args.target_pct,
                stop_pct=args.stop_pct,
                max_holding_minutes=args.max_hold_minutes,
            )
        ),
        audit_store=audit,
        candidate_provider=None,  # deliberate: monitor/exit only until research gates pass
    )
    market_stream = AlpacaStockStream(symbols, engine.on_market_event, feed=args.feed)
    trade_stream = AlpacaPaperTradeStream(engine.on_order_update)
    runtime = PaperRuntime(
        engine,
        market_stream,
        trade_stream,
        config=RuntimeConfig(reconciliation_interval_seconds=args.reconcile_seconds),
    )

    print("=" * 78)
    print("JOE BOT V0.9 — ALPACA PAPER CONNECTIVITY / POSITION ENGINE")
    print("=" * 78)
    print("Paper broker only. New entries are disabled in this runner.")
    print("Existing paper positions can still be monitored and risk-reducing exits remain enabled.")
    print(f"Symbols: {len(symbols)} | Feed: {args.feed} | Audit DB: {args.audit_db}")
    runtime.run_forever()


if __name__ == "__main__":
    main()
