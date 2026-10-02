"""Deterministic Victory Sprint 9 safety/fault-injection campaign.

These checks exercise local safety invariants only. They never instantiate an
external trading client and cannot place broker orders.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from execution.broker import AccountSnapshot, OrderIntent, OrderIntentType, OrderSide, OrderSnapshot, OrderStatus
from execution.order_manager import OrderManager
from risk.risk_engine import KillSwitchReason, ProductionRiskEngine, RiskLimits

UTC = timezone.utc


class _LocalPaperBroker:
    paper = True

    def __init__(self) -> None:
        self.submissions: list[OrderIntent] = []

    def submit_market_order(self, intent: OrderIntent) -> OrderSnapshot:
        self.submissions.append(intent)
        return OrderSnapshot(
            broker_order_id=f"local-{len(self.submissions)}",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            filled_quantity=0,
            status=OrderStatus.ACCEPTED,
            submitted_at=datetime.now(tz=UTC),
        )

    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover - protocol completeness
        return None

    def get_open_orders(self) -> list[OrderSnapshot]:
        return []

    def get_positions(self):
        return []

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(10_000, 10_000, 10_000)


def _healthy(engine: ProductionRiskEngine, now: datetime) -> None:
    engine.update_health(
        broker_connected=True,
        websocket_stable=True,
        model_ready=True,
        features_valid=True,
        position_state_consistent=True,
        unexpected_volatility=False,
        broker_trading_blocked=False,
        clock_drift_seconds=0.0,
    )
    engine.mark_market_event(now)


def _entry_approved(engine: ProductionRiskEngine, now: datetime) -> bool:
    return engine.approve_entry(
        account_equity=10_000,
        proposed_notional=500,
        open_positions=0,
        gross_exposure=0,
        net_exposure=0,
        sector_exposure=0,
        direction="LONG",
        now=now,
    ).approved


def run_fault_injection() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: str) -> None:
        cases.append({"name": name, "passed": bool(passed), "details": details})

    now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)

    risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=5))
    risk.reset_session(10_000)
    _healthy(risk, now - timedelta(seconds=6))
    record(
        "stale_market_data_blocks_entries",
        not _entry_approved(risk, now) and KillSwitchReason.MARKET_DATA_STALE in risk.active_kill_switches(now),
        "stale market timestamp must fail closed",
    )

    risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=30))
    risk.reset_session(10_000)
    _healthy(risk, now)
    risk.update_health(broker_connected=False)
    record(
        "broker_disconnect_blocks_entries",
        not _entry_approved(risk, now) and KillSwitchReason.BROKER_DISCONNECTED in risk.active_kill_switches(now),
        "broker disconnect must disable entries",
    )

    risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=30))
    risk.reset_session(10_000)
    _healthy(risk, now)
    risk.update_health(position_state_consistent=False)
    record(
        "position_mismatch_blocks_entries",
        not _entry_approved(risk, now) and KillSwitchReason.POSITION_STATE_INCONSISTENT in risk.active_kill_switches(now),
        "unreconciled position state must disable entries",
    )

    risk = ProductionRiskEngine(RiskLimits(stale_market_data_seconds=30))
    risk.reset_session(10_000)
    _healthy(risk, now)
    risk.manual_halt("fault injection")
    record(
        "entry_halt_never_blocks_exit",
        (not _entry_approved(risk, now)) and risk.approve_exit().approved,
        "manual/automatic entry halts must preserve risk-reducing exits",
    )

    broker = _LocalPaperBroker()
    manager = OrderManager(broker)
    intent = OrderIntent("AAPL", OrderSide.BUY, 2, OrderIntentType.ENTRY, "fault-idempotent-1", "fault injection")
    first = manager.submit(intent)
    second = manager.submit(intent)
    record(
        "duplicate_client_order_id_is_idempotent",
        len(broker.submissions) == 1 and first == second,
        "same intent/client_order_id must never submit twice",
    )

    passed = sum(1 for case in cases if case["passed"])
    return {
        "campaign": "fault-injection",
        "external_orders": False,
        "cases": cases,
        "passed_cases": passed,
        "total_cases": len(cases),
        "pass_rate": passed / len(cases) if cases else 0.0,
        "all_passed": passed == len(cases),
    }
