"""V0.9 event-driven paper-trading coordinator.

The coordinator deliberately receives candidate proposals through an injected
provider.  That keeps live state/execution independent from the alpha/Laya
implementation while allowing the exact V0.7/V0.8 stack to be plugged in after
its empirical gates pass.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from database.db import AuditStore
from execution.broker import (
    AccountSnapshot,
    BrokerProtocol,
    OrderIntent,
    OrderIntentType,
    OrderSide,
    OrderSnapshot,
    OrderStatus,
)
from execution.exit_engine import ExitEngine
from execution.order_manager import OrderManager
from market.stream import BarEvent, MarketEvent, QuoteEvent
from risk.position_manager import LivePositionManager, ManagedPosition
from risk.risk_engine import ProductionRiskEngine


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class EntryProposal:
    symbol: str
    direction: str
    quantity: int
    reference_price: float
    stop_price: float
    target_price: float
    expected_ev_bps: float
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sector: str = "UNKNOWN"
    spread_bps: float = 0.0
    liquidity_dollars: float = float("inf")
    volatility_fraction: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "direction", self.direction.upper())
        if self.direction not in {"LONG", "SHORT"}:
            raise ValueError("direction must be LONG or SHORT")
        if self.quantity < 1:
            raise ValueError("quantity must be >= 1")
        if min(self.reference_price, self.stop_price, self.target_price) <= 0:
            raise ValueError("proposal prices must be positive")

    @property
    def proposed_notional(self) -> float:
        return self.quantity * self.reference_price


CandidateProvider = Callable[[BarEvent, "PaperEngineState"], Sequence[EntryProposal]]


@dataclass(frozen=True)
class PaperEngineState:
    account: AccountSnapshot
    positions: tuple[dict[str, Any], ...]
    entries_enabled: bool
    kill_switches: tuple[str, ...]


class PaperTradingEngine:
    def __init__(
        self,
        *,
        broker: BrokerProtocol,
        risk_engine: ProductionRiskEngine,
        exit_engine: ExitEngine,
        audit_store: AuditStore,
        candidate_provider: CandidateProvider | None = None,
        order_manager: OrderManager | None = None,
        position_manager: LivePositionManager | None = None,
    ) -> None:
        if not getattr(broker, "paper", False):
            raise ValueError("V0.9 paper engine refuses non-paper brokers")
        self.broker = broker
        self.risk_engine = risk_engine
        self.exit_engine = exit_engine
        self.audit = audit_store
        self.candidate_provider = candidate_provider
        self.orders = order_manager or OrderManager(broker)
        self.positions = position_manager or LivePositionManager()
        self.account: AccountSnapshot | None = None
        self.started = False
        self.latest_spreads: dict[str, float] = {}
        self._pending_exit_symbols: set[str] = set()

    def start(self, *, now: datetime | None = None) -> None:
        now = now or utc_now()
        account = self.broker.get_account()
        self.account = account
        broker_positions = self.broker.get_positions()
        self.positions.adopt_broker_positions(broker_positions, opened_at=now)
        self.risk_engine.reset_session(account.equity)
        self.risk_engine.update_health(broker_connected=True, broker_trading_blocked=account.trading_blocked)
        self.started = True
        self.audit.append(
            "engine_started",
            {
                "account": asdict(account),
                "positions": [asdict(item) for item in broker_positions],
                "paper": True,
            },
            timestamp=now,
        )

    def stop(self, reason: str = "operator shutdown", *, now: datetime | None = None) -> None:
        self.risk_engine.manual_halt(reason)
        self.audit.append("engine_stopped", {"reason": reason}, timestamp=now or utc_now())
        self.started = False

    def _require_started(self) -> None:
        if not self.started or self.account is None:
            raise RuntimeError("paper engine has not been started")

    def state(self, now: datetime | None = None) -> PaperEngineState:
        self._require_started()
        kills = tuple(reason.value for reason in self.risk_engine.active_kill_switches(now))
        return PaperEngineState(
            account=self.account,  # type: ignore[arg-type]
            positions=tuple(position.to_dict() for position in self.positions.positions()),
            entries_enabled=not kills,
            kill_switches=kills,
        )

    def on_market_event(self, event: MarketEvent) -> None:
        self._require_started()
        self.risk_engine.mark_market_event(event.received_at or event.timestamp)
        self.risk_engine.update_health(websocket_stable=True)
        if isinstance(event, QuoteEvent):
            spread = event.spread_bps
            if spread is not None:
                self.latest_spreads[event.symbol] = spread
            mid = event.mid
            if mid is not None:
                self.positions.mark_price(event.symbol, mid)
                self._evaluate_exit(event.symbol, mid, event.timestamp)
            return

        self.positions.mark_price(event.symbol, event.close)
        self._evaluate_exit(event.symbol, event.close, event.timestamp)
        if self.candidate_provider is None:
            return
        # Do not open a fresh position in a symbol while an exit is pending.
        proposals = self.candidate_provider(event, self.state(event.timestamp))
        for proposal in proposals:
            self._process_entry(proposal, now=event.timestamp)

    def _evaluate_exit(self, symbol: str, price: float, now: datetime) -> None:
        position = self.positions.get(symbol)
        if position is None or symbol in self._pending_exit_symbols:
            return
        decision = self.exit_engine.evaluate(position, price=price, now=now)
        if not decision.should_exit:
            return
        risk_decision = self.risk_engine.approve_exit()
        if not risk_decision.approved:
            raise RuntimeError("risk engine violated exit invariant")
        side = OrderSide.SELL if position.direction == "LONG" else OrderSide.BUY
        client_id = f"exit-{symbol.lower()}-{uuid.uuid4().hex[:20]}"
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            quantity=max(1, int(round(position.absolute_quantity))),
            intent_type=OrderIntentType.EXIT,
            client_order_id=client_id,
            reason=f"exit_engine:{decision.reason.value if decision.reason else 'unknown'}",
            metadata={"trigger_price": price, "exit_reason": decision.reason.value if decision.reason else None},
        )
        snapshot = self.orders.submit(intent)
        self._pending_exit_symbols.add(symbol)
        self.audit.append(
            "exit_order_submitted",
            {"intent": asdict(intent), "broker_order": snapshot.to_dict()},
            symbol=symbol,
            timestamp=now,
        )

    def _process_entry(self, proposal: EntryProposal, *, now: datetime) -> None:
        self._require_started()
        if proposal.symbol in self._pending_exit_symbols:
            return
        if self.positions.get(proposal.symbol) is not None or self.orders.has_live_entry(proposal.symbol):
            self.audit.append(
                "entry_rejected",
                {"reason": "symbol already held or has a live entry order", "proposal": asdict(proposal)},
                decision_id=proposal.decision_id,
                symbol=proposal.symbol,
                timestamp=now,
            )
            return
        spread = self.latest_spreads.get(proposal.symbol, proposal.spread_bps)
        decision = self.risk_engine.approve_entry(
            account_equity=self.account.equity,  # type: ignore[union-attr]
            proposed_notional=proposal.proposed_notional,
            open_positions=len(self.positions.positions()),
            gross_exposure=self.positions.gross_exposure(),
            net_exposure=self.positions.net_exposure(),
            sector_exposure=self.positions.sector_exposure(proposal.sector),
            direction=proposal.direction,
            spread_bps=spread,
            liquidity_dollars=proposal.liquidity_dollars,
            volatility_fraction=proposal.volatility_fraction,
            now=now,
        )
        self.audit.append(
            "risk_decision",
            {"proposal": asdict(proposal), "risk": asdict(decision)},
            decision_id=proposal.decision_id,
            symbol=proposal.symbol,
            timestamp=now,
        )
        if not decision.approved:
            return

        side = OrderSide.BUY if proposal.direction == "LONG" else OrderSide.SELL
        intent = OrderIntent(
            symbol=proposal.symbol,
            side=side,
            quantity=proposal.quantity,
            intent_type=OrderIntentType.ENTRY,
            client_order_id=f"entry-{proposal.symbol.lower()}-{proposal.decision_id[:20]}",
            reason=f"positive_ev:{proposal.expected_ev_bps:.2f}bps",
            metadata={
                "decision_id": proposal.decision_id,
                "direction": proposal.direction,
                "stop_price": proposal.stop_price,
                "target_price": proposal.target_price,
                "sector": proposal.sector,
                "expected_ev_bps": proposal.expected_ev_bps,
                **proposal.metadata,
            },
        )
        snapshot = self.orders.submit(intent)
        self.audit.append(
            "entry_order_submitted",
            {"intent": asdict(intent), "broker_order": snapshot.to_dict()},
            decision_id=proposal.decision_id,
            symbol=proposal.symbol,
            timestamp=now,
        )

    def on_order_update(self, snapshot: OrderSnapshot) -> None:
        self._require_started()
        managed = self.orders.apply_update(snapshot)
        intent = managed.intent
        decision_id = intent.metadata.get("decision_id") if isinstance(intent.metadata, dict) else None
        self.audit.append(
            "order_update",
            snapshot.to_dict(),
            decision_id=decision_id,
            symbol=snapshot.symbol,
            timestamp=snapshot.updated_at or utc_now(),
        )
        if snapshot.status == OrderStatus.FILLED:
            self.positions.apply_filled_order(intent, snapshot)
            if intent.intent_type == OrderIntentType.EXIT:
                self._pending_exit_symbols.discard(intent.symbol)
            # Broker position endpoints can lag a fill WebSocket update briefly.
            # The periodic runtime reconciliation checks broker truth after that
            # propagation window instead of creating an immediate false kill switch.

    def reconcile_positions(self, *, timestamp: datetime | None = None) -> dict[str, Any]:
        self._require_started()
        broker_positions = self.broker.get_positions()
        result = self.positions.reconcile(broker_positions)
        self.risk_engine.update_health(position_state_consistent=result["consistent"])
        self.audit.append("position_reconciliation", result, timestamp=timestamp or utc_now())
        return result

    def reconcile_account(self, *, timestamp: datetime | None = None) -> AccountSnapshot:
        self._require_started()
        self.account = self.broker.get_account()
        self.risk_engine.update_equity(self.account.equity)
        self.risk_engine.update_health(
            broker_connected=True,
            broker_trading_blocked=self.account.trading_blocked,
        )
        self.audit.append("account_reconciliation", asdict(self.account), timestamp=timestamp or utc_now())
        return self.account

    def broker_disconnected(self, error: str | None = None) -> None:
        self.risk_engine.update_health(broker_connected=False)
        self.audit.append("broker_disconnected", {"error": error}, timestamp=utc_now())

    def market_stream_unstable(self, error: str | None = None) -> None:
        self.risk_engine.update_health(websocket_stable=False)
        self.audit.append("market_stream_unstable", {"error": error}, timestamp=utc_now())
