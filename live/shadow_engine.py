"""V0.95 shadow-production engine: full decisions, zero external orders."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from execution.broker import OrderIntentType, OrderSnapshot, OrderStatus
from execution.shadow_broker import ShadowBroker
from live.paper_engine import PaperTradingEngine
from market.stream import MarketEvent


class ShadowTradingEngine(PaperTradingEngine):
    def __init__(self, *, broker: ShadowBroker, **kwargs: Any) -> None:
        if not getattr(broker, "shadow", False) or getattr(broker, "external_orders", True):
            raise ValueError("ShadowTradingEngine requires a network-isolated ShadowBroker")
        super().__init__(broker=broker, **kwargs)
        self.broker: ShadowBroker = broker
        self._entry_fills: dict[str, OrderSnapshot] = {}
        self._entry_decisions: dict[str, str | None] = {}

    def start(self, *, now: datetime | None = None) -> None:
        super().start(now=now)
        self.audit.append(
            "shadow_mode_started",
            {"external_orders": False, "broker": "ShadowBroker"},
            timestamp=now,
        )

    def on_market_event(self, event: MarketEvent) -> None:
        self._require_started()
        # First let already-accepted hypothetical orders fill from the new bar.
        for update in self.broker.on_market_event(event):
            self.on_order_update(update)
        super().on_market_event(event)
        # Keep account marks synchronized after each live-market event.
        self.account = self.broker.get_account()
        self.risk_engine.update_equity(self.account.equity)

    def on_order_update(self, snapshot: OrderSnapshot) -> None:
        managed_before = self.orders.get(snapshot.client_order_id)
        super().on_order_update(snapshot)
        if snapshot.status != OrderStatus.FILLED or managed_before is None:
            return
        intent = managed_before.intent
        decision_id = intent.metadata.get("decision_id") if isinstance(intent.metadata, dict) else None
        if intent.intent_type == OrderIntentType.ENTRY:
            self._entry_fills[snapshot.symbol] = snapshot
            self._entry_decisions[snapshot.symbol] = decision_id
            expected = intent.metadata.get("reference_price") if isinstance(intent.metadata, dict) else None
            slippage_bps = None
            if expected and snapshot.average_fill_price:
                sign = 1.0 if intent.side.value == "BUY" else -1.0
                slippage_bps = sign * (snapshot.average_fill_price / float(expected) - 1.0) * 10_000.0
            fill_latency_seconds = None
            if snapshot.submitted_at is not None and snapshot.updated_at is not None:
                fill_latency_seconds = (snapshot.updated_at - snapshot.submitted_at).total_seconds()
            self.audit.append(
                "shadow_entry_filled",
                {
                    "expected_price": expected,
                    "actual_fill_price": snapshot.average_fill_price,
                    "fill_slippage_bps": slippage_bps,
                    "signal_to_fill_seconds": fill_latency_seconds,
                    "quantity": snapshot.filled_quantity,
                    "target_price": intent.metadata.get("target_price"),
                    "stop_price": intent.metadata.get("stop_price"),
                    "expected_ev_bps": intent.metadata.get("expected_ev_bps"),
                },
                decision_id=decision_id,
                symbol=snapshot.symbol,
                timestamp=snapshot.updated_at,
            )
        else:
            entry = self._entry_fills.pop(snapshot.symbol, None)
            original_decision_id = self._entry_decisions.pop(snapshot.symbol, None)
            realized_return = None
            if entry is not None and entry.average_fill_price and snapshot.average_fill_price:
                if entry.side.value == "BUY":
                    realized_return = snapshot.average_fill_price / entry.average_fill_price - 1.0
                else:
                    realized_return = (entry.average_fill_price - snapshot.average_fill_price) / entry.average_fill_price
            self.audit.append(
                "shadow_trade_closed",
                {
                    "entry_price": None if entry is None else entry.average_fill_price,
                    "exit_price": snapshot.average_fill_price,
                    "realized_return": realized_return,
                    "realized_bps": None if realized_return is None else realized_return * 10_000.0,
                    "exit_reason": intent.metadata.get("exit_reason"),
                    "quantity": snapshot.filled_quantity,
                },
                decision_id=original_decision_id,
                symbol=snapshot.symbol,
                timestamp=snapshot.updated_at,
            )

    def _process_entry(self, proposal, *, now: datetime) -> None:
        # Preserve the expected decision-time reference price in the order audit.
        metadata = dict(proposal.metadata)
        metadata["reference_price"] = proposal.reference_price
        proposal = type(proposal)(
            symbol=proposal.symbol,
            direction=proposal.direction,
            quantity=proposal.quantity,
            reference_price=proposal.reference_price,
            stop_price=proposal.stop_price,
            target_price=proposal.target_price,
            expected_ev_bps=proposal.expected_ev_bps,
            decision_id=proposal.decision_id,
            sector=proposal.sector,
            spread_bps=proposal.spread_bps,
            liquidity_dollars=proposal.liquidity_dollars,
            volatility_fraction=proposal.volatility_fraction,
            metadata=metadata,
        )
        self.audit.append(
            "shadow_candidate",
            {"proposal": asdict(proposal), "external_order": False},
            decision_id=proposal.decision_id,
            symbol=proposal.symbol,
            timestamp=now,
        )
        super()._process_entry(proposal, now=now)
