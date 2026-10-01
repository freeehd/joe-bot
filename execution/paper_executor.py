"""Legacy convenience wrapper over the V0.9 Alpaca paper-only broker."""

from __future__ import annotations

import uuid

from execution.broker import AlpacaPaperBroker, OrderIntent, OrderIntentType, OrderSide


class PaperExecutor:
    def __init__(self, broker: AlpacaPaperBroker | None = None) -> None:
        self.broker = broker or AlpacaPaperBroker()

    def buy(self, symbol, quantity):
        return self.broker.submit_market_order(
            OrderIntent(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=int(quantity),
                intent_type=OrderIntentType.ENTRY,
                client_order_id=f"legacy-buy-{uuid.uuid4().hex}",
                reason="legacy paper executor buy",
            )
        )

    def sell(self, symbol, quantity):
        return self.broker.submit_market_order(
            OrderIntent(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=int(quantity),
                intent_type=OrderIntentType.ENTRY,
                client_order_id=f"legacy-sell-{uuid.uuid4().hex}",
                reason="legacy paper executor sell",
            )
        )
