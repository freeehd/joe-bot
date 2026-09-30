import os

from dotenv import load_dotenv

from alpaca.trading.client import (
    TradingClient
)

from alpaca.trading.requests import (
    MarketOrderRequest
)

from alpaca.trading.enums import (
    OrderSide,
    TimeInForce
)


load_dotenv()


class PaperExecutor:

    def __init__(self):

        api_key = os.getenv(
            "ALPACA_API_KEY"
        )

        secret_key = os.getenv(
            "ALPACA_SECRET_KEY"
        )

        self.client = TradingClient(
            api_key,
            secret_key,
            paper=True
        )


    def buy(
        self,
        symbol,
        quantity
    ):

        order = MarketOrderRequest(

            symbol=symbol,

            qty=quantity,

            side=OrderSide.BUY,

            time_in_force=TimeInForce.DAY
        )


        result = self.client.submit_order(
            order_data=order
        )

        return result


    def sell(
        self,
        symbol,
        quantity
    ):

        order = MarketOrderRequest(

            symbol=symbol,

            qty=quantity,

            side=OrderSide.SELL,

            time_in_force=TimeInForce.DAY
        )


        result = self.client.submit_order(
            order_data=order
        )

        return result