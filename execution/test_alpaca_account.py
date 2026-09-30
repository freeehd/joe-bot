import os

from dotenv import load_dotenv

from alpaca.trading.client import (
    TradingClient
)


load_dotenv()


client = TradingClient(

    os.getenv("ALPACA_API_KEY"),

    os.getenv("ALPACA_SECRET_KEY"),

    paper=True
)


account = client.get_account()


print(
    "Account status:",
    account.status
)

print(
    "Equity:",
    account.equity
)

print(
    "Buying power:",
    account.buying_power
)