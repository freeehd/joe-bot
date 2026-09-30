from execution.paper_executor import (
    PaperExecutor
)


executor = PaperExecutor()


print(
    "Submitting PAPER order..."
)


order = executor.buy(

    symbol="AAPL",

    quantity=1
)


print(order)