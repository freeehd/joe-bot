from risk.risk_engine import RiskEngine


risk = RiskEngine()


fake_decision = {

    "approved": True,

    "action": "long"
}


result = risk.approve_trade(

    decision=fake_decision,

    account_equity=10000,

    daily_pnl=-20,

    open_positions=0
)


print(result)


quantity = risk.calculate_position_size(

    account_equity=10000,

    entry_price=100,

    stop_price=99.50
)


print(
    "Allowed quantity:",
    quantity
)