class RiskEngine:

    def __init__(
        self,
        max_positions=1,
        max_daily_loss_percent=1.0,
        risk_per_trade_percent=0.25
    ):

        self.max_positions = max_positions

        self.max_daily_loss_percent = (
            max_daily_loss_percent
        )

        self.risk_per_trade_percent = (
            risk_per_trade_percent
        )


    def approve_trade(
        self,
        decision,
        account_equity,
        daily_pnl,
        open_positions
    ):

        if not decision["approved"]:

            return {
                "approved": False,
                "reason":
                "Strategy did not approve trade."
            }


        if open_positions >= self.max_positions:

            return {
                "approved": False,
                "reason":
                "Maximum open positions reached."
            }


        max_daily_loss = (
            account_equity *
            (self.max_daily_loss_percent / 100)
        )


        if daily_pnl <= -max_daily_loss:

            return {
                "approved": False,
                "reason":
                "Daily loss limit reached."
            }


        return {
            "approved": True,
            "reason":
            "Risk checks passed."
        }


    def calculate_position_size(
        self,
        account_equity,
        entry_price,
        stop_price
    ):

        risk_amount = (
            account_equity *
            (self.risk_per_trade_percent / 100)
        )

        risk_per_share = abs(
            entry_price - stop_price
        )

        if risk_per_share <= 0:
            return 0

        quantity = (
            risk_amount /
            risk_per_share
        )

        return int(quantity)