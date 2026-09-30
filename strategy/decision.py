class TradeDecision:

    def evaluate(
        self,
        laya_result,
        alpha_probability
    ):

        answers = laya_result["answers"]

        action_data = answers["action"]

        action = action_data["choice"]

        # Laya normally provides confidence
        confidence = action_data.get(
            "confidence",
            0
        )

        risk_probability = answers[
            "high_risk"
        ]["noul"]

        decision = {
            "action": action,
            "laya_confidence": confidence,
            "risk_probability": risk_probability,
            "alpha_probability": alpha_probability,
            "approved": False,
            "reason": ""
        }

        # Quant model requirement
        if alpha_probability < 0.65:

            decision["reason"] = (
                "XGBoost probability too low."
            )

            return decision

        # Laya uncertainty requirement
        if confidence < 0.65:

            decision["reason"] = (
                "Laya confidence too low."
            )

            return decision

        # Laya risk classifier
        if risk_probability > 0.60:

            decision["reason"] = (
                "Laya classified setup as high risk."
            )

            return decision

        if action == "wait":

            decision["reason"] = (
                "Laya recommends waiting."
            )

            return decision

        decision["approved"] = True

        decision["reason"] = (
            "Alpha and Laya both approve."
        )

        return decision