from config.settings import (
    TRADING_CAPITAL,
    MAX_CAPITAL_DEPLOYED,
    MAX_POSITIONS,
    MAX_POSITION_PERCENT,
    MIN_POSITION_DOLLARS,
    MIN_ALPHA_PROBABILITY,
)


class CapitalAllocator:

    def allocate(self, ranked_candidates):

        max_deployment = (
            TRADING_CAPITAL
            * MAX_CAPITAL_DEPLOYED
        )

        max_single_position = (
            TRADING_CAPITAL
            * MAX_POSITION_PERCENT
        )

        # =====================================
        # ELIGIBILITY
        # =====================================

        eligible = []

        for candidate in ranked_candidates:

            if candidate["alpha_probability"] < MIN_ALPHA_PROBABILITY:
                continue

            if candidate["score"] < 0.45:
                continue

            if candidate["direction"] not in (
                "LONG",
                "SHORT"
            ):
                continue

            eligible.append(candidate)

        eligible = eligible[:MAX_POSITIONS]

        if not eligible:

            return {
                "positions": [],
                "capital_deployed": 0.0,
                "cash_reserved": TRADING_CAPITAL,
            }

        # =====================================
        # DON'T AUTOMATICALLY USE ALL CAPITAL
        # =====================================

        # Opportunity strength determines how
        # much of max deployable capital is used.
        best_score = eligible[0]["score"]

        if best_score >= 0.80:
            deployment_factor = 1.00

        elif best_score >= 0.70:
            deployment_factor = 0.80

        elif best_score >= 0.60:
            deployment_factor = 0.60

        elif best_score >= 0.50:
            deployment_factor = 0.40

        else:
            deployment_factor = 0.25

        capital_budget = (
            max_deployment
            * deployment_factor
        )

        # =====================================
        # SCORE-WEIGHTED DISTRIBUTION
        # =====================================

        total_score = sum(
            item["score"]
            for item in eligible
        )

        positions = []

        for candidate in eligible:

            weight = (
                candidate["score"]
                / total_score
            )

            allocation = (
                capital_budget
                * weight
            )

            allocation = min(
                allocation,
                max_single_position
            )

            if allocation < MIN_POSITION_DOLLARS:
                continue

            allocation = round(
                allocation,
                2
            )

            estimated_quantity = (
                allocation
                / candidate["price"]
            )

            positions.append(
                {
                    "symbol": candidate["symbol"],
                    "direction": candidate["direction"],
                    "price": candidate["price"],
                    "allocation": allocation,
                    "estimated_quantity": estimated_quantity,
                    "alpha_probability":
                        candidate["alpha_probability"],
                    "long_probability":
                        candidate["long_probability"],
                    "short_probability":
                        candidate["short_probability"],
                    "score":
                        candidate["score"],
                }
            )

        # IMPORTANT:
        # calculate deployed capital from
        # positions that ACTUALLY survived.

        deployed = round(
            sum(
                position["allocation"]
                for position in positions
            ),
            2
        )

        cash = round(
            TRADING_CAPITAL - deployed,
            2
        )

        return {
            "positions": positions,
            "capital_deployed": deployed,
            "cash_reserved": cash,
        }