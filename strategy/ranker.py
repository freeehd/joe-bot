class CandidateRanker:

    def calculate_score(
        self,
        candidate
    ):

        alpha = candidate[
            "alpha_probability"
        ]


        relative_volume = candidate[
            "relative_volume"
        ]


        return_1m = candidate[
            "return_1m"
        ]


        return_5m = candidate[
            "return_5m"
        ]


        ema_distance = candidate[
            "ema_distance"
        ]


        vwap_distance = candidate[
            "vwap_distance"
        ]


        direction = candidate[
            "direction"
        ]


        # ==================================
        # ALPHA SCORE
        # ==================================

        alpha_score = min(
            alpha / 0.50,
            1.0
        )


        # ==================================
        # VOLUME SCORE
        # ==================================

        volume_score = min(
            relative_volume / 2.0,
            1.0
        )


        # ==================================
        # MOMENTUM SCORE
        # ==================================

        movement = (

            abs(return_1m)

            +

            abs(return_5m)
        )


        momentum_score = min(
            movement / 0.01,
            1.0
        )


        # ==================================
        # DIRECTION CONFIRMATION
        # ==================================

        trend_score = 0.0


        if direction == "LONG":

            if (
                return_5m > 0
                and
                ema_distance > 0
                and
                vwap_distance > 0
            ):

                trend_score = 1.0

            elif (
                ema_distance > 0
                or
                vwap_distance > 0
            ):

                trend_score = 0.5


        elif direction == "SHORT":

            if (
                return_5m < 0
                and
                ema_distance < 0
                and
                vwap_distance < 0
            ):

                trend_score = 1.0

            elif (
                ema_distance < 0
                or
                vwap_distance < 0
            ):

                trend_score = 0.5


        # ==================================
        # LONG/SHORT SEPARATION
        # ==================================

        long_p = candidate[
            "long_probability"
        ]


        short_p = candidate[
            "short_probability"
        ]


        probability_gap = abs(
            long_p - short_p
        )


        certainty_score = min(
            probability_gap / 0.30,
            1.0
        )


        # ==================================
        # FINAL
        # ==================================

        score = (

            alpha_score
            *
            0.45

            +

            volume_score
            *
            0.15

            +

            momentum_score
            *
            0.15

            +

            trend_score
            *
            0.15

            +

            certainty_score
            *
            0.10
        )


        return float(
            score
        )


    def rank(
        self,
        candidates
    ):

        ranked = []


        for candidate in candidates:

            if not candidate[
                "active"
            ]:

                continue


            candidate = (
                candidate.copy()
            )


            candidate[
                "score"
            ] = (

                self.calculate_score(
                    candidate
                )
            )


            ranked.append(
                candidate
            )


        ranked.sort(

            key=lambda x:
                x["score"],

            reverse=True
        )


        return ranked