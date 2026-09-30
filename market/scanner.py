from market.alpaca_client import (
    get_bars
)

from features.feature_engine import (
    create_features
)

from models.alpha_model import (
    AlphaModel
)
from config.settings import (
    LOOKBACK_DAYS,
    MIN_RELATIVE_VOLUME,
    MIN_ABS_5M_MOVE,
    MIN_DIRECTIONAL_EDGE,
)

class MarketScanner:

    def __init__(self):

        print(
            "Loading XGBoost models "
            "for scanner..."
        )

        self.alpha_model = (
            AlphaModel()
        )

        print(
            "Scanner ready."
        )


    def scan_symbol(
        self,
        symbol
    ):

        try:

            print(
                f"Scanning {symbol}...",
                end=" "
            )


            # ==================================
            # DATA
            # ==================================

            df = get_bars(
                symbol,
                days=LOOKBACK_DAYS
            )


            if (
                df is None
                or
                len(df) == 0
            ):

                print(
                    "NO DATA"
                )

                return None


            if "symbol" in df.index.names:

                df = df.reset_index(
                    level="symbol",
                    drop=True
                )


            # ==================================
            # FEATURES
            # ==================================

            df = create_features(
                df
            )


            if len(df) == 0:

                print(
                    "NO FEATURES"
                )

                return None


            latest = df.iloc[-1]


            # ==================================
            # ALPHA
            # ==================================

            alpha = (
                self.alpha_model.predict(
                    latest
                )
            )


            long_probability = (
                alpha[
                    "long_probability"
                ]
            )


            short_probability = (
                alpha[
                    "short_probability"
                ]
            )


            # ==================================
            # FEATURE VALUES
            # ==================================

            price = float(
                latest["close"]
            )


            return_1m = float(
                latest["return_1m"]
            )


            return_3m = float(
                latest["return_3m"]
            )


            return_5m = float(
                latest["return_5m"]
            )


            relative_volume = float(
                latest[
                    "relative_volume"
                ]
            )


            ema_distance = float(
                latest[
                    "ema_distance"
                ]
            )


            vwap_distance = float(
                latest[
                    "vwap_distance"
                ]
            )


            candle_range = float(
                latest["range"]
            )


         # ==================================
        # DIRECTION
        # ==================================

        probability_gap = abs(
            long_probability
            -
            short_probability
        )


        if (
            probability_gap
            <
            MIN_DIRECTIONAL_EDGE
        ):

            direction = "UNCLEAR"

            alpha_probability = max(
                long_probability,
                short_probability
            )


        elif (
            long_probability
            >
            short_probability
        ):

            direction = "LONG"

            alpha_probability = (
                long_probability
            )


        else:

            direction = "SHORT"

            alpha_probability = (
                short_probability
            )

            # ==================================
            # ACTIVITY FILTER
            # ==================================

            active = True

            rejection_reason = None
            if direction == "UNCLEAR":

                active = False

                rejection_reason = (
                    "directional probability gap too small"
                )

            if (
                relative_volume
                <
                MIN_RELATIVE_VOLUME
            ):

                active = False

                rejection_reason = (
                    "low relative volume"
                )


            if (
                abs(return_5m)
                <
                MIN_ABS_5M_MOVE
            ):

                active = False

                rejection_reason = (
                    "insufficient movement"
                )


            # ==================================
            # RESULT
            # ==================================

            result = {

                "symbol":
                    symbol,

                "price":
                    price,

                "long_probability":
                    long_probability,

                "short_probability":
                    short_probability,

                "alpha_probability":
                    alpha_probability,

                "direction":
                    direction,

                "return_1m":
                    return_1m,

                "return_3m":
                    return_3m,

                "return_5m":
                    return_5m,

                "relative_volume":
                    relative_volume,

                "ema_distance":
                    ema_distance,

                "vwap_distance":
                    vwap_distance,

                "range":
                    candle_range,

                "active":
                    active,

                "rejection_reason":
                    rejection_reason,
                "probability_gap":
                     probability_gap,
            }


            print(

                f"L={long_probability:.2%} "

                f"S={short_probability:.2%} "

                f"DIR={direction} "

                f"5m={return_5m:.2%} "

                f"RVOL={relative_volume:.2f}"
            )


            return result


        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            return None


    def scan_universe(
        self,
        symbols
    ):

        results = []


        for symbol in symbols:

            result = (
                self.scan_symbol(
                    symbol
                )
            )


            if result is not None:

                results.append(
                    result
                )


        return results