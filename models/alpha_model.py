import joblib
import pandas as pd


LONG_MODEL_PATH = (
    "data/models/xgboost_long.pkl"
)

SHORT_MODEL_PATH = (
    "data/models/xgboost_short.pkl"
)


FEATURES = [
    "return_1m",
    "return_3m",
    "return_5m",
    "ema_distance",
    "vwap_distance",
    "relative_volume",
    "range",
]


class AlphaModel:

    def __init__(self):

        print(
            "Loading LONG XGBoost model..."
        )

        self.long_model = (
            joblib.load(
                LONG_MODEL_PATH
            )
        )


        print(
            "Loading SHORT XGBoost model..."
        )

        self.short_model = (
            joblib.load(
                SHORT_MODEL_PATH
            )
        )


        print(
            "XGBoost models loaded."
        )


    def _prepare(
        self,
        feature_row
    ):

        if isinstance(
            feature_row,
            pd.Series
        ):

            feature_row = (
                feature_row
                .to_frame()
                .T
            )


        return feature_row[
            FEATURES
        ]


    def predict(
        self,
        feature_row
    ):

        X = self._prepare(
            feature_row
        )


        long_probability = (
            self.long_model
            .predict_proba(X)[0][1]
        )


        short_probability = (
            self.short_model
            .predict_proba(X)[0][1]
        )


        return {

            "long_probability":
                float(
                    long_probability
                ),

            "short_probability":
                float(
                    short_probability
                ),
        }