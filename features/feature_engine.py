import pandas as pd


def create_features(df: pd.DataFrame):

    df = df.copy()

    # Returns
    df["return_1m"] = df["close"].pct_change(1)
    df["return_3m"] = df["close"].pct_change(3)
    df["return_5m"] = df["close"].pct_change(5)

    # Moving averages
    df["ema_5"] = df["close"].ewm(span=5).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()

    df["ema_distance"] = (
        df["ema_5"] - df["ema_20"]
    ) / df["close"]

    # VWAP
    df["vwap_distance"] = (
        df["close"] - df["vwap"]
    ) / df["vwap"]

    # Volume
    df["volume_avg_20"] = df["volume"].rolling(20).mean()

    df["relative_volume"] = (
        df["volume"] / df["volume_avg_20"]
    )

    # Candle range
    df["range"] = (
        df["high"] - df["low"]
    ) / df["close"]

    return df

def create_labels(df):

    df = df.copy()

    # ==========================================
    # FUTURE PRICE
    # ==========================================

    df["future_close_5m"] = (
        df["close"].shift(-5)
    )


    df["future_return_5m"] = (
        df["future_close_5m"]
        /
        df["close"]
        -
        1
    )


    # ==========================================
    # LONG TARGET
    # ==========================================

    # +0.20% within the 5-minute closing horizon

    df["long_target"] = (
        df["future_return_5m"]
        >
        0.002
    ).astype(int)


    # ==========================================
    # SHORT TARGET
    # ==========================================

    # -0.20% within the 5-minute closing horizon

    df["short_target"] = (
        df["future_return_5m"]
        <
        -0.002
    ).astype(int)


    return df.dropna()