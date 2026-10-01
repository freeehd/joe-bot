import pandas as pd


def _session_groups(df: pd.DataFrame):
    if isinstance(df.index, pd.DatetimeIndex):
        return df.groupby(df.index.normalize(), group_keys=False, sort=False)
    return None


def _create_features_single_session(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1m"] = df["close"].pct_change(1)
    df["return_3m"] = df["close"].pct_change(3)
    df["return_5m"] = df["close"].pct_change(5)

    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_distance"] = (df["ema_5"] - df["ema_20"]) / df["close"]

    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    df["volume_avg_20"] = df["volume"].rolling(20).mean()
    df["relative_volume"] = df["volume"] / df["volume_avg_20"]

    df["range"] = (df["high"] - df["low"]) / df["close"]

    return df


def create_features(df: pd.DataFrame, *, session_aware: bool = False) -> pd.DataFrame:
    """Create the v0.3 feature set.

    ``session_aware=False`` preserves compatibility with the existing checked-in
    binary models. New research/training code should use ``session_aware=True``
    so returns, EMAs and rolling volume do not bridge overnight gaps.
    """

    if not session_aware:
        return _create_features_single_session(df)

    groups = _session_groups(df)
    if groups is None:
        return _create_features_single_session(df)

    return groups.apply(_create_features_single_session)


def create_labels(df):
    """Legacy v0.3 labels retained only for reproducibility of old models."""

    df = df.copy()
    df["future_close_5m"] = df["close"].shift(-5)
    df["future_return_5m"] = df["future_close_5m"] / df["close"] - 1
    df["long_target"] = (df["future_return_5m"] > 0.002).astype(int)
    df["short_target"] = (df["future_return_5m"] < -0.002).astype(int)
    return df.dropna()
