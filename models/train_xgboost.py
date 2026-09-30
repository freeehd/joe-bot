import os
import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)

from market.alpaca_client import (
    get_bars
)

from market.universe import (
    SYMBOLS
)

from features.feature_engine import (
    create_features,
    create_labels,
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


TRAINING_DAYS = 60


def download_training_data():

    frames = []

    print()
    print("=" * 70)
    print("BUILDING MULTI-STOCK TRAINING DATASET")
    print("=" * 70)

    for symbol in SYMBOLS:

        try:

            print(
                f"Downloading {symbol}...",
                end=" "
            )

            df = get_bars(
                symbol,
                days=TRAINING_DAYS
            )

            if df is None or len(df) == 0:

                print("NO DATA")
                continue

            if "symbol" in df.index.names:

                df = df.reset_index(
                    level="symbol",
                    drop=True
                )

            df = create_features(df)

            df = create_labels(df)

            df["training_symbol"] = symbol

            frames.append(df)

            print(
                f"{len(df)} rows"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

    if not frames:

        raise RuntimeError(
            "No training data downloaded."
        )

    combined = pd.concat(
        frames,
        axis=0
    )

    # Sort chronologically.
    combined = combined.sort_index()

    return combined


def print_distribution(name, y):

    print()
    print("=" * 70)
    print(f"{name} TARGET DISTRIBUTION")
    print("=" * 70)

    print(
        y.value_counts()
    )

    print()

    print(
        y.value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )


def build_model():

    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def evaluate(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print()
    print("=" * 70)
    print(f"{name} MODEL")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    try:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        print(
            f"ROC-AUC: {auc:.4f}"
        )

    except ValueError:

        print(
            "ROC-AUC unavailable."
        )


def main():

    df = download_training_data()

    print()
    print(
        f"TOTAL TRAINING ROWS: "
        f"{len(df):,}"
    )

    X = df[FEATURES]

    y_long = df[
        "long_target"
    ]

    y_short = df[
        "short_target"
    ]

    print_distribution(
        "LONG",
        y_long
    )

    print_distribution(
        "SHORT",
        y_short
    )

    # =====================================
    # TIME-ORDERED SPLIT
    # =====================================

    split = int(
        len(df) * 0.80
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_long_train = (
        y_long.iloc[:split]
    )

    y_long_test = (
        y_long.iloc[split:]
    )

    y_short_train = (
        y_short.iloc[:split]
    )

    y_short_test = (
        y_short.iloc[split:]
    )

    # =====================================
    # LONG
    # =====================================

    print()
    print(
        "Training LONG model..."
    )

    long_model = build_model()

    long_model.fit(
        X_train,
        y_long_train
    )

    evaluate(
        "LONG",
        long_model,
        X_test,
        y_long_test
    )

    # =====================================
    # SHORT
    # =====================================

    print()
    print(
        "Training SHORT model..."
    )

    short_model = build_model()

    short_model.fit(
        X_train,
        y_short_train
    )

    evaluate(
        "SHORT",
        short_model,
        X_test,
        y_short_test
    )

    # =====================================
    # SAVE
    # =====================================

    os.makedirs(
        "data/models",
        exist_ok=True
    )

    joblib.dump(
        long_model,
        "data/models/xgboost_long.pkl"
    )

    joblib.dump(
        short_model,
        "data/models/xgboost_short.pkl"
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Saved:"
    )

    print(
        "data/models/xgboost_long.pkl"
    )

    print(
        "data/models/xgboost_short.pkl"
    )


if __name__ == "__main__":

    main()