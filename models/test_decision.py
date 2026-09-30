from market.alpaca_client import get_bars
from features.feature_engine import create_features

from models.alpha_model import AlphaModel
from models.laya_engine import LayaEngine

from strategy.decision import TradeDecision


SYMBOL = "NVDA"


df = get_bars(
    SYMBOL,
    days=5
)

if "symbol" in df.index.names:

    df = df.reset_index(
        level="symbol",
        drop=True
    )


df = create_features(df)

latest = df.iloc[-1]


alpha_model = AlphaModel()

laya = LayaEngine()

decision_engine = TradeDecision()


alpha_probability = (
    alpha_model.predict_probability(
        latest
    )
)


laya_result = (
    laya.evaluate_trade(
        SYMBOL,
        latest,
        alpha_probability
    )
)


decision = (
    decision_engine.evaluate(
        laya_result,
        alpha_probability
    )
)


print()
print("============================")
print("TRADING DECISION")
print("============================")

print(
    "XGBoost:",
    f"{alpha_probability:.2%}"
)

print(
    "Laya action:",
    decision["action"]
)

print(
    "Laya confidence:",
    f"{decision['laya_confidence']:.2%}"
)

print(
    "Risk probability:",
    f"{decision['risk_probability']:.2%}"
)

print(
    "Trade approved:",
    decision["approved"]
)

print(
    "Reason:",
    decision["reason"]
)