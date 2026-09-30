from market.alpaca_client import get_bars
from features.feature_engine import create_features
from models.alpha_model import AlphaModel


df = get_bars("NVDA", days=5)

if "symbol" in df.index.names:
    df = df.reset_index(level="symbol", drop=True)

df = create_features(df)

latest = df.iloc[-1]

model = AlphaModel()

probability = model.predict_probability(latest)

print()
print("NVDA probability:")
print(f"{probability:.2%}")