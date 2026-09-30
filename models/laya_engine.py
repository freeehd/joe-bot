import laya


class LayaEngine:

    def __init__(self):

        print("Loading Laya...")

        self.agent = laya.load(
            "convaiinnovations/laya"
        )

        print("Laya loaded.")

    def evaluate_trade(
        self,
        symbol,
        features,
        alpha_probability
    ):

        state = f"""
Symbol: {symbol}

Short-term market state:

1 minute return:
{features['return_1m']:.6f}

3 minute return:
{features['return_3m']:.6f}

5 minute return:
{features['return_5m']:.6f}

EMA distance:
{features['ema_distance']:.6f}

VWAP distance:
{features['vwap_distance']:.6f}

Relative volume:
{features['relative_volume']:.3f}

Current candle range:
{features['range']:.6f}

Quantitative model probability
of a favorable short-term move:

{alpha_probability:.3f}
"""

        questions = {

            "action": {
                "type": "choice",

                "instructions":
                "Determine the best short-term trading action based only on the supplied market state.",

                "criteria": {
                    "long":
                    "Strong evidence supports entering a short-term long trade.",

                    "wait":
                    "Evidence is insufficient, conflicting, or does not justify a trade.",

                    "short":
                    "Strong evidence supports entering a short-term short trade."
                }
            },

            "high_risk": {
                "type": "noul",

                "instructions":
                "Is this setup unusually risky or uncertain?"
            }
        }

        result = self.agent.predict(
            state,
            questions
        )

        return result