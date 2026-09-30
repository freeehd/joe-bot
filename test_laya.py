import laya

agent = laya.load("convaiinnovations/laya")

state = """
Price momentum is positive.
Volume is increasing.
Price is above VWAP.
Spread is tight.
Market is trending upward.
"""

questions = {
    "action": {
        "type": "choice",
        "instructions": "What action should the trading system consider?",
        "criteria": {
            "long": "Conditions strongly favor an upward move",
            "wait": "There is insufficient edge",
            "short": "Conditions strongly favor a downward move"
        }
    }
}

result = agent.predict(state, questions)

print(result)