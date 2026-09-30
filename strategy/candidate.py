from dataclasses import dataclass


@dataclass
class Candidate:

    symbol: str

    alpha_probability: float

    relative_volume: float

    momentum: float

    volatility: float

    spread_percent: float

    score: float = 0.0