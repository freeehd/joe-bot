"""Symbol universes used by legacy scanning and research pipelines."""

LEGACY_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA",
    "AVGO", "NFLX", "PLTR", "MU", "INTC", "QCOM", "ARM", "SPY", "QQQ",
]

# Phase B/C starter universe: 50 liquid US equities. Market benchmarks live in
# MARKET_CONTEXT_SYMBOLS so model rows and context rows are explicit concepts.
RESEARCH_UNIVERSE_50 = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA",
    "AVGO", "NFLX", "PLTR", "MU", "INTC", "QCOM", "ARM", "CSCO", "IBM",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "XOM", "CVX", "COP",
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE",
    "WMT", "COST", "HD", "LOW",
    "CAT", "DE", "GE", "BA",
    "CRM", "ORCL", "NOW", "ADBE",
    "UBER", "DIS", "KO", "NKE", "T",
]

MARKET_CONTEXT_SYMBOLS = ["SPY", "QQQ"]

# Preserve the v0.3 scanner's exact universe until it is retired.
SYMBOLS = LEGACY_SYMBOLS
