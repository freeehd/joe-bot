"""Symbol universes used by legacy scanning and research pipelines."""

LEGACY_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA",
    "AVGO", "NFLX", "PLTR", "MU", "INTC", "QCOM", "ARM", "SPY", "QQQ",
]

# Phase B starter universe: 50 heavily traded large-cap US names plus SPY/QQQ
# market benchmarks. This is deliberately static and versionable; a later phase
# will replace it with a liquidity/price/spread screened dynamic universe.
RESEARCH_UNIVERSE_50 = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA",
    "AVGO", "NFLX", "PLTR", "MU", "INTC", "QCOM", "ARM", "SPY", "QQQ",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "XOM", "CVX", "COP",
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE",
    "WMT", "COST", "HD", "LOW",
    "CAT", "DE", "GE", "BA",
    "CRM", "ORCL", "NOW", "ADBE",
    "UBER", "DIS", "KO", "NKE", "T",
]

# Preserve the v0.3 scanner's exact universe until it is retired.
SYMBOLS = LEGACY_SYMBOLS
