"""Static broad-sector metadata for the current research universe.

This is research metadata, not a live security-master. Unknown symbols are
classified as ``UNKNOWN`` so callers can choose conservative exposure handling.
"""

from __future__ import annotations

SECTOR_BY_SYMBOL = {
    # Information Technology
    "AAPL": "INFORMATION_TECHNOLOGY", "MSFT": "INFORMATION_TECHNOLOGY",
    "NVDA": "INFORMATION_TECHNOLOGY", "AMD": "INFORMATION_TECHNOLOGY",
    "AVGO": "INFORMATION_TECHNOLOGY", "PLTR": "INFORMATION_TECHNOLOGY",
    "MU": "INFORMATION_TECHNOLOGY", "INTC": "INFORMATION_TECHNOLOGY",
    "QCOM": "INFORMATION_TECHNOLOGY", "ARM": "INFORMATION_TECHNOLOGY",
    "CSCO": "INFORMATION_TECHNOLOGY", "IBM": "INFORMATION_TECHNOLOGY",
    "CRM": "INFORMATION_TECHNOLOGY", "ORCL": "INFORMATION_TECHNOLOGY",
    "NOW": "INFORMATION_TECHNOLOGY", "ADBE": "INFORMATION_TECHNOLOGY",
    # Communication Services
    "META": "COMMUNICATION_SERVICES", "GOOGL": "COMMUNICATION_SERVICES",
    "NFLX": "COMMUNICATION_SERVICES", "DIS": "COMMUNICATION_SERVICES",
    "T": "COMMUNICATION_SERVICES",
    # Consumer Discretionary
    "AMZN": "CONSUMER_DISCRETIONARY", "TSLA": "CONSUMER_DISCRETIONARY",
    "HD": "CONSUMER_DISCRETIONARY", "LOW": "CONSUMER_DISCRETIONARY",
    "NKE": "CONSUMER_DISCRETIONARY",
    # Consumer Staples
    "WMT": "CONSUMER_STAPLES", "COST": "CONSUMER_STAPLES", "KO": "CONSUMER_STAPLES",
    # Financials
    "JPM": "FINANCIALS", "BAC": "FINANCIALS", "WFC": "FINANCIALS",
    "GS": "FINANCIALS", "MS": "FINANCIALS", "V": "FINANCIALS", "MA": "FINANCIALS",
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    # Health Care
    "LLY": "HEALTH_CARE", "UNH": "HEALTH_CARE", "JNJ": "HEALTH_CARE",
    "MRK": "HEALTH_CARE", "ABBV": "HEALTH_CARE", "PFE": "HEALTH_CARE",
    # Industrials
    "CAT": "INDUSTRIALS", "DE": "INDUSTRIALS", "GE": "INDUSTRIALS",
    "BA": "INDUSTRIALS", "UBER": "INDUSTRIALS",
    # Context ETFs
    "SPY": "ETF", "QQQ": "ETF",
}


def sector_for(symbol: str) -> str:
    return SECTOR_BY_SYMBOL.get(symbol.upper(), "UNKNOWN")
