# ==========================================
# TRADING CAPITAL
# ==========================================

# Amount of PAPER capital this bot is allowed
# to manage.
TRADING_CAPITAL = 500.00


# Maximum percentage of trading capital
# that may be deployed at one time.
#
# 0.70 = maximum 70% invested.
MAX_CAPITAL_DEPLOYED = 0.70


# Maximum number of simultaneous candidate
# positions.
MAX_POSITIONS = 3


# Maximum percentage of total trading capital
# allowed in one stock.
#
# $500 * 0.30 = max $150 per stock.
MAX_POSITION_PERCENT = 0.30


# Don't bother allocating extremely tiny
# dollar positions.
MIN_POSITION_DOLLARS = 20.00


# ==========================================
# CANDIDATE FILTERS
# ==========================================

# IMPORTANT:
# This is intentionally low for now because
# your current XGBoost model has been producing
# low probabilities.
#
# We will recalibrate this later using
# backtesting.
MIN_ALPHA_PROBABILITY = 0.05


# Minimum relative volume.
MIN_RELATIVE_VOLUME = 0.50


# Minimum absolute 5-minute movement.
#
# 0.0005 = 0.05%
MIN_ABS_5M_MOVE = 0.0005


# Number of days of minute data retrieved
# when evaluating each stock.
LOOKBACK_DAYS = 5

# Minimum difference between LONG and SHORT
# model probabilities.

MIN_DIRECTIONAL_EDGE = 0.05