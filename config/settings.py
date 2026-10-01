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
# ==========================================
# V0.4 RESEARCH LABELS (NO TRADING)
# ==========================================

# Fixed barriers are the first reproducible Phase A baseline.
# Phase A can switch to ATR-aware barriers with LABEL_USE_ATR=True.
LABEL_HORIZON_BARS = 10
LABEL_TARGET_PCT = 0.003      # +0.30%
LABEL_STOP_PCT = 0.0015       # -0.15%
LABEL_USE_ATR = False
LABEL_ATR_PERIOD = 14
LABEL_ATR_TARGET_MULTIPLIER = 1.0
LABEL_ATR_STOP_MULTIPLIER = 0.5

# V0.4 research dataset / validation.
V04_TRAINING_DAYS = 60
V04_TRAIN_FRACTION = 0.70
V04_CALIBRATION_FRACTION = 0.15

# ==========================================
# PHASE B HISTORICAL DATASET
# ==========================================

# Batch requests keep provider calls manageable while still avoiding one HTTP
# request per symbol.
RESEARCH_BATCH_SIZE = 10

# Alpaca IEX is widely accessible; use "sip" when the account/data entitlement
# supports it. The selected feed is always recorded in the dataset manifest.
RESEARCH_DATA_FEED = "iex"

# Adjust historical bars for splits/dividends/spin-offs. Raw data remains
# reproducible because the exact adjustment mode is versioned in the manifest.
RESEARCH_DATA_ADJUSTMENT = "all"

# Initial short-horizon research is regular-session only. Extended-hours and
# overnight behavior should later be modeled as separate regimes.
RESEARCH_REGULAR_HOURS_ONLY = True
