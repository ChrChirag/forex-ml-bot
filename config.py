import os
from dotenv import load_dotenv

load_dotenv()

#IBKR connection 
IBKR_HOST      = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT      = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

#Instruments 
# Add/remove pairs here. Each pair needs its own backtest + trained model.
# After changing this list, re-run: backtest.py → train_model.py → train_rl.py
INSTRUMENTS = [
    "EUR_USD",   #Euro / US Dollar — London + NY sessions
    "AUD_USD",   #Australian Dollar / USD — Asian + NY sessions
]
GRANULARITY   = "M15"
CANDLES_COUNT = 100

# Strategy parameters
EMA_SLOW  = 21
RSI_PERIOD = 14

RSI_BUY_THRESHOLD  = 50
RSI_SELL_THRESHOLD = 50

# Per-trade risk management
TAKE_PROFIT_PCT    = 0.003
STOP_LOSS_PCT      = 0.0015
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_UNITS = 10000   # hard cap to prevent oversized orders on large paper accounts

#Daily caps 
DAILY_PROFIT_TARGET_PCT = 0.02
DAILY_LOSS_LIMIT_PCT    = 0.015

# Feature engineering 
# Backtest uses 1h data; live bot uses 15m data
BACKTEST_TP_PCT         = 0.005
BACKTEST_SL_PCT         = 0.0025
BACKTEST_CANDLES_PER_HOUR = 1     # 1h data → 1 candle per hour
LIVE_CANDLES_PER_HOUR     = 4     # 15m data → 4 candles per hour

#Feature flags 
ML_FILTER_ENABLED      = True    # XGBoost entry filter
SESSION_FILTER_ENABLED = True    # Only trade during active market sessions
RL_EXIT_ENABLED        = False   # RL exit manager — set True after train_rl.py

#Scheduler 
CANDLE_CHECK_INTERVAL_MINUTES = 15