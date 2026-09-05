# forex-ml-bot
Automated forex trading bot with XGBoost ML signal filtering, live IBKR CFD execution, and weekly model retraining. Built during an AI degree summer project.
# Forex ML Trading Bot

An automated forex trading system that combines a rules-based signal engine 
with an XGBoost confidence filter and live order execution through 
Interactive Brokers. Built as a summer project during my AI degree at the 
University of Groningen.

## What it does

Every 15 minutes the bot:
1. Fetches live EUR/USD and AUD/USD price data from IBKR
2. Computes EMA(9/21) and RSI(14) indicators
3. Generates a BUY, SELL, or HOLD signal
4. Passes the signal through an XGBoost classifier — only trades with 
   predicted win probability above a calibrated threshold are approved
5. Places a bracket order (entry + take-profit + stop-loss) via the IBKR API
6. Enforces a daily P&L cap that halts trading if gains or losses 
   breach defined limits

Live signals are logged to CSV and used to retrain the model weekly, 
so the filter improves as real market data accumulates.

## Architecture
Market Data (IBKR)
↓
Signal Engine (EMA + RSI)
↓
XGBoost ML Filter → SKIP if P(win) < threshold
↓
Risk Guard (daily P&L cap)
↓
Order Engine → IBKR CFD


## Stack

- **Python** — core language
- **ib_insync** — Interactive Brokers API wrapper
- **XGBoost** — signal confidence classifier
- **stable-baselines3** — PPO-based RL exit manager (framework built, 
  activates once sufficient live data accumulates)
- **yfinance** — historical data for backtesting and training
- **pandas / numpy** — data processing

## File structure

| File | Purpose |
|------|---------|
| `main.py` | Scheduler loop — connects to IBKR and runs cycles |
| `broker.py` | IBKR API wrapper (candles, orders, positions) |
| `strategy.py` | Signal generation and position sizing |
| `indicators.py` | EMA and RSI calculations |
| `features.py` | Feature engineering for the ML model |
| `ml_filter.py` | Loads trained model and filters live signals |
| `risk.py` | Daily P&L guard |
| `sessions.py` | Forex session awareness (London, NY, Asian) |
| `data.py` | Candle data parsing |
| `backtest.py` | Historical replay to generate training data |
| `train_model.py` | XGBoost training with walk-forward validation |
| `train_rl.py` | RL exit manager training (PPO) |
| `label_live.py` | Labels live signals with outcomes for retraining |
| `rl_env.py` | Gymnasium environment for RL exit training |
| `rl_manager.py` | Live RL exit manager |
| `config.py` | All parameters in one place |

## Setup

**Requirements:**
- Interactive Brokers account with CFD trading permissions enabled
- TWS (Trader Workstation) running locally with API enabled on port 7497
- Python 3.11+

**Install:**
```bash
git clone https://github.com/yourusername/forex-ml-bot
cd forex-ml-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your IBKR credentials in .env
```

**Run the full pipeline:**
```bash
# 1. Build training dataset from historical data
python backtest.py

# 2. Train the ML signal filter
python train_model.py

# 3. Train the RL exit manager (optional)
python train_rl.py

# 4. Start the live bot (TWS must be open)
python main.py
```

**Retrain weekly with live data:**
```bash
python label_live.py   # label live signal outcomes
python train_model.py  # retrain with expanded dataset
```

## Current model performance

| Pair | AUC | Threshold | Notes |
|------|-----|-----------|-------|
| EUR/USD | 0.538 | 0.70 | Improving — ~6 weeks live data |
| AUD/USD | 0.601 | 0.51 | Best model — more active in Asian session |

AUC of 0.5 = random. Both models are above random and improving as 
live trade data accumulates. The infrastructure is the foundation — 
performance improves with time and data.

## Important notes

- **EU regulatory note**: MiFID II prohibits leveraged spot forex for 
  retail clients in the EU. The bot uses CFD contracts to comply.
- **Paper trading first**: Run on IBKR paper account before any real capital.
- **Not financial advice**: This is an engineering and ML project. 
  Past paper performance does not guarantee real returns.

## Roadmap

- [ ] Retrain on 6 months of live 15-minute data
- [ ] Enable RL exit manager once sufficient live trades accumulate  
- [ ] Add GBP/USD and USD/JPY once models improve
- [ ] Deploy on VPS for 24/7 operation without laptop dependency

## License
MIT
