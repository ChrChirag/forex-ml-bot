"""
Train the RL exit manager for each instrument using PPO.

Uses 2 years of hourly historical data. The agent learns when to close
a position early vs letting it run to TP/SL.

Usage:
    python train_rl.py

Run this AFTER backtest.py and train_model.py.
Saved models go to models/{instrument}_rl_exit.zip
"""
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from config     import INSTRUMENTS, EMA_FAST, EMA_SLOW, RSI_PERIOD
from indicators import add_indicators
from strategy   import generate_signal
from rl_env     import ForexExitEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

YF_TICKERS = {
    'EUR_USD': 'EURUSD=X',
    'GBP_USD': 'GBPUSD=X',
    'USD_JPY': 'USDJPY=X',
    'AUD_USD': 'AUDUSD=X',
    'USD_CHF': 'USDCHF=X',
}

TP_PCT = 0.005
SL_PCT = 0.0025
TIMESTEPS = 80_000   # increase for better results, decrease to train faster


def fetch_data(instrument: str) -> pd.DataFrame:
    ticker = YF_TICKERS.get(instrument, instrument.replace('_', '') + '=X')
    logger.info(f"Downloading {ticker} 1h data...")
    raw = yf.download(ticker, period='2y', interval='1h',
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    for col in ['datetime', 'date']:
        if col in raw.columns:
            raw = raw.rename(columns={col: 'time'})
            break

    raw['time'] = pd.to_datetime(raw['time']).dt.tz_localize(None)
    raw = raw[['time', 'open', 'high', 'low', 'close']].dropna()
    raw = raw.sort_values('time').reset_index(drop=True)
    return add_indicators(raw, EMA_FAST, EMA_SLOW, RSI_PERIOD)


def find_signals(df: pd.DataFrame) -> list:
    """Return list of (candle_index, 'BUY'/'SELL') for all signal-firing bars."""
    signals = []
    warmup  = max(EMA_SLOW, RSI_PERIOD, 100) + 5
    for i in range(warmup, len(df) - 50):
        s = generate_signal(df.iloc[:i + 1])
        if s != 'HOLD':
            signals.append((i, s))
    logger.info(f"Found {len(signals)} signal entry points.")
    return signals


def train(instrument: str):
    logger.info(f"\n{'━'*40}\nTraining RL exit manager for {instrument}\n{'━'*40}")

    df      = fetch_data(instrument)
    signals = find_signals(df)

    if len(signals) < 20:
        logger.warning(f"Only {len(signals)} signals — need 20+ to train RL. Skipping.")
        return

    # Chronological 80/20 split
    split        = int(len(signals) * 0.8)
    train_sigs   = signals[:split]
    eval_sigs    = signals[split:]

    train_env = ForexExitEnv(df, train_sigs, TP_PCT, SL_PCT)
    check_env(train_env, warn=True)

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate    = 3e-4,
        n_steps          = 1024,
        batch_size       = 64,
        n_epochs         = 10,
        gamma            = 0.99,
        clip_range       = 0.2,
        verbose          = 0,
    )

    model.learn(total_timesteps=TIMESTEPS,
                progress_bar=False)

    out = Path("models") / f"{instrument}_rl_exit"
    model.save(str(out))
    logger.info(f"Model saved → {out}.zip")

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_env = ForexExitEnv(df, eval_sigs, TP_PCT, SL_PCT)
    obs, _   = eval_env.reset()
    total, n = 0.0, 0
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action)
        total += reward
        if done:
            obs, _ = eval_env.reset()
            n += 1

    avg = total / max(n, 1)
    logger.info(f"{instrument}: RL eval — {n} episodes, avg P&L {avg*100:+.3f}% per trade")


if __name__ == '__main__':
    Path("models").mkdir(exist_ok=True)
    for instrument in INSTRUMENTS:
        try:
            train(instrument)
        except Exception as e:
            logger.error(f"{instrument}: RL training failed — {e}")