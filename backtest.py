"""
Backtest using yfinance 1-hour data (2 years) — no API key needed.

Using hourly data gives ~11,000 candles vs ~5,000 for 15m,
producing 3-4x more labeled signals for ML training.

The model trains on 1h features (candles_per_hour=1) and runs live
on 15m features (candles_per_hour=4). RSI/EMA/ATR are scale-independent
so they transfer well; we retrain on live 15m data after a few months.

Usage:
    python backtest.py
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import INSTRUMENTS, EMA_FAST, EMA_SLOW, RSI_PERIOD
from indicators import add_indicators
from strategy import generate_signal
from features import compute_features, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Backtest-specific settings (hourly data → larger pip targets)
BACKTEST_TP_PCT = 0.005  # 0.5%  ≈ 56 pips at 1.12 — reachable in 24h
BACKTEST_SL_PCT = 0.0025  # 0.25% ≈ 28 pips         — realistic hourly noise
BACKTEST_CPH = 1  # candles_per_hour for 1h data
MAX_LOOK_FORWARD = 48  # 48 hourly candles = 2 trading days
MIN_SIGNAL_GAP = 4  # minimum 4 candles between signals

YF_TICKERS = {
    "EUR_USD": ["EURUSD=X", "EUR=X"],
    "GBP_USD": ["GBPUSD=X", "GBP=X"],
    "USD_JPY": ["USDJPY=X", "JPY=X"],
}


def fetch_history(instrument: str) -> pd.DataFrame:
    """Download 2 years of 1-hour candles from yfinance."""
    tickers = YF_TICKERS.get(instrument, [instrument.replace("_", "") + "=X"])
    df = pd.DataFrame()

    for ticker in tickers:
        if not df.empty:
            break
        for kwargs in [
            dict(period="2y", interval="1h", auto_adjust=True, progress=False),
            dict(period="730d", interval="1h", auto_adjust=True, progress=False),
        ]:
            try:
                logger.info(f"Downloading {ticker} 1h (2y) from yfinance...")
                tmp = yf.download(ticker, **kwargs)
                if not tmp.empty and len(tmp) > 500:
                    df = tmp
                    logger.info(f"Got {len(df)} rows from {ticker}")
                    break
            except Exception as e:
                logger.warning(f"{ticker} failed: {e}")

    if df.empty:
        raise ValueError("No data returned. Check your internet connection.")

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    for col in ["datetime", "date"]:
        if col in df.columns:
            df = df.rename(columns={col: "time"})
            break

    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close"]].dropna()
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(
        f"Cleaned: {len(df)} candles  ({df['time'].iloc[0]} → {df['time'].iloc[-1]})"
    )
    return df


def label_outcome(df, signal_idx, signal, tp_pct, sl_pct, max_forward):
    entry = df["close"].iloc[signal_idx]
    tp = entry * (1 + tp_pct) if signal == "BUY" else entry * (1 - tp_pct)
    sl = entry * (1 - sl_pct) if signal == "BUY" else entry * (1 + sl_pct)

    end_idx = min(signal_idx + 1 + max_forward, len(df))
    for j in range(signal_idx + 1, end_idx):
        high = df["high"].iloc[j]
        low = df["low"].iloc[j]
        if signal == "BUY":
            if high >= tp:
                return 1
            if low <= sl:
                return 0
        else:
            if low <= tp:
                return 1
            if high >= sl:
                return 0
    return None


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df, EMA_FAST, EMA_SLOW, RSI_PERIOD)
    rows = []
    warmup = max(EMA_SLOW, RSI_PERIOD, 100) + 5
    last_signal_idx = -MIN_SIGNAL_GAP

    for i in range(warmup, len(df) - MAX_LOOK_FORWARD - 1):
        if i - last_signal_idx < MIN_SIGNAL_GAP:
            continue

        window = df.iloc[: i + 1]
        signal = generate_signal(window)
        if signal == "HOLD":
            continue

        features = compute_features(window, signal, candles_per_hour=BACKTEST_CPH)
        if features is None:
            continue

        label = label_outcome(
            df, i, signal, BACKTEST_TP_PCT, BACKTEST_SL_PCT, MAX_LOOK_FORWARD
        )
        if label is None:
            continue

        features["timestamp"] = df["time"].iloc[i]
        features["signal"] = signal
        features["y"] = label
        rows.append(features)
        last_signal_idx = i

        if len(rows) % 20 == 0:
            logger.info(f"  {len(rows)} labeled signals so far...")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = Path("training_data")
    out_dir.mkdir(exist_ok=True)

    logger.info("Using 1h yfinance data (2 years) for training.")
    logger.info(
        f"Backtest TP={BACKTEST_TP_PCT * 100:.2f}%  SL={BACKTEST_SL_PCT * 100:.3f}%  "
        f"MaxLookForward={MAX_LOOK_FORWARD}h"
    )

    for instrument in INSTRUMENTS:
        df = fetch_history(instrument)
        dataset = build_dataset(df)

        if dataset.empty:
            logger.warning(f"{instrument}: 0 signals found.")
            continue

        out_path = out_dir / f"{instrument}_signals.csv"
        dataset.to_csv(out_path, index=False)

        wins = int(dataset["y"].sum())
        total = len(dataset)

        logger.info(
            f"\n{'─' * 40}"
            f"\n{instrument} results:"
            f"\n  Candles       : {len(df)}"
            f"\n  Total signals : {total}"
            f"\n  Wins (TP hit) : {wins}"
            f"\n  Losses (SL)   : {total - wins}"
            f"\n  Win rate      : {wins / total:.1%}"
            f"\n  Saved to      : {out_path}"
            f"\n{'─' * 40}"
        )
