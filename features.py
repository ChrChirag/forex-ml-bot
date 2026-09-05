"""
Feature engineering for the XGBoost confidence filter.

candles_per_hour controls how far back the return lookbacks go:
  - 15m data (live trading): candles_per_hour=4
  - 1h  data (backtest):     candles_per_hour=1
"""

import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_width(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / sma


def compute_features(
    df: pd.DataFrame,
    signal: str,
    candles_per_hour: int = 4,  # 4 for 15m live data, 1 for 1h backtest data
) -> dict:
    """
    Build the feature vector for the latest candle.
    df must already have ema_fast, ema_slow, rsi columns attached.
    """
    if len(df) < 30:
        return None

    df = df.copy()
    df["atr"] = atr(df, period=14)
    df["bb_w"] = bollinger_width(df["close"], period=20)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = curr["close"]

    # Lookbacks in candles, corrected for timeframe
    lb_1h = candles_per_hour  # 1 hour back
    lb_4h = candles_per_hour * 4  # 4 hours back
    lb_24h = candles_per_hour * 24  # 24 hours back

    def safe_return(n):
        if len(df) > n:
            return price / df["close"].iloc[-(n + 1)] - 1
        return 0.0

    return {
        # Trend
        "ema_ratio": curr["ema_fast"] / curr["ema_slow"],
        "ema_gap_pct": (curr["ema_fast"] - curr["ema_slow"]) / price,
        # Momentum / RSI
        "rsi": curr["rsi"],
        "rsi_slope_3": curr["rsi"] - df["rsi"].iloc[-4],
        "rsi_distance": abs(curr["rsi"] - prev["rsi"]),
        # Volatility
        "atr_pct": curr["atr"] / price,
        "bb_width": curr["bb_w"],
        # Price action (timeframe-corrected)
        "return_1h": safe_return(lb_1h),
        "return_4h": safe_return(lb_4h),
        "return_24h": safe_return(lb_24h),
        # Time
        "hour": curr["time"].hour,
        "day_of_week": curr["time"].dayofweek,
        # Signal direction
        "direction": 1 if signal == "BUY" else -1,
    }


FEATURE_COLUMNS = [
    "ema_ratio",
    "ema_gap_pct",
    "rsi",
    "rsi_slope_3",
    "rsi_distance",
    "atr_pct",
    "bb_width",
    "return_1h",
    "return_4h",
    "return_24h",
    "hour",
    "day_of_week",
    "direction",
]
