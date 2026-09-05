import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EWM with com=period-1).
    Returns values in the 0–100 range.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    rsi_period: int,
) -> pd.DataFrame:
    """Attach indicator columns to a candle DataFrame and return a copy."""
    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["rsi"]      = rsi(df["close"], rsi_period)
    return df
