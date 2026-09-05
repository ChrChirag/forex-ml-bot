import logging
import pandas as pd
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame) -> str:
    """
    Returns 'BUY', 'SELL', or 'HOLD'.

    Entry rules:
    ─ LONG : EMA_fast > EMA_slow  AND  RSI < threshold  AND  RSI rising
    ─ SHORT: EMA_fast < EMA_slow  AND  RSI > threshold  AND  RSI falling

    Why looser than a strict crossover:
    The original crossover (prev RSI < 40 AND curr RSI >= 40) fires maybe
    once every few weeks on M15 data — far too rare to build a training set.
    This version fires whenever price is in a pullback that is turning around,
    which is the same idea but gives us 100–400 signals over 60 days instead of 1.
    """
    if len(df) < 3:
        return "HOLD"

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    uptrend = curr["ema_fast"] > curr["ema_slow"]
    downtrend = curr["ema_fast"] < curr["ema_slow"]

    # RSI pulled back below threshold and is now turning up → long entry
    rsi_buy = curr["rsi"] < RSI_BUY_THRESHOLD and curr["rsi"] > prev["rsi"]
    # RSI bounced above threshold and is now turning down → short entry
    rsi_sell = curr["rsi"] > RSI_SELL_THRESHOLD and curr["rsi"] < prev["rsi"]

    if uptrend and rsi_buy:
        logger.debug(
            f"BUY  | RSI {prev['rsi']:.1f}→{curr['rsi']:.1f}↑ "
            f"| EMA {curr['ema_fast']:.5f}>{curr['ema_slow']:.5f}"
        )
        return "BUY"

    if downtrend and rsi_sell:
        logger.debug(
            f"SELL | RSI {prev['rsi']:.1f}→{curr['rsi']:.1f}↓ "
            f"| EMA {curr['ema_fast']:.5f}<{curr['ema_slow']:.5f}"
        )
        return "SELL"

    return "HOLD"


def trade_levels(signal, price, tp_pct, sl_pct):
    if signal == "BUY":
        return price * (1 + tp_pct), price * (1 - sl_pct)
    if signal == "SELL":
        return price * (1 - tp_pct), price * (1 + sl_pct)
    return None, None


def position_units(signal, balance, price, risk_pct, sl_pct):
    from config import MAX_POSITION_UNITS

    raw = (balance * risk_pct) / (price * sl_pct)
    units = int(round(min(raw, MAX_POSITION_UNITS)))  # cap at max
    return units if signal == "BUY" else -units
