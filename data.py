import logging
import pandas as pd

logger = logging.getLogger(__name__)


def candles_to_df(candles: list) -> pd.DataFrame:
    """Parse OANDA candle JSON into a tidy DataFrame (complete candles only)."""
    rows = []
    for c in candles:
        if c.get("complete", False):
            rows.append({
                "time":  pd.Timestamp(c["time"]),
                "open":  float(c["mid"]["o"]),
                "high":  float(c["mid"]["h"]),
                "low":   float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No complete candles in response.")
        return df

    # Make timezone-aware for market-age checks in main
    df["time"] = df["time"].dt.tz_convert("UTC") if df["time"].dt.tz else df["time"].dt.tz_localize("UTC")
    return df.reset_index(drop=True)


def get_market_data(broker, instrument: str, granularity: str, count: int) -> pd.DataFrame:
    raw = broker.get_candles(instrument, granularity, count)
    return candles_to_df(raw)
