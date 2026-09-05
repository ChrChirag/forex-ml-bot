"""
Label live signals from signal_log.csv with ML features and trade outcomes.

Run this periodically (weekly is fine) to convert your live trading log
into retraining data. After running, re-run train_model.py — it will
automatically use the combined dataset (backtest + live), giving the model
real 15-minute market experience instead of just the hourly backtest proxy.

Workflow:
    1. python label_live.py        ← run this
    2. python train_model.py       ← retrain with expanded dataset
    3. python main.py              ← bot now uses better model

Reads:
    signal_log.csv                                 (written by the live bot)
    training_data/{instrument}_signals.csv         (existing backtest data)

Writes:
    training_data/{instrument}_live_signals.csv    (labeled live signals)
    training_data/{instrument}_combined.csv        (backtest + live, use for retrain)
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import INSTRUMENTS, EMA_FAST, EMA_SLOW, RSI_PERIOD
from indicators import add_indicators
from features import compute_features, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SIGNAL_LOG = Path("signal_log.csv")
TRAINING_DIR = Path("training_data")
MAX_LOOK_FORWARD = 96  # 96 × 15m = 24h to resolve a trade

YF_TICKERS = {
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "AUD_USD": "AUDUSD=X",
}


# ── Data fetching ─────────────────────────────────────────────────────────────


def fetch_m15(instrument: str) -> pd.DataFrame:
    """Download M15 data (last 57 days) and add indicators."""
    ticker = YF_TICKERS.get(instrument, instrument.replace("_", "") + "=X")
    logger.info(f"Downloading {ticker} M15 data...")

    for kwargs in [
        dict(period="57d", interval="15m", auto_adjust=True, progress=False),
        dict(
            period="55d",
            interval="15m",
            auto_adjust=True,
            progress=False,
            threads=False,
        ),
    ]:
        try:
            raw = yf.download(ticker, **kwargs)
            if not raw.empty and len(raw) > 100:
                break
        except Exception as e:
            logger.warning(f"Download attempt failed: {e}")

    if raw.empty:
        raise ValueError(f"Could not download data for {instrument}")

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    for col in ["datetime", "date"]:
        if col in raw.columns:
            raw = raw.rename(columns={col: "time"})
            break

    raw["time"] = pd.to_datetime(raw["time"]).dt.tz_localize(None)
    raw = raw[["time", "open", "high", "low", "close"]].dropna()
    raw = raw.sort_values("time").reset_index(drop=True)
    raw = add_indicators(raw, EMA_FAST, EMA_SLOW, RSI_PERIOD)

    logger.info(
        f"Got {len(raw)} M15 candles ({raw['time'].iloc[0]} → {raw['time'].iloc[-1]})"
    )
    return raw


# ── Signal matching ───────────────────────────────────────────────────────────


def find_candle_idx(df: pd.DataFrame, timestamp_str: str):
    """
    Find the DataFrame row index matching a signal timestamp.
    Allows up to 30-minute tolerance (handles weekend gaps, DST shifts etc.)
    Returns None if no close match found.
    """
    try:
        ts = pd.Timestamp(timestamp_str)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
    except Exception:
        return None

    diffs = (df["time"] - ts).abs()
    idx = diffs.idxmin()
    if diffs[idx] > pd.Timedelta(minutes=30):
        return None
    return int(idx)


# ── Outcome labeling ──────────────────────────────────────────────────────────


def label_outcome(
    df: pd.DataFrame,
    signal_idx: int,
    signal: str,
    tp_price: float,
    sl_price: float,
) -> int | None:
    """
    Walk forward from signal_idx checking if TP or SL was hit first.
    Returns 1 (TP hit = win), 0 (SL hit = loss), or None (unresolved).
    """
    end = min(signal_idx + 1 + MAX_LOOK_FORWARD, len(df))
    for j in range(signal_idx + 1, end):
        high = df["high"].iloc[j]
        low = df["low"].iloc[j]
        if signal == "BUY":
            if high >= tp_price:
                return 1
            if low <= sl_price:
                return 0
        else:  # SELL
            if low <= tp_price:
                return 1
            if high >= sl_price:
                return 0
    return None  # neither level hit within look-forward window


# ── Feature computation ───────────────────────────────────────────────────────


def extract_features(df: pd.DataFrame, signal_idx: int, signal: str) -> dict | None:
    """
    Compute the same feature vector the ML model sees, using the live M15 window.
    candles_per_hour=4 because this is 15-minute data.
    """
    if signal_idx < 30:
        return None
    window = df.iloc[: signal_idx + 1]
    return compute_features(window, signal, candles_per_hour=4)


# ── Per-instrument labeling ───────────────────────────────────────────────────


def label_instrument(instrument: str, live_signals: pd.DataFrame) -> pd.DataFrame:
    """Label all live signals for one instrument. Returns labeled DataFrame."""
    instrument_signals = live_signals[live_signals["instrument"] == instrument].copy()
    if instrument_signals.empty:
        logger.info(f"{instrument}: No live signals in log.")
        return pd.DataFrame()

    logger.info(f"{instrument}: {len(instrument_signals)} live signals to label.")

    try:
        df = fetch_m15(instrument)
    except Exception as e:
        logger.error(f"{instrument}: Could not fetch price data — {e}")
        return pd.DataFrame()

    rows = []
    stats = {"labeled": 0, "unresolved": 0, "no_candle": 0, "no_features": 0}

    for _, sig in instrument_signals.iterrows():
        # Locate signal candle in price data
        idx = find_candle_idx(df, sig["timestamp"])
        if idx is None:
            stats["no_candle"] += 1
            continue

        # Compute features
        features = extract_features(df, idx, sig["signal"])
        if features is None:
            stats["no_features"] += 1
            continue

        # Determine outcome
        try:
            tp = float(sig["tp_price"])
            sl = float(sig["sl_price"])
        except (ValueError, TypeError):
            continue

        y = label_outcome(df, idx, sig["signal"], tp, sl)
        if y is None:
            stats["unresolved"] += 1
            continue

        features["timestamp"] = sig["timestamp"]
        features["signal"] = sig["signal"]
        features["y"] = y
        rows.append(features)
        stats["labeled"] += 1

    logger.info(
        f"{instrument}: labeled={stats['labeled']} | "
        f"unresolved={stats['unresolved']} | "
        f"no_candle={stats['no_candle']} | "
        f"no_features={stats['no_features']}"
    )
    return pd.DataFrame(rows)


# ── Combine with backtest data ────────────────────────────────────────────────


def combine_datasets(instrument: str, live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge newly labeled live signals with the existing backtest signals.
    The combined dataset is what train_model.py should train on.
    """
    backtest_path = TRAINING_DIR / f"{instrument}_signals.csv"
    if backtest_path.exists():
        backtest_df = pd.read_csv(backtest_path)
        combined = pd.concat([backtest_df, live_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "signal"])
        logger.info(
            f"{instrument}: backtest={len(backtest_df)} + "
            f"live={len(live_df)} = {len(combined)} combined signals"
        )
        return combined
    else:
        logger.warning(
            f"{instrument}: No backtest file found — using live signals only."
        )
        return live_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not SIGNAL_LOG.exists():
        logger.error(
            f"{SIGNAL_LOG} not found. "
            "Make sure the bot has been running and logging signals."
        )
        raise SystemExit(1)

    # Load and filter signal log.
    # IMPORTANT: filter on order_status == 'Filled', not just 'approved'.
    # An ML-approved signal can still be rejected by the broker (e.g. size
    # limits) — training on a rejected order would teach the model from a
    # trade that never actually happened. Older logs (before order_status
    # was added) fall back to the approved-only filter.
    log = pd.read_csv(SIGNAL_LOG)

    if "order_status" in log.columns:
        skipped_unfilled = log[
            (log["approved"] == True)
            & (log["order_status"] != "Filled")
            & (log["signal"].isin(["BUY", "SELL"]))
        ]
        if len(skipped_unfilled) > 0:
            logger.info(
                f"Excluding {len(skipped_unfilled)} approved-but-not-filled "
                f"signals (rejected/cancelled at the broker) from training data."
            )
        approved = log[
            (log["approved"] == True)
            & (log["order_status"] == "Filled")
            & (log["signal"].isin(["BUY", "SELL"]))
        ].copy()
    else:
        logger.warning(
            "signal_log.csv has no 'order_status' column (old format) — "
            "cannot distinguish filled vs rejected orders. Using approved-only filter."
        )
        approved = log[
            (log["approved"] == True) & (log["signal"].isin(["BUY", "SELL"]))
        ].copy()

    if approved.empty:
        logger.warning(
            "No filled BUY/SELL trades in signal_log.csv yet.\n"
            "Run the bot for at least a few days before labeling."
        )
        raise SystemExit(0)

    logger.info(f"Found {len(approved)} filled trades across all instruments.")
    TRAINING_DIR.mkdir(exist_ok=True)

    summary = []
    for instrument in INSTRUMENTS:
        logger.info(f"\n{'─' * 40}\nProcessing {instrument}\n{'─' * 40}")
        live_df = label_instrument(instrument, approved)

        if live_df.empty:
            summary.append((instrument, 0, 0, "n/a"))
            continue

        # Save live signals
        live_path = TRAINING_DIR / f"{instrument}_live_signals.csv"
        live_df.to_csv(live_path, index=False)
        logger.info(f"Saved live signals → {live_path}")

        # Combine with backtest and save
        combined = combine_datasets(instrument, live_df)
        combined_path = TRAINING_DIR / f"{instrument}_signals.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"Updated training file → {combined_path}")

        wins = int(live_df["y"].sum())
        total = len(live_df)
        summary.append((instrument, total, wins, f"{wins / total:.1%}"))

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print("LIVE SIGNAL LABELING COMPLETE")
    print(f"{'═' * 50}")
    print(f"{'Instrument':<12} {'Labeled':>8} {'Wins':>6} {'Win rate':>10}")
    print(f"{'─' * 40}")
    for inst, total, wins, wr in summary:
        print(f"{inst:<12} {total:>8} {wins:>6} {wr:>10}")
    print(f"\nNext step: run  python train_model.py  to retrain with the expanded dataset.")
