import logging
import time
import csv
from datetime import date
from pathlib import Path

import pandas as pd

from config import (
    IBKR_HOST,
    IBKR_PORT,
    IBKR_CLIENT_ID,
    INSTRUMENTS,
    GRANULARITY,
    CANDLES_COUNT,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    RISK_PER_TRADE_PCT,
    DAILY_PROFIT_TARGET_PCT,
    DAILY_LOSS_LIMIT_PCT,
    CANDLE_CHECK_INTERVAL_MINUTES,
    ML_FILTER_ENABLED,
    SESSION_FILTER_ENABLED,
    RL_EXIT_ENABLED,
    LIVE_CANDLES_PER_HOUR,
)
from broker import IBKRBroker
from data import get_market_data
from indicators import add_indicators
from strategy import generate_signal, trade_levels, position_units
from risk import DailyRiskGuard
from ml_filter import MLFilter
from features import compute_features
from sessions import is_active as session_active, session_info
from rl_manager import RLExitManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Shared objects ────────────────────────────────────────────────────────────
broker = IBKRBroker(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
guard = DailyRiskGuard(DAILY_PROFIT_TARGET_PCT, DAILY_LOSS_LIMIT_PCT)

ml_filters = {i: MLFilter(i) if ML_FILTER_ENABLED else None for i in INSTRUMENTS}
rl_managers = {i: RLExitManager(i) if RL_EXIT_ENABLED else None for i in INSTRUMENTS}

SIGNAL_LOG = Path("signal_log.csv")
SIGNAL_HEADERS = [
    "timestamp",
    "instrument",
    "signal",
    "session",
    "ml_probability",
    "approved",
    "rl_closed",
    "order_status",
    "entry_price",
    "tp_price",
    "sl_price",
]


def log_signal(row: dict):
    new_file = not SIGNAL_LOG.exists()
    with open(SIGNAL_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIGNAL_HEADERS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ── Main cycle ────────────────────────────────────────────────────────────────
def run_cycle():
    logger.info("━━━ Cycle start ━━━")

    # New day → reset risk guard
    if guard.trading_date != date.today():
        balance = broker.get_account_balance()
        guard.reset_for_day(balance)

    current_balance = broker.get_account_balance()
    if not guard.check(current_balance):
        if guard.is_halted:
            logger.info("Daily limit hit — closing all positions.")
            broker.close_all_positions()
            for rl in rl_managers.values():
                if rl:
                    rl.on_close()
        return

    for instrument in INSTRUMENTS:
        try:
            # ── Session filter ────────────────────────────────────────────
            if SESSION_FILTER_ENABLED and not session_active(instrument):
                logger.info(f"{instrument}: {session_info(instrument)} — skipping.")
                continue

            # ── Fetch candles + indicators ────────────────────────────────
            df = get_market_data(broker, instrument, GRANULARITY, CANDLES_COUNT)
            if df.empty or len(df) < max(EMA_SLOW, RSI_PERIOD) + 5:
                logger.warning(f"{instrument}: Not enough candle data.")
                continue

            now_utc = pd.Timestamp.now("UTC")
            candle_age = now_utc - df["time"].iloc[-1]
            if candle_age > pd.Timedelta(hours=2):
                logger.info(
                    f"{instrument}: Market closed ({candle_age} since last candle)."
                )
                continue

            df = add_indicators(df, EMA_FAST, EMA_SLOW, RSI_PERIOD)

            # ── Check open position ───────────────────────────────────────
            open_trades = broker.get_open_trades()
            has_position = any(t["instrument"] == instrument for t in open_trades)

            if has_position:
                rl = rl_managers.get(instrument)
                if rl and rl.should_close(df):
                    logger.info(f"{instrument}: RL exit triggered.")
                    broker.close_all_positions()
                    rl.on_close()
                    log_signal(
                        {
                            "timestamp": df["time"].iloc[-1].isoformat(),
                            "instrument": instrument,
                            "signal": "RL_CLOSE",
                            "session": session_info(instrument),
                            "ml_probability": "",
                            "approved": True,
                            "rl_closed": True,
                            "entry_price": "",
                            "tp_price": "",
                            "sl_price": "",
                        }
                    )
                else:
                    logger.info(f"{instrument}: Position open — holding.")
                continue

            # ── Generate signal ───────────────────────────────────────────
            signal = generate_signal(df)
            logger.info(f"{instrument}: signal = {signal} | {session_info(instrument)}")

            if signal == "HOLD":
                continue

            # ── ML confidence filter ──────────────────────────────────────
            ml = ml_filters.get(instrument)
            approved, prob = (True, None) if ml is None else ml.approve(df, signal)
            if prob is not None:
                logger.info(
                    f"{instrument}: ML P={prob:.3f} "
                    f"(threshold {ml.threshold:.2f}) → "
                    f"{'APPROVED' if approved else 'SKIPPED'}"
                )

            # ── Calculate trade levels ────────────────────────────────────
            price = df["close"].iloc[-1]
            tp, sl = trade_levels(signal, price, TAKE_PROFIT_PCT, STOP_LOSS_PCT)
            units = position_units(
                signal, current_balance, price, RISK_PER_TRADE_PCT, STOP_LOSS_PCT
            )

            # ── Calculate trade levels ────────────────────────────────────
            price = df["close"].iloc[-1]
            tp, sl = trade_levels(signal, price, TAKE_PROFIT_PCT, STOP_LOSS_PCT)
            units = position_units(
                signal, current_balance, price, RISK_PER_TRADE_PCT, STOP_LOSS_PCT
            )

            if not approved:
                log_signal(
                    {
                        "timestamp": df["time"].iloc[-1].isoformat(),
                        "instrument": instrument,
                        "signal": signal,
                        "session": session_info(instrument),
                        "ml_probability": f"{prob:.4f}" if prob is not None else "",
                        "approved": False,
                        "rl_closed": False,
                        "order_status": "not_attempted",
                        "entry_price": f"{price:.5f}",
                        "tp_price": f"{tp:.5f}",
                        "sl_price": f"{sl:.5f}",
                    }
                )
                continue

            logger.info(
                f"{instrument}: {signal} | price={price:.5f} "
                f"TP={tp:.5f} SL={sl:.5f} units={units}"
            )
            order_result = broker.place_market_order(instrument, units, tp, sl)

            # Log AFTER placing the order so order_status reflects what
            # actually happened at the broker, not just the ML decision.
            log_signal(
                {
                    "timestamp": df["time"].iloc[-1].isoformat(),
                    "instrument": instrument,
                    "signal": signal,
                    "session": session_info(instrument),
                    "ml_probability": f"{prob:.4f}" if prob is not None else "",
                    "approved": True,
                    "rl_closed": False,
                    "order_status": order_result.get("status", "unknown"),
                    "entry_price": f"{price:.5f}",
                    "tp_price": f"{tp:.5f}",
                    "sl_price": f"{sl:.5f}",
                }
            )

            # Only notify the RL manager if a real position actually opened —
            # an approved signal can still be rejected by the broker (e.g.
            # size limits), and the RL manager must never track a phantom trade.
            rl = rl_managers.get(instrument)
            if rl and order_result.get("filled"):
                rl.on_open(price, signal)

        except Exception as e:
            logger.error(f"{instrument}: Error — {e}", exc_info=True)

    logger.info("━━━ Cycle end ━━━\n")


# ── Scheduler ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ml_on = [k for k, v in ml_filters.items() if v and v.is_active]
    rl_on = [k for k, v in rl_managers.items() if v and v.model]
    logger.info(
        f"Bot starting\n"
        f"  Instruments    : {INSTRUMENTS}\n"
        f"  Session filter : {'ON' if SESSION_FILTER_ENABLED else 'OFF'}\n"
        f"  ML filter      : {ml_on or 'none'}\n"
        f"  RL exit        : {rl_on or 'none (run train_rl.py to enable)'}\n"
        f"  Daily limits   : +{DAILY_PROFIT_TARGET_PCT * 100:.1f}% / "
        f"-{DAILY_LOSS_LIMIT_PCT * 100:.1f}%"
    )

    RECONNECT_WAIT = 60  # seconds to wait before reconnecting after disconnect
    run_cycle()
    last_run = time.time()
    interval = CANDLE_CHECK_INTERVAL_MINUTES * 60

    while True:
        try:
            # Reconnect broker if disconnected (e.g. after TWS daily restart)
            if not broker.ib.isConnected():
                logger.warning("TWS disconnected — waiting 60s then reconnecting...")
                time.sleep(RECONNECT_WAIT)
                try:
                    broker.ib.disconnect()
                except Exception:
                    pass
                broker._contracts = {}  # clear cached contracts
                broker._connect()
                logger.info("Reconnected to TWS successfully.")

            broker.ib.sleep(1)

            if time.time() - last_run >= interval:
                run_cycle()
                last_run = time.time()

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            broker.ib.disconnect()
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e} — retrying in 60s...")
            time.sleep(RECONNECT_WAIT)
