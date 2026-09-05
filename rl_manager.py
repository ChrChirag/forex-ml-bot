"""
Live RL exit manager.

Loaded by main.py for each instrument. When an RL model exists, it predicts
whether to close the current open position on every 15-minute cycle.

Falls back silently to TP/SL-only mode if no model is found —
the bot works fine without it.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RLExitManager:
    """
    Manages the exit of a single open position using a trained PPO model.

    Usage in main.py:
        rl = RLExitManager('EUR_USD')

        # When a trade opens:
        rl.on_open(entry_price, signal)

        # Every cycle while position is open:
        if rl.should_close(df):
            broker.close_all_positions()

        # When trade closes (TP/SL hit):
        rl.on_close()
    """

    def __init__(self, instrument: str):
        self.instrument  = instrument
        self.model       = None
        self.entry_price: float | None = None
        self.direction:   int   | None = None
        self.bars_held:   int          = 0
        self._load()

    def _load(self):
        model_path = Path("models") / f"{self.instrument}_rl_exit.zip"
        if not model_path.exists():
            logger.info(
                f"{self.instrument}: No RL exit model — "
                "using TP/SL only. Run train_rl.py to enable."
            )
            return
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path))
            logger.info(f"{self.instrument}: RL exit manager loaded.")
        except Exception as e:
            logger.warning(f"{self.instrument}: Could not load RL model: {e}")

    @property
    def is_active(self) -> bool:
        return self.model is not None and self.entry_price is not None

    # ── Position lifecycle ────────────────────────────────────────────────────

    def on_open(self, entry_price: float, signal: str):
        self.entry_price = entry_price
        self.direction   = 1 if signal == 'BUY' else -1
        self.bars_held   = 0
        logger.info(
            f"{self.instrument}: RL manager tracking "
            f"{'LONG' if self.direction == 1 else 'SHORT'} @ {entry_price:.5f}"
        )

    def on_close(self):
        self.entry_price = None
        self.direction   = None
        self.bars_held   = 0

    # ── Decision ─────────────────────────────────────────────────────────────

    def should_close(self, df: pd.DataFrame) -> bool:
        """
        Returns True if the RL agent recommends closing now.
        Returns False if no model, no open position, or agent says HOLD.
        """
        if not self.is_active:
            return False

        self.bars_held += 1
        obs    = self._obs(df)
        action, _ = self.model.predict(obs, deterministic=True)

        pnl = self._pnl(df)
        logger.debug(
            f"{self.instrument}: RL obs bars={self.bars_held} "
            f"pnl={pnl*100:+.3f}% → {'CLOSE' if action == 1 else 'HOLD'}"
        )

        if action == 1:
            logger.info(
                f"{self.instrument}: RL recommends CLOSE "
                f"(bars_held={self.bars_held}, P&L={pnl*100:+.3f}%)"
            )
            return True
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pnl(self, df: pd.DataFrame) -> float:
        price = float(df['close'].iloc[-1])
        return (price / self.entry_price - 1) * self.direction

    def _obs(self, df: pd.DataFrame) -> np.ndarray:
        curr = df.iloc[-1]
        p    = float(curr['close'])

        ema_fast = float(curr.get('ema_fast', p))
        ema_slow = float(curr.get('ema_slow', p))
        rsi      = float(curr.get('rsi', 50))

        return np.float32([
            np.clip(self._pnl(df), -0.05, 0.05),
            min(self.bars_held / 48, 1.0),
            rsi / 100.0,
            np.clip(ema_fast / ema_slow if ema_slow else 1.0, 0.99, 1.01),
            float(self.direction),
        ])