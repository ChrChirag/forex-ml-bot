"""
Gymnasium environment for RL-based forex exit management.

The RL agent observes the state of an open position every bar and decides:
  0 → HOLD  (keep the position open)
  1 → CLOSE (close the position now at market price)

TP and SL still exist as hard limits — the RL agent can also close early
to cut losses faster or lock in partial gains before the full TP is reached.

State vector (5 features):
  [unrealized_pnl_pct, bars_held_norm, rsi_norm, ema_ratio, direction]

Reward: realized P&L when the position closes (any reason).
        Small time penalty per bar to discourage aimless holding.
"""
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

MAX_BARS     = 48        # max bars to hold before forced close (48h on 1h data)
TIME_PENALTY = 0.00005  # tiny penalty per bar to encourage decisive exits


class ForexExitEnv(gym.Env):
    """
    Episodes: replay historical signal entry points, observe + manage position.
    Each episode ends when TP, SL, manual close, or MAX_BARS is reached.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        signal_indices: list,
        tp_pct: float = 0.005,
        sl_pct: float = 0.0025,
    ):
        super().__init__()
        self.df             = df.reset_index(drop=True)
        self.signal_indices = signal_indices  # list of (idx, 'BUY'/'SELL')
        self.tp_pct         = tp_pct
        self.sl_pct         = sl_pct

        # 5-dimensional observation space
        self.observation_space = spaces.Box(
            low=np.float32( [-0.05, 0.0, 0.0, 0.990, -1.0]),
            high=np.float32([ 0.05, 1.0, 1.0, 1.010,  1.0]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # 0=HOLD, 1=CLOSE

        self.entry_idx   = 0
        self.entry_price = 1.0
        self.direction   = 1
        self.bars_held   = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx, signal      = random.choice(self.signal_indices)
        self.entry_idx   = idx
        self.entry_price = float(self.df['close'].iloc[idx])
        self.direction   = 1 if signal == 'BUY' else -1
        self.bars_held   = 0
        return self._obs(), {}

    def step(self, action: int):
        self.bars_held += 1
        i = self.entry_idx + self.bars_held

        # End of dataset
        if i >= len(self.df):
            return self._obs(), float(self._pnl_at(-1)), True, False, {}

        row  = self.df.iloc[i]
        high = float(row['high'])
        low  = float(row['low'])

        # ── Forced closes: TP / SL ────────────────────────────────────────
        if self.direction == 1:   # LONG
            if high >= self.entry_price * (1 + self.tp_pct):
                return self._obs(), self.tp_pct, True, False, {}
            if low  <= self.entry_price * (1 - self.sl_pct):
                return self._obs(), -self.sl_pct, True, False, {}
        else:                      # SHORT
            if low  <= self.entry_price * (1 - self.tp_pct):
                return self._obs(), self.tp_pct, True, False, {}
            if high >= self.entry_price * (1 + self.sl_pct):
                return self._obs(), -self.sl_pct, True, False, {}

        # ── Agent closes ──────────────────────────────────────────────────
        if action == 1:
            pnl = self._pnl_at(i) - 0.0001   # subtract approx spread
            return self._obs(), float(pnl), True, False, {}

        # ── Max bars reached: forced exit ─────────────────────────────────
        if self.bars_held >= MAX_BARS:
            pnl = self._pnl_at(i) - 0.0001
            return self._obs(), float(pnl), True, False, {}

        # ── Continue holding ──────────────────────────────────────────────
        return self._obs(), -TIME_PENALTY, False, False, {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pnl_at(self, idx: int) -> float:
        price = float(self.df['close'].iloc[idx])
        return (price / self.entry_price - 1) * self.direction

    def _obs(self) -> np.ndarray:
        i   = min(self.entry_idx + self.bars_held, len(self.df) - 1)
        row = self.df.iloc[i]
        p   = float(row['close'])

        ema_fast = float(row.get('ema_fast', p))
        ema_slow = float(row.get('ema_slow', p))
        rsi      = float(row.get('rsi', 50))

        pnl_now = (p / self.entry_price - 1) * self.direction

        return np.float32([
            np.clip(pnl_now,  -0.05, 0.05),
            self.bars_held / MAX_BARS,
            rsi / 100.0,
            np.clip(ema_fast / ema_slow if ema_slow else 1.0, 0.99, 1.01),
            float(self.direction),
        ])