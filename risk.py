import logging
from datetime import date

logger = logging.getLogger(__name__)


class DailyRiskGuard:
    """
    Tracks the day's starting balance and stops the bot once either:
    • Daily profit target is hit (e.g. +5%) → lock in gains
    • Daily loss limit is hit (e.g. −3%)    → protect capital

    Call reset_for_day() once at the start of each trading day.
    Call check() before every trade cycle; it returns False when the bot should halt.
    """

    def __init__(self, profit_target_pct: float, loss_limit_pct: float):
        self.profit_target_pct = profit_target_pct
        self.loss_limit_pct    = loss_limit_pct
        self._start_balance: float | None = None
        self._trading_date:  date  | None = None
        self.is_halted = False

    @property
    def trading_date(self) -> date | None:
        return self._trading_date

    def reset_for_day(self, balance: float):
        self._start_balance = balance
        self._trading_date  = date.today()
        self.is_halted      = False
        logger.info(
            f"New trading day — starting balance: {balance:.2f} | "
            f"Target: +{self.profit_target_pct*100:.0f}% | "
            f"Floor: −{self.loss_limit_pct*100:.0f}%"
        )

    def check(self, current_balance: float) -> bool:
        """
        Returns True if trading may continue.
        Returns False (and sets is_halted) when a limit is hit.
        """
        if self.is_halted:
            return False

        if self._start_balance is None:
            return True  # guard not initialised yet — allow first cycle

        pnl_pct = (current_balance - self._start_balance) / self._start_balance

        if pnl_pct >= self.profit_target_pct:
            logger.info(
                f"✅ Daily profit target hit: {pnl_pct*100:+.2f}% ≥ "
                f"+{self.profit_target_pct*100:.0f}% — halting for the day."
            )
            self.is_halted = True
            return False

        if pnl_pct <= -self.loss_limit_pct:
            logger.info(
                f"🛑 Daily loss limit hit: {pnl_pct*100:+.2f}% ≤ "
                f"−{self.loss_limit_pct*100:.0f}% — halting for the day."
            )
            self.is_halted = True
            return False

        logger.info(
            f"Daily P&L: {pnl_pct*100:+.2f}% "
            f"(target +{self.profit_target_pct*100:.0f}% / "
            f"floor −{self.loss_limit_pct*100:.0f}%)"
        )
        return True
