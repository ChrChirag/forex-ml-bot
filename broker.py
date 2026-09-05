"""
Interactive Brokers broker wrapper using ib_insync.

SETUP:
1. Install TWS from interactivebrokers.co.uk
2. Log in with paper trading credentials
3. Edit → Global Configuration → API → Settings
   - Enable ActiveX and Socket Clients
   - Socket port: 7497
   - Untick Read-Only API
4. From WSL: cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
   Put that IP in .env as IBKR_HOST
"""

import logging
import time
from ib_insync import IB, MarketOrder, LimitOrder, StopOrder

logger = logging.getLogger(__name__)


class IBKRBroker:
    def __init__(self, host: str, port: int, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self._contracts = {}  # cache: avoids re-qualifying every cycle
        self._connect()

    def _connect(self):
        for attempt in range(5):
            client_id = self.client_id + attempt
            try:
                self.ib.connect(self.host, self.port, clientId=client_id, timeout=20)
                self.ib.RequestTimeout = 30  # 30s timeout on all API calls
                if attempt > 0:
                    logger.info(f"Connected using clientId {client_id}")
                return
            except Exception as e:
                if "already in use" in str(e) or "326" in str(e):
                    logger.warning(
                        f"clientId {client_id} in use, trying {client_id + 1}..."
                    )
                    self.ib = IB()
                    continue
                logger.error(f"Cannot connect to TWS: {e}")
                logger.error(
                    "Make sure TWS is running and API is enabled on port 7497."
                )
                raise
        raise ConnectionError("Could not connect — all client IDs in use. Restart TWS.")

    def _contract(self, instrument: str):
        """
        Return a CFD contract for order placement.

        EU/MiFID II regulations prohibit leveraged spot FX trading for retail
        clients on IBKR Ireland accounts. CFDs are the correct instrument for
        EU-based leveraged forex trading (30:1 cap on major pairs).
        """
        if instrument not in self._contracts:
            from ib_insync import Contract

            pair = instrument.replace("_", "")
            base = pair[:3]
            quote = pair[3:]
            c = Contract()
            c.secType = "CFD"
            c.symbol = base
            c.currency = quote
            c.exchange = "SMART"
            self._contracts[instrument] = c
            logger.debug(f"Created CFD contract for {instrument}")
        return self._contracts[instrument]

    def _data_contract(self, instrument: str):
        """
        Return the underlying Forex contract for historical data requests.
        IBKR does not provide historical data for CFD contracts directly —
        the warning 2127 tells us to use the underlying CASH contract instead.
        Orders still use the CFD contract above.
        """
        from ib_insync import Contract

        pair = instrument.replace("_", "")
        base = pair[:3]
        quote = pair[3:]
        c = Contract()
        c.secType = "CASH"
        c.symbol = base
        c.currency = quote
        c.exchange = "IDEALPRO"
        return c

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        for v in self.ib.accountValues():
            if v.tag == "NetLiquidation" and v.currency == "BASE":
                return float(v.value)
        for v in self.ib.accountValues():
            if v.tag == "NetLiquidation":
                return float(v.value)
        raise ValueError("Could not get account balance from TWS.")

    # ── Market data ───────────────────────────────────────────────────────────

    def get_candles(self, instrument: str, granularity: str, count: int) -> list:
        BAR_SIZE = {
            "M1": "1 min",
            "M5": "5 mins",
            "M15": "15 mins",
            "M30": "30 mins",
            "H1": "1 hour",
            "H4": "4 hours",
            "D": "1 day",
        }
        MINS = {
            "1 min": 1,
            "5 mins": 5,
            "15 mins": 15,
            "30 mins": 30,
            "1 hour": 60,
            "4 hours": 240,
            "1 day": 1440,
        }
        bar_size = BAR_SIZE.get(granularity, "15 mins")
        total_min = count * MINS.get(bar_size, 15)
        cal_days = max(2, int(total_min / (60 * 16) * 1.5) + 1)
        duration = f"{min(cal_days, 30)} D"

        bars = self.ib.reqHistoricalData(
            self._data_contract(instrument),  # underlying CASH contract for data
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )

        # Small pause between requests to avoid IBKR pacing violations
        time.sleep(1)

        result = []
        for bar in bars[-count:]:
            t = bar.date
            result.append(
                {
                    "complete": True,
                    "time": t.isoformat() if hasattr(t, "isoformat") else str(t),
                    "mid": {
                        "o": str(bar.open),
                        "h": str(bar.high),
                        "l": str(bar.low),
                        "c": str(bar.close),
                    },
                }
            )
        return result

    # ── Positions & orders ────────────────────────────────────────────────────

    def get_open_trades(self) -> list:
        result = []
        for pos in self.ib.positions():
            # Check both CFD (current) and CASH (legacy) position types
            if pos.contract.secType in ("CFD", "CASH") and abs(pos.position) > 0:
                symbol = f"{pos.contract.symbol}_{pos.contract.currency}"
                result.append({"instrument": symbol, "units": pos.position})
        return result

    def place_market_order(
        self,
        instrument: str,
        units: int,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> dict:
        contract = self._contract(instrument)
        action = "BUY" if units > 0 else "SELL"
        close_action = "SELL" if units > 0 else "BUY"
        qty = abs(units)
        pid = self.ib.client.getReqId()

        parent = MarketOrder(
            action=action,
            totalQuantity=qty,
            orderId=pid,
            transmit=False,
            tif="DAY",
        )
        tp_order = LimitOrder(
            action=close_action,
            totalQuantity=qty,
            lmtPrice=round(take_profit_price, 5),
            tif="GTC",
            orderId=pid + 1,
            parentId=pid,
            transmit=False,
        )
        sl_order = StopOrder(
            action=close_action,
            totalQuantity=qty,
            stopPrice=round(stop_loss_price, 5),
            tif="GTC",
            orderId=pid + 2,
            parentId=pid,
            transmit=True,
        )

        t1 = self.ib.placeOrder(contract, parent)
        t2 = self.ib.placeOrder(contract, tp_order)
        t3 = self.ib.placeOrder(contract, sl_order)

        # Poll up to 10s for the parent order to settle (filled or rejected).
        # This confirms whether a real position actually opened —
        # an "approved" ML signal can still be rejected by the broker
        # (e.g. size limits), and we must not treat that as a real trade.
        for _ in range(10):
            self.ib.sleep(1)
            if t1.orderStatus.status in (
                "Filled",
                "Cancelled",
                "Inactive",
                "ApiCancelled",
            ):
                break

        status = t1.orderStatus.status
        filled = status == "Filled"

        if filled:
            logger.info(
                f"Order FILLED: {action} {qty} {instrument} "
                f"@ {t1.orderStatus.avgFillPrice:.5f} "
                f"TP={take_profit_price:.5f} SL={stop_loss_price:.5f}"
            )
        else:
            logger.warning(
                f"Order NOT FILLED (status={status}): "
                f"{action} {qty} {instrument} — no position opened."
            )

        return {
            "filled": filled,
            "status": status,
            "parent": t1.order.orderId,
            "tp": t2.order.orderId,
            "sl": t3.order.orderId,
        }

    def close_all_positions(self):
        self.ib.reqGlobalCancel()
        self.ib.sleep(0.5)
        for pos in self.ib.positions():
            if pos.contract.secType == "CASH" and abs(pos.position) > 0:
                action = "SELL" if pos.position > 0 else "BUY"
                self.ib.placeOrder(pos.contract, MarketOrder(action, abs(pos.position)))
                logger.info(f"Closed {pos.contract.localSymbol} {pos.position}")
        self.ib.sleep(1)
