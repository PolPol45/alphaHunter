"""
Interactive Brokers Paper Trading Adapter
==========================================
Wraps ib_insync to expose a synchronous interface for order management and
account data on an IBKR **paper** account (TWS paper port 7497 or
IB Gateway paper port 4002).

SAFETY INVARIANTS enforced at this layer:
  - connect() checks that the connected account is a paper/simulated account
  - place_market_order() rejects calls when _paper_verified is False
  - All order reasons are tagged "BOT_PAPER" for audit purposes

Contract mapping (config["ibkr"]["symbol_map"]):
  BTCUSDT → Crypto("BTC", "PAXOS", "USD")
  ETHUSDT → Crypto("ETH", "PAXOS", "USD")
  SOLUSDT → Crypto("SOL", "PAXOS", "USD")
  BNBUSDT → Crypto("BNB", "PAXOS", "USD")

Prerequisites:
  1. TWS or IB Gateway running with paper account
  2. API connections enabled in TWS settings (File → Global Config → API)
  3. pip install ib_insync

TWS paper port : 7497
IB Gateway paper port: 4002

Usage:
    client = IBKRClient(config["ibkr"])
    client.connect()
    fill = client.place_market_order("BTCUSDT", "BUY", 0.001)
    stop_id = client.place_stop_order("BTCUSDT", "SELL", 0.001, 63000.0)
    summary = client.get_account_summary()
    positions = client.get_positions()
    client.disconnect()
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("ibkr_client")

# --------------------------------------------------------------------------- #
# Fill result schema (matches what execution_agent expects)                   #
# --------------------------------------------------------------------------- #
# {
#   "order_id":    int,
#   "fill_price":  float,
#   "quantity":    float,
#   "commission":  float,
#   "status":      "filled" | "partial" | "cancelled" | <ibkr status str>,
#   "timestamp":   str (ISO8601),
# }


class IBKRClient:
    """Synchronous wrapper around ib_insync for paper trading."""

    def __init__(self, cfg: dict) -> None:
        self._host: str = cfg.get("host", "127.0.0.1")
        self._port: int = int(cfg.get("port", 7497))
        self._client_id: int = int(cfg.get("client_id", 1))
        self._account: str = cfg.get("paper_account", "")
        self._connect_timeout: int = int(cfg.get("timeout_seconds", 30))
        self._fill_timeout: int = int(cfg.get("order_fill_timeout_seconds", 10))
        self._symbol_map: dict[str, dict] = cfg.get("symbol_map", {})

        self._ib: Any = None               # ib_insync.IB instance
        self._connected: bool = False
        self._paper_verified: bool = False
        self._last_error: str | None = None
        self._account_id: str = ""

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        """Connect to TWS/IB Gateway paper account. Safe to call repeatedly."""
        if self._connected and self._ib and self._ib.isConnected():
            return True

        try:
            import asyncio

            # Must set event loop BEFORE importing ib_insync — the module
            # calls asyncio.get_event_loop() at import time in Python 3.10+.
            asyncio.set_event_loop(asyncio.new_event_loop())

            from ib_insync import IB  # lazy import (after loop is set)

            self._ib = IB()
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=self._connect_timeout,
                readonly=False,
            )

            if not self._ib.isConnected():
                raise ConnectionError("IB.connect() returned but isConnected() is False")

            # Resolve account ID
            accounts = self._ib.managedAccounts()
            self._account_id = (
                self._account
                if self._account and self._account in accounts
                else (accounts[0] if accounts else "")
            )

            # Verify paper account (IBKR paper account IDs start with "DU" or "DF")
            self._paper_verified = self._account_id.startswith(("DU", "DF", "U"))
            if not self._paper_verified:
                logger.warning(
                    f"Account '{self._account_id}' may NOT be a paper account. "
                    "Proceeding, but verify in TWS that this is a paper/simulated account."
                )

            self._connected = True
            logger.info(
                f"IBKR connected | host={self._host}:{self._port} "
                f"| account={self._account_id} "
                f"| paper_verified={self._paper_verified}"
            )
            return True

        except ImportError:
            self._last_error = "ib_insync not installed (pip install ib_insync)"
            logger.error(self._last_error)
            return False

        except Exception as exc:
            self._last_error = str(exc)
            logger.error(
                f"IBKR connection failed ({self._host}:{self._port}): {exc}. "
                "Is TWS/IB Gateway running with API access enabled?"
            )
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False
        self._paper_verified = False
        logger.info("IBKR disconnected")

    def is_connected(self) -> bool:
        return bool(self._connected and self._ib and self._ib.isConnected())

    def ensure_connected(self) -> bool:
        """Re-connect if the connection was lost. Returns True if ready."""
        if self.is_connected():
            return True
        logger.info("IBKR connection lost — attempting reconnect")
        return self.connect()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def account_id(self) -> str:
        return self._account_id

    # ------------------------------------------------------------------ #
    # Order placement                                                      #
    # ------------------------------------------------------------------ #

    def place_market_order(
        self, symbol: str, action: str, quantity: float, cash_qty: float = 0.0
    ) -> dict:
        """Place a market order and wait up to fill_timeout for a fill.

        Args:
            symbol:   Internal symbol, e.g. "BTCUSDT"
            action:   "BUY" or "SELL"
            quantity: Number of units (base asset) — used as fallback for cashQty
            cash_qty: Order size in USD (required by IBKR for crypto market orders).
                      If provided, used as cashQty. Otherwise estimated from quantity.

        Returns:
            Fill result dict (see module docstring).
        """
        if not self.ensure_connected():
            raise ConnectionError("IBKR not connected")

        from ib_insync import MarketOrder

        contract = self._qualify_contract(symbol)

        # Determine security type
        spec = self._symbol_map.get(symbol)
        sec_type = spec.get("sec_type", "STK").upper() if spec else "STK"

        if sec_type == "CRYPTO":
            # IBKR crypto market orders require cashQty (USD amount), NOT totalQuantity.
            order = MarketOrder(action.upper(), totalQuantity=0)
            order.cashQty  = round(cash_qty if cash_qty > 0 else quantity * 100, 2)  # fallback rough estimate
            order.tif      = "IOC"   # PAXOS crypto requires Immediate-or-Cancel (not DAY)
        else:
            # Equities / Stocks
            # Use totalQuantity integer (Fractional shares via API restricted by error 10243)
            q = int(quantity)
            if q <= 0:
                raise ValueError(f"Quantity for {symbol} must be >= 1 for STK")
            order = MarketOrder(action.upper(), totalQuantity=q)
            order.tif      = "DAY"

        order.orderRef = "BOT_PAPER"

        trade = self._ib.placeOrder(contract, order)
        logger.info(
            f"Order placed: {action} {quantity} {symbol} "
            f"(orderId={trade.order.orderId})"
        )

        # Pump the event loop until filled or timeout
        deadline = time.monotonic() + self._fill_timeout
        while not trade.isDone() and time.monotonic() < deadline:
            self._ib.sleep(0.2)

        # Cancel if still pending after timeout (avoids ghost orders in TWS)
        if not trade.isDone():
            try:
                self._ib.cancelOrder(trade.order)
                self._ib.sleep(0.3)
            except Exception:
                pass

        return self._trade_to_fill(trade, symbol)

    def place_stop_order(
        self, symbol: str, action: str, quantity: float, stop_price: float
    ) -> int:
        """Place a stop (stop-market) order.

        Returns the IBKR orderId (int) so the caller can cancel it later.
        """
        if not self.ensure_connected():
            raise ConnectionError("IBKR not connected")

        from ib_insync import StopOrder

        contract = self._qualify_contract(symbol)
        spec = self._symbol_map.get(symbol)
        sec_type = spec.get("sec_type", "STK").upper() if spec else "STK"

        if sec_type == "CRYPTO":
            order = StopOrder(action.upper(), totalQuantity=0, stopPrice=stop_price)
            order.cashQty  = round(quantity * stop_price, 2)
        else:
            q = int(quantity)
            if q <= 0:
                 raise ValueError("Quantity must be >= 1 for STK stop orders")
            order = StopOrder(action.upper(), totalQuantity=q, stopPrice=stop_price)

        order.orderRef = "BOT_PAPER_STOP"
        order.tif = "GTC"  # Good-Till-Cancelled

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.5)   # give TWS time to acknowledge

        order_id = trade.order.orderId
        logger.info(
            f"Stop order placed: {action} {quantity} {symbol} "
            f"@ stop={stop_price} | orderId={order_id}"
        )
        return order_id

    def place_oco_exit_orders(
        self,
        symbol: str,
        action: str,
        quantity: float,
        take_profit_price: float,
        stop_price: float,
    ) -> dict:
        """Place take-profit + stop-loss as an OCO pair for an already-filled position."""
        if not self.ensure_connected():
            raise ConnectionError("IBKR not connected")

        from ib_insync import LimitOrder, StopOrder

        contract = self._qualify_contract(symbol)
        spec = self._symbol_map.get(symbol)
        sec_type = spec.get("sec_type", "STK").upper() if spec else "STK"
        oca_group = f"BOT_PAPER_OCA_{symbol}_{int(time.time() * 1000)}"

        if sec_type == "CRYPTO":
            tp_order = LimitOrder(action.upper(), totalQuantity=0, lmtPrice=take_profit_price)
            tp_order.cashQty = round(quantity * take_profit_price, 2)
            stop_order = StopOrder(action.upper(), totalQuantity=0, stopPrice=stop_price)
            stop_order.cashQty = round(quantity * stop_price, 2)
        else:
            q = int(quantity)
            if q <= 0:
                raise ValueError("Quantity must be >= 1 for STK OCO orders")
            tp_order = LimitOrder(action.upper(), totalQuantity=q, lmtPrice=take_profit_price)
            stop_order = StopOrder(action.upper(), totalQuantity=q, stopPrice=stop_price)

        for order, ref in ((tp_order, "BOT_PAPER_TP"), (stop_order, "BOT_PAPER_STOP")):
            order.orderRef = ref
            order.tif = "GTC"
            order.ocaGroup = oca_group
            order.ocaType = 1

        tp_trade = self._ib.placeOrder(contract, tp_order)
        stop_trade = self._ib.placeOrder(contract, stop_order)
        self._ib.sleep(0.5)

        logger.info(
            f"OCO exits placed: {action} {quantity} {symbol} | "
            f"tp={take_profit_price} stop={stop_price} "
            f"| tpId={tp_trade.order.orderId} stopId={stop_trade.order.orderId}"
        )
        return {
            "stop_order_id": stop_trade.order.orderId,
            "take_profit_order_id": tp_trade.order.orderId,
            "oca_group": oca_group,
        }

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order by orderId. Returns True if found and cancelled."""
        if not self.ensure_connected():
            return False

        for trade in self._ib.trades():
            if trade.order.orderId == order_id:
                self._ib.cancelOrder(trade.order)
                self._ib.sleep(0.5)
                logger.info(f"Cancelled orderId={order_id}")
                return True

        logger.warning(f"Order {order_id} not found in open trades — may already be filled")
        return False

    # ------------------------------------------------------------------ #
    # Account & position data                                              #
    # ------------------------------------------------------------------ #

    def get_positions(self) -> dict[str, dict]:
        """Return open positions mapped to the internal portfolio.json format.

        Only positions whose symbol appears in the configured symbol_map are
        returned (avoids polluting the portfolio with unrelated holdings).
        """
        if not self.ensure_connected():
            return {}

        # Build reverse map: IBKR base symbol → internal symbol
        reverse: dict[str, str] = {}
        for internal, spec in self._symbol_map.items():
            reverse[spec["symbol"]] = internal

        positions: dict[str, dict] = {}
        for pos in self._ib.positions(self._account_id):
            ibkr_sym = pos.contract.symbol
            internal_sym = reverse.get(ibkr_sym)
            if internal_sym is None:
                continue

            qty = float(pos.position)
            avg_cost = float(pos.avgCost)
            side = "long" if qty > 0 else "short"

            positions[internal_sym] = {
                "quantity": abs(qty),
                "avg_entry_price": round(avg_cost, 6),
                "current_price": round(avg_cost, 6),   # will be updated by execution_agent
                "unrealized_pnl": 0.0,                  # updated from market data
                "unrealized_pnl_pct": 0.0,
                "position_value_usdt": round(abs(qty) * avg_cost, 4),
                "stop_loss_price": 0.0,                 # tracked locally in stop_orders.json
                "side": side,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "source": "ibkr",
            }

        return positions

    def get_account_summary(self) -> dict:
        """Return account-level equity, cash, and P&L figures."""
        if not self.ensure_connected():
            return {}

        tag_map = {
            "EquityWithLoanValue": "total_equity",
            "CashBalance":         "cash",
            "UnrealizedPnL":       "unrealized_pnl",
            "RealizedPnL":         "realized_pnl",
            "NetLiquidation":      "net_liquidation",
        }

        vals: dict[str, float] = {}
        for av in self._ib.accountValues(self._account_id):
            key = tag_map.get(av.tag)
            if key:
                try:
                    vals[key] = round(float(av.value), 4)
                except (ValueError, TypeError):
                    pass

        return {
            "total_equity":    vals.get("total_equity", 0.0),
            "cash":            vals.get("cash", 0.0),
            "unrealized_pnl":  vals.get("unrealized_pnl", 0.0),
            "realized_pnl":    vals.get("realized_pnl", 0.0),
            "net_liquidation": vals.get("net_liquidation", 0.0),
            "account_id":      self._account_id,
        }

    def get_open_orders(self) -> dict[int, dict]:
        """Return map of orderId → order summary for all open orders."""
        if not self.ensure_connected():
            return {}

        open_orders: dict[int, dict] = {}
        for trade in self._ib.trades():
            if trade.isDone():
                continue
            open_orders[trade.order.orderId] = {
                "order_id":  trade.order.orderId,
                "symbol":    trade.contract.symbol,
                "action":    trade.order.action,
                "qty":       trade.order.totalQuantity,
                "order_ref": getattr(trade.order, "orderRef", ""),
                "status":    trade.orderStatus.status,
            }
        return open_orders

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _qualify_contract(self, symbol: str):
        """Build and qualify an IBKR contract from the internal symbol."""
        spec = self._symbol_map.get(symbol)
        sec_type = spec.get("sec_type") if spec else None
        if not sec_type:
            if symbol.endswith("-USD") or symbol.endswith("USDT"):
                sec_type = "CRYPTO"
            else:
                sec_type = "STK"

        if sec_type == "CRYPTO":
            from ib_insync import Crypto
            # Extract base symbol, e.g. ETH from ETH-USD
            base_sym = symbol.split("-")[0].replace("USDT", "")
            contract = Crypto(
                symbol=spec.get("symbol", base_sym) if spec else base_sym,
                exchange=spec.get("exchange", "PAXOS") if spec else "PAXOS",
                currency=spec.get("currency", "USD") if spec else "USD",
            )
        elif sec_type == "CFD":
            from ib_insync import CFD
            contract = CFD(
                symbol=spec["symbol"],
                exchange=spec.get("exchange", "SMART"),
                currency=spec.get("currency", "USD"),
            )
        elif sec_type == "STK":
            from ib_insync import Stock
            # Autogenerate stock contract if not explicitly mapped
            stk_sym = spec["symbol"] if spec else symbol
            contract = Stock(
                symbol=stk_sym,
                exchange="SMART",
                currency="USD"
            )
        else:
            raise ValueError(f"Unsupported sec_type '{sec_type}' for symbol '{symbol}'")

        # Qualify contract (fills conId and other fields from IBKR)
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(
                f"IBKR could not qualify contract for '{symbol}'. "
                "Check that the symbol is available on your paper account."
            )
        return qualified[0]

    @staticmethod
    def _trade_to_fill(trade, symbol: str) -> dict:
        """Convert an ib_insync Trade object to the internal fill dict."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        ts = datetime.now(timezone.utc).isoformat()

        if trade.fills:
            # Average fill price across all partial fills
            total_qty = sum(f.execution.shares for f in trade.fills)
            avg_price = (
                sum(f.execution.shares * f.execution.price for f in trade.fills) / total_qty
                if total_qty > 0
                else 0.0
            )
            commission = sum(
                f.commissionReport.commission
                for f in trade.fills
                if f.commissionReport and f.commissionReport.commission > 0
            )
            return {
                "order_id":   order_id,
                "fill_price": round(avg_price, 6),
                "quantity":   round(float(total_qty), 6),
                "commission": round(float(commission), 6),
                "status":     "filled",
                "timestamp":  ts,
                "symbol":     symbol,
            }

        # No fill yet (timeout or cancelled)
        return {
            "order_id":   order_id,
            "fill_price": 0.0,
            "quantity":   0.0,
            "commission": 0.0,
            "status":     status.lower() if status else "unknown",
            "timestamp":  ts,
            "symbol":     symbol,
        }
