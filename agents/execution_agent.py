"""
Execution Agent — Triple Portfolio
====================================
Manages three independent sub-portfolios:

  Retail        ($20K)    — executed from validated_signals["retail"]
  Institutional ($880K)   — executed from validated_signals["institutional"]
  Alpha Hunter  ($100K)   — executed from alpha_signals.json (US equities, simulation)

Retail + Institutional share the same IBKR connection.
Alpha Hunter runs in pure simulation (US equity universe not in IBKR crypto map).
Stop-loss orders are placed on IBKR immediately after each fill (retail/inst).

SAFETY INVARIANTS:
  1. config.trading.paper_trading must be True (hardcoded assert)
  2. IBKRClient verifies account starts with "DU"/"DF"
  3. All orders tagged BOT_PAPER / BOT_PAPER_STOP

Reads:  data/validated_signals.json, data/market_data.json,
        data/alpha_signals.json,
        data/portfolio_retail.json, data/portfolio_institutional.json,
        data/portfolio_alpha.json,
        data/stop_orders_retail.json, data/stop_orders_institutional.json
Writes: (same files, updated)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR


class ExecutionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("execution_agent")
        self._sim = self.config["simulation"]
        self._turnover_cfg = self.config.get("turnover_control", {})

        trading_cfg = self.config.get("trading", {})
        ibkr_cfg = {"host": "127.0.0.1", "port": 7497, **self.config.get("ibkr", {})}
        self._ibkr         = None
        self._ibkr_enabled = bool(
            trading_cfg.get("ibkr_paper_enabled", ibkr_cfg.get("enabled", False))
        )
        if self._ibkr_enabled:
            from adapters.ibkr_client import IBKRClient
            self._ibkr = IBKRClient(ibkr_cfg)
            if self._ibkr.connect():
                self.logger.info(
                    f"IBKR paper adapter ready | account={self._ibkr.account_id}"
                )
            else:
                self.logger.warning(
                    f"IBKR connect failed ({self._ibkr.last_error}) — will use simulation fallback"
                )

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        assert self.config["trading"]["paper_trading"] is True, (
            "SAFETY VIOLATION: paper_trading must be True."
        )
        self.mark_running()
        try:
            validated  = self.read_json(DATA_DIR / "validated_signals.json")
            market_doc = self.read_json(DATA_DIR / "market_data.json")

            if not validated:
                raise ValueError("validated_signals.json is empty or missing")

            port_retail = self.read_json(DATA_DIR / "portfolio_retail.json")
            port_inst   = self.read_json(DATA_DIR / "portfolio_institutional.json")

            if not port_retail:
                port_retail = self._init_portfolio("retail")
                self.logger.info("Initialised new retail portfolio")
            if not port_inst:
                port_inst = self._init_portfolio("institutional")
                self.logger.info("Initialised new institutional portfolio")

            stop_retail = self.read_json(DATA_DIR / "stop_orders_retail.json")
            stop_inst   = self.read_json(DATA_DIR / "stop_orders_institutional.json")

            ibkr_live = (
                self._ibkr_enabled and self._ibkr and self._ibkr.ensure_connected()
            )

            now = self._now_iso()

            # ── Retail execution ──────────────────────────────────────── #
            retail_enabled = bool(self.config.get("retail", {}).get("enabled", True))
            if retail_enabled:
                self._run_cycle(
                    mode         = "retail",
                    validated    = validated.get("retail", {}),
                    market_doc   = market_doc,
                    portfolio    = port_retail,
                    stop_orders  = stop_retail,
                    ibkr_live    = ibkr_live,
                )
            port_retail["timestamp"]      = now
            port_retail["execution_mode"] = "ibkr" if ibkr_live else "simulation"
            self._set_account_metadata(port_retail, "retail", ibkr_live)

            # ── Institutional execution ───────────────────────────────── #
            self._run_cycle(
                mode         = "institutional",
                validated    = validated.get("institutional", {}),
                market_doc   = market_doc,
                portfolio    = port_inst,
                stop_orders  = stop_inst,
                ibkr_live    = ibkr_live,
            )
            port_inst["timestamp"]      = now
            port_inst["execution_mode"] = "ibkr" if ibkr_live else "simulation"
            self._set_account_metadata(port_inst, "institutional", ibkr_live)

            # ── Alpha Hunter portfolio (US equities, always simulation) ── #
            alpha_cfg = self.config.get("alpha_hunter", {})
            if alpha_cfg.get("enabled", False):
                alpha_signals_doc = self.read_json(DATA_DIR / "alpha_signals.json")
                port_alpha        = self.read_json(DATA_DIR / "portfolio_alpha.json")
                if not port_alpha:
                    port_alpha = self._init_alpha_portfolio()
                    self.logger.info("Initialised new alpha portfolio")

                if alpha_signals_doc:
                    self._run_alpha_cycle(
                        signals_doc=alpha_signals_doc,
                        portfolio=port_alpha,
                    )

                port_alpha["timestamp"]      = now
                port_alpha["execution_mode"] = "simulation"
                port_alpha["account_scope"] = "local_simulation"
                port_alpha["equity_basis"] = "portfolio_mark_to_market"
                port_alpha["allocation_note"] = (
                    "Alpha Hunter uses an isolated simulated portfolio. "
                    "Equity and cash are computed from local positions only."
                )
                self.write_json(DATA_DIR / "portfolio_alpha.json", port_alpha)
                self.update_shared_state("data_freshness.portfolio_alpha", port_alpha["timestamp"])
                self.logger.info(
                    f"[alpha][sim] Equity={port_alpha['total_equity']:.2f} | "
                    f"Cash={port_alpha['cash']:.2f} | "
                    f"Positions={len([p for p in port_alpha['positions'].values() if p.get('quantity', 0) > 0])} | "
                    f"DD={port_alpha['drawdown_pct']*100:.2f}%"
                )

            # ── Persist ──────────────────────────────────────────────── #
            self.write_json(DATA_DIR / "portfolio_retail.json",       port_retail)
            self.write_json(DATA_DIR / "portfolio_institutional.json", port_inst)
            self.write_json(DATA_DIR / "stop_orders_retail.json",     stop_retail)
            self.write_json(DATA_DIR / "stop_orders_institutional.json", stop_inst)

            self.update_shared_state("data_freshness.portfolio_retail",       port_retail["timestamp"])
            self.update_shared_state("data_freshness.portfolio_institutional", port_inst["timestamp"])

            exec_mode = "ibkr" if ibkr_live else "simulation"
            self.logger.info(
                f"[{exec_mode}] Retail   Equity={port_retail['total_equity']:.2f} | "
                f"Cash={port_retail['cash']:.2f} | DD={port_retail['drawdown_pct']*100:.2f}%"
            )
            self.logger.info(
                f"[{exec_mode}] Inst.    Equity={port_inst['total_equity']:.2f} | "
                f"Cash={port_inst['cash']:.2f} | DD={port_inst['drawdown_pct']*100:.2f}%"
            )
            self.mark_done()
            return True

        except AssertionError:
            raise
        except Exception as exc:
            self.mark_error(exc)
            return False

    # ------------------------------------------------------------------ #
    # Per-mode execution cycle                                             #
    # ------------------------------------------------------------------ #

    def _run_cycle(
        self,
        mode:        str,
        validated:   dict,
        market_doc:  dict,
        portfolio:   dict,
        stop_orders: dict,
        ibkr_live:   bool,
    ) -> None:
        cfg = self.config[mode]

        # P2 — reset daily_start_equity at midnight
        self._maybe_reset_daily_equity(portfolio)

        if ibkr_live:
            self._sync_from_ibkr(portfolio, market_doc)
            self._check_stop_order_status(stop_orders, portfolio)

        # P1 — check take-profit and trailing stop BEFORE new entries
        self._check_exits(mode, portfolio, market_doc, stop_orders, ibkr_live)

        # P1 fix: enforce max_open_positions and cash_reserve_floor
        _exec_cfg = self.config.get("execution", {})
        _max_open = int(_exec_cfg.get("max_open_positions", 12))
        _cash_floor_pct = float(_exec_cfg.get("cash_reserve_floor_pct", 0.10))
        _equity = float(portfolio.get("total_equity") or portfolio.get("cash", 0))
        _cash_floor_abs = _equity * _cash_floor_pct
        _open_count = sum(1 for p in portfolio.get("positions", {}).values() if float(p.get("quantity", 0)) > 0)

        for symbol, vsig in validated.items():
            if not vsig.get("approved"):
                continue
            # Skip new BUY entries if portfolio is full or cash below floor
            if vsig.get("signal_type") == "BUY" and symbol not in portfolio.get("positions", {}):
                if _open_count >= _max_open:
                    self.logger.info(f"[{mode}] Max open positions ({_max_open}) reached — skipping {symbol}")
                    continue
                if portfolio.get("cash", 0) <= _cash_floor_abs:
                    self.logger.info(f"[{mode}] Cash floor hit (have {portfolio['cash']:.0f}, floor {_cash_floor_abs:.0f}) — skipping {symbol}")
                    continue
                _open_count += 1  # reserve slot optimistically
            if ibkr_live:
                if self._ibkr_enabled:
                    # ibkr_client natively converts unknown symbols to US STK on SMART
                    self._execute_ibkr_signal(symbol, vsig, portfolio, stop_orders, mode)
                else:
                    self.logger.warning(f"[{mode}] IBKR unavailable — using simulation fallback")
                    self._simulate_fill(symbol, vsig, portfolio, mode)
            else:
                self._simulate_fill(symbol, vsig, portfolio, mode)

        if ibkr_live:
            try:
                summary = self._ibkr.get_account_summary()
                # Apportion account summary between retail and institutional
                total_ib_equity = summary.get("total_equity", 0)
                if total_ib_equity > 0:
                    # Scale by portfolio's share of configured capital
                    ret_cap  = self.config["retail"]["capital"]
                    inst_cap = self.config["institutional"]["capital"]
                    total_cfg = ret_cap + inst_cap
                    share = ret_cap / total_cfg if mode == "retail" else inst_cap / total_cfg
                    portfolio["total_equity"] = round(total_ib_equity * share, 4)
                    portfolio["cash"]         = round(summary.get("cash", portfolio["cash"]) * share, 4)
            except Exception as exc:
                self.logger.warning(f"[{mode}] Could not read IBKR account summary: {exc}")
                self._recalculate_equity(portfolio, market_doc)
        else:
            self._update_sim_positions(portfolio, market_doc)
            self._recalculate_equity(portfolio, market_doc)

        self._update_drawdown(portfolio, mode)

    # ================================================================== #
    # IBKR execution path (shared by both modes)                          #
    # ================================================================== #

    def _sync_from_ibkr(self, portfolio: dict, market_doc: dict) -> None:
        try:
            ibkr_positions = self._ibkr.get_positions()
        except Exception as exc:
            self.logger.warning(f"IBKR get_positions failed: {exc}")
            return

        assets = market_doc.get("assets", {})
        for symbol, ibkr_pos in ibkr_positions.items():
            existing = portfolio["positions"].get(symbol, {})
            ibkr_pos["stop_loss_price"]    = existing.get("stop_loss_price", 0.0)
            ibkr_pos["opened_at"]          = existing.get("opened_at", ibkr_pos["opened_at"])
            # Fix A — preserve P1 exit fields across IBKR sync cycles
            ibkr_pos["entry_atr"]          = existing.get("entry_atr", 0.0)
            ibkr_pos["trailing_high"]      = existing.get("trailing_high", ibkr_pos.get("avg_entry_price", 0.0))
            ibkr_pos["take_profit_price"]  = existing.get("take_profit_price", 0.0)

            current_price = float(
                assets.get(symbol, {}).get("last_price", ibkr_pos["avg_entry_price"])
            )
            ibkr_pos["current_price"] = round(current_price, 4)
            qty   = ibkr_pos["quantity"]
            entry = ibkr_pos["avg_entry_price"]
            side  = ibkr_pos["side"]
            pnl   = ((current_price - entry) if side == "long" else (entry - current_price)) * qty
            ibkr_pos["unrealized_pnl"]     = round(pnl, 4)
            ibkr_pos["unrealized_pnl_pct"] = round(pnl / (entry * qty) if entry * qty > 0 else 0, 6)
            ibkr_pos["position_value_usdt"] = round(current_price * qty, 4)
            portfolio["positions"][symbol]  = ibkr_pos

        for symbol in list(portfolio["positions"].keys()):
            if symbol not in ibkr_positions and portfolio["positions"][symbol].get("quantity", 0) > 0:
                self.logger.info(f"Position {symbol} closed externally on IBKR — zeroing out")
                portfolio["positions"][symbol]["quantity"] = 0.0
                portfolio["positions"][symbol]["unrealized_pnl"] = 0.0
                portfolio["positions"][symbol]["position_value_usdt"] = 0.0

    def _check_stop_order_status(self, stop_orders: dict, portfolio: dict) -> None:
        try:
            open_ids = set(self._ibkr.get_open_orders().keys())
        except Exception as exc:
            self.logger.warning(f"Could not read open orders from IBKR: {exc}")
            return

        for symbol, order_meta in list(stop_orders.items()):
            meta = self._normalise_exit_orders(order_meta)
            tracked_ids = {
                oid
                for oid in (meta["stop_order_id"], meta["take_profit_order_id"])
                if oid is not None
            }
            if tracked_ids and tracked_ids.intersection(open_ids):
                continue
            if tracked_ids:
                self.logger.info(
                    f"Exit orders for {symbol} no longer open "
                    f"(stop={meta['stop_order_id']}, tp={meta['take_profit_order_id']})"
                )
                pos = portfolio["positions"].get(symbol, {})
                if pos.get("quantity", 0) > 0:
                    close_price = float(pos.get("current_price") or pos.get("avg_entry_price", 0))
                    tp_price = float(pos.get("take_profit_price", 0) or 0)
                    side = pos.get("side", "long")
                    tp_hit = (
                        tp_price > 0
                        and (
                            (side == "long" and close_price >= tp_price)
                            or (side == "short" and close_price <= tp_price)
                        )
                    )
                    reason = "TP_HIT" if tp_hit else "SL_HIT"
                    if reason == "SL_HIT":
                        close_price = float(pos.get("stop_loss_price") or close_price)
                    self._record_close_trade(symbol, close_price, pos, portfolio, reason)
                    portfolio["positions"][symbol] = self._zeroed_position(
                        pos,
                        close_price,
                        pos.get("side", "long"),
                    )
                del stop_orders[symbol]

    @staticmethod
    def _normalise_exit_orders(order_meta: dict | int | float | str | None) -> dict:
        if isinstance(order_meta, dict):
            stop_id = order_meta.get("stop_order_id")
            tp_id = order_meta.get("take_profit_order_id")
            return {
                "stop_order_id": int(stop_id) if stop_id is not None else None,
                "take_profit_order_id": int(tp_id) if tp_id is not None else None,
                "oca_group": order_meta.get("oca_group"),
            }
        if order_meta in (None, "", 0):
            return {
                "stop_order_id": None,
                "take_profit_order_id": None,
                "oca_group": None,
            }
        return {
            "stop_order_id": int(order_meta),
            "take_profit_order_id": None,
            "oca_group": None,
        }

    @staticmethod
    def _risk_pct_from_stop(fill_price: float, stop_price: float) -> float:
        if fill_price <= 0 or stop_price <= 0:
            return 0.0
        return abs(fill_price - stop_price) / fill_price

    @classmethod
    def _build_exit_levels(cls, signal_type: str, fill_price: float, vsig: dict) -> dict:
        raw_risk_pct = vsig.get("stop_loss_pct")
        if raw_risk_pct is not None:
            risk_pct = float(raw_risk_pct)
        else:
            risk_pct = cls._risk_pct_from_stop(
                fill_price,
                float(vsig.get("stop_loss_price", 0) or 0),
            )
        if risk_pct <= 0:
            risk_pct = 0.03

        r_multiple = float(vsig.get("r_multiple", 1.2) or 1.2)
        if signal_type == "BUY":
            stop_price = fill_price * (1.0 - risk_pct)
            take_profit = fill_price * (1.0 + risk_pct * r_multiple)
        else:
            stop_price = fill_price * (1.0 + risk_pct)
            take_profit = fill_price * (1.0 - risk_pct * r_multiple)

        return {
            "risk_pct": risk_pct,
            "r_multiple": r_multiple,
            "stop_loss_price": round(stop_price, 4),
            "take_profit_price": round(take_profit, 4),
            "stop_loss": round(stop_price, 4),
            "take_profit": round(take_profit, 4),
        }

    def _execute_ibkr_signal(
        self, symbol: str, vsig: dict, portfolio: dict, stop_orders: dict, mode: str
    ) -> None:
        # P3 — Greeks enforcement (institutional only)
        level = self._greeks_enforcement_level(mode, vsig)
        if level >= 2:
            self.logger.warning(f"[{mode}] Greeks violation (level={level}) — skipping {symbol}")
            return
        if level == 1:
            self.logger.info(f"[{mode}] Greeks warning for {symbol} (level=1) — proceeding")

        signal_type = vsig["signal_type"]
        quantity    = float(vsig["quantity"])
        allow_short = (
            bool(self.config.get(mode, {}).get("allow_short", False))
            and bool(self.config.get("ibkr", {}).get("allow_short_orders", False))
        )

        existing_pos = portfolio["positions"].get(symbol, {})
        existing_qty = float(existing_pos.get("quantity", 0))
        existing_side = existing_pos.get("side", "long")
        desired_side = "long" if signal_type == "BUY" else "short"

        allow_trade, block_reason = self._turnover_gate(
            mode=mode,
            symbol=symbol,
            vsig=vsig,
            portfolio=portfolio,
            existing_pos=existing_pos,
            desired_side=desired_side,
        )
        if not allow_trade:
            self.logger.info(f"[{mode}] Turnover gate blocked {symbol}: {block_reason}")
            return

        if existing_qty > 0 and existing_side == desired_side:
            self.logger.debug(f"[{mode}] Already {desired_side} {symbol} — skipping duplicate {signal_type}")
            return

        if existing_qty > 0 and existing_side != desired_side:
            current_price = float(vsig.get("entry_price", existing_pos.get("current_price", 0)))
            self.logger.info(f"[{mode}] {signal_type} signal → reversing {symbol} from {existing_side} to {desired_side}")
            self._close_ibkr_position(symbol, current_price, f"SIGNAL_{signal_type}", portfolio, stop_orders, mode)
            existing_pos = portfolio["positions"].get(symbol, {})
            existing_qty = float(existing_pos.get("quantity", 0))

        if signal_type == "SELL" and existing_qty <= 0 and not allow_short:
            self.logger.debug(f"[{mode}] SELL signal for {symbol} with shorting disabled — skipped")
            return

        if quantity <= 0:
            return

        if symbol in stop_orders:
            existing_exit_orders = self._normalise_exit_orders(stop_orders[symbol])
            try:
                for order_id in (
                    existing_exit_orders["stop_order_id"],
                    existing_exit_orders["take_profit_order_id"],
                ):
                    if order_id is not None:
                        self._ibkr.cancel_order(order_id)
                del stop_orders[symbol]
            except Exception as exc:
                self.logger.warning(f"[{mode}] Could not cancel old exits for {symbol}: {exc}")

        cash_qty = float(vsig.get("position_size_usdt", 0.0))
        try:
            fill = self._ibkr.place_market_order(symbol, signal_type, quantity, cash_qty=cash_qty)
        except Exception as exc:
            self.logger.error(f"[{mode}] IBKR order failed for {symbol}: {exc}")
            return

        if fill["status"] != "filled" or fill["fill_price"] == 0.0:
            self.logger.warning(f"[{mode}] Order for {symbol} not filled (status={fill['status']})")
            return

        fill_price = fill["fill_price"]
        fill_quantity = float(fill.get("quantity") or quantity)
        commission = fill["commission"]
        exit_levels = self._build_exit_levels(signal_type, fill_price, vsig)

        trade = {
            "id":            str(uuid.uuid4()),
            "symbol":        symbol,
            "side":          signal_type,
            "quantity":      round(fill_quantity, 6),
            "fill_price":    round(fill_price, 4),
            "notional":      round(fill_price * fill_quantity, 4),
            "commission":    round(commission, 6),
            "stop_loss_price": exit_levels["stop_loss_price"],
            "take_profit_price": exit_levels["take_profit_price"],
            "stop_loss":     exit_levels["stop_loss"],
            "take_profit":   exit_levels["take_profit"],
            "r_multiple":    exit_levels["r_multiple"],
            "timestamp":     self._now_iso(),
            "reason":        f"SIGNAL_{signal_type}",
            "source":        "ibkr",
            "agent_source":  vsig.get("agent_source"),
            "strategy_bucket": vsig.get("strategy_bucket"),
            "ibkr_order_id": fill["order_id"],
            "mode":          mode,
        }
        portfolio.setdefault("trades", []).append(trade)

        pos = portfolio["positions"].setdefault(
            symbol, self._empty_position(symbol, fill_price, signal_type)
        )
        old_qty   = float(pos.get("quantity", 0))
        old_entry = float(pos.get("avg_entry_price", fill_price))
        new_qty   = old_qty + fill_quantity
        pos["side"]               = desired_side
        pos["avg_entry_price"]    = round((old_qty * old_entry + fill_quantity * fill_price) / new_qty if new_qty > 0 else fill_price, 6)
        pos["quantity"]           = round(new_qty, 6)
        pos["stop_loss_price"]    = exit_levels["stop_loss_price"]
        pos["take_profit_price"]  = exit_levels["take_profit_price"]
        pos["stop_loss"]          = exit_levels["stop_loss"]
        pos["take_profit"]        = exit_levels["take_profit"]
        pos["r_multiple"]         = exit_levels["r_multiple"]
        pos["current_price"]      = round(fill_price, 4)
        pos["position_value_usdt"] = round(new_qty * fill_price, 4)
        pos["agent_source"]       = vsig.get("agent_source", pos.get("agent_source"))
        pos["strategy_bucket"]    = vsig.get("strategy_bucket", pos.get("strategy_bucket"))
        # P1 — store exit params only on the initial fill
        if old_qty == 0:
            pos["entry_atr"]         = round(float(vsig.get("atr", fill_price * 0.02)), 6)
            pos["trailing_high"]     = round(fill_price, 4)

        exit_action = "SELL" if signal_type == "BUY" else "BUY"
        try:
            stop_orders[symbol] = self._ibkr.place_oco_exit_orders(
                symbol,
                exit_action,
                fill_quantity,
                exit_levels["take_profit_price"],
                exit_levels["stop_loss_price"],
            )
            self.logger.info(
                f"[{mode}] IBKR FILLED {signal_type} {symbol}: "
                f"qty={fill_quantity} @ {fill_price:.4f} | "
                f"stop={exit_levels['stop_loss_price']:.4f} | "
                f"tp={exit_levels['take_profit_price']:.4f}"
            )
        except Exception as exc:
            self.logger.error(
                f"[{mode}] Could not place OCO exits for {symbol}: {exc}"
            )

    # ================================================================== #
    # Simulation fallback                                                  #
    # ================================================================== #

    def _fetch_missing_prices(self, symbols: list[str], assets: dict) -> dict[str, float]:
        """Bulk-fetch live prices for symbols absent from market_data (equities, yfinance-format crypto)."""
        missing = [s for s in symbols if s not in assets]
        prices: dict[str, float] = {}
        if not missing:
            return prices
        # During backtest: skip yfinance entirely — use last_price from snapshot or entry price
        ctx = self.read_json(DATA_DIR / "backtest_context.json")
        if ctx.get("enabled"):
            return prices
        try:
            import yfinance as yf
            tickers = " ".join(missing)
            df = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                return prices
            close = df["Close"] if "Close" in df.columns else df.xs("Close", axis=1, level=0) if isinstance(df.columns, type(df.columns)) and df.columns.nlevels > 1 else df
            if hasattr(close, "columns"):
                for sym in missing:
                    if sym in close.columns:
                        val = close[sym].dropna()
                        if not val.empty:
                            prices[sym] = float(val.iloc[-1])
            else:
                # single ticker — close is a Series
                val = close.dropna()
                if not val.empty and len(missing) == 1:
                    prices[missing[0]] = float(val.iloc[-1])
        except Exception as exc:
            self.logger.warning(f"yfinance bulk price fetch failed: {exc}")
        return prices

    def _update_sim_positions(self, portfolio: dict, market_doc: dict) -> None:
        assets   = market_doc.get("assets", {})
        to_close = []

        # One bulk yfinance call for all open positions not in market_data (equities / yfinance crypto)
        open_symbols = [
            sym for sym, pos in portfolio.get("positions", {}).items()
            if pos.get("quantity", 0) > 0
        ]
        live_prices = self._fetch_missing_prices(open_symbols, assets)
        if live_prices:
            self.logger.info(f"yfinance price refresh: {list(live_prices.keys())}")

        for symbol, pos in portfolio.get("positions", {}).items():
            if pos.get("quantity", 0) <= 0:
                continue
            # Priority: Binance market_data → yfinance live → last known price
            if symbol in assets:
                current_price = float(assets[symbol].get("last_price", pos.get("avg_entry_price", 0)))
            elif symbol in live_prices:
                current_price = live_prices[symbol]
            else:
                current_price = float(pos.get("current_price") or pos.get("avg_entry_price", 0))
            pos["current_price"] = round(current_price, 4)
            # Refresh VWAP from latest market data (ISSUE-001)
            fresh_vwap = assets.get(symbol, {}).get("vwap")
            if fresh_vwap:
                pos["vwap_at_entry"] = fresh_vwap
            qty   = float(pos["quantity"])
            entry = float(pos["avg_entry_price"])
            side  = pos.get("side", "long")
            pnl   = ((current_price - entry) if side == "long" else (entry - current_price)) * qty
            pos["unrealized_pnl"]      = round(pnl, 4)
            pos["unrealized_pnl_pct"]  = round(pnl / (entry * qty) if entry * qty > 0 else 0, 6)
            pos["position_value_usdt"] = round(current_price * qty, 4)

            stop = pos.get("stop_loss_price", 0)
            if stop and stop > 0:
                triggered = (side == "long" and current_price <= stop) or (
                    side == "short" and current_price >= stop
                )
                if triggered:
                    # Hold period minimo: non stoppare prima di min_holding_hours
                    # Questo evita che il rumore a breve termine chiuda posizioni valide
                    opened_at = pos.get("opened_at")
                    min_hold_h = float(
                        self._turnover_cfg.get("min_holding_hours", 24.0)
                    )
                    hours_open = self._hours_since(opened_at, self._now()) if opened_at else None
                    if hours_open is not None and hours_open < min_hold_h:
                        # Ancora nel hold period: blocca lo stop a meno che la perdita
                        # non sia estrema (>2× ATR = segnale di crollo vero)
                        atr = float(pos.get("entry_atr", entry * 0.02) or entry * 0.02)
                        extreme_loss = abs(current_price - entry) > 4.0 * atr
                        if not extreme_loss:
                            continue  # non chiudere, lascia respirare la posizione
                    to_close.append((symbol, current_price, "SL_HIT"))

        for symbol, price, reason in to_close:
            self._close_sim_position(symbol, price, reason, portfolio)

    def _simulate_fill(self, symbol: str, vsig: dict, portfolio: dict, mode: str) -> None:
        # P3 — Greeks enforcement (institutional only)
        level = self._greeks_enforcement_level(mode, vsig)
        if level >= 2:
            self.logger.warning(f"[{mode}] Greeks violation (level={level}) — skipping {symbol}")
            return
        if level == 1:
            self.logger.info(f"[{mode}] Greeks warning for {symbol} (level=1) — proceeding")

        signal_type   = vsig["signal_type"]
        entry_price   = float(vsig["entry_price"])
        quantity      = float(vsig["quantity"])
        size_usdt     = float(vsig["position_size_usdt"])
        allow_short   = bool(self.config.get(mode, {}).get("allow_short", False))

        existing_pos  = portfolio["positions"].get(symbol, {})
        existing_qty  = float(existing_pos.get("quantity", 0))
        existing_side = existing_pos.get("side", "long")
        desired_side  = "long" if signal_type == "BUY" else "short"

        allow_trade, block_reason = self._turnover_gate(
            mode=mode,
            symbol=symbol,
            vsig=vsig,
            portfolio=portfolio,
            existing_pos=existing_pos,
            desired_side=desired_side,
        )
        if not allow_trade:
            self.logger.info(f"[{mode}] Turnover gate blocked {symbol}: {block_reason}")
            return

        if existing_qty > 0 and existing_side == desired_side:
            self.logger.debug(f"[{mode}][sim] Already {desired_side} {symbol} — skipping duplicate {signal_type}")
            return

        if existing_qty > 0 and existing_side != desired_side:
            self.logger.info(f"[{mode}][sim] {signal_type} signal → reversing {symbol} from {existing_side} to {desired_side}")
            self._close_sim_position(symbol, entry_price, f"SIGNAL_{signal_type}", portfolio)
            existing_pos  = portfolio["positions"].get(symbol, {})
            existing_qty  = float(existing_pos.get("quantity", 0))

        if signal_type == "SELL" and existing_qty <= 0 and not allow_short:
            self.logger.debug(f"[{mode}][sim] SELL signal for {symbol} with shorting disabled — skipped")
            return

        if quantity <= 0 or size_usdt <= 0:
            return

        sim_params = self._sim_params()
        slip      = sim_params["slippage_pct"]
        comm_rate = sim_params["commission_pct"]
        side = "long" if signal_type == "BUY" else "short"
        fill_price = entry_price * (1 + slip if signal_type == "BUY" else 1 - slip)
        notional   = fill_price * quantity
        commission = notional * comm_rate
        cash_delta = self._open_cash_delta(side, notional, commission)
        exit_levels = self._build_exit_levels(signal_type, fill_price, vsig)

        if side == "long":
            cash_needed = abs(cash_delta)
            if portfolio["cash"] < cash_needed:
                self.logger.warning(
                    f"[{mode}][sim] Insufficient cash for {symbol}: "
                    f"need {cash_needed:.2f}, have {portfolio['cash']:.2f}"
                )
                return

        portfolio["cash"] = round(portfolio["cash"] + cash_delta, 4)

        pos = portfolio["positions"].setdefault(
            symbol, self._empty_position(symbol, fill_price, signal_type)
        )
        old_qty   = float(pos.get("quantity", 0))
        old_entry = float(pos.get("avg_entry_price", fill_price))
        new_qty   = old_qty + quantity
        pos["side"]               = side
        pos["avg_entry_price"]    = round((old_qty * old_entry + quantity * fill_price) / new_qty if new_qty > 0 else fill_price, 4)
        pos["quantity"]           = round(new_qty, 6)
        pos["stop_loss_price"]    = exit_levels["stop_loss_price"]
        pos["take_profit_price"]  = exit_levels["take_profit_price"]
        pos["stop_loss"]          = exit_levels["stop_loss"]
        pos["take_profit"]        = exit_levels["take_profit"]
        pos["r_multiple"]         = exit_levels["r_multiple"]
        pos["current_price"]      = round(fill_price, 4)
        pos["position_value_usdt"] = round(new_qty * fill_price, 4)
        pos["agent_source"]       = vsig.get("agent_source", pos.get("agent_source"))
        pos["strategy_bucket"]    = vsig.get("strategy_bucket", pos.get("strategy_bucket"))
        # P1 — store exit params only on the initial fill (old_qty == 0)
        if old_qty == 0:
            pos["entry_atr"]        = round(float(vsig.get("atr", fill_price * 0.02)), 6)
            pos["trailing_high"]    = round(fill_price, 4)
            pos["vwap_at_entry"]    = vsig.get("vwap")  # VWAP trailing stop (ISSUE-001)

        portfolio.setdefault("trades", []).append({
            "id":            str(uuid.uuid4()),
            "symbol":        symbol,
            "side":          signal_type,
            "quantity":      round(quantity, 6),
            "fill_price":    round(fill_price, 4),
            "notional":      round(notional, 4),
            "commission":    round(commission, 4),
            "stop_loss_price": exit_levels["stop_loss_price"],
            "take_profit_price": exit_levels["take_profit_price"],
            "stop_loss":     exit_levels["stop_loss"],
            "take_profit":   exit_levels["take_profit"],
            "r_multiple":    exit_levels["r_multiple"],
            "timestamp":     self._now_iso(),
            "reason":        f"SIGNAL_{signal_type}",
            "source":        "simulation",
            "agent_source":  vsig.get("agent_source"),
            "strategy_bucket": vsig.get("strategy_bucket"),
            "mode":          mode,
        })
        self.logger.info(
            f"[{mode}][sim] FILLED {signal_type} {symbol} ({side}): "
            f"qty={quantity:.6f} @ {fill_price:.4f} | notional={notional:.2f}"
        )

    def _close_sim_position(
        self, symbol: str, price: float, reason: str, portfolio: dict
    ) -> None:
        pos = portfolio["positions"].get(symbol)
        if not pos or pos.get("quantity", 0) <= 0:
            return

        qty       = float(pos["quantity"])
        sim_params = self._sim_params()
        slip      = sim_params["slippage_pct"]
        comm_rate = sim_params["commission_pct"]
        side      = pos.get("side", "long")
        fill_price = price * (1 - slip if side == "long" else 1 + slip)
        notional   = fill_price * qty
        commission = notional * comm_rate
        entry      = float(pos["avg_entry_price"])
        realized   = ((fill_price - entry) if side == "long" else (entry - fill_price)) * qty - commission

        portfolio["cash"] = round(
            portfolio["cash"] + self._close_cash_delta(side, notional, commission),
            4,
        )
        self._record_close_trade(symbol, fill_price, pos, portfolio, reason, commission)
        portfolio["positions"][symbol] = self._zeroed_position(pos, fill_price, side)
        self.logger.info(
            f"[sim] CLOSED {symbol} ({reason}): qty={qty:.6f} @ {fill_price:.4f} | pnl={realized:.4f}"
        )

    # ================================================================== #
    # P1 — Exit logic: take-profit & trailing stop                        #
    # ================================================================== #

    def _check_exits(
        self,
        mode: str,
        portfolio: dict,
        market_doc: dict,
        stop_orders: dict,
        ibkr_live: bool,
    ) -> None:
        """Evaluate open positions for take-profit and trailing stop exits.
        Stop-loss is handled separately (IBKR stop orders / _update_sim_positions).
        Exit priority: TRAILING_STOP → TAKE_PROFIT.
        """
        assets  = market_doc.get("assets", {})
        to_exit = []

        for symbol, pos in portfolio.get("positions", {}).items():
            if pos.get("quantity", 0) <= 0:
                continue

            current_price = float(assets.get(symbol, {}).get("last_price", pos.get("current_price", 0)))
            if current_price <= 0:
                continue

            entry = float(pos.get("avg_entry_price", 0))
            if entry <= 0:
                continue

            side      = pos.get("side", "long")
            tp = float(pos.get("take_profit_price", 0))
            if tp <= 0:
                risk_pct = self._risk_pct_from_stop(
                    entry,
                    float(pos.get("stop_loss_price", 0) or 0),
                )
                if risk_pct <= 0:
                    continue
                r_multiple = float(pos.get("r_multiple", 1.2) or 1.2)
                if side == "long":
                    tp = entry * (1.0 + risk_pct * r_multiple)
                else:
                    tp = entry * (1.0 - risk_pct * r_multiple)
                pos["take_profit_price"] = round(tp, 4)

            # VWAP trailing stop (ISSUE-001 — Beat-the-Market paper §3)
            # Trail = max(SL_price, VWAP) for long; min(SL_price, VWAP) for short
            # Activates as soon as position is open (no gain threshold required)
            vwap = assets.get(symbol, {}).get("vwap") or pos.get("vwap_at_entry")
            sl_price = float(pos.get("stop_loss_price", 0) or 0)
            if vwap and vwap > 0 and sl_price > 0:
                # VWAP_TRAIL only activates after minimum holding period (24h)
                opened_at = pos.get("opened_at") or pos.get("timestamp")
                hours_held = self._hours_since(opened_at, self._now()) if opened_at else 0
                if hours_held is not None and hours_held >= 24:
                    if side == "long":
                        vwap_trail = max(sl_price, vwap * 0.95)  # 5% buffer below VWAP
                        if current_price < vwap_trail:
                            to_exit.append((symbol, current_price, "VWAP_TRAIL"))
                            continue
                    else:
                        vwap_trail = min(sl_price, vwap * 1.05)
                        if current_price > vwap_trail:
                            to_exit.append((symbol, current_price, "VWAP_TRAIL"))
                            continue
            else:
                # Fallback: fixed trailing stop — activates only after +2% gain
                gain_pct = ((current_price - entry) / entry) if side == "long" else ((entry - current_price) / entry)
                if gain_pct >= 0.02:
                    trail_high = float(pos.get("trailing_high", current_price))
                    if side == "long":
                        new_high = max(trail_high, current_price)
                    else:
                        new_high = min(trail_high, current_price)
                    pos["trailing_high"] = round(new_high, 4)
                    trail_stop = new_high * (1 - 0.025) if side == "long" else new_high * (1 + 0.025)
                    triggered = (side == "long" and current_price <= trail_stop) or \
                                (side == "short" and current_price >= trail_stop)
                    if triggered:
                        to_exit.append((symbol, current_price, "TRAILING_STOP"))
                        continue

            # Take-profit check
            if (side == "long" and current_price >= tp) or (side == "short" and current_price <= tp):
                to_exit.append((symbol, tp, "TP_HIT"))

        for symbol, price, reason in to_exit:
            self.logger.info(f"[{mode}] EXIT → {symbol} @ {price:.4f} ({reason})")
            if ibkr_live:
                self._close_ibkr_position(symbol, price, reason, portfolio, stop_orders, mode)
            else:
                self._close_sim_position(symbol, price, reason, portfolio)

    def _get_current_atr(self, symbol: str, market_doc: dict, fallback: float) -> float:
        """Estimate 14-period ATR from 4h OHLCV candles in market_doc."""
        ohlcv = market_doc.get("assets", {}).get(symbol, {}).get("ohlcv_4h", [])
        if len(ohlcv) < 2:
            return fallback
        candles = ohlcv[-15:]
        trs = [
            max(
                candles[i]["h"] - candles[i]["l"],
                abs(candles[i]["h"] - candles[i - 1]["c"]),
                abs(candles[i]["l"] - candles[i - 1]["c"]),
            )
            for i in range(1, len(candles))
        ]
        return sum(trs) / len(trs) if trs else fallback

    def _close_ibkr_position(
        self,
        symbol: str,
        price: float,
        reason: str,
        portfolio: dict,
        stop_orders: dict,
        mode: str,
    ) -> None:
        pos = portfolio["positions"].get(symbol)
        if not pos or pos.get("quantity", 0) <= 0:
            return
        qty  = float(pos["quantity"])
        side = pos.get("side", "long")
        close_action = "SELL" if side == "long" else "BUY"
        cash_qty = round(qty * price, 2)

        # Cancel the standing exit orders first
        if symbol in stop_orders:
            exit_orders = self._normalise_exit_orders(stop_orders[symbol])
            try:
                for order_id in (
                    exit_orders["stop_order_id"],
                    exit_orders["take_profit_order_id"],
                ):
                    if order_id is not None:
                        self._ibkr.cancel_order(order_id)
                del stop_orders[symbol]
            except Exception as exc:
                self.logger.warning(f"[{mode}] Could not cancel exit orders for {symbol}: {exc}")

        try:
            fill = self._ibkr.place_market_order(symbol, close_action, qty, cash_qty=cash_qty)
        except Exception as exc:
            self.logger.error(f"[{mode}] IBKR close failed for {symbol}: {exc}")
            return

        close_price = (
            fill["fill_price"]
            if fill.get("status") == "filled" and fill.get("fill_price", 0) > 0
            else price
        )
        commission = float(fill.get("commission", 0.0))
        entry      = float(pos.get("avg_entry_price", close_price))
        realized   = ((close_price - entry) if side == "long" else (entry - close_price)) * qty - commission
        self._record_close_trade(symbol, close_price, pos, portfolio, reason, commission)
        portfolio["positions"][symbol] = self._zeroed_position(pos, close_price, side)
        self.logger.info(
            f"[{mode}] IBKR CLOSED {symbol} ({reason}): "
            f"qty={qty:.6f} @ {close_price:.4f} | pnl={realized:.4f}"
        )

    # ================================================================== #
    # P2 — Daily circuit-breaker equity reset                             #
    # ================================================================== #

    def _maybe_reset_daily_equity(self, portfolio: dict) -> None:
        """Reset daily_start_equity to current equity when the UTC date rolls over."""
        today = self._now().date().isoformat()
        if portfolio.get("daily_reset_date", "") != today:
            portfolio["daily_start_equity"] = portfolio.get("total_equity", portfolio.get("cash", 0))
            portfolio["daily_reset_date"]   = today
            self.logger.info(f"Daily equity reset → {portfolio['daily_start_equity']:.2f}")

    # ================================================================== #
    # P3 — Greeks enforcement (institutional only)                        #
    # ================================================================== #

    def _greeks_enforcement_level(self, mode: str, vsig: dict) -> int:
        """Return violation count: 0=ok, 1=warn+proceed, 2+=skip entry."""
        if mode != "institutional":
            return 0
        limits = self.config.get("institutional", {}).get("greeks_limits", {})
        if not limits:
            return 0

        violations = 0
        if abs(float(vsig.get("delta", 0))) > limits.get("delta_max", 1.0):
            violations += 1
        if abs(float(vsig.get("gamma", 0))) > limits.get("gamma_max", 1.0):
            violations += 1
        if abs(float(vsig.get("vega", 0))) > limits.get("vega_max", 1.0):
            violations += 1
        if float(vsig.get("theta", 0)) < limits.get("theta_min", -1.0):
            violations += 1
        return violations

    @staticmethod
    def _position_market_contribution(side: str, quantity: float, current_price: float) -> float:
        notional = current_price * quantity
        return notional if side == "long" else -notional

    @staticmethod
    def _open_cash_delta(side: str, notional: float, commission: float) -> float:
        return -(notional + commission) if side == "long" else (notional - commission)

    @staticmethod
    def _close_cash_delta(side: str, notional: float, commission: float) -> float:
        return (notional - commission) if side == "long" else -(notional + commission)

    # ================================================================== #
    # Shared helpers                                                       #
    # ================================================================== #

    def _recalculate_equity(self, portfolio: dict, market_doc: dict) -> None:
        assets = market_doc.get("assets", {})
        pos_value = sum(
            self._position_market_contribution(
                pos.get("side", "long"),
                float(pos["quantity"]),
                float(assets.get(sym, {}).get("last_price", pos.get("current_price", 0))),
            )
            for sym, pos in portfolio.get("positions", {}).items()
            if pos.get("quantity", 0) > 0
        )
        portfolio["total_equity"] = round(portfolio["cash"] + pos_value, 4)

    def _update_drawdown(self, portfolio: dict, mode: str) -> None:
        total   = portfolio["total_equity"]
        peak    = float(portfolio.get("peak_equity", total))
        if total > peak:
            peak = total
        portfolio["peak_equity"]    = round(peak, 4)
        portfolio["drawdown_pct"]   = round((peak - total) / peak, 6) if peak > 0 else 0.0

        initial = float(self.config[mode]["capital"])
        portfolio["total_pnl"]      = round(total - initial, 4)
        portfolio["total_pnl_pct"]  = round((total - initial) / initial, 6) if initial > 0 else 0.0

    def _record_close_trade(
        self, symbol: str, fill_price: float, pos: dict,
        portfolio: dict, reason: str, commission: float = 0.0
    ) -> None:
        qty   = float(pos.get("quantity", 0))
        side  = pos.get("side", "long")
        entry = float(pos.get("avg_entry_price", fill_price))
        realized = ((fill_price - entry) if side == "long" else (entry - fill_price)) * qty - commission
        portfolio.setdefault("trades", []).append({
            "id":           str(uuid.uuid4()),
            "symbol":       symbol,
            "side":         "SELL" if side == "long" else "BUY",
            "quantity":     round(qty, 6),
            "fill_price":   round(fill_price, 4),
            "notional":     round(fill_price * qty, 4),
            "commission":   round(commission, 6),
            "realized_pnl": round(realized, 4),
            "timestamp":    self._now_iso(),
            "reason":       reason,
            "agent_source": pos.get("agent_source"),
            "strategy_bucket": pos.get("strategy_bucket"),
        })
        portfolio["realized_pnl"] = round(portfolio.get("realized_pnl", 0.0) + realized, 4)

    @staticmethod
    def _empty_position(symbol: str, price: float, signal_type: str) -> dict:
        return {
            "quantity": 0.0,
            "avg_entry_price": 0.0,
            "current_price": price,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "position_value_usdt": 0.0,
            "stop_loss_price": 0.0,
            "take_profit_price": 0.0,
            "r_multiple": 0.0,
            "trailing_high": price,
            "entry_atr": 0.0,
            "side": "long" if signal_type == "BUY" else "short",
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _zeroed_position(pos: dict, fill_price: float, side: str) -> dict:
        return {
            "quantity": 0.0,
            "avg_entry_price": 0.0,
            "current_price": fill_price,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "position_value_usdt": 0.0,
            "stop_loss_price": 0.0,
            "take_profit_price": 0.0,
            "r_multiple": 0.0,
            "side": side,
            "opened_at": pos.get("opened_at"),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ================================================================== #
    # Alpha Hunter — US equity simulation portfolio                      #
    # ================================================================== #

    def _init_alpha_portfolio(self) -> dict:
        capital = float(self.config.get("alpha_hunter", {}).get("capital", 100_000.0))
        return {
            "mode":               "alpha",
            "timestamp":          self._now_iso(),
            "cash":               capital,
            "total_equity":       capital,
            "peak_equity":        capital,
            "daily_start_equity": capital,
            "drawdown_pct":       0.0,
            "initial_capital":    capital,
            "realized_pnl":       0.0,
            "total_pnl":          0.0,
            "total_pnl_pct":      0.0,
            "execution_mode":     "simulation",
            "positions":          {},
            "trades":             [],
        }

    def _run_alpha_cycle(self, signals_doc: dict, portfolio: dict) -> None:
        """
        Inline risk validation + simulation execution for the Alpha Hunter portfolio.
        No IBKR — all trades are simulated.
        """
        cfg    = self.config.get("alpha_hunter", {})
        cap    = float(cfg.get("capital", 100_000.0))
        max_pos = int(cfg.get("max_open_trades", 5))
        stake  = float(cfg.get("stake_pct", 0.20))
        res    = float(cfg.get("cash_reserve_pct", 0.10))
        min_sz = float(cfg.get("min_trade_usd", 500.0))
        thr    = float(cfg.get("signal_threshold", 0.58))
        max_dd = float(cfg.get("max_drawdown_pct", 0.10))
        max_dl = float(cfg.get("max_daily_loss_pct", 0.04))
        sl_pct = float(cfg.get("stop_loss_pct", 0.03))
        tp_pct = float(cfg.get("take_profit_pct", 0.08))
        slip   = float(cfg.get("slippage_pct", 0.0002))
        fee    = float(cfg.get("fee_taker", 0.0005))
        trail_pct  = float(cfg.get("trailing_stop_pct", 0.02))
        trail_act  = float(cfg.get("trailing_stop_activation_pct", 0.015))
        allow_short = bool(cfg.get("allow_short", False))
        top_candidates = int(cfg.get("top_candidates", max_pos))

        # ── P2: daily reset ──────────────────────────────────────────── #
        self._maybe_reset_daily_equity(portfolio)

        # ── Circuit breakers ─────────────────────────────────────────── #
        dd = float(portfolio.get("drawdown_pct", 0.0))
        if dd >= max_dd:
            self.logger.warning(
                f"[alpha] Max drawdown {dd*100:.1f}% reached — blocking new entries"
            )
            self._alpha_update_positions(portfolio, signals_doc)
            self._alpha_recalculate_equity(portfolio)
            self._alpha_update_drawdown(portfolio, cap)
            return

        daily_eq  = float(portfolio.get("daily_start_equity", cap))
        total_eq  = float(portfolio.get("total_equity", cap))
        daily_loss = (daily_eq - total_eq) / daily_eq if daily_eq > 0 else 0.0
        if daily_loss >= max_dl:
            self.logger.warning(
                f"[alpha] Daily loss {daily_loss*100:.1f}% reached — blocking new entries"
            )
            self._alpha_update_positions(portfolio, signals_doc)
            self._alpha_recalculate_equity(portfolio)
            self._alpha_update_drawdown(portfolio, cap)
            return

        # ── P1: check exits on existing positions ────────────────────── #
        self._alpha_check_exits(portfolio, signals_doc, sl_pct, tp_pct, trail_pct, trail_act, slip, fee)

        # ── Count open positions ──────────────────────────────────────── #
        open_count = sum(
            1 for p in portfolio.get("positions", {}).values()
            if p.get("quantity", 0) > 0
        )

        raw_signals = signals_doc.get("signals", {})
        selected_symbols = self._alpha_selected_symbols(
            signals_doc, raw_signals, thr, top_candidates
        )

        # ── Process BUY signals ──────────────────────────────────────── #
        for symbol, sig in sorted(
            raw_signals.items(),
            key=lambda kv: kv[1].get("buy_score", 0),
            reverse=True,
        ):
            if sig.get("signal_type") != "BUY":
                continue
            if sig.get("score", 0) < thr:
                continue
            existing = portfolio["positions"].get(symbol, {})
            if existing.get("quantity", 0) > 0:
                existing_side = existing.get("side", "long")
                if existing_side == "long":
                    continue
                exit_price = float(sig.get("last_price", existing.get("current_price", 0)))
                if exit_price <= 0:
                    continue
                close_fill = exit_price * (1.0 + slip)
                self._alpha_close(symbol, close_fill, "SIGNAL_BUY", portfolio, fee)
                open_count = max(0, open_count - 1)

            if symbol not in selected_symbols:
                continue
            if open_count >= max_pos:
                break

            entry_price = float(sig.get("last_price", 0))
            if entry_price <= 0:
                continue

            if self._alpha_open_position(
                symbol=symbol,
                signal=sig,
                portfolio=portfolio,
                side="long",
                stake=stake,
                reserve_pct=res,
                min_trade_usd=min_sz,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                slippage_pct=slip,
                fee_rate=fee,
            ):
                open_count += 1

        # ── Process SELL signals (close longs and open shorts if enabled) ── #
        for symbol, sig in sorted(
            raw_signals.items(),
            key=lambda kv: kv[1].get("sell_score", 0),
            reverse=True,
        ):
            if sig.get("signal_type") != "SELL":
                continue
            if sig.get("score", 0) < thr:
                continue

            pos = portfolio["positions"].get(symbol, {})
            if pos.get("quantity", 0) > 0:
                if pos.get("side") == "short":
                    continue
                exit_price = float(sig.get("last_price", pos.get("current_price", 0)))
                if exit_price <= 0:
                    continue
                close_fill = exit_price * (1.0 - slip)
                self._alpha_close(symbol, close_fill, "SIGNAL_SELL", portfolio, fee)
                open_count = max(0, open_count - 1)

            if not allow_short:
                continue
            if symbol not in selected_symbols:
                continue
            if open_count >= max_pos:
                break

            if self._alpha_open_position(
                symbol=symbol,
                signal=sig,
                portfolio=portfolio,
                side="short",
                stake=stake,
                reserve_pct=res,
                min_trade_usd=min_sz,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                slippage_pct=slip,
                fee_rate=fee,
            ):
                open_count += 1

        # ── Update unrealised PnL from latest prices ─────────────────── #
        self._alpha_update_positions(portfolio, signals_doc)
        self._alpha_recalculate_equity(portfolio)
        self._alpha_update_drawdown(portfolio, cap)

    @staticmethod
    def _alpha_selected_symbols(
        signals_doc: dict,
        raw_signals: dict,
        threshold: float,
        limit: int,
    ) -> set[str]:
        limit = max(0, int(limit))
        scanner_doc = signals_doc.get("scanner", {})
        ranked_candidates = scanner_doc.get("top_candidates", [])
        if ranked_candidates:
            return {
                row.get("symbol")
                for row in ranked_candidates[:limit]
                if row.get("signal_type") in {"BUY", "SELL"} and row.get("symbol")
            }

        ranked: list[tuple[str, float, float]] = []
        for symbol, signal in raw_signals.items():
            signal_type = signal.get("signal_type", "HOLD")
            score = float(signal.get("score", 0.0))
            if signal_type not in {"BUY", "SELL"} or score < threshold:
                continue
            ranked.append(
                (
                    symbol,
                    score,
                    max(
                        float(signal.get("buy_score", 0.0)),
                        float(signal.get("sell_score", 0.0)),
                    ),
                )
            )

        ranked.sort(key=lambda row: (row[1], row[2]), reverse=True)
        return {symbol for symbol, _, _ in ranked[:limit]}

    def _alpha_open_position(
        self,
        symbol: str,
        signal: dict,
        portfolio: dict,
        side: str,
        stake: float,
        reserve_pct: float,
        min_trade_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        slippage_pct: float,
        fee_rate: float,
    ) -> bool:
        entry_price = float(signal.get("last_price", 0))
        if entry_price <= 0:
            return False

        available = portfolio["cash"] * (1.0 - reserve_pct)
        size_usd = min(available * stake, portfolio["cash"] - min_trade_usd)
        if size_usd < min_trade_usd:
            self.logger.debug(f"[alpha] Insufficient cash for {symbol}")
            return False

        fill = entry_price * (1.0 + slippage_pct if side == "long" else 1.0 - slippage_pct)
        qty = size_usd / fill
        notional = fill * qty
        commission = notional * fee_rate
        cash_delta = self._open_cash_delta(side, notional, commission)

        if side == "long" and portfolio["cash"] < abs(cash_delta):
            self.logger.debug(f"[alpha] Insufficient cash for {symbol}")
            return False

        stop_price = float(
            signal.get(
                "stop_loss",
                fill * (1 - stop_loss_pct if side == "long" else 1 + stop_loss_pct),
            )
        )
        take_profit = float(
            signal.get(
                "take_profit",
                fill * (1 + take_profit_pct if side == "long" else 1 - take_profit_pct),
            )
        )

        portfolio["cash"] = round(portfolio["cash"] + cash_delta, 4)
        pos = portfolio["positions"].setdefault(
            symbol, self._empty_position(symbol, fill, "BUY" if side == "long" else "SELL")
        )
        pos.update({
            "quantity": round(qty, 6),
            "avg_entry_price": round(fill, 4),
            "current_price": round(fill, 4),
            "stop_loss_price": round(stop_price, 4),
            "take_profit_price": round(take_profit, 4),
            "trailing_high": round(fill, 4),
            "entry_atr": round(float(signal.get("atr", fill * 0.02)), 6),
            "position_value_usdt": round(notional, 4),
            "unrealized_pnl": 0.0,
            "side": side,
        })

        trade_side = "BUY" if side == "long" else "SELL"
        portfolio.setdefault("trades", []).append({
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": trade_side,
            "quantity": round(qty, 6),
            "fill_price": round(fill, 4),
            "notional": round(notional, 4),
            "commission": round(commission, 4),
            "stop_loss_price": round(stop_price, 4),
            "take_profit_price": round(take_profit, 4),
            "score": signal.get("score", 0),
            "iv_hv_ratio": signal.get("iv_hv_ratio", 1.0),
            "rsi": signal.get("rsi", 50.0),
            "timestamp": self._now_iso(),
            "reason": f"SIGNAL_{trade_side}",
            "source": "alpha_simulation",
            "mode": "alpha",
        })
        self.logger.info(
            f"[alpha][sim] {trade_side} {symbol} ({side}): qty={qty:.4f} @ {fill:.2f} | "
            f"notional={notional:.0f} | stop={stop_price:.2f} | "
            f"score={signal.get('score', 0):.3f} iv/hv={signal.get('iv_hv_ratio', 1):.2f}"
        )
        return True

    # ── Alpha helper: update position prices from latest signals ─────── #

    def _alpha_update_positions(self, portfolio: dict, signals_doc: dict) -> None:
        raw = signals_doc.get("signals", {})
        for symbol, pos in portfolio.get("positions", {}).items():
            if pos.get("quantity", 0) <= 0:
                continue
            last = float(raw.get(symbol, {}).get("last_price", 0))
            if last <= 0:
                last = float(pos.get("current_price", pos.get("avg_entry_price", 0)))
            pos["current_price"] = round(last, 4)
            qty   = float(pos["quantity"])
            entry = float(pos["avg_entry_price"])
            side  = pos.get("side", "long")
            pnl   = ((last - entry) if side == "long" else (entry - last)) * qty
            pos["unrealized_pnl"]       = round(pnl, 4)
            pos["unrealized_pnl_pct"]   = round(pnl / (entry * qty), 6) if entry * qty > 0 else 0.0
            pos["position_value_usdt"]  = round(last * qty, 4)

    # ── Alpha helper: check exits (trailing stop + take-profit + stop) ── #

    def _alpha_check_exits(
        self, portfolio: dict, signals_doc: dict,
        sl_pct: float, tp_pct: float,
        trail_pct: float, trail_act: float,
        slip: float, fee: float,
    ) -> None:
        raw = signals_doc.get("signals", {})
        to_close: list[tuple[str, float, str]] = []

        for symbol, pos in portfolio.get("positions", {}).items():
            if pos.get("quantity", 0) <= 0:
                continue

            last = float(raw.get(symbol, {}).get("last_price", 0))
            if last <= 0:
                last = float(pos.get("current_price", 0))
            if last <= 0:
                continue

            entry  = float(pos.get("avg_entry_price", 0))
            side   = pos.get("side", "long")
            stop   = float(pos.get("stop_loss_price", 0))
            tp     = float(pos.get("take_profit_price", 0))

            # Hard stop
            if stop > 0 and side == "long" and last <= stop:
                to_close.append((symbol, last, "STOP_LOSS"))
                continue
            if stop > 0 and side == "short" and last >= stop:
                to_close.append((symbol, last, "STOP_LOSS"))
                continue

            # Take-profit
            if tp > 0:
                if side == "long"  and last >= tp:
                    to_close.append((symbol, last, "TAKE_PROFIT")); continue
                if side == "short" and last <= tp:
                    to_close.append((symbol, last, "TAKE_PROFIT")); continue

            # Trailing stop (activates after trail_act gain)
            if entry > 0:
                gain = ((last - entry) / entry) if side == "long" else ((entry - last) / entry)
                if gain >= trail_act:
                    high = float(pos.get("trailing_high", last))
                    if side == "long":
                        high = max(high, last)
                        ts   = high * (1.0 - trail_pct)
                        if last <= ts:
                            to_close.append((symbol, last, "TRAILING_STOP"))
                            continue
                    else:
                        high = min(high, last)
                        ts   = high * (1.0 + trail_pct)
                        if last >= ts:
                            to_close.append((symbol, last, "TRAILING_STOP"))
                            continue
                    pos["trailing_high"] = round(high, 4)

        for symbol, price, reason in to_close:
            self.logger.info(f"[alpha] EXIT {symbol} @ {price:.2f} ({reason})")
            side = portfolio.get("positions", {}).get(symbol, {}).get("side", "long")
            fill = price * (1.0 - slip if side == "long" else 1.0 + slip)
            self._alpha_close(symbol, fill, reason, portfolio, fee)

    def _alpha_close(
        self, symbol: str, fill_price: float, reason: str,
        portfolio: dict, fee: float,
    ) -> None:
        pos = portfolio["positions"].get(symbol)
        if not pos or pos.get("quantity", 0) <= 0:
            return
        qty       = float(pos["quantity"])
        side      = pos.get("side", "long")
        notional  = fill_price * qty
        commission = notional * fee
        entry      = float(pos.get("avg_entry_price", fill_price))
        realized   = ((fill_price - entry) if side == "long" else (entry - fill_price)) * qty - commission

        portfolio["cash"] = round(
            portfolio["cash"] + self._close_cash_delta(side, notional, commission),
            4,
        )
        portfolio["realized_pnl"] = round(portfolio.get("realized_pnl", 0.0) + realized, 4)
        portfolio.setdefault("trades", []).append({
            "id":           str(uuid.uuid4()),
            "symbol":       symbol,
            "side":         "SELL" if side == "long" else "BUY",
            "quantity":     round(qty, 6),
            "fill_price":   round(fill_price, 4),
            "notional":     round(notional, 4),
            "commission":   round(commission, 4),
            "realized_pnl": round(realized, 4),
            "timestamp":    self._now_iso(),
            "reason":       reason,
            "source":       "alpha_simulation",
            "mode":         "alpha",
        })
        portfolio["positions"][symbol] = self._zeroed_position(pos, fill_price, side)
        self.logger.info(
            f"[alpha][sim] CLOSED {symbol} ({reason}): "
            f"qty={qty:.4f} @ {fill_price:.2f} | realized_pnl={realized:.2f}"
        )

    def _alpha_recalculate_equity(self, portfolio: dict) -> None:
        pos_val = sum(
            self._position_market_contribution(
                p.get("side", "long"),
                float(p["quantity"]),
                float(p.get("current_price", p.get("avg_entry_price", 0))),
            )
            for p in portfolio.get("positions", {}).values()
            if p.get("quantity", 0) > 0
        )
        portfolio["total_equity"] = round(portfolio["cash"] + pos_val, 4)

    def _alpha_update_drawdown(self, portfolio: dict, initial_capital: float) -> None:
        total = portfolio["total_equity"]
        peak  = float(portfolio.get("peak_equity", total))
        if total > peak:
            peak = total
        portfolio["peak_equity"]   = round(peak, 4)
        portfolio["drawdown_pct"]  = round((peak - total) / peak, 6) if peak > 0 else 0.0
        portfolio["total_pnl"]     = round(total - initial_capital, 4)
        portfolio["total_pnl_pct"] = round((total - initial_capital) / initial_capital, 6) if initial_capital > 0 else 0.0

    def _set_account_metadata(self, portfolio: dict, mode: str, ibkr_live: bool) -> None:
        if ibkr_live:
            total_cfg = float(self.config["retail"]["capital"] + self.config["institutional"]["capital"])
            cfg_cap = float(self.config[mode]["capital"])
            portfolio["account_scope"] = "shared_ibkr_account"
            portfolio["equity_basis"] = "proportional_split_of_shared_ibkr_account"
            portfolio["configured_capital_share"] = round(cfg_cap / total_cfg, 6) if total_cfg > 0 else 0.0
            portfolio["allocation_note"] = (
                "Equity and cash are proportional allocations of one shared IBKR paper account. "
                "They are not isolated sub-account balances."
            )
            return

        portfolio["account_scope"] = "local_simulation"
        portfolio["equity_basis"] = "portfolio_mark_to_market"
        portfolio["configured_capital_share"] = 1.0
        portfolio["allocation_note"] = (
            "Equity and cash are computed from this portfolio's own simulated positions."
        )

    # ================================================================== #
    # Existing portfolio init                                             #
    # ================================================================== #

    def _init_portfolio(self, mode: str) -> dict:
        capital = float(self.config[mode]["capital"])
        return {
            "mode":               mode,
            "timestamp":          self._now_iso(),
            "cash":               capital,
            "total_equity":       capital,
            "peak_equity":        capital,
            "daily_start_equity": capital,
            "drawdown_pct":       0.0,
            "initial_capital":    capital,
            "realized_pnl":       0.0,
            "total_pnl":          0.0,
            "total_pnl_pct":      0.0,
            "execution_mode":     "simulation",
            "account_scope":      "local_simulation",
            "equity_basis":       "portfolio_mark_to_market",
            "configured_capital_share": 1.0,
            "allocation_note":    (
                "Equity and cash are computed from this portfolio's own simulated positions."
            ),
            "positions":          {},
            "trades":             [],
        }

    def _now(self) -> datetime:
        ctx = self.read_json(DATA_DIR / "backtest_context.json")
        if ctx.get("enabled") and ctx.get("simulated_now"):
            return datetime.fromisoformat(ctx["simulated_now"])
        return datetime.now(timezone.utc)

    def _now_iso(self) -> str:
        return self._now().isoformat()

    def _sim_params(self) -> dict:
        """Simulation cost model with optional backtest overrides."""
        ctx = self.read_json(DATA_DIR / "backtest_context.json")
        if ctx.get("enabled"):
            fees = ctx.get("fees", {})
            slip = ctx.get("slippage_model", {})
            slippage_bps = float(slip.get("default_bps", 10.0))
            return {
                "slippage_pct": slippage_bps / 10_000.0,
                "commission_pct": float(fees.get("commission_pct", self._sim.get("commission_pct", 0.001))),
            }
        return {
            "slippage_pct": float(self._sim.get("slippage_pct", 0.001)),
            "commission_pct": float(self._sim.get("commission_pct", 0.001)),
        }

    def _turnover_gate(
        self,
        mode: str,
        symbol: str,
        vsig: dict,
        portfolio: dict,
        existing_pos: dict,
        desired_side: str,
    ) -> tuple[bool, str | None]:
        cfg = self._turnover_cfg
        if not bool(cfg.get("enabled", True)):
            return True, None

        min_holding_hours = float(cfg.get("min_holding_hours", 24.0))
        cooldown_after_close_hours = float(cfg.get("cooldown_after_close_hours", 12.0))
        frequent_window_hours = float(cfg.get("frequent_trade_window_hours", 72.0))
        max_trades_per_symbol = int(cfg.get("max_trades_per_symbol", 4))
        rebalance_threshold_pct = float(cfg.get("rebalance_threshold_pct", 0.05))

        now = self._now()
        existing_qty = float(existing_pos.get("quantity", 0.0) or 0.0)
        existing_side = existing_pos.get("side", "long")

        # 1) Minimum holding period before reversal
        if existing_qty > 0 and existing_side != desired_side:
            opened_at = existing_pos.get("opened_at")
            hours_open = self._hours_since(opened_at, now)
            if hours_open is not None and hours_open < min_holding_hours:
                return False, f"min_holding_period_not_reached ({hours_open:.1f}h<{min_holding_hours:.1f}h)"

        # 2) Rebalance threshold for same-side updates
        if existing_qty > 0 and existing_side == desired_side:
            current_notional = abs(existing_qty * float(existing_pos.get("current_price", existing_pos.get("avg_entry_price", 0.0) or 0.0)))
            target_notional = abs(float(vsig.get("position_size_usdt", 0.0)))
            if current_notional > 0:
                diff_pct = abs(target_notional - current_notional) / current_notional
                if diff_pct < rebalance_threshold_pct:
                    return False, f"rebalance_threshold_not_met ({diff_pct:.3f}<{rebalance_threshold_pct:.3f})"

        trades = portfolio.get("trades", []) or []
        symbol_trades = [t for t in trades if t.get("symbol") == symbol and t.get("timestamp")]
        symbol_trades_sorted = sorted(symbol_trades, key=lambda t: str(t.get("timestamp")), reverse=True)

        # 3) Cooldown after last close
        for t in symbol_trades_sorted:
            reason = str(t.get("reason", ""))
            if reason.startswith("TAKE_PROFIT") or reason.startswith("STOP_LOSS") or reason.startswith("SIGNAL_"):
                hours_since_close = self._hours_since(t.get("timestamp"), now)
                if hours_since_close is not None and hours_since_close < cooldown_after_close_hours and existing_qty <= 0:
                    return False, f"cooldown_after_close ({hours_since_close:.1f}h<{cooldown_after_close_hours:.1f}h)"
                break

        # 4) Frequent-trade penalty
        recent_count = 0
        for t in symbol_trades_sorted:
            hours = self._hours_since(t.get("timestamp"), now)
            if hours is None:
                continue
            if hours <= frequent_window_hours:
                recent_count += 1
        if recent_count >= max_trades_per_symbol:
            return False, f"frequent_trade_penalty ({recent_count}>={max_trades_per_symbol} in {frequent_window_hours:.0f}h)"

        return True, None

    @staticmethod
    def _hours_since(ts: str | None, now: datetime) -> float | None:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts)
            delta = now - dt
            return max(0.0, delta.total_seconds() / 3600.0)
        except Exception:
            return None
