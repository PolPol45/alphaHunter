"""
Report Agent — Dual Portfolio
===============================
Generates a structured hourly JSON report covering both portfolios.

Reads:  data/portfolio_retail.json, data/portfolio_institutional.json,
        data/signals.json, data/validated_signals.json,
        data/market_data.json, shared_state.json
Writes: reports/report_YYYY-MM-DD_HH-MM-SS.json
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR, REPORTS_DIR, SHARED_STATE_PATH

MAX_REPORTS = 48


class ReportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("report_agent")
        self._last_report_trades_retail = 0
        self._last_report_trades_inst   = 0

    def run(self) -> bool:
        self.mark_running()
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)

            port_retail = self.read_json(DATA_DIR / "portfolio_retail.json")
            port_inst   = self.read_json(DATA_DIR / "portfolio_institutional.json")
            signals_doc = self.read_json(DATA_DIR / "signals.json")
            validated   = self.read_json(DATA_DIR / "validated_signals.json")
            market_doc  = self.read_json(DATA_DIR / "market_data.json")
            state       = self.read_json(SHARED_STATE_PATH)

            report = self._generate_report(
                port_retail, port_inst, signals_doc, validated, market_doc, state
            )

            ts          = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            report_path = REPORTS_DIR / f"report_{ts}.json"
            self.write_json(report_path, report)
            self._cleanup_old_reports()

            r_eq  = report["retail"]["portfolio_summary"]["total_equity"]
            r_pnl = report["retail"]["portfolio_summary"]["total_pnl_pct"]
            i_eq  = report["institutional"]["portfolio_summary"]["total_equity"]
            i_pnl = report["institutional"]["portfolio_summary"]["total_pnl_pct"]
            exec_mode = report["data_sources"].get("execution_mode", "?")

            self.logger.info(
                f"Report saved → {report_path.name} | "
                f"Retail={r_eq:.2f}({r_pnl*100:+.2f}%) | "
                f"Inst={i_eq:.2f}({i_pnl*100:+.2f}%) | exec={exec_mode}"
            )
            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    # ------------------------------------------------------------------ #
    # Report construction                                                  #
    # ------------------------------------------------------------------ #

    def _generate_report(
        self,
        port_retail: dict,
        port_inst:   dict,
        signals_doc: dict,
        validated:   dict,
        market_doc:  dict,
        state:       dict,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()

        # Combined equity
        r_eq = port_retail.get("total_equity", 0)
        i_eq = port_inst.get("total_equity", 0)

        return {
            "report_id":    str(uuid.uuid4()),
            "generated_at": now,
            "combined": {
                "total_equity":   round(r_eq + i_eq, 2),
                "retail_equity":  round(r_eq, 2),
                "inst_equity":    round(i_eq, 2),
            },
            "retail": {
                "portfolio_summary": self._portfolio_summary(port_retail),
                "open_positions":    self._open_positions(port_retail),
                "new_trades":        self._new_trades(port_retail, "retail"),
                "signal_summary":    self._signal_summary(
                    signals_doc.get("retail", {}), validated.get("retail", {})
                ),
            },
            "institutional": {
                "portfolio_summary": self._portfolio_summary(port_inst),
                "open_positions":    self._open_positions(port_inst),
                "new_trades":        self._new_trades(port_inst, "institutional"),
                "signal_summary":    self._signal_summary(
                    signals_doc.get("institutional", {}), validated.get("institutional", {})
                ),
            },
            "system_health": self._system_health(state),
            "data_sources":  self._data_sources(port_retail, port_inst, market_doc, state),
        }

    # ------------------------------------------------------------------ #
    # Section builders                                                     #
    # ------------------------------------------------------------------ #

    def _portfolio_summary(self, portfolio: dict) -> dict:
        return {
            "total_equity":    portfolio.get("total_equity",    0.0),
            "cash":            portfolio.get("cash",            0.0),
            "initial_capital": portfolio.get("initial_capital", 0.0),
            "realized_pnl":    portfolio.get("realized_pnl",    0.0),
            "total_pnl":       portfolio.get("total_pnl",       0.0),
            "total_pnl_pct":   portfolio.get("total_pnl_pct",   0.0),
            "peak_equity":     portfolio.get("peak_equity",     0.0),
            "drawdown_pct":    portfolio.get("drawdown_pct",    0.0),
            "account_scope":   portfolio.get("account_scope",   "unknown"),
            "equity_basis":    portfolio.get("equity_basis",    "unknown"),
            "configured_capital_share": portfolio.get("configured_capital_share"),
            "allocation_note": portfolio.get("allocation_note"),
        }

    def _open_positions(self, portfolio: dict) -> dict:
        return {
            sym: pos
            for sym, pos in portfolio.get("positions", {}).items()
            if pos.get("quantity", 0) > 0
        }

    def _new_trades(self, portfolio: dict, mode: str) -> list:
        all_trades = portfolio.get("trades", [])
        if mode == "retail":
            new = all_trades[self._last_report_trades_retail:]
            self._last_report_trades_retail = len(all_trades)
        else:
            new = all_trades[self._last_report_trades_inst:]
            self._last_report_trades_inst = len(all_trades)
        return new

    def _signal_summary(self, signals: dict, validated: dict) -> dict:
        non_hold = [s for s in signals.values() if s.get("signal_type") != "HOLD"]
        approved = [s for s in validated.values() if s.get("approved")]
        rejected = [s for s in validated.values() if not s.get("approved")]

        reasons: dict[str, int] = {}
        for sig in rejected:
            r = sig.get("rejection_reason", "UNKNOWN")
            reasons[r] = reasons.get(r, 0) + 1

        return {
            "signals_generated": len(non_hold),
            "signals_approved":  len(approved),
            "signals_rejected":  len(rejected),
            "rejection_reasons": reasons,
            "current_signals": {
                sym: {"type": sig.get("signal_type"), "score": sig.get("score")}
                for sym, sig in signals.items()
            },
        }

    def _system_health(self, state: dict) -> dict:
        sys_info   = state.get("system", {})
        started_at = sys_info.get("started_at")
        uptime     = None
        if started_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                uptime = int((datetime.now(timezone.utc) - start_dt).total_seconds())
            except ValueError:
                pass
        return {
            "cycle_count":    sys_info.get("cycle_count", 0),
            "uptime_seconds": uptime,
            "paper_trading":  sys_info.get("paper_trading", True),
            "agent_statuses": state.get("agents", {}),
        }

    def _data_sources(
        self,
        port_retail: dict,
        port_inst:   dict,
        market_doc:  dict,
        state:       dict,
    ) -> dict:
        adapters_state = state.get("adapters", {})
        openbb_state   = adapters_state.get("openbb", {})
        ibkr_state     = adapters_state.get("ibkr", {})
        wm_state       = adapters_state.get("world_monitor", {})

        exec_mode = port_retail.get("execution_mode", port_inst.get("execution_mode", "unknown"))
        return {
            "market_data_source": market_doc.get("data_source", "unknown"),
            "execution_mode":     exec_mode,
            "portfolio_accounting": {
                "retail_scope": port_retail.get("account_scope", "unknown"),
                "institutional_scope": port_inst.get("account_scope", "unknown"),
                "retail_equity_basis": port_retail.get("equity_basis", "unknown"),
                "institutional_equity_basis": port_inst.get("equity_basis", "unknown"),
            },
            "openbb": {
                "connected":  openbb_state.get("connected", market_doc.get("data_source") == "openbb"),
                "provider":   openbb_state.get("provider", self.config.get("openbb", {}).get("provider", "?")),
                "last_error": openbb_state.get("last_error"),
            },
            "ibkr": {
                "connected":  ibkr_state.get("connected", exec_mode == "ibkr"),
                "account_id": ibkr_state.get("account_id", ""),
                "last_error": ibkr_state.get("last_error"),
            },
            "world_monitor": {
                "connected":  wm_state.get("connected", False),
                "mode":       wm_state.get("mode", "stub"),
                "last_error": wm_state.get("last_error"),
                "events_in_last_cycle": len(market_doc.get("world_events", [])),
            },
        }

    # ------------------------------------------------------------------ #
    # Housekeeping                                                         #
    # ------------------------------------------------------------------ #

    def _cleanup_old_reports(self) -> None:
        reports = sorted(REPORTS_DIR.glob("report_*.json"))
        for old in reports[: max(0, len(reports) - MAX_REPORTS)]:
            try:
                old.unlink()
            except OSError as exc:
                self.logger.warning(f"Could not delete {old.name}: {exc}")
