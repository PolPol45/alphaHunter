"""
Portfolio Manager
=================
Implements the FigJam sub-portfolio structure.

Personal ($20k total):
  crypto    $5k  — active (mirrors retail portfolio proportionally)
  bull      $5k  — idle until BullStrategyAgent (Phase 3)
  bear      $5k  — idle until BearStrategyAgent (Phase 3)
  free_ride $5k  — idle until LLM Ensemble (Phase 4)

Institutional ($980k total):
  crypto          $150k — active (mirrors institutional portfolio proportionally)
  bull            $250k — idle until Phase 3
  bear            $250k — idle until Phase 3
  free_ride       $250k — idle until Phase 4
  analyst_reserve  $80k — idle until Analyst Bot (Phase 4)

For active sub-portfolios, equity/cash/positions are scaled proportionally
from the parent portfolio file. Idle sub-portfolios hold initial capital with
no positions.

Writes to: data/sub_portfolios.json  (single file, all 9 sub-portfolios)
           data/portfolio_personal_<key>.json  (one per sub-portfolio)
           data/portfolio_inst_<key>.json
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent, DATA_DIR

logger = logging.getLogger("portfolio_manager")

# Strategy label → emoji for dashboard display
_STRATEGY_ICONS = {
    "crypto_momentum": "₿",
    "bull_undervalue":  "🐂",
    "bear_short":       "🐻",
    "llm_ensemble":     "🤖",
    "analyst_bot":      "📊",
}


class PortfolioManager(BaseAgent):
    """
    Reads existing retail/institutional portfolio files and writes
    sub-portfolio views derived from config['portfolios'].
    Runs once per cycle after ExecutionAgent.
    """

    def __init__(self) -> None:
        super().__init__("portfolio_manager")
        self._port_cfg: dict = self.config.get("portfolios", {})

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        self.mark_running()
        try:
            personal_cfg = self._port_cfg.get("personal", {})
            inst_cfg     = self._port_cfg.get("institutional", {})

            parent_retail = self.read_json(DATA_DIR / "portfolio_retail.json") or {}
            parent_inst   = self.read_json(DATA_DIR / "portfolio_institutional.json") or {}

            sub_portfolios: dict[str, dict] = {}

            # Personal sub-portfolios
            p_total = personal_cfg.get("total_capital", 20000.0)
            for key, sub_cfg in personal_cfg.get("sub", {}).items():
                sp_id   = f"personal_{key}"
                capital = sub_cfg["capital"]
                active  = sub_cfg.get("active", False)
                sp = self._build_sub_portfolio(
                    sp_id     = sp_id,
                    group     = "personal",
                    key       = key,
                    capital   = capital,
                    group_total = p_total,
                    strategy  = sub_cfg.get("strategy", ""),
                    description = sub_cfg.get("description", ""),
                    active    = active,
                    parent    = parent_retail if active else {},
                    parent_capital = personal_cfg.get("total_capital", 20000.0),
                )
                sub_portfolios[sp_id] = sp
                self.write_json(DATA_DIR / f"portfolio_{sp_id}.json", sp)

            # Institutional sub-portfolios
            i_total = inst_cfg.get("total_capital", 980000.0)
            for key, sub_cfg in inst_cfg.get("sub", {}).items():
                sp_id   = f"inst_{key}"
                capital = sub_cfg["capital"]
                active  = sub_cfg.get("active", False)
                sp = self._build_sub_portfolio(
                    sp_id     = sp_id,
                    group     = "institutional",
                    key       = key,
                    capital   = capital,
                    group_total = i_total,
                    strategy  = sub_cfg.get("strategy", ""),
                    description = sub_cfg.get("description", ""),
                    active    = active,
                    parent    = parent_inst if active else {},
                    parent_capital = inst_cfg.get("total_capital", 980000.0),
                )
                sub_portfolios[sp_id] = sp
                self.write_json(DATA_DIR / f"portfolio_{sp_id}.json", sp)

            # Write combined snapshot for the dashboard
            snapshot = self._build_snapshot(sub_portfolios, personal_cfg, inst_cfg)
            self.write_json(DATA_DIR / "sub_portfolios.json", snapshot)
            self.update_shared_state("data_freshness.sub_portfolios", snapshot["generated_at"])

            active_count = sum(1 for sp in sub_portfolios.values() if sp["active"])
            self.logger.info(
                f"Sub-portfolios synced: {len(sub_portfolios)} total, "
                f"{active_count} active"
            )
            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            self.logger.error(f"PortfolioManager error: {exc}", exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_sub_portfolio(
        self,
        *,
        sp_id: str,
        group: str,
        key: str,
        capital: float,
        group_total: float,
        strategy: str,
        description: str,
        active: bool,
        parent: dict,
        parent_capital: float,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()

        if not active or not parent:
            return {
                "id":              sp_id,
                "group":           group,
                "key":             key,
                "name":            key.replace("_", " ").title(),
                "strategy":        strategy,
                "icon":            _STRATEGY_ICONS.get(strategy, "•"),
                "description":     description,
                "initial_capital": capital,
                "cash":            capital,
                "total_equity":    capital,
                "peak_equity":     capital,
                "realized_pnl":    0.0,
                "unrealized_pnl":  0.0,
                "total_pnl":       0.0,
                "total_pnl_pct":   0.0,
                "drawdown_pct":    0.0,
                "active":          False,
                "status":          "idle",
                "positions":       {},
                "last_updated":    now,
            }

        # Scale factor: this sub-portfolio's share of the parent
        frac = capital / parent_capital if parent_capital else 0.0

        parent_equity    = float(parent.get("total_equity",    capital))
        parent_cash      = float(parent.get("cash",            capital))
        parent_peak      = float(parent.get("peak_equity",     capital))
        parent_r_pnl     = float(parent.get("realized_pnl",    0.0))
        parent_dd        = float(parent.get("drawdown_pct",    0.0))

        total_equity   = round(parent_equity  * frac, 4)
        cash           = round(parent_cash    * frac, 4)
        peak_equity    = round(parent_peak    * frac, 4)
        realized_pnl   = round(parent_r_pnl  * frac, 4)
        total_pnl      = round(total_equity - capital, 4)
        total_pnl_pct  = round(total_pnl / capital, 6) if capital else 0.0
        unrealized_pnl = round(total_pnl - realized_pnl, 4)

        # Scale positions proportionally
        positions = self._scale_positions(parent.get("positions", {}), frac)

        return {
            "id":              sp_id,
            "group":           group,
            "key":             key,
            "name":            key.replace("_", " ").title(),
            "strategy":        strategy,
            "icon":            _STRATEGY_ICONS.get(strategy, "•"),
            "description":     description,
            "initial_capital": capital,
            "cash":            cash,
            "total_equity":    total_equity,
            "peak_equity":     peak_equity,
            "realized_pnl":    realized_pnl,
            "unrealized_pnl":  unrealized_pnl,
            "total_pnl":       total_pnl,
            "total_pnl_pct":   total_pnl_pct,
            "drawdown_pct":    parent_dd,
            "active":          True,
            "status":          "active",
            "positions":       positions,
            "last_updated":    now,
        }

    def _scale_positions(self, positions: dict, frac: float) -> dict:
        """Return a proportionally scaled copy of parent positions."""
        scaled: dict = {}
        for sym, pos in positions.items():
            qty = float(pos.get("quantity", 0))
            if qty == 0:
                continue
            scaled[sym] = {
                "quantity":           round(qty * frac, 8),
                "avg_entry_price":    pos.get("avg_entry_price", 0),
                "current_price":      pos.get("current_price", 0),
                "unrealized_pnl":     round(float(pos.get("unrealized_pnl", 0)) * frac, 4),
                "unrealized_pnl_pct": pos.get("unrealized_pnl_pct", 0),
                "position_value":     round(float(pos.get("position_value_usdt", 0)) * frac, 4),
                "stop_loss_price":    pos.get("stop_loss_price", 0),
                "take_profit_price":  pos.get("take_profit_price", 0),
                "side":               pos.get("side", "long"),
            }
        return scaled

    def _build_snapshot(
        self,
        sub_portfolios: dict[str, dict],
        personal_cfg: dict,
        inst_cfg: dict,
    ) -> dict:
        """Build the combined snapshot written to sub_portfolios.json."""
        def group_summary(prefix: str, total_capital: float) -> dict:
            subs = {k: v for k, v in sub_portfolios.items() if k.startswith(prefix)}
            total_eq  = sum(sp["total_equity"] for sp in subs.values())
            total_pnl = sum(sp["total_pnl"] for sp in subs.values())
            pnl_pct   = (total_pnl / total_capital) if total_capital else 0.0
            return {
                "total_capital":  total_capital,
                "total_equity":   round(total_eq, 2),
                "total_pnl":      round(total_pnl, 2),
                "total_pnl_pct":  round(pnl_pct, 6),
                "sub_portfolios": subs,
            }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "personal":     group_summary("personal_", personal_cfg.get("total_capital", 20000.0)),
            "institutional": group_summary("inst_",    inst_cfg.get("total_capital", 980000.0)),
        }
