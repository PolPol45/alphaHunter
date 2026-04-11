"""
Orchestrator — Multi-Agent Trading Bot Entry Point

Coordinates the full agent pipeline in a synchronous loop:

  MarketDataAgent → TechnicalAnalysisAgent → RiskAgent → ExecutionAgent
                                                        → ReportAgent (hourly)

Design decisions:
  - Synchronous: agents run sequentially to enforce data dependency order
    (TA needs market data, Risk needs signals, Execution needs validated signals)
  - If an upstream agent fails, downstream agents are skipped for that cycle
    (never execute unvalidated signals)
  - ReportAgent runs independently of pipeline failures
  - SIGINT / SIGTERM trigger a clean shutdown

Usage:
  python orchestrator.py

Always runs in PAPER TRADING mode. See config.json to change cycle interval.
"""

import json
import logging
import os
import pathlib
import signal
import sys
import time
from datetime import datetime, timezone

# Ensure trading_bot/ is on the path so `agents.*` imports resolve
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agents.market_data_agent import MarketDataAgent
from agents.macro_analyzer_agent import MacroAnalyzerAgent
from agents.sector_analyzer_agent import SectorAnalyzerAgent
from agents.stock_analyzer_agent import StockAnalyzerAgent
from agents.news_data_agent import NewsDataAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.report_agent import ReportAgent
from agents.alpha_hunter_agent import AlphaHunterAgent
from agents.bull_strategy_agent import BullStrategyAgent
from agents.bear_strategy_agent import BearStrategyAgent
from agents.crypto_strategy_agent import CryptoStrategyAgent
from agents.portfolio_manager import PortfolioManager
from agents.backtesting_agent import BacktestingAgent
from agents.base_agent import (
    BASE_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR, SHARED_STATE_PATH, CONFIG_PATH
)

BANNER = """
╔══════════════════════════════════════════════════════╗
║        CRYPTO TRADING BOT — PAPER TRADING MODE       ║
║   No real orders. No real money. Simulation only.    ║
╚══════════════════════════════════════════════════════╝
"""


class Orchestrator:
    def __init__(self):
        self._setup_directories()
        self.config = self._load_config()
        self.logger = self._setup_logger()

        self._shutdown_requested = False
        self._last_report_time = 0.0
        self._start_time = time.time()
        self._mode = self.config.get("orchestrator", {}).get("mode", "live_cycle")

        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Instantiate agents
        self.market_agent    = MarketDataAgent()
        self.macro_agent     = MacroAnalyzerAgent()
        self.sector_agent    = SectorAnalyzerAgent()
        self.stock_agent     = StockAnalyzerAgent()
        self.news_agent      = NewsDataAgent()
        self.ta_agent        = TechnicalAnalysisAgent()
        self.risk_agent      = RiskAgent()
        self.execution_agent = ExecutionAgent()
        self.report_agent    = ReportAgent()

        # Alpha Hunter — runs independently (own data source)
        alpha_cfg = self.config.get("alpha_hunter", {})
        self.alpha_agent: AlphaHunterAgent | None = (
            AlphaHunterAgent() if alpha_cfg.get("enabled", False) else None
        )

        bull_cfg = self.config.get("bull_strategy", {})
        self.bull_agent: BullStrategyAgent | None = (
            BullStrategyAgent() if bull_cfg.get("enabled", False) else None
        )

        bear_cfg = self.config.get("bear_strategy", {})
        self.bear_agent: BearStrategyAgent | None = (
            BearStrategyAgent() if bear_cfg.get("enabled", False) else None
        )

        crypto_cfg = self.config.get("crypto_strategy", {})
        self.crypto_agent: CryptoStrategyAgent | None = (
            CryptoStrategyAgent() if crypto_cfg.get("enabled", False) else None
        )

        # Portfolio Manager — syncs sub-portfolio views after execution
        self.portfolio_manager = PortfolioManager()
        self.backtesting_agent = BacktestingAgent()

        # Initialize shared state
        self._init_shared_state()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _setup_directories(self) -> None:
        for d in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] [orchestrator] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh = logging.FileHandler(LOGS_DIR / "orchestrator.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)

        return logger

    def _init_shared_state(self) -> None:
        state = {
            "system": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_cycle": None,
                "cycle_count": 0,
                "paper_trading": self.config["trading"]["paper_trading"],
                "status": "running",
            },
            "agents": {
                "market_data_agent":          {"status": "idle", "last_run": None, "last_error": None},
                "macro_analyzer_agent":       {"status": "idle", "last_run": None, "last_error": None},
                "sector_analyzer_agent":      {"status": "idle", "last_run": None, "last_error": None},
                "stock_analyzer_agent":       {"status": "idle", "last_run": None, "last_error": None},
                "news_data_agent":            {"status": "idle", "last_run": None, "last_error": None},
                "technical_analysis_agent":   {"status": "idle", "last_run": None, "last_error": None},
                "risk_agent":                 {"status": "idle", "last_run": None, "last_error": None},
                "execution_agent":            {"status": "idle", "last_run": None, "last_error": None},
                "report_agent":               {"status": "idle", "last_run": None, "last_error": None},
                "alpha_hunter_agent":         {"status": "idle", "last_run": None, "last_error": None},
                "bull_strategy_agent":        {"status": "idle", "last_run": None, "last_error": None},
                "bear_strategy_agent":        {"status": "idle", "last_run": None, "last_error": None},
                "crypto_strategy_agent":      {"status": "idle", "last_run": None, "last_error": None},
                "portfolio_manager":          {"status": "idle", "last_run": None, "last_error": None},
            },
            "data_freshness": {
                "market_data":       None,
                "news_feed":         None,
                "macro_snapshot":    None,
                "insider_activity":  None,
                "signals":           None,
                "validated_signals": None,
                "portfolio":         None,
            },
            "alerts": [],
        }
        tmp = SHARED_STATE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, SHARED_STATE_PATH)

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        print(BANNER)
        if self._mode == "backtest":
            self.logger.info("Starting orchestrator in BACKTEST mode")
            ok = self._run_agent("backtesting_agent", self.backtesting_agent)
            if not ok:
                self.logger.error("BacktestingAgent failed")
            self._shutdown()
            return

        self.logger.info(
            f"Starting orchestrator | "
            f"Assets: {self.config['assets']} | "
            f"Cycle interval: {self.config['orchestrator']['cycle_interval_seconds']}s | "
            f"Paper trading: {self.config['trading']['paper_trading']}"
        )

        cycle_count = 0

        while not self._shutdown_requested:
            cycle_start = time.time()
            cycle_count += 1

            self.logger.info(f"── Cycle #{cycle_count} start ──────────────────")
            self._run_cycle(cycle_count)
            self.logger.info(f"── Cycle #{cycle_count} done ──────────────────")

            elapsed = time.time() - cycle_start
            sleep_time = max(0.0, self.config["orchestrator"]["cycle_interval_seconds"] - elapsed)

            if not self._shutdown_requested:
                self.logger.info(
                    f"Cycle took {elapsed:.1f}s. Sleeping {sleep_time:.1f}s until next cycle."
                )
                # Sleep in short chunks so SIGINT is handled promptly
                deadline = time.time() + sleep_time
                while not self._shutdown_requested and time.time() < deadline:
                    time.sleep(min(1.0, deadline - time.time()))

        self._shutdown()

    # ------------------------------------------------------------------ #
    # Single cycle                                                         #
    # ------------------------------------------------------------------ #

    def _run_cycle(self, cycle_count: int) -> None:
        # Step 1: Market data — foundation of everything
        market_ok = self._run_agent("market_data_agent", self.market_agent)
        if not market_ok:
            self.logger.warning("MarketDataAgent failed — skipping TA, Risk, Execution this cycle")
            self._update_cycle_state(cycle_count)
            return

        # Step 2: Macro analysis
        macro_ok = self._run_agent("macro_analyzer_agent", self.macro_agent)
        if not macro_ok:
            self.logger.warning("MacroAnalyzerAgent failed — pipeline continues but regime data may be stale")

        # Step 2.5: Sector analysis
        sector_ok = self._run_agent("sector_analyzer_agent", self.sector_agent)
        if not sector_ok:
            self.logger.warning("SectorAnalyzerAgent failed — pipeline continues but sector data may be stale")

        # Step 2.6: Stock analysis
        stock_ok = self._run_agent("stock_analyzer_agent", self.stock_agent)
        if not stock_ok:
            self.logger.warning("StockAnalyzerAgent failed — pipeline continues but stock data may be stale")

        # Step 3: News analysis
        news_ok = self._run_agent("news_data_agent", self.news_agent)
        if not news_ok:
            self.logger.warning("NewsDataAgent failed — skipping TA, Risk, Execution this cycle")
            self._update_cycle_state(cycle_count)
            return

        # Step 3: Technical analysis
        ta_ok = self._run_agent("technical_analysis_agent", self.ta_agent)
        if not ta_ok:
            self.logger.warning("TechnicalAnalysisAgent failed — skipping Risk, Execution this cycle")
            self._update_cycle_state(cycle_count)
            return

        # Step 4: Strategy agents
        # Bull and Bear are now instantaneous consumers (they read stock_scores.json). We run them every cycle.
        # Crypto still queries external data directly, so we run it on cadence.
        strategy_cadence = self.config.get("orchestrator", {}).get("strategy_cadence", 10)
        run_crypto = (cycle_count % strategy_cadence == 1)  # ciclo 1, 11, 21, ...

        if self.bull_agent is not None:
            self._run_agent("bull_strategy_agent", self.bull_agent)
            
        if self.bear_agent is not None:
            self._run_agent("bear_strategy_agent", self.bear_agent)

        if run_crypto:
            self.logger.info(f"Crypto strategy scan cycle ({cycle_count})")
            if self.crypto_agent is not None:
                self._run_agent("crypto_strategy_agent", self.crypto_agent)
        else:
            self.logger.info(
                f"Crypto strategy scan skipped (cycle {cycle_count}, next at cycle "
                f"{(cycle_count // strategy_cadence + 1) * strategy_cadence + 1})"
            )

        # Step 4.5: Risk validation (Aggregates strategy signals)
        risk_ok = self._run_agent("risk_agent", self.risk_agent)
        if not risk_ok:
            self.logger.warning("RiskAgent failed — skipping Execution this cycle")
            self._update_cycle_state(cycle_count)
            return

        # Step 5a: Alpha Hunter (ogni ciclo — dati equity veloci)
        if self.alpha_agent is not None:
            self._run_agent("alpha_hunter_agent", self.alpha_agent)

        # Step 5b: Execution (crypto + alpha portfolios)
        self._run_agent("execution_agent", self.execution_agent)

        # Step 5c: Portfolio Manager — sync sub-portfolio views from parent portfolios
        self._run_agent("portfolio_manager", self.portfolio_manager)

        # Step 6: Report (hourly, independent of pipeline failures)
        report_interval = self.config["orchestrator"]["report_interval_seconds"]
        if time.time() - self._last_report_time >= report_interval:
            self.logger.info("Hourly report interval reached — running ReportAgent")
            self._run_agent("report_agent", self.report_agent)
            self._last_report_time = time.time()

        self._update_cycle_state(cycle_count)

    def _run_agent(self, name: str, agent) -> bool:
        try:
            return agent.run()
        except Exception as e:
            self.logger.error(f"Unhandled exception in {name}: {e}", exc_info=True)
            return False

    def _update_cycle_state(self, cycle_count: int) -> None:
        try:
            state_path = SHARED_STATE_PATH
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            state["system"]["cycle_count"] = cycle_count
            state["system"]["last_cycle"] = datetime.now(timezone.utc).isoformat()

            tmp = state_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, state_path)
        except Exception as e:
            self.logger.warning(f"Could not update shared state: {e}")

    # ------------------------------------------------------------------ #
    # Shutdown                                                             #
    # ------------------------------------------------------------------ #

    def _handle_shutdown(self, signum, frame) -> None:
        self.logger.info(f"Shutdown signal received (signal {signum})")
        self._shutdown_requested = True

    def _shutdown(self) -> None:
        self.logger.info("Shutting down cleanly...")
        try:
            with open(SHARED_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            state["system"]["status"] = "stopped"
            tmp = SHARED_STATE_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, SHARED_STATE_PATH)
        except Exception as e:
            self.logger.warning(f"Could not write final state: {e}")

        uptime = time.time() - self._start_time
        self.logger.info(f"Orchestrator stopped. Uptime: {uptime:.0f}s")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    orch = Orchestrator()
    orch.run()
