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
import threading
import time
from datetime import datetime, timedelta, timezone

# Ensure trading_bot/ is on the path so `agents.*` imports resolve
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agents.market_data_agent import MarketDataAgent
from agents.macro_analyzer_agent import MacroAnalyzerAgent
from agents.sector_analyzer_agent import SectorAnalyzerAgent
from agents.stock_analyzer_agent import StockAnalyzerAgent
from agents.news_data_agent import NewsDataAgent
from agents.telegram_sentiment_agent import TelegramSentimentAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.report_agent import ReportAgent
from agents.alpha_hunter_agent import AlphaHunterAgent
from agents.bull_strategy_agent import BullStrategyAgent
from agents.pairs_arbitrage_agent import PairsArbitrageAgent
from agents.bear_strategy_agent import BearStrategyAgent
from agents.crypto_strategy_agent import CryptoStrategyAgent
from agents.portfolio_manager import PortfolioManager
from agents.backtesting_agent import BacktestingAgent
from agents.feature_store_agent import FeatureStoreAgent
from agents.ml_strategy_agent import MLStrategyAgent
from agents.adaptive_learner import AdaptiveLearner
from agents.auditor_agent import AuditorAgent
from agents.ml_cross_sectional_agent import MLCrossSectionalAgent
from agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
from agents.universe_discovery_agent import UniverseDiscoveryAgent
from agents.universe_hygiene_agent import UniverseHygieneAgent
from agents.alternative_data_agent import AlternativeDataAgent
from agents.base_agent import (
    BASE_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR, SHARED_STATE_PATH, CONFIG_PATH
)
from backtesting.auto_ml_pipeline import AutoMLBacktestPipeline

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
        self.telegram_agent  = TelegramSentimentAgent()
        self.pairs_agent     = PairsArbitrageAgent()
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

        # ML learning pipeline
        self.feature_store_agent = FeatureStoreAgent()
        self.ml_strategy_agent = MLStrategyAgent()
        self.ml_cross_sectional_agent = MLCrossSectionalAgent()

        # Sentiment, Universe, Auditor
        self.sentiment_agent = SentimentAnalyzerAgent()
        self.alternative_data_agent = AlternativeDataAgent()
        self.universe_discovery_agent = UniverseDiscoveryAgent()
        self.universe_hygiene_agent = UniverseHygieneAgent()
        self.auditor_agent = AuditorAgent()

        # Adaptive Learner — aggiorna parametri di rischio ogni ciclo
        self.adaptive_learner = AdaptiveLearner()

        # Portfolio Manager — syncs sub-portfolio views after execution
        self.portfolio_manager = PortfolioManager()
        self.backtesting_agent = BacktestingAgent()

        # Auto-backtest state
        ab_cfg = self.config.get("orchestrator", {}).get("auto_backtest", {})
        self._auto_backtest_enabled = bool(ab_cfg.get("enabled", False))
        self._auto_backtest_interval_days = int(ab_cfg.get("interval_days", 7))
        self._auto_backtest_rolling_days = int(ab_cfg.get("rolling_window_days", 180))
        self._auto_ml_on_completion = bool(ab_cfg.get("auto_ml_on_completion", True))
        self._auto_backtest_thread: threading.Thread | None = None
        self._auto_backtest_lock = threading.Lock()

        # Background slow agents — StockAnalyzer (20s) + FeatureStore (37s) + SectorAnalyzer (36s)
        # Run on their own cadence to keep main cycle under 60s target.
        orch_cfg = self.config.get("orchestrator", {})
        self._stock_bg_cadence = int(orch_cfg.get("stock_bg_cadence", 3))    # every 3 cycles (~3 min)
        self._feature_bg_cadence = int(orch_cfg.get("feature_bg_cadence", 5))  # every 5 cycles (~5 min)
        self._sector_bg_cadence = int(orch_cfg.get("sector_bg_cadence", 10))  # every 10 cycles (~10 min)
        self._stock_bg_thread: threading.Thread | None = None
        self._feature_bg_thread: threading.Thread | None = None
        self._sector_bg_thread: threading.Thread | None = None
        self._stock_bg_lock = threading.Lock()
        self._feature_bg_lock = threading.Lock()
        self._sector_bg_lock = threading.Lock()

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
                "feature_store_agent":        {"status": "idle", "last_run": None, "last_error": None},
                "ml_strategy_agent":          {"status": "idle", "last_run": None, "last_error": None},
                "ml_cross_sectional_agent":   {"status": "idle", "last_run": None, "last_error": None},
                "sentiment_analyzer_agent":   {"status": "idle", "last_run": None, "last_error": None},
                "universe_discovery_agent":   {"status": "idle", "last_run": None, "last_error": None},
                "universe_hygiene_agent":     {"status": "idle", "last_run": None, "last_error": None},
                "auditor_agent":              {"status": "idle", "last_run": None, "last_error": None},
                "adaptive_learner":           {"status": "idle", "last_run": None, "last_error": None},
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
            _backtest_mode = self.config.get("orchestrator", {}).get("mode") == "backtest"
            _interval = 0.0 if _backtest_mode else float(
                self.config["orchestrator"]["cycle_interval_seconds"]
            )
            sleep_time = max(0.0, _interval - elapsed)

            if not self._shutdown_requested:
                if _backtest_mode:
                    self.logger.info(f"Cycle took {elapsed:.1f}s. Backtest mode — no sleep.")
                else:
                    self.logger.info(
                        f"Cycle took {elapsed:.1f}s. Sleeping {sleep_time:.1f}s until next cycle."
                    )
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

        # Step 2.5: Sector analysis — background thread, cadence every N cycles
        if cycle_count % self._sector_bg_cadence == 1:
            self._launch_background_agent(
                "sector_analyzer_agent", self.sector_agent,
                self._sector_bg_thread, self._sector_bg_lock,
                "_sector_bg_thread",
            )
        else:
            self.logger.info(
                f"SectorAnalyzerAgent skipped (cycle {cycle_count}, "
                f"next at cycle {(cycle_count // self._sector_bg_cadence + 1) * self._sector_bg_cadence + 1})"
            )

        # Step 2.6: Stock analysis — background thread, cadence every N cycles
        if cycle_count % self._stock_bg_cadence == 1:
            self._launch_background_agent(
                "stock_analyzer_agent", self.stock_agent,
                self._stock_bg_thread, self._stock_bg_lock,
                "_stock_bg_thread",
            )
        else:
            self.logger.info(
                f"StockAnalyzerAgent skipped (cycle {cycle_count}, "
                f"next at cycle {(cycle_count // self._stock_bg_cadence + 1) * self._stock_bg_cadence + 1})"
            )

        # Step 2.7: Feature store — background thread, cadence every N cycles
        if cycle_count % self._feature_bg_cadence == 1:
            self._launch_background_agent(
                "feature_store_agent", self.feature_store_agent,
                self._feature_bg_thread, self._feature_bg_lock,
                "_feature_bg_thread",
            )
        else:
            self.logger.info(
                f"FeatureStoreAgent skipped (cycle {cycle_count}, "
                f"next at cycle {(cycle_count // self._feature_bg_cadence + 1) * self._feature_bg_cadence + 1})"
            )

        # Step 2.74: Alternative data — Fear&Greed + xStocks L2 orderbook imbalance
        self._run_agent("alternative_data_agent", self.alternative_data_agent)

        # Step 2.75: Sentiment analysis — VADER NLP on RSS feeds
        self._run_agent("sentiment_analyzer_agent", self.sentiment_agent)

        # Step 2.8: ML strategy — re-trains every refresh_days (default 30d), reads feature history
        ml_ok = self._run_agent("ml_strategy_agent", self.ml_strategy_agent)

        # Step 2.85: ML cross-sectional — weekly dataset builder + pipeline
        if cycle_count % 7 == 1:
            self._run_agent("ml_cross_sectional_agent", self.ml_cross_sectional_agent)
        if not ml_ok:
            self.logger.warning("MLStrategyAgent failed — RiskAgent will proceed without ML boost")

        # Step 3: News analysis
        news_ok = self._run_agent("news_data_agent", self.news_agent)

        # Step 3c: Pairs arbitrage (ogni ciclo — veloce, no network calls)
        self._run_agent("pairs_arbitrage_agent", self.pairs_agent)

        # Step 3b: Telegram sentiment (ogni 10 cicli ~10 min, non ogni minuto)
        _tg_enabled = self.config.get("telegram_sentiment", {}).get("enabled", False)
        if _tg_enabled and cycle_count % 10 == 0:
            self._run_agent("telegram_sentiment_agent", self.telegram_agent)
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
            # ISSUE-006: Inject intraday momentum SPY/QQQ/IWM into bull_signals ETF bucket
            self._inject_intraday_momentum_signals()

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

        # Step 5d: Adaptive Learner — aggiorna parametri di rischio in base alle performance
        self._run_agent("adaptive_learner", self.adaptive_learner)

        # Step 5e: Universe Discovery + Hygiene (ogni 10 cicli)
        if cycle_count % 10 == 1:
            self._run_agent("universe_discovery_agent", self.universe_discovery_agent)
            self._run_agent("universe_hygiene_agent", self.universe_hygiene_agent)

        # Step 5f: Auditor (ogni ciclo — usa LLM se abilitato in config)
        self._run_agent("auditor_agent", self.auditor_agent)

        # Step 6: Report (hourly, independent of pipeline failures)
        report_interval = self.config["orchestrator"]["report_interval_seconds"]
        if time.time() - self._last_report_time >= report_interval:
            self.logger.info("Hourly report interval reached — running ReportAgent")
            self._run_agent("report_agent", self.report_agent)
            self._last_report_time = time.time()

        self._update_cycle_state(cycle_count)

        # Step 7: Periodic auto-backtest (non-blocking background thread)
        if self._auto_backtest_enabled:
            self._maybe_trigger_auto_backtest()

    def _inject_intraday_momentum_signals(self) -> None:
        """ISSUE-006: Intraday Momentum SPY/QQQ/IWM — live orchestrator integration.
        Reads market_data.json, computes 20-period noise band breakout for broad market ETFs,
        and prepends valid picks to bull_signals.json ETF bucket before RiskAgent runs."""
        import math
        try:
            def _rj(p):
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return {}
            mkt = _rj(DATA_DIR / "market_data.json")
            bull = _rj(DATA_DIR / "bull_signals.json")
            assets = mkt.get("assets", {})
            ETF_SYMS = ["SPY", "QQQ", "IWM"]
            intraday_picks = []
            for sym in ETF_SYMS:
                data = assets.get(sym, {})
                ohlcv = data.get("ohlcv_4h") or data.get("ohlcv_1d") or []
                closes = [float(c["c"]) for c in ohlcv if c.get("c") and float(c["c"]) > 0]
                if len(closes) < 22:
                    continue
                last = closes[-1]
                window = closes[-21:-1]
                mu = sum(window) / len(window)
                variance = sum((x - mu) ** 2 for x in window) / len(window)
                sigma = math.sqrt(variance)
                upper = mu + 1.5 * sigma
                lower = mu - 1.5 * sigma
                mom_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0
                if last > upper and mom_5d > 0.005:
                    score = round(min(1.0, 0.70 + mom_5d * 5), 3)
                    intraday_picks.append({
                        "symbol": sym,
                        "composite_score": score,
                        "_signal_type": "BUY",
                        "_agent_source": "bull",
                        "_ta_last_price": round(last, 4),
                        "_noise_bands": {"upper": round(upper, 4), "lower": round(lower, 4), "sigma": round(sigma, 4)},
                        "_tier_size_mult": 0.5,
                        "_tier_sl_mult": 1.5,
                        "_tier_tp_mult": 2.0,
                        "momentum_5d": round(mom_5d * 100, 2),
                        "source": "intraday_momentum_spy",
                    })
            if intraday_picks:
                bull.setdefault("allocations", {}).setdefault("etf_50_pct", [])
                bull["allocations"]["etf_50_pct"] = intraday_picks + bull["allocations"]["etf_50_pct"]
                (DATA_DIR / "bull_signals.json").write_text(json.dumps(bull, indent=2), encoding="utf-8")
                self.logger.info(f"ISSUE-006: Injected {len(intraday_picks)} intraday momentum picks: {[p['symbol'] for p in intraday_picks]}")
        except Exception as e:
            self.logger.warning(f"_inject_intraday_momentum_signals failed: {e}")

    def _launch_background_agent(
        self,
        name: str,
        agent,
        current_thread: "threading.Thread | None",
        lock: threading.Lock,
        thread_attr: str,
    ) -> None:
        """Launch agent in a daemon thread if no prior run is still active."""
        with lock:
            t = getattr(self, thread_attr)
            if t is not None and t.is_alive():
                self.logger.info(f"{name} background thread still running — skipping launch")
                return

            self.logger.info(f"Launching {name} in background thread")

            def _run():
                try:
                    agent.run()
                except Exception as exc:
                    self.logger.error(f"Background {name} error: {exc}", exc_info=True)

            new_thread = threading.Thread(target=_run, name=name, daemon=True)
            setattr(self, thread_attr, new_thread)
            new_thread.start()

    def _maybe_trigger_auto_backtest(self) -> None:
        """Start a background backtest if the interval has elapsed and no backtest is running."""
        with self._auto_backtest_lock:
            if self._auto_backtest_thread is not None and self._auto_backtest_thread.is_alive():
                return  # Already running

            last_run = self._read_auto_backtest_last_run()
            if last_run is not None:
                next_due = last_run + timedelta(days=self._auto_backtest_interval_days)
                if datetime.now(timezone.utc) < next_due:
                    return

            self.logger.info(
                "Auto-backtest due — launching background thread "
                f"(interval={self._auto_backtest_interval_days}d, window={self._auto_backtest_rolling_days}d)"
            )
            self._auto_backtest_thread = threading.Thread(
                target=self._run_auto_backtest_and_automl,
                name="auto-backtest",
                daemon=True,
            )
            self._auto_backtest_thread.start()

    def _run_auto_backtest_and_automl(self) -> None:
        """Background worker: update config dates, run backtest, then run AutoML."""
        try:
            # Build rolling window dates
            end_dt = datetime.now(timezone.utc).date()
            start_dt = end_dt - timedelta(days=self._auto_backtest_rolling_days)

            # Temporarily override backtesting dates in the agent's config
            self.backtesting_agent.config["backtesting"]["start_date"] = start_dt.isoformat()
            self.backtesting_agent.config["backtesting"]["end_date"] = end_dt.isoformat()

            self.logger.info(f"Auto-backtest starting | window={start_dt} → {end_dt}")
            ok = self.backtesting_agent.run()

            now_iso = datetime.now(timezone.utc).isoformat()
            self._write_auto_backtest_last_run(now_iso, ok)

            if not ok:
                self.logger.error("Auto-backtest failed — skipping AutoML update")
                return

            self.logger.info("Auto-backtest completed — running AutoML weight update")
            if self._auto_ml_on_completion:
                self._run_automl()
        except Exception as exc:
            self.logger.error(f"Auto-backtest background thread error: {exc}", exc_info=True)

    def _run_automl(self) -> None:
        """Run AutoMLBacktestPipeline to update learned_strategy_weights.json."""
        try:
            pipeline = AutoMLBacktestPipeline(
                repo_dir=BASE_DIR,
                workspace_dir=BASE_DIR.parent,
            )
            result = pipeline.run()
            if result.get("status") == "ok":
                self.logger.info(
                    "AutoML weight update done | reports=%d | weights=%s",
                    result.get("reports_count", 0),
                    result.get("strategy_weights", {}),
                )
            else:
                self.logger.warning(f"AutoML skipped: {result.get('reason', 'unknown')}")
        except Exception as exc:
            self.logger.error(f"AutoML pipeline error: {exc}", exc_info=True)

    def _read_auto_backtest_last_run(self) -> datetime | None:
        state_path = DATA_DIR / "auto_backtest_state.json"
        if not state_path.exists():
            return None
        try:
            doc = json.loads(state_path.read_text(encoding="utf-8"))
            last = doc.get("last_run_at")
            if not last:
                return None
            return datetime.fromisoformat(last)
        except Exception:
            return None

    def _write_auto_backtest_last_run(self, iso_ts: str, success: bool) -> None:
        state_path = DATA_DIR / "auto_backtest_state.json"
        doc = {"last_run_at": iso_ts, "last_run_success": success}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        os.replace(tmp, state_path)

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
        if self._auto_backtest_thread is not None and self._auto_backtest_thread.is_alive():
            self.logger.info("Waiting for auto-backtest background thread to finish (max 300s)...")
            self._auto_backtest_thread.join(timeout=300)
            if self._auto_backtest_thread.is_alive():
                self.logger.warning("Auto-backtest thread did not finish in time — proceeding with shutdown")
        for attr, label in [("_stock_bg_thread", "StockAnalyzer"), ("_feature_bg_thread", "FeatureStore"), ("_sector_bg_thread", "SectorAnalyzer")]:
            t = getattr(self, attr)
            if t is not None and t.is_alive():
                self.logger.info(f"Waiting for {label} background thread (max 60s)...")
                t.join(timeout=60)
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
