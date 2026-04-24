from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.base_agent import BaseAgent, DATA_DIR, REPORTS_DIR
from backtesting.cross_sectional_ml_pipeline import CrossSectionalMLPipeline
from backtesting.ml_dataset_builder import WeeklyMLDatasetBuilder


class MLStrategyAgent(BaseAgent):
    """
    Production-facing ML strategy agent.

    Responsibilities:
    - refresh cross-sectional model predictions on configurable cadence
    - produce `data/ml_signals.json` with ranking + long/short weights
    - keep runtime mode configurable (`informative_only` or active strategy mode)
    """

    def __init__(self) -> None:
        super().__init__("ml_strategy_agent")
        self._cfg = self.config.get("ml_strategy", {})
        self._enabled = bool(self._cfg.get("enabled", True))
        self._refresh_days = int(self._cfg.get("refresh_days", 30))

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self._enabled:
                self.logger.info("MLStrategyAgent disabled by config")
                self.mark_done()
                return True

            if not self._is_refresh_due():
                self.logger.info("MLStrategyAgent refresh not due yet")
                self.mark_done()
                return True

            fs_cfg = self.config.get("feature_store", {})
            features_default = fs_cfg.get("history_output_path", "cross_sectional_features_history.jsonl")
            features_path = DATA_DIR / str(self._cfg.get("features_input_path", features_default))
            dataset_path = DATA_DIR / str(self._cfg.get("dataset_output_path", "ml_dataset_weekly.jsonl"))
            dataset_meta = DATA_DIR / str(self._cfg.get("dataset_meta_path", "ml_dataset_weekly.meta.json"))
            signals_path = DATA_DIR / str(self._cfg.get("signals_output_path", "ml_signals.json"))

            window_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            summary_name = str(self._cfg.get("summary_output_pattern", "ml_cross_sectional_summary_{window}.json")).format(window=window_label)
            summary_path = REPORTS_DIR / summary_name

            builder = WeeklyMLDatasetBuilder(
                features_path=features_path,
                output_dataset_path=dataset_path,
                output_meta_path=dataset_meta,
                benchmark_default=str(self._cfg.get("benchmark_default", "SPY")),
            )
            builder.run()

            pipeline = CrossSectionalMLPipeline(
                dataset_path=dataset_path,
                signals_path=signals_path,
                summary_path=summary_path,
                config=dict(self._cfg),
            )
            summary = pipeline.run()

            # Enrich ml_signals with strategy parameters used by RiskAgent.
            signals_doc = self.read_json(signals_path) or {}
            signals_doc["strategy"] = {
                "prediction_horizon_weeks": int(self._cfg.get("prediction_horizon_weeks", 1)),
                "portfolio_size": int(self._cfg.get("portfolio_size", 20)),
                "max_long_positions": int(self._cfg.get("max_long_positions", 10)),
                "max_short_positions": int(self._cfg.get("max_short_positions", 10)),
                "stop_loss_pct": float(self._cfg.get("stop_loss_pct", 0.03)),
                "take_profit_pct": float(self._cfg.get("take_profit_pct", 0.07)),
                "leverage_cap": float(self._cfg.get("leverage_cap", 0.7)),
            }
            signals_doc["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.write_json(signals_path, signals_doc)

            ts = datetime.now(timezone.utc).isoformat()
            self.write_json(DATA_DIR / "ml_strategy_state.json", {"last_run_at": ts, "status": summary.get("status", "ok")})
            self.update_shared_state("data_freshness.ml_signals", ts)
            self.update_shared_state("data_freshness.ml_strategy", ts)
            self.logger.info(
                "MLStrategyAgent completed | status=%s | folds=%s",
                summary.get("status"),
                summary.get("fold_count", 0),
            )
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    def _is_refresh_due(self) -> bool:
        state = self.read_json(DATA_DIR / "ml_strategy_state.json") or {}
        last = state.get("last_run_at")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
        except Exception:
            return True
        return datetime.now(timezone.utc) >= (last_dt + timedelta(days=self._refresh_days))
