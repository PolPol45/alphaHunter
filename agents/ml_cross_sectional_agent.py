from __future__ import annotations

from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR, REPORTS_DIR
from backtesting.cross_sectional_ml_pipeline import CrossSectionalMLPipeline
from backtesting.ml_dataset_builder import WeeklyMLDatasetBuilder


class MLCrossSectionalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ml_cross_sectional_agent")
        self._cfg = self.config.get("ml_cross_sectional", {})
        self._enabled = bool(self._cfg.get("enabled", True))

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self._enabled:
                self.logger.info("ML cross-sectional disabled by config")
                self.mark_done()
                return True

            fs_cfg = self.config.get("feature_store", {})
            default_features_path = DATA_DIR / str(fs_cfg.get("history_output_path", "cross_sectional_features_history.jsonl"))
            features_path = DATA_DIR / str(self._cfg.get("features_input_path", default_features_path.name))
            dataset_path = DATA_DIR / str(self._cfg.get("dataset_output_path", "ml_dataset_weekly.parquet"))
            meta_path = DATA_DIR / str(self._cfg.get("dataset_meta_path", "ml_dataset_weekly.meta.json"))
            signals_path = DATA_DIR / str(self._cfg.get("signals_output_path", "ml_signals.json"))

            window_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            summary_name = str(self._cfg.get("summary_output_pattern", "ml_cross_sectional_summary_{window}.json")).format(window=window_label)
            summary_path = REPORTS_DIR / summary_name

            builder = WeeklyMLDatasetBuilder(
                features_path=features_path,
                output_dataset_path=dataset_path,
                output_meta_path=meta_path,
                benchmark_default=str(self._cfg.get("benchmark_default", "SPY")),
            )
            build_result = builder.run()

            pipeline = CrossSectionalMLPipeline(
                dataset_path=dataset_path,
                signals_path=signals_path,
                summary_path=summary_path,
                config=self._cfg,
            )
            summary = pipeline.run()

            generated_at = datetime.now(timezone.utc).isoformat()
            self.update_shared_state("data_freshness.ml_dataset_weekly", generated_at)
            self.update_shared_state("data_freshness.ml_signals", generated_at)
            self.update_shared_state("data_freshness.ml_cross_sectional_summary", generated_at)
            self.logger.info(
                "ML cross-sectional done | dataset_rows=%s folds=%s status=%s",
                build_result.rows,
                summary.get("fold_count", 0),
                summary.get("status"),
            )
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False
