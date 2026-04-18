import json
import tempfile
import unittest
from pathlib import Path

from backtesting.auto_ml_pipeline import AutoMLBacktestPipeline


def _mk_report(start: str, end: str, bull_pnl: float, bear_pnl: float, crypto_pnl: float, ml_pnl: float) -> dict:
    return {
        "report_type": "backtest",
        "window": {"start_date": start, "end_date": end},
        "summary": {"start_equity": 900000.0, "final_equity": 910000.0},
        "metrics": {"max_drawdown": 0.12, "turnover": 40.0},
        "strategy_breakdown": {
            "bull": {"trades": 120, "realized_pnl": bull_pnl, "win_rate": 0.55, "turnover": 0.9},
            "bear": {"trades": 80, "realized_pnl": bear_pnl, "win_rate": 0.50, "turnover": 0.8},
            "crypto": {"trades": 100, "realized_pnl": crypto_pnl, "win_rate": 0.57, "turnover": 1.0},
            "ml": {"trades": 90, "realized_pnl": ml_pnl, "win_rate": 0.56, "turnover": 0.7},
        },
    }


class AutoMLPipelineTests(unittest.TestCase):
    def test_pipeline_generates_learned_weights_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "trading_bot"
            (repo / "reports").mkdir(parents=True)
            (repo / "data").mkdir(parents=True)

            reports = [
                _mk_report("2025-01-01", "2025-03-31", 8000, 1500, 5000, 2500),
                _mk_report("2025-04-01", "2025-06-30", 7000, -1000, 4000, 1800),
                _mk_report("2025-07-01", "2025-09-30", 6000, -500, 3500, 1600),
            ]
            for i, r in enumerate(reports):
                fp = repo / "reports" / f"backtest_report_2025-0{i+1}-01_2025-0{i+3}-30.json"
                fp.write_text(json.dumps(r), encoding="utf-8")

            pipeline = AutoMLBacktestPipeline(repo_dir=repo, workspace_dir=root, min_train_reports=2)
            out = pipeline.run()
            self.assertEqual(out["status"], "ok")
            self.assertIn("strategy_weights", out)

            learned = json.loads((repo / "data" / "learned_strategy_weights.json").read_text(encoding="utf-8"))
            self.assertIn("strategy_weights", learned)
            for k in ("bull", "bear", "crypto", "ml"):
                self.assertIn(k, learned["strategy_weights"])
                self.assertGreaterEqual(float(learned["strategy_weights"][k]), 0.7)
                self.assertLessEqual(float(learned["strategy_weights"][k]), 1.3)


if __name__ == "__main__":
    unittest.main()
