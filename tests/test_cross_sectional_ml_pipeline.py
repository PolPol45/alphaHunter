from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtesting.cross_sectional_ml_pipeline import CrossSectionalMLPipeline


class CrossSectionalMLPipelineTests(unittest.TestCase):
    def test_rolling_splits_are_temporally_disjoint(self) -> None:
        cfg = {"split": {"train_years": 1, "val_years": 1, "test_years": 1}}
        pipe = CrossSectionalMLPipeline(pathlib.Path("a"), pathlib.Path("b"), pathlib.Path("c"), cfg)
        dates = pd.Series(pd.date_range("2020-01-03", periods=180, freq="W-FRI", tz="UTC"))
        splits = pipe._rolling_splits(dates)
        self.assertGreaterEqual(len(splits), 1)
        s = splits[0]
        self.assertEqual(len(s.train_dates & s.val_dates), 0)
        self.assertEqual(len(s.train_dates & s.test_dates), 0)
        self.assertEqual(len(s.val_dates & s.test_dates), 0)

    def test_pipeline_generates_signals_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            ds = base / "ml_dataset_weekly.jsonl"
            sig = base / "ml_signals.json"
            rep = base / "ml_report.json"

            dates = pd.date_range("2023-01-06", periods=80, freq="W-FRI", tz="UTC")
            symbols = [f"S{i:03d}" for i in range(30)]
            rows = []
            rng = np.random.default_rng(42)
            for d in dates:
                for i, s in enumerate(symbols):
                    m = rng.normal(0, 0.05)
                    liq = rng.uniform(1e6, 1e8)
                    target = 0.02 * m + rng.normal(0, 0.01)
                    rows.append({
                        "date_t": d,
                        "symbol": s,
                        "benchmark": "SPY",
                        "close": 100 + i,
                        "momentum_30d": m,
                        "median_dollar_volume_30d": liq,
                        "volatility_30d": abs(rng.normal(0.02, 0.01)),
                        "beta": rng.normal(1.0, 0.2),
                        "target_excess_return_t_plus_1": target,
                    })
            with open(ds, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(pd.Series(row).to_json(date_format="iso") + "\n")

            pipe = CrossSectionalMLPipeline(ds, sig, rep, {"runtime_mode": "informative_only", "split": {"train_years": 1, "val_years": 1, "test_years": 1}})
            summary = pipe.run()
            self.assertEqual(summary["status"], "ok")
            self.assertTrue(sig.exists())
            self.assertTrue(rep.exists())


if __name__ == "__main__":
    unittest.main()
