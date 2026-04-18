from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pandas as pd

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtesting.ml_dataset_builder import WeeklyMLDatasetBuilder


class WeeklyMLDatasetBuilderTests(unittest.TestCase):
    def test_excess_return_uses_t_plus_1_without_lookahead(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            features = base / "cross_sectional_features.jsonl"
            rows = [
                {"date": "2026-01-02", "symbol": "AAA", "benchmark": "SPY", "close": 100.0, "momentum_30d": 0.1},
                {"date": "2026-01-09", "symbol": "AAA", "benchmark": "SPY", "close": 110.0, "momentum_30d": 0.1},
                {"date": "2026-01-02", "symbol": "SPY", "benchmark": "SPY", "close": 100.0, "momentum_30d": 0.1},
                {"date": "2026-01-09", "symbol": "SPY", "benchmark": "SPY", "close": 105.0, "momentum_30d": 0.1},
            ]
            with open(features, "w", encoding="utf-8") as h:
                for r in rows:
                    h.write(pd.Series(r).to_json() + "\n")

            out = base / "ml_dataset_weekly.parquet"
            meta = base / "ml_dataset_weekly.meta.json"
            builder = WeeklyMLDatasetBuilder(features, out, meta, benchmark_default="SPY")
            result = builder.run()
            if result.output_path.suffix == ".parquet":
                df = pd.read_parquet(result.output_path)
            else:
                df = pd.read_json(result.output_path, lines=True)
            a = df[df["symbol"] == "AAA"].iloc[0]
            # AAA return = +10%, SPY return = +5% => excess = +5%
            self.assertAlmostEqual(float(a["target_excess_return_t_plus_1"]), 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
