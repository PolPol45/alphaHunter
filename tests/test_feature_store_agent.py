from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import datetime, timezone

import pandas as pd

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.feature_store_agent import FeatureStoreAgent


class FeatureStoreAgentTests(unittest.TestCase):
    def test_market_features_include_expected_fields(self) -> None:
        idx = pd.date_range(end=datetime.now(timezone.utc), periods=200, freq="D")
        closes = pd.Series(range(100, 300), index=idx, dtype=float)
        volumes = pd.Series([1_000_000.0] * len(idx), index=idx)
        highs = closes * 1.01
        lows = closes * 0.99

        history = pd.DataFrame({"Close": closes, "Volume": volumes, "High": highs, "Low": lows}, index=idx)
        feat = FeatureStoreAgent._market_features(history)

        self.assertIn("momentum_7d", feat)
        self.assertIn("momentum_30d", feat)
        self.assertIn("momentum_90d", feat)
        self.assertIn("volatility_30d", feat)
        self.assertIn("drawdown_rolling_90d", feat)
        self.assertIn("median_dollar_volume_30d", feat)


if __name__ == "__main__":
    unittest.main()
