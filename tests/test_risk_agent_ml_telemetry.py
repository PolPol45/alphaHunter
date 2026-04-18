from __future__ import annotations

import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.risk_agent import RiskAgent


class RiskAgentMLTelemetryTests(unittest.TestCase):
    def test_ml_telemetry_is_informative_only(self) -> None:
        payload = {
            "runtime_mode": "informative_only",
            "latest_date": "2026-01-01",
            "latest_ranking": {
                "top_decile_long": [{"symbol": "A"}],
                "bottom_decile_short": [{"symbol": "B"}],
            },
        }
        t = RiskAgent._ml_telemetry(payload)
        self.assertTrue(t["ingested_for_telemetry_only"])
        self.assertEqual(t["long_count"], 1)
        self.assertEqual(t["short_count"], 1)

    def test_extract_ml_returns_trade_candidates_when_active(self) -> None:
        agent = RiskAgent()
        payload = {
            "runtime_mode": "long_short_parallel",
            "latest_ranking": {
                "top_decile_long": [{"symbol": "AAPL", "predicted_excess_return": 0.02, "weight": 0.6}],
                "bottom_decile_short": [{"symbol": "TSLA", "predicted_excess_return": -0.01, "weight": -0.4}],
            },
        }
        out = agent._extract_ml(payload)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["_agent_source"], "ml_long")
        self.assertEqual(out[1]["_agent_source"], "ml_short")


if __name__ == "__main__":
    unittest.main()
