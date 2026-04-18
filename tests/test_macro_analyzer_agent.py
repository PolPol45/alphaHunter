from __future__ import annotations

import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.macro_analyzer_agent import MacroAnalyzerAgent


class MacroAnalyzerAgentTests(unittest.TestCase):
    def test_builds_market_regime_from_macro_snapshot(self) -> None:
        agent = MacroAnalyzerAgent()
        snapshot = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "market_bias": -0.25,
            "series": {
                "fed_funds": {"status": "ok", "value": 4.5},
                "cpi_yoy": {"status": "ok", "value": 3.2},
            },
            "advanced_macro": {"real_rate_10y": 1.9},
            "liquidity": {"liquidity_proxy_score": -0.1},
            "risk_flags": [{"code": "REAL_RATE_PRESSURE"}],
        }
        market = agent._from_snapshot(snapshot)
        self.assertEqual(market["regime"], "RISK_OFF")
        self.assertEqual(market["score"], -0.25)
        self.assertEqual(market["macro_factors"]["fed_proxy"], 4.5)
        self.assertEqual(market["source"], "macro_snapshot")


if __name__ == "__main__":
    unittest.main()
