from __future__ import annotations

import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.alpha_hunter_agent import AlphaHunterAgent
from agents.bear_strategy_agent import BearStrategyAgent
from agents.bull_strategy_agent import BullStrategyAgent


class PR34StrategyEnhancementsTests(unittest.TestCase):
    def test_bull_ml_sets(self) -> None:
        ml_doc = {
            "latest_ranking": {
                "top_decile_long": [{"symbol": "AAPL"}],
                "bottom_decile_short": [{"symbol": "TSLA"}],
            }
        }
        longs, shorts = BullStrategyAgent._ml_sets(ml_doc)
        self.assertIn("AAPL", longs)
        self.assertIn("TSLA", shorts)

    def test_bull_dynamic_threshold_bounds(self) -> None:
        agent = BullStrategyAgent()
        agent._dynamic_threshold_enabled = True
        agent.read_json = lambda *_args, **_kwargs: {
            "advanced_macro": {"liquidity_proxy_score": -0.2},
            "risk_flags": [{"name": "X"}, {"name": "Y"}, {"name": "Z"}],
        }
        threshold = agent._dynamic_threshold(macro_bias=-0.4, macro_multiplier=0.8)
        self.assertGreaterEqual(threshold, 0.45)
        self.assertLessEqual(threshold, 0.90)

    def test_bear_ml_alpha_overlap_sets(self) -> None:
        agent = BearStrategyAgent()
        ml = {"latest_ranking": {"bottom_decile_short": [{"symbol": "NVDA"}, {"symbol": "META"}]}}
        alpha = {"signals": {"NVDA": {"signal_type": "SELL"}, "MSFT": {"signal_type": "HOLD"}}}
        self.assertEqual(agent._ml_short_symbols(ml), {"NVDA", "META"})
        self.assertEqual(agent._alpha_short_symbols(alpha), {"NVDA"})

    def test_alpha_ml_blend(self) -> None:
        agent = AlphaHunterAgent()
        agent._cfg["ml_blend_enabled"] = True
        agent._cfg["ml_blend_weight"] = 0.2
        buy, sell, contrib = agent._blend_with_ml(0.5, 0.2, 0.04)
        self.assertGreater(buy, 0.5)
        self.assertAlmostEqual(sell, 0.2, places=6)
        self.assertGreater(contrib, 0.0)


if __name__ == "__main__":
    unittest.main()

