from __future__ import annotations

import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.crypto_strategy_agent import CryptoStrategyAgent
from agents.risk_agent import RiskAgent


class PR34RolloutControlsTests(unittest.TestCase):
    def test_bear_rollout_short_enabled(self) -> None:
        agent = RiskAgent()
        agent.config.setdefault("bear_strategy", {})["rollout"] = {"stage": "hedge_only"}
        agent._bear_rollout_cfg = agent.config["bear_strategy"]["rollout"]
        self.assertFalse(agent._bear_short_enabled(risk_off=True))

        agent.config["bear_strategy"]["rollout"] = {"stage": "cautious_short"}
        agent._bear_rollout_cfg = agent.config["bear_strategy"]["rollout"]
        self.assertFalse(agent._bear_short_enabled(risk_off=False))
        self.assertTrue(agent._bear_short_enabled(risk_off=True))

    def test_rank_overlap_ratio(self) -> None:
        ratio = RiskAgent._rank_overlap_ratio({"A", "B", "C"}, {"B", "C", "D"})
        self.assertAlmostEqual(ratio, 2 / 3, places=6)

    def test_crypto_cost_guard(self) -> None:
        agent = CryptoStrategyAgent()
        good = {"median_dollar_volume_30d": 2_500_000.0, "spread_proxy": 0.01}
        bad = {"median_dollar_volume_30d": 10_000.0, "spread_proxy": 0.12}
        self.assertTrue(agent._passes_cost_guard(good))
        self.assertFalse(agent._passes_cost_guard(bad))

    def test_controls_rollout_modes(self) -> None:
        agent = RiskAgent()
        agent._controls_rollout_cfg = {"mode": "off"}
        agent._controls_mode = agent._controls_mode_name()
        self.assertFalse(agent._corr_gate_is_active())
        off_mult = agent._sizing_multiplier(score=0.8, vol_pct=0.3, corr_penalty=0.9)

        agent._controls_rollout_cfg = {"mode": "enforce"}
        agent._controls_mode = agent._controls_mode_name()
        self.assertTrue(agent._corr_gate_is_active())
        on_mult = agent._sizing_multiplier(score=0.8, vol_pct=0.3, corr_penalty=0.9)
        self.assertGreater(off_mult, on_mult)


if __name__ == "__main__":
    unittest.main()
