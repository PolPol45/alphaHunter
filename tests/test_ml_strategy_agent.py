from __future__ import annotations

import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.ml_strategy_agent import MLStrategyAgent


class MLStrategyAgentTests(unittest.TestCase):
    def test_refresh_due_without_state(self) -> None:
        agent = MLStrategyAgent()
        agent.read_json = lambda path: {}
        self.assertTrue(agent._is_refresh_due())


if __name__ == "__main__":
    unittest.main()
