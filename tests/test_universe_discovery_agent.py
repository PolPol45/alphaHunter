from __future__ import annotations

import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.universe_discovery_agent import UniverseDiscoveryAgent


class FakeDiscoveryAgent(UniverseDiscoveryAgent):
    @staticmethod
    def _describe_symbol(symbol: str):
        return {
            "symbol": symbol,
            "long_name": symbol,
            "quote_type": "EQUITY",
            "exchange": "NMS",
            "sector": "Technology",
            "candidate_origin": "discovery_provider",
        }


class UniverseDiscoveryAgentTests(unittest.TestCase):
    def test_run_writes_candidates(self) -> None:
        agent = FakeDiscoveryAgent()
        captured = {}
        agent.write_json = lambda path, data: captured.__setitem__(path.name, data)
        agent.update_shared_state = lambda *args, **kwargs: None
        self.assertTrue(agent.run())
        self.assertIn("universe_discovery_candidates.json", captured)
        self.assertIn("candidates", captured["universe_discovery_candidates.json"])


if __name__ == "__main__":
    unittest.main()
