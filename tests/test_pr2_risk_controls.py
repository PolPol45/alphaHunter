import pathlib
import sys
import unittest


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.risk_agent import RiskAgent


class PR2RiskControlsTests(unittest.TestCase):
    def test_pearson_corr_is_high_for_aligned_series(self) -> None:
        a = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02]
        b = [0.015, 0.018, -0.008, 0.028, 0.012, -0.019]
        corr = RiskAgent._pearson_corr(a, b)
        self.assertIsNotNone(corr)
        self.assertGreater(corr, 0.9)

    def test_sizing_multiplier_penalizes_high_correlation(self) -> None:
        agent = RiskAgent()
        low_corr = agent._sizing_multiplier(score=0.75, vol_pct=0.02, corr_penalty=0.1)
        high_corr = agent._sizing_multiplier(score=0.75, vol_pct=0.02, corr_penalty=0.9)
        self.assertGreater(low_corr, high_corr)

    def test_extract_bear_hedge_only_mode(self) -> None:
        agent = RiskAgent()
        agent.config.setdefault("bear_strategy", {})["mode"] = "bear_hedge_only"

        bear_sig = {
            "allocations": {
                "hedge_etfs": [{"symbol": "VIXY"}, {"symbol": "SDS"}],
                "short_candidates": [{"symbol": "XYZ"}],
                "bankruptcy_risk": [{"symbol": "ABC"}],
            }
        }
        out = agent._extract_bear(bear_sig)
        self.assertEqual([x["symbol"] for x in out], ["VIXY", "SDS"])
        self.assertTrue(all(x.get("_signal_type") == "BUY" for x in out))


if __name__ == "__main__":
    unittest.main()
