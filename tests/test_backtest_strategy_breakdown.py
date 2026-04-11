import pathlib
import sys
import unittest


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.backtesting_agent import BacktestingAgent


class BacktestStrategyBreakdownTests(unittest.TestCase):
    def test_strategy_breakdown_groups_trades(self) -> None:
        agent = BacktestingAgent.__new__(BacktestingAgent)
        trades = [
            {"strategy_bucket": "bull", "notional": 1000, "realized_pnl": 10},
            {"strategy_bucket": "bull", "notional": 1200, "realized_pnl": -5},
            {"strategy_bucket": "bear", "notional": 800, "realized_pnl": 4},
            {"strategy_bucket": "crypto", "notional": 900, "realized_pnl": 0},
            {"notional": 500, "realized_pnl": -1},  # unknown bucket
        ]
        res = agent._strategy_breakdown(trades, avg_equity=10_000)

        self.assertEqual(res["bull"]["trades"], 2)
        self.assertEqual(res["bear"]["trades"], 1)
        self.assertEqual(res["crypto"]["trades"], 1)
        self.assertEqual(res["unknown"]["trades"], 1)
        self.assertAlmostEqual(res["bull"]["realized_pnl"], 5.0, places=6)
        self.assertAlmostEqual(res["bull"]["turnover"], 0.22, places=6)


if __name__ == "__main__":
    unittest.main()
