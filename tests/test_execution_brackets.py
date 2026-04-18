import logging
import pathlib
import sys
import unittest
from datetime import datetime, timezone


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.execution_agent import ExecutionAgent


class ExecutionBracketTests(unittest.TestCase):
    def _agent_stub(self) -> ExecutionAgent:
        agent = ExecutionAgent.__new__(ExecutionAgent)
        agent.config = {
            "retail": {"allow_short": True, "capital": 10_000.0},
            "institutional": {"allow_short": True, "capital": 10_000.0},
            "simulation": {"slippage_pct": 0.0, "commission_pct": 0.0},
        }
        agent._sim = agent.config["simulation"]
        agent._turnover_cfg = {"enabled": False, "min_holding_hours": 0}
        agent.logger = logging.getLogger("test.execution.brackets")
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        agent._now = lambda: now
        agent._now_iso = lambda: now.isoformat()
        agent._greeks_enforcement_level = lambda mode, vsig: 0
        agent._turnover_gate = lambda **kwargs: (True, None)
        agent.read_json = lambda path: {}
        agent._sim_params = lambda: {"slippage_pct": 0.0, "commission_pct": 0.0}
        return agent

    @staticmethod
    def _portfolio() -> dict:
        return {
            "cash": 10_000.0,
            "total_equity": 10_000.0,
            "realized_pnl": 0.0,
            "positions": {},
            "trades": [],
        }

    def test_build_exit_levels_from_stop_loss_pct(self) -> None:
        levels = ExecutionAgent._build_exit_levels(
            "BUY",
            875.0,
            {"stop_loss_pct": 0.022},
        )

        self.assertAlmostEqual(levels["stop_loss_price"], 855.75, places=4)
        self.assertAlmostEqual(levels["take_profit_price"], 913.5, places=4)
        self.assertEqual(levels["r_multiple"], 2.0)

    def test_simulate_fill_sets_explicit_stop_and_take_profit(self) -> None:
        agent = self._agent_stub()
        portfolio = self._portfolio()

        agent._simulate_fill(
            "NVDA",
            {
                "signal_type": "BUY",
                "entry_price": 875.0,
                "quantity": 5.0,
                "position_size_usdt": 4375.0,
                "stop_loss_pct": 0.022,
                "agent_source": "risk_agent",
                "strategy_bucket": "bull",
            },
            portfolio,
            "retail",
        )

        pos = portfolio["positions"]["NVDA"]
        trade = portfolio["trades"][-1]
        self.assertAlmostEqual(pos["stop_loss_price"], 855.75, places=4)
        self.assertAlmostEqual(pos["take_profit_price"], 913.5, places=4)
        self.assertEqual(pos["r_multiple"], 2.0)
        self.assertAlmostEqual(trade["stop_loss_price"], 855.75, places=4)
        self.assertAlmostEqual(trade["take_profit_price"], 913.5, places=4)

    def test_simulation_closes_position_on_stop_loss_hit(self) -> None:
        agent = self._agent_stub()
        portfolio = self._portfolio()
        agent._simulate_fill(
            "NVDA",
            {
                "signal_type": "BUY",
                "entry_price": 875.0,
                "quantity": 5.0,
                "position_size_usdt": 4375.0,
                "stop_loss_pct": 0.022,
            },
            portfolio,
            "retail",
        )

        agent._update_sim_positions(
            portfolio,
            {"assets": {"NVDA": {"last_price": 850.0}}},
        )

        self.assertEqual(portfolio["positions"]["NVDA"]["quantity"], 0.0)
        self.assertEqual(portfolio["trades"][-1]["reason"], "SL_HIT")
        self.assertLess(portfolio["trades"][-1]["realized_pnl"], 0.0)

    def test_simulation_closes_position_on_take_profit_hit_at_two_r(self) -> None:
        agent = self._agent_stub()
        portfolio = self._portfolio()
        agent._simulate_fill(
            "NVDA",
            {
                "signal_type": "BUY",
                "entry_price": 875.0,
                "quantity": 5.0,
                "position_size_usdt": 4375.0,
                "stop_loss_pct": 0.022,
            },
            portfolio,
            "retail",
        )

        risk = (
            portfolio["positions"]["NVDA"]["avg_entry_price"]
            - portfolio["positions"]["NVDA"]["stop_loss_price"]
        ) * portfolio["positions"]["NVDA"]["quantity"]

        agent._check_exits(
            mode="retail",
            portfolio=portfolio,
            market_doc={"assets": {"NVDA": {"last_price": 920.0}}},
            stop_orders={},
            ibkr_live=False,
        )

        self.assertEqual(portfolio["positions"]["NVDA"]["quantity"], 0.0)
        self.assertEqual(portfolio["trades"][-1]["reason"], "TP_HIT")
        self.assertAlmostEqual(portfolio["trades"][-1]["realized_pnl"], 2.0 * risk, places=4)


if __name__ == "__main__":
    unittest.main()
