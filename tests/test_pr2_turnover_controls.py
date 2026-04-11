import pathlib
import sys
import unittest
from datetime import datetime, timezone, timedelta


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.execution_agent import ExecutionAgent


class PR2TurnoverControlsTests(unittest.TestCase):
    def _agent_stub(self) -> ExecutionAgent:
        agent = ExecutionAgent.__new__(ExecutionAgent)
        agent._turnover_cfg = {
            "enabled": True,
            "rebalance_threshold_pct": 0.05,
            "min_holding_hours": 24,
            "cooldown_after_close_hours": 12,
            "frequent_trade_window_hours": 72,
            "max_trades_per_symbol": 4,
        }
        return agent

    def test_blocks_reversal_before_min_holding_period(self) -> None:
        now = datetime.now(timezone.utc)
        agent = self._agent_stub()
        agent._now = lambda: now

        ok, reason = agent._turnover_gate(
            mode="retail",
            symbol="BTCUSDT",
            vsig={"position_size_usdt": 1000},
            portfolio={"trades": []},
            existing_pos={
                "quantity": 1.0,
                "side": "long",
                "opened_at": (now - timedelta(hours=2)).isoformat(),
                "current_price": 100.0,
                "avg_entry_price": 100.0,
            },
            desired_side="short",
        )
        self.assertFalse(ok)
        self.assertIn("min_holding_period_not_reached", reason or "")

    def test_blocks_entry_after_recent_close_cooldown(self) -> None:
        now = datetime.now(timezone.utc)
        agent = self._agent_stub()
        agent._now = lambda: now

        ok, reason = agent._turnover_gate(
            mode="retail",
            symbol="ETHUSDT",
            vsig={"position_size_usdt": 1000},
            portfolio={
                "trades": [
                    {
                        "symbol": "ETHUSDT",
                        "timestamp": (now - timedelta(hours=3)).isoformat(),
                        "reason": "TAKE_PROFIT",
                    }
                ]
            },
            existing_pos={"quantity": 0.0},
            desired_side="long",
        )
        self.assertFalse(ok)
        self.assertIn("cooldown_after_close", reason or "")


if __name__ == "__main__":
    unittest.main()
