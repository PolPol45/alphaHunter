import pathlib
import sys
import unittest


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.execution_agent import ExecutionAgent


class ExecutionShortHelpersTests(unittest.TestCase):
    def test_cash_deltas_match_long_and_short_lifecycle(self) -> None:
        self.assertEqual(
            ExecutionAgent._open_cash_delta("long", 1_000.0, 5.0),
            -1_005.0,
        )
        self.assertEqual(
            ExecutionAgent._close_cash_delta("long", 1_100.0, 5.0),
            1_095.0,
        )
        self.assertEqual(
            ExecutionAgent._open_cash_delta("short", 1_000.0, 5.0),
            995.0,
        )
        self.assertEqual(
            ExecutionAgent._close_cash_delta("short", 900.0, 5.0),
            -905.0,
        )

    def test_position_market_contribution_marks_short_as_negative_notional(self) -> None:
        self.assertEqual(
            ExecutionAgent._position_market_contribution("long", 2.0, 100.0),
            200.0,
        )
        self.assertEqual(
            ExecutionAgent._position_market_contribution("short", 2.0, 100.0),
            -200.0,
        )

    def test_alpha_selected_symbols_uses_scanner_output_when_present(self) -> None:
        selected = ExecutionAgent._alpha_selected_symbols(
            {
                "scanner": {
                    "top_candidates": [
                        {"symbol": "MSFT", "signal_type": "BUY"},
                        {"symbol": "TSLA", "signal_type": "SELL"},
                        {"symbol": "QQQ", "signal_type": "HOLD"},
                    ]
                }
            },
            raw_signals={},
            threshold=0.58,
            limit=2,
        )

        self.assertEqual(selected, {"MSFT", "TSLA"})


if __name__ == "__main__":
    unittest.main()
