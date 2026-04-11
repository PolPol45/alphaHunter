import pathlib
import sys
import unittest


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

try:
    from agents.risk_agent import RiskAgent
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    RiskAgent = None
    IMPORT_ERROR = exc


@unittest.skipIf(RiskAgent is None, f"risk agent deps unavailable: {IMPORT_ERROR}")
class RiskScannerSelectionTests(unittest.TestCase):
    def test_select_candidates_returns_top_ranked_symbols(self) -> None:
        selected = RiskAgent._select_candidates(
            signals={
                "BTCUSDT": {"signal_type": "BUY", "score": 0.78, "buy_score": 0.78},
                "ETHUSDT": {"signal_type": "BUY", "score": 0.66, "buy_score": 0.66},
                "SOLUSDT": {"signal_type": "SELL", "score": 0.74, "sell_score": 0.74},
                "BNBUSDT": {"signal_type": "BUY", "score": 0.59, "buy_score": 0.59},
            },
            positions={"ETHUSDT": {"quantity": 1.0, "side": "long"}},
            limit=2,
            threshold=0.60,
            allow_short=True,
        )

        self.assertEqual(selected, ["BTCUSDT", "SOLUSDT"])

    def test_select_candidates_ignores_new_shorts_when_disabled(self) -> None:
        selected = RiskAgent._select_candidates(
            signals={
                "BTCUSDT": {"signal_type": "SELL", "score": 0.81, "sell_score": 0.81},
                "ETHUSDT": {"signal_type": "BUY", "score": 0.73, "buy_score": 0.73},
            },
            positions={},
            limit=3,
            threshold=0.60,
            allow_short=False,
        )

        self.assertEqual(selected, ["ETHUSDT"])


if __name__ == "__main__":
    unittest.main()
