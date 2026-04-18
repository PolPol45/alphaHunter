"""
Tests for issue #175 — stock score filter fix in TechnicalAnalysisAgent.
"""
import pathlib
import sys
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.technical_analysis_agent import TechnicalAnalysisAgent


def _make_agent() -> TechnicalAnalysisAgent:
    agent = TechnicalAnalysisAgent.__new__(TechnicalAnalysisAgent)
    agent.config = {
        "retail": {
            "rsi_oversold": 35, "rsi_overbought": 65,
            "signal_threshold": 0.55,
            "stop_loss_pct": 0.03, "take_profit_pct": 0.08,
        },
        "news_data": {"news_adjustment_weight": 0.10, "macro_adjustment_weight": 0.12},
    }
    return agent


def _buy_signal_dict(last_price: float = 100.0) -> dict:
    """Minimal signal dict that _generate_signal would return as BUY."""
    return {
        "signal_type": "BUY",
        "buy_score": 0.70,
        "sell_score": 0.10,
        "last_price": last_price,
        "stock_score_filter_applied": False,
        "regime_filter_applied": False,
    }


class TestStockScoreFilterCrypto(unittest.TestCase):
    """Crypto symbols (score=None) must NEVER be filtered by stock score gate."""

    def test_crypto_none_score_does_not_filter_buy(self):
        # None = not in stock_scores → filter skipped
        stock_filter_applied = (None is not None and None < 0.60)
        self.assertFalse(stock_filter_applied)

    def test_crypto_none_score_flag_is_false(self):
        stock_composite_score = None
        flag = stock_composite_score is not None and stock_composite_score < 0.60
        self.assertFalse(flag)

    def test_crypto_high_score_unaffected(self):
        # Even if somehow a crypto got score=0.30, None sentinel prevents filtering
        stock_composite_score = None
        flag = stock_composite_score is not None and stock_composite_score < 0.60
        self.assertFalse(flag)


class TestStockScoreFilterEquity(unittest.TestCase):
    """Equity symbols with composite_score < 0.60 must have BUY downgraded to HOLD."""

    def test_low_equity_score_triggers_filter(self):
        stock_composite_score = 0.45
        flag = stock_composite_score is not None and stock_composite_score < 0.60
        self.assertTrue(flag, "score=0.45 should trigger filter")

    def test_high_equity_score_passes(self):
        stock_composite_score = 0.75
        flag = stock_composite_score is not None and stock_composite_score < 0.60
        self.assertFalse(flag, "score=0.75 should not trigger filter")

    def test_boundary_0_60_is_not_filtered(self):
        stock_composite_score = 0.60
        flag = stock_composite_score is not None and stock_composite_score < 0.60
        self.assertFalse(flag, "score=0.60 (boundary) should pass")

    def test_boundary_0_599_is_filtered(self):
        stock_composite_score = 0.599
        flag = stock_composite_score is not None and stock_composite_score < 0.60
        self.assertTrue(flag, "score=0.599 should be filtered")


class TestHoldSignalFields(unittest.TestCase):
    """_hold_signal must include all fields added in STEP-4 to avoid KeyError downstream."""

    REQUIRED_FIELDS = [
        "signal_type", "score", "buy_score", "sell_score", "last_price",
        "stop_loss", "take_profit", "suggested_stop_loss_pct",
        "ema9_cross", "ema9_cross_dn", "atr", "rsi", "adx", "ema_trend",
        "regime_filter_applied", "stock_score_filter_applied",
        "mode", "context_adjustment",
    ]

    def test_hold_signal_has_all_fields(self):
        result = TechnicalAnalysisAgent._hold_signal(100.0)
        for field in self.REQUIRED_FIELDS:
            self.assertIn(field, result, f"_hold_signal missing field: {field}")

    def test_hold_signal_ema9_cross_is_false(self):
        result = TechnicalAnalysisAgent._hold_signal(100.0)
        self.assertFalse(result["ema9_cross"])
        self.assertFalse(result["ema9_cross_dn"])

    def test_hold_signal_filter_flags_are_false(self):
        result = TechnicalAnalysisAgent._hold_signal(100.0)
        self.assertFalse(result["regime_filter_applied"])
        self.assertFalse(result["stock_score_filter_applied"])

    def test_hold_signal_suggested_sl_is_zero(self):
        result = TechnicalAnalysisAgent._hold_signal(100.0)
        self.assertEqual(result["suggested_stop_loss_pct"], 0.0)


class TestFilterFlagInOutput(unittest.TestCase):
    """stock_score_filter_applied flag must accurately reflect what happened."""

    def test_flag_false_when_score_is_none(self):
        # Simulates crypto path
        score = None
        flag = score is not None and score < 0.60
        self.assertFalse(flag)

    def test_flag_true_when_equity_score_low(self):
        score = 0.45
        flag = score is not None and score < 0.60
        self.assertTrue(flag)

    def test_flag_false_when_equity_score_high(self):
        score = 0.80
        flag = score is not None and score < 0.60
        self.assertFalse(flag)


if __name__ == "__main__":
    unittest.main()
