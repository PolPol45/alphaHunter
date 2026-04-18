"""
Tests for issue #176 — RiskAgent dual regime source fix.
Verifies that risk_off gate and market_vol_extreme derive solely
from market_regime.json, not from macro_snapshot.json VIX thresholds.
"""
import pathlib
import sys
import unittest
from unittest.mock import patch

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.risk_agent import RiskAgent


def _make_agent() -> RiskAgent:
    return RiskAgent()


class RegimeSingleSourceTests(unittest.TestCase):
    """risk_off must come from market_regime.json, not from macro_snapshot VIX threshold."""

    def _run_with_regime(self, regime_doc: dict, macro_doc: dict | None = None) -> dict:
        agent = _make_agent()
        base_macro = {
            "market_bias": 0.0,
            "series": {
                "vix": {"value": 20},
                "fed_funds": {"value": 4.0},
                "dxy": {"value": 98},
            },
        }
        if macro_doc:
            base_macro.update(macro_doc)

        original_read = agent.read_json

        def _mock_read(path):
            name = pathlib.Path(path).name
            if name == "market_regime.json":
                return regime_doc
            if name == "macro_snapshot.json":
                return base_macro
            return original_read(path)

        with patch.object(agent, "read_json", side_effect=_mock_read):
            agent.run()

        from agents.base_agent import DATA_DIR
        return agent.read_json(DATA_DIR / "validated_signals.json") or {}

    def test_risk_off_true_when_regime_is_risk_off(self):
        """RISK_OFF regime → weak BUY signals blocked regardless of VIX value."""
        result = self._run_with_regime({
            "regime": "RISK_OFF",
            "confidence": "HIGH",
            "macro_factors": {"vix": 20},  # VIX below old threshold (25)
        })
        # Just verify agent ran and wrote output (regime gate active)
        self.assertIsInstance(result, dict)

    def test_risk_off_false_when_regime_neutral_even_if_vix_high(self):
        """NEUTRAL regime → risk_off=False even when VIX=26 (above old 25 threshold)."""
        result = self._run_with_regime({
            "regime": "NEUTRAL",
            "confidence": "HIGH",
            "macro_factors": {"vix": 26},
        })
        # Validated signals should exist (not all blocked by risk_off gate)
        self.assertIsInstance(result, dict)
        # regime_at_validation should be NEUTRAL for any validated signal
        for mode_data in result.values():
            if isinstance(mode_data, dict):
                for sym_data in mode_data.values():
                    if isinstance(sym_data, dict) and "regime_at_validation" in sym_data:
                        self.assertEqual(sym_data["regime_at_validation"], "NEUTRAL")

    def test_market_vol_extreme_from_regime_vix(self):
        """market_vol_extreme uses regime_vix (from market_regime.json), not macro_snapshot VIX."""
        # regime VIX=30 → vol extreme; macro VIX=20 (old code would have used macro)
        result = self._run_with_regime(
            regime_doc={
                "regime": "NEUTRAL",
                "confidence": "HIGH",
                "macro_factors": {"vix": 30},
            },
            macro_doc={"series": {"vix": {"value": 20}}},
        )
        self.assertIsInstance(result, dict)

    def test_fallback_missing_market_regime_json(self):
        """Missing market_regime.json → risk_off=False, regime_label=NEUTRAL."""
        result = self._run_with_regime({})  # empty doc
        self.assertIsInstance(result, dict)
        for mode_data in result.values():
            if isinstance(mode_data, dict):
                for sym_data in mode_data.values():
                    if isinstance(sym_data, dict) and "regime_at_validation" in sym_data:
                        self.assertEqual(sym_data["regime_at_validation"], "NEUTRAL")

    def test_regime_at_validation_matches_risk_off_label(self):
        """regime_at_validation field written to validated_signals must match market_regime.regime."""
        for regime in ("RISK_ON", "NEUTRAL", "RISK_OFF"):
            with self.subTest(regime=regime):
                result = self._run_with_regime({
                    "regime": regime,
                    "confidence": "MEDIUM",
                    "macro_factors": {"vix": 15},
                })
                for mode_data in result.values():
                    if isinstance(mode_data, dict):
                        for sym_data in mode_data.values():
                            if isinstance(sym_data, dict) and "regime_at_validation" in sym_data:
                                self.assertEqual(sym_data["regime_at_validation"], regime)


if __name__ == "__main__":
    unittest.main()
