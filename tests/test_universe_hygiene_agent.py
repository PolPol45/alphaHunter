from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.universe_hygiene_agent import UniverseHygieneAgent


def _history(days: int, close: float, volume: float, stale_days: int = 0) -> pd.DataFrame:
    end = datetime.now(timezone.utc) - timedelta(days=stale_days)
    idx = pd.date_range(end=end, periods=days, freq="D")
    return pd.DataFrame({"Close": [close] * days, "Volume": [volume] * days}, index=idx)


class _FakeUniverseHygieneAgent(UniverseHygieneAgent):
    def __init__(self, histories: dict[str, pd.DataFrame | None]) -> None:
        super().__init__()
        self._histories = histories

    def _collect_symbols(self) -> list[str]:
        return list(self._histories.keys())

    def _fetch_history(self, symbol: str):
        return self._histories.get(symbol)


class UniverseHygieneAgentTests(unittest.TestCase):
    def test_excludes_delisted_or_no_data(self) -> None:
        agent = _FakeUniverseHygieneAgent({"DEAD": None})
        row = agent._evaluate_symbol("DEAD")
        self.assertEqual(row["status"], "excluded")
        self.assertIn("NO_DATA", row["reason_codes"])

    def test_excludes_illiquid_symbols(self) -> None:
        agent = _FakeUniverseHygieneAgent({"ILLQ": _history(days=120, close=10.0, volume=100.0)})
        row = agent._evaluate_symbol("ILLQ")
        self.assertEqual(row["status"], "excluded")
        self.assertIn("ILLIQUID", row["reason_codes"])

    def test_excludes_stale_symbols(self) -> None:
        agent = _FakeUniverseHygieneAgent({"STALE": _history(days=120, close=120.0, volume=5_000_000.0, stale_days=20)})
        row = agent._evaluate_symbol("STALE")
        self.assertEqual(row["status"], "excluded")
        self.assertIn("STALE_DATA", row["reason_codes"])

    def test_collect_symbols_returns_candidate_origin_map(self) -> None:
        agent = UniverseHygieneAgent()
        symbols, origin = agent._collect_symbols()
        self.assertIsInstance(symbols, list)
        self.assertIsInstance(origin, dict)


if __name__ == "__main__":
    unittest.main()
