"""
Tests for issue #178 — SectorAnalyzerAgent per-cycle price cache.
"""
import json
import pathlib
import sys
import time
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.sector_analyzer_agent import SectorAnalyzerAgent, _CACHE_FILE


def _make_agent(ttl: float = 3600) -> SectorAnalyzerAgent:
    a = SectorAnalyzerAgent()
    a._cache_ttl = ttl
    return a


def _dummy_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = np.linspace(100, 110, n)
    return pd.DataFrame({"Close": closes, "Open": closes, "High": closes, "Low": closes, "Volume": 1e6}, index=idx)


class CacheWriteReadTests(unittest.TestCase):

    def setUp(self):
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()

    def tearDown(self):
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()

    def test_cache_file_written_after_first_download(self):
        agent = _make_agent()
        with patch.object(agent, "_fetch_history", return_value=_dummy_df()) as mock_fetch:
            agent._fetch_history_cached("SPY", 100)
        self.assertTrue(_CACHE_FILE.exists(), "cache file must exist after first download")
        cache = json.loads(_CACHE_FILE.read_text())
        self.assertIn("SPY_100d", cache)
        mock_fetch.assert_called_once()

    def test_second_call_within_ttl_uses_cache(self):
        agent = _make_agent(ttl=3600)
        with patch.object(agent, "_fetch_history", return_value=_dummy_df()) as mock_fetch:
            agent._fetch_history_cached("SPY", 100)
            agent._fetch_history_cached("SPY", 100)  # should hit cache
        mock_fetch.assert_called_once()  # NOT called twice

    def test_cache_miss_after_ttl_expired(self):
        agent = _make_agent(ttl=0.01)  # 10ms TTL
        with patch.object(agent, "_fetch_history", return_value=_dummy_df()) as mock_fetch:
            agent._fetch_history_cached("SPY", 100)
            time.sleep(0.05)
            agent._fetch_history_cached("SPY", 100)  # TTL expired → fresh fetch
        self.assertEqual(mock_fetch.call_count, 2)

    def test_force_refresh_bypasses_cache(self):
        agent = _make_agent(ttl=3600)
        with patch.object(agent, "_fetch_history", return_value=_dummy_df()) as mock_fetch:
            agent._fetch_history_cached("SPY", 100)
            agent._fetch_history_cached("SPY", 100, force=True)
        self.assertEqual(mock_fetch.call_count, 2)

    def test_breadth_cache_written_and_hit(self):
        agent = _make_agent(ttl=3600)
        members = ["AAPL", "MSFT"]
        with patch.object(agent, "_calculate_breadth", return_value=0.75) as mock_breadth:
            v1 = agent._calculate_breadth_cached(members)
            v2 = agent._calculate_breadth_cached(members)
        mock_breadth.assert_called_once()
        self.assertEqual(v1, 0.75)
        self.assertEqual(v2, 0.75)

    def test_breadth_cache_force_refresh(self):
        agent = _make_agent(ttl=3600)
        members = ["AAPL", "MSFT"]
        with patch.object(agent, "_calculate_breadth", return_value=0.75) as mock_breadth:
            agent._calculate_breadth_cached(members)
            agent._calculate_breadth_cached(members, force=True)
        self.assertEqual(mock_breadth.call_count, 2)

    def test_corrupt_cache_entry_falls_back_to_fetch(self):
        agent = _make_agent(ttl=3600)
        # Write a corrupt cache entry
        _CACHE_FILE.write_text(json.dumps({"SPY_100d": {"ts": time.time(), "records": "CORRUPT"}}))
        with patch.object(agent, "_fetch_history", return_value=_dummy_df()) as mock_fetch:
            df = agent._fetch_history_cached("SPY", 100)
        mock_fetch.assert_called_once()
        self.assertIsNotNone(df)


class CacheTTLConfigTests(unittest.TestCase):
    def test_cache_ttl_loaded_from_config(self):
        agent = SectorAnalyzerAgent()
        # Default from config.json should be 3600
        self.assertEqual(agent._cache_ttl, 3600.0)


if __name__ == "__main__":
    unittest.main()
