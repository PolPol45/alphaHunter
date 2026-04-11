import pathlib
import sys
import unittest


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from adapters.fred_client import FredClient
from adapters.openinsider_client import OpenInsiderClient
from agents.base_agent import BaseAgent
from agents.news_data_agent import NewsDataAgent


class FredClientParseTests(unittest.TestCase):
    def test_parse_csv_filters_missing_values(self) -> None:
        csv_text = "DATE,VALUE\n2026-01-01,4.50\n2026-02-01,.\n2026-03-01,4.25\n"
        rows = FredClient.parse_csv(csv_text)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["value"], 4.25)
        self.assertTrue(rows[-1]["date"].startswith("2026-03-01"))

    def test_parse_csv_supports_series_named_columns(self) -> None:
        csv_text = "DATE,FEDFUNDS\n2026-01-01,4.50\n2026-02-01,4.25\n"
        rows = FredClient.parse_csv(csv_text)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["value"], 4.25)

    def test_connect_sets_down_state_on_probe_failure(self) -> None:
        client = FredClient({"enabled": True, "probe_series_id": "FEDFUNDS"})
        client._fetch_csv = lambda series_id: (_ for _ in ()).throw(RuntimeError("HTTP 503"))

        self.assertFalse(client.connect())
        self.assertEqual(client.state, "down")
        self.assertIn("HTTP 503", client.last_error)


class OpenInsiderParseTests(unittest.TestCase):
    def test_parse_filings_and_detect_clusters(self) -> None:
        html_doc = """
        <table>
          <tr>
            <th>X</th><th>Filed</th><th>TradeDate</th><th>Ticker</th><th>Company</th>
            <th>Insider</th><th>Title</th><th>TradeType</th><th>Type</th>
            <th>Price</th><th>Qty</th><th>Owned</th><th>Value</th>
          </tr>
          <tr>
            <td>1</td><td>2026-04-09 10:00:00</td><td>2026-04-08</td><td>AAPL</td><td>Apple</td>
            <td>Jane Doe</td><td>CEO</td><td>X</td><td>P - Purchase</td>
            <td>180.50</td><td>2000</td><td>10000</td><td>361000</td>
          </tr>
          <tr>
            <td>2</td><td>2026-04-08 12:00:00</td><td>2026-04-07</td><td>AAPL</td><td>Apple</td>
            <td>John Doe</td><td>CFO</td><td>X</td><td>P - Purchase</td>
            <td>181.00</td><td>1500</td><td>12000</td><td>271500</td>
          </tr>
          <tr>
            <td>3</td><td>2026-04-08 09:00:00</td><td>2026-04-07</td><td>TSLA</td><td>Tesla</td>
            <td>Max Roe</td><td>Director</td><td>X</td><td>S - Sale</td>
            <td>240.00</td><td>1000</td><td>8000</td><td>240000</td>
          </tr>
        </table>
        """
        filings = OpenInsiderClient.parse_filings(html_doc, lookback_days=60)
        clusters = OpenInsiderClient.detect_clusters(filings, min_filings=2, min_total_value=250000)

        self.assertEqual(len(filings), 2)
        self.assertEqual(clusters[0]["symbol"], "AAPL")
        self.assertEqual(clusters[0]["filing_count"], 2)
        self.assertGreater(clusters[0]["total_value_usd"], 600000)

    def test_get_activity_marks_source_down_on_fetch_error(self) -> None:
        client = OpenInsiderClient({"enabled": True})
        client._fetch_html = lambda: (_ for _ in ()).throw(RuntimeError("Connection refused"))

        activity = client.get_activity()

        self.assertEqual(activity["clusters"], [])
        self.assertFalse(client.is_connected())
        self.assertFalse(client.reachable)
        self.assertEqual(client.state, "down")
        self.assertIn("Connection refused", client.last_error)


class _StubFinnhub:
    last_error = None

    def is_connected(self) -> bool:
        return True

    def get_events(self) -> list[dict]:
        return [
            {
                "title": "[Macro] CPI cools more than expected",
                "summary": "Inflation slows and markets rally",
                "timestamp": "2026-04-10T08:00:00+00:00",
                "category": "macro",
                "event_type": "economic_calendar",
                "source": "finnhub/calendar",
                "symbols_affected": ["BTCUSDT", "ETHUSDT"],
                "sentiment": "bullish",
            }
        ]


class _StubYF:
    last_error = None

    def is_connected(self) -> bool:
        return True

    def get_news(self) -> list[dict]:
        return [
            {
                "symbol": "BTC-USD",
                "headline": "Bitcoin adoption accelerates after ETF inflows",
                "summary": "Institutional flows continue.",
                "published_at": "2026-04-10T09:15:00+00:00",
                "url": "https://example.com/btc",
                "source": "yfinance/BTC-USD",
                "category": "crypto",
            }
        ]


class _StubOpenInsider:
    last_error = None

    def is_connected(self) -> bool:
        return True

    def get_status(self) -> dict:
        return {
            "connected": True,
            "reachable": True,
            "state": "connected",
            "last_error": None,
            "last_success_at": "2026-04-10T07:00:00+00:00",
        }

    def get_activity(self) -> dict:
        return {
            "recent_filings": [
                {
                    "filed_at": "2026-04-10T07:00:00+00:00",
                    "symbol": "MSFT",
                    "insider": "Satya Nadella",
                    "title": "CEO",
                    "trade_type": "P - Purchase",
                    "price": 420.0,
                    "quantity": 1000.0,
                    "value_usd": 420000.0,
                }
            ],
            "clusters": [
                {
                    "symbol": "MSFT",
                    "filing_count": 2,
                    "unique_insiders": 2,
                    "insiders": ["Satya Nadella", "Amy Hood"],
                    "total_value_usd": 640000.0,
                    "latest_filed_at": "2026-04-10T07:00:00+00:00",
                    "avg_price": 418.0,
                    "alert_type": "INSIDER_CLUSTER_BUY",
                }
            ],
        }


class _StubFred:
    last_error = None
    state = "connected"

    def is_connected(self) -> bool:
        return True

    def get_series(self, series_id: str, limit: int = 24) -> list[dict]:
        if series_id == "FEDFUNDS":
            return [
                {"date": "2026-03-01T00:00:00+00:00", "value": 4.5},
                {"date": "2026-04-01T00:00:00+00:00", "value": 4.25},
            ]
        if series_id == "CPIAUCSL":
            rows = []
            for i in range(14):
                rows.append({"date": f"2025-{(i % 12) + 1:02d}-01T00:00:00+00:00", "value": 300 + i})
            return rows
        if series_id == "SP500":
            return [
                {"date": "2026-04-09T00:00:00+00:00", "value": 5200.0},
                {"date": "2026-04-10T00:00:00+00:00", "value": 5260.0},
            ]
        if series_id == "VIXCLS":
            return [
                {"date": "2026-04-09T00:00:00+00:00", "value": 18.0},
                {"date": "2026-04-10T00:00:00+00:00", "value": 19.0},
            ]
        return []


class FakeNewsDataAgent(NewsDataAgent):
    def __init__(self) -> None:
        BaseAgent.__init__(self, "news_data_agent")
        self._cfg = {
            "max_items": 20,
            "top_alerts": 5,
            "lookback_hours": 72,
            "high_impact_threshold": 0.72,
            "macro_adjustment_weight": 0.12,
            "news_adjustment_weight": 0.10,
            "compatibility_allow_stub": False,
        }
        self._max_items = 20
        self._top_alerts = 5
        self._lookback_hours = 72
        self._high_impact_threshold = 0.72
        self._macro_adjustment_weight = 0.12
        self._news_adjustment_weight = 0.10
        self._compatibility_allow_stub = False
        self._market_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._alpha_symbols = ["AAPL", "MSFT"]
        self._finnhub = _StubFinnhub()
        self._yfinance_news = _StubYF()
        self._openinsider = _StubOpenInsider()
        self._fred = _StubFred()


class NewsDataAgentTests(unittest.TestCase):
    def test_dedupe_prefers_latest_item(self) -> None:
        items = [
            {
                "headline": "Bitcoin adoption accelerates",
                "source": "yfinance/BTC-USD",
                "symbol": "BTCUSDT",
                "published_at": "2026-04-10T07:00:00+00:00",
            },
            {
                "headline": "Bitcoin adoption accelerates fast",
                "source": "yfinance/BTC-USD",
                "symbol": "BTCUSDT",
                "published_at": "2026-04-10T08:00:00+00:00",
            },
        ]
        deduped = NewsDataAgent._dedupe_items(items)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["published_at"], "2026-04-10T08:00:00+00:00")

    def test_run_generates_all_phase2_documents(self) -> None:
        agent = FakeNewsDataAgent()
        captured: dict[str, dict] = {}

        agent.write_json = lambda path, data: captured.__setitem__(path.name, data)
        agent.update_shared_state = lambda *args, **kwargs: None
        agent.read_json = lambda path: {"world_events": []} if path.name == "market_data.json" else {}
        agent._market_snapshot_from_yfinance = lambda metric, spec: {
            "label": spec.get("label", metric),
            "provider": "yfinance",
            "symbol": spec.get("symbol"),
            "value": 100.0,
            "previous_value": 99.0,
            "change_pct": 1.01,
            "unit": spec.get("unit", "index"),
            "fetched_at": "2026-04-10T00:00:00+00:00",
            "status": "ok",
        }

        self.assertTrue(agent.run())
        self.assertIn("news_feed.json", captured)
        self.assertIn("macro_snapshot.json", captured)
        self.assertIn("insider_activity.json", captured)
        self.assertGreaterEqual(len(captured["news_feed.json"]["items"]), 2)
        self.assertGreaterEqual(len(captured["insider_activity.json"]["clusters"]), 1)
        self.assertIn("series", captured["macro_snapshot.json"])
        self.assertIn("live_sources", captured["news_feed.json"]["source_status"])
        self.assertEqual(captured["macro_snapshot.json"]["source_status"]["fred"]["healthy_series"], 7)

    def test_scoring_keeps_bullish_crypto_headline_positive(self) -> None:
        text = "Bitcoin traders set $88K target as market bias finally tilts toward bulls"
        sentiment = NewsDataAgent._score_sentiment(text)

        self.assertGreater(sentiment, 0.0)

    def test_general_macro_headline_without_crypto_match_has_lower_relevance(self) -> None:
        agent = FakeNewsDataAgent()
        low = agent._score_relevance(
            "Britain's Tesco warns on food inflation amid Iran war",
            None,
            "general",
            False,
            [],
        )
        high = agent._score_relevance(
            "Bitcoin ETF inflows rise after CPI cools",
            "BTCUSDT",
            "crypto",
            False,
            ["BTCUSDT"],
        )

        self.assertLess(low, high)

    def test_stub_world_monitor_is_excluded_from_real_feed(self) -> None:
        agent = FakeNewsDataAgent()
        items = agent._fallback_world_events(
            [{"source": "worldmonitor/stub", "title": "demo", "category": "macro"}],
            "2026-04-10T00:00:00+00:00",
        )
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
