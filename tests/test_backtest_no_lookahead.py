from datetime import date

from agents.backtesting_agent import BacktestingAgent


def test_filter_no_lookahead_removes_future_candles():
    agent = BacktestingAgent()
    day = date(2026, 1, 10)
    cutoff_ts = 1768089599  # 2026-01-10 23:59:59 UTC approx

    payload = {
        "timestamp": "2026-01-10T00:00:00+00:00",
        "data_source": "test",
        "assets": {
            "BTCUSDT": {
                "last_price": 100.0,
                "ohlcv_1d": [
                    {"t": cutoff_ts - 10, "o": 1, "h": 1, "l": 1, "c": 100, "v": 1},
                    {"t": cutoff_ts + 10, "o": 1, "h": 1, "l": 1, "c": 200, "v": 1},
                ],
                "ohlcv_4h": [
                    {"t": cutoff_ts - 20, "o": 1, "h": 1, "l": 1, "c": 90, "v": 1},
                    {"t": cutoff_ts + 20, "o": 1, "h": 1, "l": 1, "c": 300, "v": 1},
                ],
                "orderbook": {"bids": [], "asks": []},
                "volume_24h": 1,
            }
        },
        "world_events": [],
    }

    filtered = agent._filter_no_lookahead(payload, day)
    c1 = filtered["assets"]["BTCUSDT"]["ohlcv_1d"]
    c4 = filtered["assets"]["BTCUSDT"]["ohlcv_4h"]

    assert len(c1) == 1
    assert len(c4) == 1
    assert c1[0]["c"] == 100
    assert c4[0]["c"] == 90
