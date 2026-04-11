from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from agents.base_agent import BASE_DIR, DATA_DIR


class HistoricalDataStore:
    """Simple deterministic historical snapshot store for backtesting.

    It supports two sources:
    1) Pre-generated daily snapshots in `backtest_data/market_YYYY-MM-DD.json`
    2) Deterministic generation from the latest `data/market_data.json`
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.base_dir = BASE_DIR
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.bt_dir = self.base_dir / "backtest_data"
        self.bt_dir.mkdir(parents=True, exist_ok=True)

    def freeze_universe(self, universe: list[str]) -> list[str]:
        # Keeps ordering stable and preserves delisted-like placeholders.
        return list(dict.fromkeys(universe))

    def get_snapshot_at(self, day: date, universe: list[str]) -> dict:
        fp = self.bt_dir / f"market_{day.isoformat()}.json"
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["assets"] = self._filter_assets(payload.get("assets", {}), universe)
            return payload

        payload = self._generate_snapshot(day, universe)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def _filter_assets(self, assets: dict, universe: list[str]) -> dict:
        out: dict = {}
        for s in universe:
            if s in assets:
                out[s] = assets[s]
        return out

    def _generate_snapshot(self, day: date, universe: list[str]) -> dict:
        seed = int(day.strftime("%Y%m%d"))
        rng = random.Random(seed)

        live = self._read_json(self.data_dir / "market_data.json")
        live_assets = live.get("assets", {}) if live else {}

        assets: dict = {}
        for symbol in universe:
            base = live_assets.get(symbol, {})
            base_price = float(base.get("last_price", 100.0)) if base else 100.0 + (abs(hash(symbol)) % 5000)
            # Day-specific drift + noise (deterministic by symbol/day)
            srng = random.Random(seed + abs(hash(symbol)) % 10_000)
            shock = srng.gauss(0, 0.01)
            drift = (srng.random() - 0.5) * 0.004
            price = max(base_price * (1.0 + drift + shock), 0.0001)

            ohlcv_1d = self._build_ohlcv(day, price, 300, 1, srng)
            ohlcv_4h = self._build_ohlcv(day, price, 200, 4, srng)
            assets[symbol] = {
                "last_price": round(price, 6),
                "ohlcv_1d": ohlcv_1d,
                "ohlcv_4h": ohlcv_4h,
                "orderbook": self._orderbook(price, srng),
                "volume_24h": round(max(1.0, abs(price) * (5 + rng.random() * 20)), 4),
            }

        return {
            "timestamp": datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "data_source": "historical_store",
            "assets": assets,
            "world_events": [],
        }

    def _build_ohlcv(self, day: date, last_price: float, bars: int, hours_per_bar: int, rng: random.Random) -> list[dict]:
        step_seconds = hours_per_bar * 3600
        end_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=23)
        start = int(end_dt.timestamp()) - bars * step_seconds
        price = last_price
        out: list[dict] = []
        for i in range(bars):
            o = price
            chg = rng.gauss(0, 0.004 if hours_per_bar == 4 else 0.008)
            c = max(o * (1 + chg), 0.0001)
            h = max(o, c) * (1 + abs(rng.gauss(0, 0.002)))
            l = min(o, c) * (1 - abs(rng.gauss(0, 0.002)))
            v = abs(rng.gauss(50, 20))
            out.append({"t": start + i * step_seconds, "o": round(o, 6), "h": round(h, 6), "l": round(l, 6), "c": round(c, 6), "v": round(v, 4)})
            price = c
        return out

    def _orderbook(self, mid: float, rng: random.Random) -> dict:
        bids = []
        asks = []
        for i in range(1, 11):
            spread = 0.0005 * i
            bids.append([round(mid * (1 - spread), 6), round(rng.uniform(0.1, 2.0), 4)])
            asks.append([round(mid * (1 + spread), 6), round(rng.uniform(0.1, 2.0), 4)])
        return {"bids": bids, "asks": asks}

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
