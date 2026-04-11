"""
Market Data Agent
=================
Fetches two timeframes per symbol:
  ohlcv_1d  — daily candles  (used by institutional signal engine + BL)
  ohlcv_4h  — 4h candles     (used by retail signal engine as primary)

Primary source: OpenBB (yfinance)
Fallback:       Simulation (random-walk)

Output (data/market_data.json):
  {
    "timestamp":   "...",
    "data_source": "openbb" | "simulation",
    "assets": {
      "BTCUSDT": {
        "last_price": float,
        "ohlcv_1d":   [{t,o,h,l,c,v}, ...],
        "ohlcv_4h":   [{t,o,h,l,c,v}, ...],
        "orderbook":  {"bids": [...], "asks": [...]},
        "volume_24h": float
      }, ...
    },
    "world_events": [...]
  }
"""

from __future__ import annotations

import hashlib
import random
import time
import uuid
from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR, BASE_DIR
from adapters.historical_data_store import HistoricalDataStore

_SIM_BASE_PRICES = {
    "BTCUSDT": 65000.0,
    "ETHUSDT": 3500.0,
    "SOLUSDT": 150.0,
    "BNBUSDT": 600.0,
}
_SIM_VOLUME_RANGES = {
    "BTCUSDT": (5.0, 50.0),
    "ETHUSDT": (20.0, 200.0),
    "SOLUSDT": (500.0, 5000.0),
    "BNBUSDT": (50.0, 500.0),
}
_BUF_1D  = 300
_BUF_4H  = 200

# Gap D — structured simulated world events pool
_EVENT_POOL = [
    {"type": "MACRO",      "title": "Fed FOMC Meeting — Rate Decision",        "impact": "HIGH"},
    {"type": "MACRO",      "title": "US CPI Inflation Data Release",            "impact": "HIGH"},
    {"type": "MACRO",      "title": "US Non-Farm Payrolls Report",              "impact": "HIGH"},
    {"type": "MACRO",      "title": "ECB Interest Rate Decision",               "impact": "MEDIUM"},
    {"type": "MACRO",      "title": "US GDP Quarterly Data",                    "impact": "MEDIUM"},
    {"type": "MACRO",      "title": "US Dollar Index Breakout",                 "impact": "MEDIUM"},
    {"type": "CRYPTO",     "title": "BTC ETF Net Inflow Surge",                 "impact": "HIGH"},
    {"type": "CRYPTO",     "title": "Ethereum Staking Yield Update",            "impact": "LOW"},
    {"type": "CRYPTO",     "title": "On-chain: Whale Accumulation Detected",    "impact": "MEDIUM"},
    {"type": "CRYPTO",     "title": "Stablecoin Market Cap Expansion",          "impact": "LOW"},
    {"type": "REGULATORY", "title": "SEC Crypto Market Structure Guidance",     "impact": "HIGH"},
    {"type": "REGULATORY", "title": "EU MiCA Compliance Deadline",              "impact": "MEDIUM"},
    {"type": "REGULATORY", "title": "US Treasury Stablecoin Legislation",       "impact": "HIGH"},
]


class MarketDataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("market_data_agent")
        self._market_symbols = self._resolve_market_symbols()

        # Binance — primary crypto data source (public API, no auth)
        from adapters.binance_client import BinanceClient
        self._binance_client = BinanceClient(self.config.get("binance", {}))
        if self._binance_client.connect():
            self.logger.info(
                f"Binance adapter ready | scan universe={len(self._market_symbols)} symbols"
            )
        else:
            self.logger.warning(
                f"Binance connect failed ({self._binance_client.last_error}) — will try OpenBB"
            )

        # OpenBB — fallback if Binance is unreachable
        self._openbb_client = None
        self._openbb_enabled = self.config.get("openbb", {}).get("enabled", False)
        if self._openbb_enabled:
            from adapters.openbb_client import OpenBBClient
            self._openbb_client = OpenBBClient(self.config["openbb"])
            connected = self._openbb_client.connect()
            if connected:
                self.logger.info("OpenBB adapter ready (fallback)")
            else:
                self.logger.warning(
                    f"OpenBB connect failed ({self._openbb_client.last_error})"
                )

        self._wm_client = None
        wm_cfg = self.config.get("world_monitor", {})
        if wm_cfg.get("enabled", False):
            from adapters.world_monitor_client import WorldMonitorClient
            self._wm_client = WorldMonitorClient(wm_cfg)
            self._wm_client.connect()
            self.logger.info(f"World Monitor adapter ready (mode={self._wm_client.mode})")

        self._sim_state: dict[str, dict] = {}
        self._sim_initialized = False
        self._historical_store = HistoricalDataStore()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        self.mark_running()
        try:
            bt_ctx = self.read_json(DATA_DIR / "backtest_context.json")
            if bt_ctx.get("enabled"):
                payload = self._run_historical(bt_ctx)
                self.write_json(DATA_DIR / "market_data.json", payload)
                self.update_shared_state("data_freshness.market_data", payload["timestamp"])
                self.logger.info(
                    f"[historical] {len(payload['assets'])} symbols | "
                    f"ts={payload['timestamp']} | events={len(payload['world_events'])}"
                )
                self.mark_done()
                return True

            payload = {
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "data_source": "unknown",
                "assets":      {},
                "world_events": [],
            }

            if not self._try_binance(payload):
                if not self._try_openbb(payload):
                    self._run_simulation(payload)

            payload["world_events"] = self._fetch_world_events()

            self.write_json(DATA_DIR / "market_data.json", payload)
            self.update_shared_state("data_freshness.market_data", payload["timestamp"])

            prices_str = ", ".join(
                f"{s}={d['last_price']:.2f}" for s, d in payload["assets"].items()
            )
            self.logger.info(
                f"[{payload['data_source']}] {len(payload['assets'])} symbols | "
                f"{prices_str} | events={len(payload['world_events'])}"
            )
            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    def _run_historical(self, bt_ctx: dict) -> dict:
        ts = bt_ctx.get("lookahead_cutoff") or bt_ctx.get("simulated_now")
        if not ts:
            raise ValueError("backtest_context requires lookahead_cutoff/simulated_now")
        day = datetime.fromisoformat(ts).date()
        frozen_universe = bt_ctx.get("frozen_universe") or self._market_symbols
        payload = self._historical_store.get_snapshot_at(day, frozen_universe)
        payload["data_source"] = "historical_store"
        payload["world_events"] = []
        return payload

    # ------------------------------------------------------------------ #
    # Binance data fetch (primary, dual timeframe)                        #
    # ------------------------------------------------------------------ #

    def _try_binance(self, payload: dict) -> bool:
        if not self._binance_client.ensure_connected():
            return False

        failed: list[str] = []
        assets: dict = {}
        for symbol in self._market_symbols:
            try:
                candles_1d = self._binance_client.get_ohlcv(symbol)
                candles_4h = self._binance_client.get_ohlcv_4h(symbol)
                if not candles_1d:
                    raise ValueError(f"Empty 1d candles for {symbol}")
                last_price = float(candles_1d[-1]["c"])
                assets[symbol] = {
                    "last_price": last_price,
                    "ohlcv_1d":   candles_1d,
                    "ohlcv_4h":   candles_4h if candles_4h else candles_1d,
                    "orderbook":  self._generate_orderbook(last_price),
                    "volume_24h": round(float(candles_1d[-1]["v"]), 4),
                }
            except Exception as exc:
                self.logger.warning(f"Binance fetch skipped for {symbol}: {exc}")
                failed.append(symbol)

        if not assets:
            self.logger.warning("Binance returned no data — trying OpenBB")
            return False

        if failed:
            self.logger.info(
                f"Binance partial fetch: {len(assets)} ok, {len(failed)} skipped ({', '.join(failed)})"
            )

        payload["assets"]      = assets
        payload["data_source"] = "binance"
        return True

    # ------------------------------------------------------------------ #
    # OpenBB data fetch (fallback, dual timeframe)                        #
    # ------------------------------------------------------------------ #

    def _try_openbb(self, payload: dict) -> bool:
        if not self._openbb_enabled or not self._openbb_client:
            return False
        if not self._openbb_client.is_connected():
            if not self._openbb_client.connect():
                return False

        failed: list[str] = []
        assets: dict = {}
        for symbol in self._market_symbols:
            try:
                candles_1d = self._openbb_client.get_ohlcv(symbol)
                candles_4h = self._openbb_client.get_ohlcv_4h(symbol)
                if not candles_1d:
                    raise ValueError(f"Empty 1d candles for {symbol}")
                last_price = float(candles_1d[-1]["c"])
                assets[symbol] = {
                    "last_price": last_price,
                    "ohlcv_1d":   candles_1d,
                    "ohlcv_4h":   candles_4h if candles_4h else candles_1d,
                    "orderbook":  self._generate_orderbook(last_price),
                    "volume_24h": round(sum(c["v"] for c in candles_1d[-1:]), 4),
                }
            except Exception as exc:
                self.logger.warning(f"OpenBB fetch skipped for {symbol}: {exc}")
                failed.append(symbol)

        if not assets:
            self.logger.warning("OpenBB returned no data for any symbol — falling back to simulation")
            return False

        if failed:
            self.logger.info(f"OpenBB partial fetch: {len(assets)} ok, {len(failed)} skipped ({', '.join(failed)})")

        payload["assets"]      = assets
        payload["data_source"] = "openbb"
        return True

    # ------------------------------------------------------------------ #
    # Simulation fallback                                                  #
    # ------------------------------------------------------------------ #

    def _run_simulation(self, payload: dict) -> None:
        if not self._sim_initialized:
            self._bootstrap_simulation()
            self._sim_initialized = True

        for symbol in self._market_symbols:
            payload["assets"][symbol] = self._update_sim_symbol(symbol)

        payload["data_source"] = "simulation"

    def _bootstrap_simulation(self) -> None:
        self.logger.info("Bootstrapping simulation candle buffers…")
        vol = self.config["simulation"]["volatility_factor"]

        for symbol in self._market_symbols:
            price = _SIM_BASE_PRICES.get(symbol, 1000.0)
            vr    = _SIM_VOLUME_RANGES.get(symbol, (1.0, 10.0))

            candles_1d = []
            candles_4h = []
            ts_1d = int(time.time()) - (_BUF_1D * 86400)
            ts_4h = int(time.time()) - (_BUF_4H * 14400)

            # Daily candles
            p = price
            for _ in range(_BUF_1D):
                o = p
                c = max(o + o * random.gauss(0, vol * 4), o * 0.001)
                h = max(o, c) * (1 + abs(random.gauss(0, vol * 2)))
                l = min(o, c) * (1 - abs(random.gauss(0, vol * 2)))
                v = random.uniform(*vr)
                candles_1d.append({"t": ts_1d, "o": round(o,4), "h": round(h,4),
                                    "l": round(l,4), "c": round(c,4), "v": round(v,4)})
                p = c
                ts_1d += 86400

            # 4h candles (price continuity from last daily)
            p4 = price
            for _ in range(_BUF_4H):
                o = p4
                c = max(o + o * random.gauss(0, vol * 2), o * 0.001)
                h = max(o, c) * (1 + abs(random.gauss(0, vol)))
                l = min(o, c) * (1 - abs(random.gauss(0, vol)))
                v = random.uniform(*vr) / 6
                candles_4h.append({"t": ts_4h, "o": round(o,4), "h": round(h,4),
                                    "l": round(l,4), "c": round(c,4), "v": round(v,4)})
                p4 = c
                ts_4h += 14400

            self._sim_state[symbol] = {
                "last_price": p4,
                "candles_1d": candles_1d,
                "candles_4h": candles_4h,
            }
        self.logger.info("Simulation bootstrap complete.")

    def _update_sim_symbol(self, symbol: str) -> dict:
        state = self._sim_state[symbol]
        vol = self.config["simulation"]["volatility_factor"]
        vr  = _SIM_VOLUME_RANGES.get(symbol, (1.0, 10.0))

        # New 4h candle
        o = state["last_price"]
        c = max(o + o * random.gauss(0, vol * 2), o * 0.001)
        h = max(o, c) * (1 + abs(random.gauss(0, vol)))
        l = min(o, c) * (1 - abs(random.gauss(0, vol)))
        v = random.uniform(*vr) / 6
        candle_4h = {"t": int(time.time()), "o": round(o,4), "h": round(h,4),
                     "l": round(l,4), "c": round(c,4), "v": round(v,4)}
        state["candles_4h"].append(candle_4h)
        if len(state["candles_4h"]) > _BUF_4H:
            state["candles_4h"].pop(0)

        # New daily candle (same price, bigger move)
        c_d = max(o + o * random.gauss(0, vol * 4), o * 0.001)
        h_d = max(o, c_d) * (1 + abs(random.gauss(0, vol * 2)))
        l_d = min(o, c_d) * (1 - abs(random.gauss(0, vol * 2)))
        v_d = random.uniform(*vr)
        candle_1d = {"t": int(time.time()), "o": round(o,4), "h": round(h_d,4),
                     "l": round(l_d,4), "c": round(c_d,4), "v": round(v_d,4)}
        state["candles_1d"].append(candle_1d)
        if len(state["candles_1d"]) > _BUF_1D:
            state["candles_1d"].pop(0)

        state["last_price"] = c

        return {
            "last_price": round(c, 4),
            "ohlcv_4h":   state["candles_4h"].copy(),
            "ohlcv_1d":   state["candles_1d"].copy(),
            "orderbook":  self._generate_orderbook(c),
            "volume_24h": round(sum(x["v"] for x in state["candles_1d"][-1:]), 4),
        }

    def _resolve_market_symbols(self) -> list[str]:
        scanner_cfg = self.config.get("scanner", {})
        base_assets = list(self.config.get("assets", []))
        if not scanner_cfg.get("enabled", False):
            return base_assets

        crypto_universe = scanner_cfg.get("crypto_universe", [])
        # Preserve order and guarantee the core configured assets are always present.
        return list(dict.fromkeys([*base_assets, *crypto_universe]))

    # ------------------------------------------------------------------ #
    # World Monitor                                                        #
    # ------------------------------------------------------------------ #

    def _fetch_world_events(self) -> list[dict]:
        if self._wm_client:
            try:
                return self._wm_client.get_events()
            except Exception as exc:
                self.logger.warning(f"World Monitor fetch error (non-blocking): {exc}")
        return self._generate_sim_world_events()

    def _generate_sim_world_events(self) -> list[dict]:
        """Produce 2–3 deterministic macro/crypto/regulatory events per day.
        Seeded on UTC date so events are stable within a day and rotate daily.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        seed_int = int(hashlib.sha256(today.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed_int)

        n_events = rng.randint(2, 3)
        chosen   = rng.sample(_EVENT_POOL, min(n_events, len(_EVENT_POOL)))

        events = []
        for i, tmpl in enumerate(chosen):
            # Sentiment: seeded, rounded to 2 dp, range -0.8..+0.8
            sentiment = round(rng.uniform(-0.8, 0.8), 2)
            events.append({
                "id":        str(uuid.UUID(int=seed_int + i)),
                "timestamp": f"{today}T00:00:00+00:00",
                "type":      tmpl["type"],
                "title":     tmpl["title"],
                "sentiment": sentiment,
                "impact":    tmpl["impact"],
                "source":    "simulation",
            })
        return events

    # ------------------------------------------------------------------ #
    # Shared utility                                                       #
    # ------------------------------------------------------------------ #

    def _generate_orderbook(self, mid_price: float) -> dict:
        spread_pct = 0.0005
        bids, asks = [], []
        for i in range(1, 11):
            qf = 1.0 / i
            bids.append([round(mid_price * (1 - spread_pct * i), 4),
                         round(random.uniform(0.1, 2.0) * qf, 4)])
            asks.append([round(mid_price * (1 + spread_pct * i), 4),
                         round(random.uniform(0.1, 2.0) * qf, 4)])
        return {"bids": bids, "asks": asks}
