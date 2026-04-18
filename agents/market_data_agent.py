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
import json
import random
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from agents.base_agent import BaseAgent, DATA_DIR, BASE_DIR
from adapters.historical_data_store import HistoricalDataStore

_SIM_BASE_PRICES: dict[str, float] = {}  # populated at runtime from yfinance

# Fallback hardcoded per simboli comuni se yfinance non disponibile
_SIM_BASE_PRICES_FALLBACK = {
    # Crypto
    "BTCUSDT": 83000.0, "BTC-USD": 83000.0,
    "ETHUSDT": 1600.0,  "ETH-USD": 1600.0,
    "SOLUSDT": 130.0,   "SOL-USD": 130.0,
    "BNBUSDT": 580.0,   "BNB-USD": 580.0,
    "UNI-USD": 5.5, "LINK-USD": 12.0, "AVAX-USD": 20.0,
    # Large cap equity
    "AAPL": 195.0, "MSFT": 380.0, "NVDA": 875.0, "AMZN": 185.0,
    "GOOGL": 165.0, "META": 510.0, "TSLA": 170.0, "AVGO": 175.0,
    "JPM": 200.0, "V": 270.0, "MA": 460.0, "UNH": 480.0,
    "JNJ": 155.0, "XOM": 110.0, "WMT": 80.0, "PG": 165.0,
    "HD": 345.0, "BAC": 38.0, "ABBV": 195.0, "KO": 63.0,
    "PEP": 155.0, "MRK": 125.0, "TMO": 535.0, "COST": 870.0,
    "MCD": 285.0, "ABT": 105.0, "CRM": 290.0, "ACN": 315.0,
    "AMD": 155.0, "QCOM": 155.0, "TXN": 175.0, "INTC": 22.0,
    "AMAT": 185.0, "LRCX": 820.0, "ADBE": 385.0, "INTU": 620.0,
    "PYPL": 65.0, "CSCO": 48.0, "IBM": 205.0, "ORCL": 125.0,
    # ETF
    "SPY": 540.0, "QQQ": 445.0, "IWM": 195.0, "DIA": 397.0,
    "GLD": 290.0, "IAU": 47.0,  "SLV": 30.0,  "SGOL": 28.0,
    "TLT": 88.0,  "IEF": 93.0,  "SHY": 82.0,  "HYG": 77.0,
    "VTI": 245.0, "VOO": 500.0, "ARKK": 45.0, "SOXX": 215.0,
    "SMH": 215.0, "XLE": 88.0,  "XLF": 42.0,  "XLK": 210.0,
    "XLV": 140.0, "XLI": 128.0, "XLY": 195.0, "XLU": 69.0,
    "XLB": 88.0,  "XLC": 88.0,  "VGT": 530.0, "VHT": 250.0,
    "VFH": 97.0,  "VDE": 115.0, "VCR": 415.0, "VPU": 155.0,
    "VNQ": 85.0,  "VOX": 95.0,  "VIS": 225.0, "RSP": 165.0,
    # Inverse / leveraged ETF
    "SQQQ": 8.0,  "SPXS": 8.0,  "SDS": 40.0,  "SH": 60.0,
    "PSQ": 20.0,  "DOG": 30.0,  "SDOW": 25.0, "TZA": 30.0,
    "QID": 20.0,  "DRV": 25.0,  "LABD": 35.0, "FAZ": 25.0,
    "EDZ": 15.0,  "YANG": 15.0,
    # Volatility
    "VXX": 55.0, "UVXY": 22.0, "VIXY": 18.0,
    # Commodities ETF
    "USO": 75.0, "BNO": 35.0, "OIH": 265.0, "XOP": 145.0,
    "UNG": 15.0, "DBA": 22.0, "PPLT": 90.0,
    # Sector / thematic
    "BOTZ": 30.0, "IGV": 82.0, "KRE": 52.0, "KBE": 42.0,
    "XBI": 85.0, "IYH": 285.0, "QQQM": 175.0,
    # Financials
    "GS": 540.0, "MS": 115.0, "BX": 140.0, "C": 65.0,
    "WFC": 68.0, "USB": 42.0, "PNC": 155.0, "COF": 165.0,
    "SCHW": 75.0, "BLK": 890.0, "CME": 225.0, "ICE": 165.0,
    "CB": 275.0, "PGR": 255.0, "AFL": 110.0, "SPGI": 465.0,
    # Healthcare
    "REGN": 760.0, "DXCM": 80.0, "BSX": 95.0, "SYK": 355.0,
    "ZTS": 175.0, "BDX": 235.0, "ELV": 420.0, "HCA": 340.0,
    "ILMN": 155.0, "TMO": 535.0, "BMY": 55.0, "PFE": 27.0,
    "AMGN": 310.0, "MRNA": 38.0, "SRPT": 125.0,
    # Biotech / genomics
    "CRSP": 45.0, "BEAM": 22.0, "NTLA": 9.0, "EDIT": 5.0,
    "TXG": 25.0, "EXAS": 55.0, "LMND": 28.0,
    # Energy
    "COP": 95.0, "EOG": 120.0, "DVN": 35.0, "OXY": 45.0,
    "KMI": 25.0, "TRGP": 150.0, "PSX": 130.0, "MPC": 155.0,
    "BKR": 38.0,
    # Materials / Mining
    "FCX": 42.0, "NEM": 52.0, "NUE": 135.0, "STLD": 120.0,
    "CLF": 12.0, "TECK": 40.0, "ALB": 75.0, "SQM": 38.0,
    "RS": 245.0, "CCJ": 45.0, "UUUU": 7.0,
    # Industrials / Defense
    "RTX": 125.0, "BA": 175.0, "TM": 175.0, "DE": 450.0,
    "RKLB": 22.0,
    # Tech mid/small
    "NET": 115.0, "ZS": 195.0, "CRWD": 350.0, "DDOG": 120.0,
    "OKTA": 95.0, "MDB": 235.0, "SPOT": 285.0, "FSLY": 8.0,
    "HOOD": 38.0, "IONQ": 30.0,
    # Crypto miners
    "MARA": 14.0, "RIOT": 7.0, "CLSK": 9.0, "CIFR": 5.0,
    "WULF": 5.0, "HUT": 12.0, "MSTR": 380.0,
    # Distressed / speculative
    "SGML": 4.0, "BEAM": 22.0,
    # Gold miners
    "AA": 32.0,
}

_SIM_VOLUME_RANGES = {
    "BTCUSDT": (5.0, 50.0),
    "ETHUSDT": (20.0, 200.0),
    "SOLUSDT": (500.0, 5000.0),
    "BNBUSDT": (50.0, 500.0),
}


def _fetch_yfinance_prices(symbols: list[str]) -> dict[str, float]:
    """Scarica prezzi reali da yfinance per usarli come base GBM."""
    try:
        import yfinance as yf

        # Normalizza simboli per yfinance (BTCUSDT → BTC-USD)
        yf_map: dict[str, str] = {}
        for s in symbols:
            if s.endswith("USDT"):
                yf_sym = s.replace("USDT", "-USD")
            else:
                yf_sym = s
            yf_map[yf_sym] = s

        tickers = list(yf_map.keys())
        data = yf.download(tickers, period="2d", interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        prices: dict[str, float] = {}
        if data.empty:
            return prices

        close = data["Close"] if "Close" in data.columns else data.xs("Close", axis=1, level=0)
        for yf_sym, orig_sym in yf_map.items():
            try:
                col = close[yf_sym] if yf_sym in close.columns else close
                val = float(col.dropna().iloc[-1])
                if val > 0:
                    prices[orig_sym] = val
                    # Alias bidirezionale
                    if orig_sym.endswith("USDT"):
                        prices[orig_sym.replace("USDT", "-USD")] = val
                    elif orig_sym.endswith("-USD"):
                        prices[orig_sym.replace("-USD", "USDT")] = val
            except Exception:
                pass
        return prices
    except Exception:
        return {}
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
        self._sim_cycle_count: int = 0
        self._sim_anchor_interval: int = 10  # re-anchor to yfinance every N cycles
        self._historical_store = HistoricalDataStore()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        self.mark_running()
        try:
            bt_ctx = self.read_json(DATA_DIR / "backtest_context.json") or {}
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

            # Crypto symbols via Binance; equity/ETF via yfinance; simulation only as last resort
            crypto_syms = [s for s in self._market_symbols if s.endswith("USDT") or s.endswith("-USD")]
            equity_syms = [s for s in self._market_symbols if s not in crypto_syms]

            # --- Crypto: Binance primary ---
            crypto_payload: dict = {"assets": {}}
            if crypto_syms:
                if not self._try_binance_symbols(crypto_payload, crypto_syms):
                    self._try_yfinance_symbols(crypto_payload, crypto_syms)

            # --- Equity/ETF: yfinance primary ---
            equity_payload: dict = {"assets": {}}
            if equity_syms:
                if not self._try_yfinance_symbols(equity_payload, equity_syms):
                    if not self._try_openbb(equity_payload):
                        self._run_simulation_symbols(equity_payload, equity_syms)

            payload["assets"].update(crypto_payload["assets"])
            payload["assets"].update(equity_payload["assets"])

            # Fallback: if nothing worked at all, run full simulation
            if not payload["assets"]:
                self._run_simulation(payload)

            # Set data_source summary
            sources = set()
            if crypto_payload["assets"]:
                sources.add(crypto_payload.get("data_source", "binance"))
            if equity_payload["assets"]:
                sources.add(equity_payload.get("data_source", "yfinance"))
            payload["data_source"] = "+".join(sorted(sources)) if sources else "simulation"

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

    def _read_json_fallback(self, path: Path) -> dict | None:
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning(f'Fallback JSON read failed {path}: {exc}')
            return None

    def _run_historical(self, bt_ctx: dict) -> dict:
        ts = bt_ctx.get("lookahead_cutoff") or bt_ctx.get("simulated_now")
        if not ts:
            raise ValueError("backtest_context requires lookahead_cutoff/simulated_now")
        day = datetime.fromisoformat(ts).date()
        frozen_universe = bt_ctx.get("frozen_universe") or self._market_symbols
        try:
            payload = self._historical_store.get_snapshot_at(day, frozen_universe)
        except json.JSONDecodeError as exc:
            self.logger.error(f'Historical store JSON corrupt for {day}: {exc}')
            self.logger.info('Using live market_data.json fallback for backtest')
            payload = self._read_json_fallback(DATA_DIR / 'market_data.json') or {'assets': {}, 'world_events': []}
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
                c4h = candles_4h if candles_4h else candles_1d
                assets[symbol] = {
                    "last_price": last_price,
                    "ohlcv_1d":   candles_1d,
                    "ohlcv_4h":   c4h,
                    "orderbook":  self._generate_orderbook(last_price),
                    "volume_24h": round(float(candles_1d[-1]["v"]), 4),
                    "vwap":       self._compute_vwap(c4h),
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
                c4h = candles_4h if candles_4h else candles_1d
                assets[symbol] = {
                    "last_price": last_price,
                    "ohlcv_1d":   candles_1d,
                    "ohlcv_4h":   c4h,
                    "orderbook":  self._generate_orderbook(last_price),
                    "volume_24h": round(sum(c["v"] for c in candles_1d[-1:]), 4),
                    "vwap":       self._compute_vwap(c4h),
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
    # Binance — subset of symbols                                         #
    # ------------------------------------------------------------------ #

    def _try_binance_symbols(self, payload: dict, symbols: list[str]) -> bool:
        if not self._binance_client.ensure_connected():
            return False
        assets: dict = {}
        for symbol in symbols:
            try:
                candles_1d = self._binance_client.get_ohlcv(symbol)
                candles_4h = self._binance_client.get_ohlcv_4h(symbol)
                if not candles_1d:
                    raise ValueError(f"Empty 1d candles for {symbol}")
                last_price = float(candles_1d[-1]["c"])
                c4h = candles_4h if candles_4h else candles_1d
                assets[symbol] = {
                    "last_price": last_price,
                    "ohlcv_1d":   candles_1d,
                    "ohlcv_4h":   c4h,
                    "orderbook":  self._generate_orderbook(last_price),
                    "volume_24h": round(float(candles_1d[-1]["v"]), 4),
                    "vwap":       self._compute_vwap(c4h),
                }
            except Exception as exc:
                self.logger.debug(f"Binance skip {symbol}: {exc}")
        if not assets:
            return False
        payload["assets"] = assets
        payload["data_source"] = "binance"
        return True

    # ------------------------------------------------------------------ #
    # yfinance — real OHLCV for equity/ETF (and crypto fallback)          #
    # ------------------------------------------------------------------ #

    def _try_yfinance_symbols(self, payload: dict, symbols: list[str]) -> bool:
        try:
            import yfinance as yf
        except ImportError:
            return False
        if not symbols:
            return False

        # Map internal symbols → yfinance tickers
        yf_map: dict[str, str] = {}  # yf_ticker → internal symbol
        for s in symbols:
            yf_sym = s.replace("USDT", "-USD") if s.endswith("USDT") else s
            yf_map[yf_sym] = s

        yf_tickers = list(yf_map.keys())
        assets: dict = {}

        try:
            # 1d candles — 300 days
            df_1d = yf.download(yf_tickers, period="300d", interval="1d",
                                auto_adjust=True, progress=False, threads=True)
            # 4h candles — last 60 days (max yfinance allows for 1h/2h/4h is ~60d)
            df_4h = yf.download(yf_tickers, period="60d", interval="1h",
                                auto_adjust=True, progress=False, threads=True)
        except Exception as exc:
            self.logger.warning(f"yfinance download failed: {exc}")
            return False

        def _extract_candles(df, yf_sym: str, resample_rule: str | None = None) -> list[dict]:
            try:
                if df.empty:
                    return []
                # Multi-ticker vs single-ticker
                if isinstance(df.columns, pd.MultiIndex):
                    if yf_sym not in df.columns.get_level_values(1):
                        return []
                    sub = df.xs(yf_sym, axis=1, level=1).dropna(how="all")
                else:
                    sub = df.dropna(how="all")
                if resample_rule:
                    sub = sub.resample(resample_rule).agg({
                        "Open": "first", "High": "max", "Low": "min",
                        "Close": "last", "Volume": "sum"
                    }).dropna()
                candles = []
                for ts, row in sub.iterrows():
                    try:
                        candles.append({
                            "t": int(ts.timestamp()),
                            "o": round(float(row["Open"]), 4),
                            "h": round(float(row["High"]), 4),
                            "l": round(float(row["Low"]), 4),
                            "c": round(float(row["Close"]), 4),
                            "v": round(float(row["Volume"]), 2),
                        })
                    except Exception:
                        continue
                return candles
            except Exception as exc:
                self.logger.debug(f"yfinance candle extract {yf_sym}: {exc}")
                return []

        for yf_sym, orig_sym in yf_map.items():
            candles_1d = _extract_candles(df_1d, yf_sym)
            candles_4h = _extract_candles(df_4h, yf_sym, resample_rule="4h")
            if not candles_1d:
                self.logger.debug(f"yfinance: no 1d data for {yf_sym}")
                continue
            last_price = candles_1d[-1]["c"]
            c4h = candles_4h if candles_4h else candles_1d
            assets[orig_sym] = {
                "last_price": last_price,
                "ohlcv_1d":   candles_1d,
                "ohlcv_4h":   c4h,
                "orderbook":  self._generate_orderbook(last_price),
                "volume_24h": round(candles_1d[-1]["v"], 2),
                "vwap":       self._compute_vwap(c4h),
            }

        if not assets:
            self.logger.warning(f"yfinance returned no data for {symbols[:5]}...")
            return False

        self.logger.info(f"yfinance: {len(assets)}/{len(symbols)} symbols fetched")
        payload["assets"] = assets
        payload["data_source"] = "yfinance"
        return True

    def _run_simulation_symbols(self, payload: dict, symbols: list[str]) -> None:
        """Simulation fallback for a subset of symbols only."""
        if not self._sim_initialized:
            self._bootstrap_simulation()
            self._sim_initialized = True
        self._sim_cycle_count += 1
        if self._sim_cycle_count % self._sim_anchor_interval == 0:
            self._reanchor_sim_prices()
        assets: dict = {}
        for symbol in symbols:
            if symbol in self._sim_state:
                assets[symbol] = self._update_sim_symbol(symbol)
        payload["assets"] = assets
        payload["data_source"] = "simulation"

    # ------------------------------------------------------------------ #
    # Simulation fallback                                                  #
    # ------------------------------------------------------------------ #

    def _run_simulation(self, payload: dict) -> None:
        try:
            import json
        except ImportError:
            self.logger.error("json module not available - cannot run simulation")
            return
        if not self._sim_initialized:
            self._bootstrap_simulation()
            self._sim_initialized = True

        self._sim_cycle_count += 1
        if self._sim_cycle_count % self._sim_anchor_interval == 0:
            self._reanchor_sim_prices()

        for symbol in self._market_symbols:
            payload["assets"][symbol] = self._update_sim_symbol(symbol)

        payload["data_source"] = "simulation"

    def _bootstrap_simulation(self) -> None:
        self.logger.info("Bootstrapping simulation — fetching real prices from yfinance…")
        vol = self.config["simulation"]["volatility_factor"]

        # Scarica prezzi reali come base GBM
        live_prices = _fetch_yfinance_prices(self._market_symbols)
        if live_prices:
            self.logger.info(f"yfinance prices loaded for {len(live_prices)} symbols")
        else:
            self.logger.warning("yfinance unavailable — using hardcoded fallback prices")

        for symbol in self._market_symbols:
            # Priorità: 1) yfinance live, 2) fallback hardcoded, 3) 100.0
            price = (
                live_prices.get(symbol)
                or _SIM_BASE_PRICES_FALLBACK.get(symbol)
                or 100.0
            )
            vr    = _SIM_VOLUME_RANGES.get(symbol, (1.0, 10.0))

            candles_1d = []
            candles_4h = []
            ts_1d = int(time.time()) - (_BUF_1D * 86400)
            ts_4h = int(time.time()) - (_BUF_4H * 14400)

            # Daily candles — vol calibrato per timeframe giornaliero
            # vol è volatilità per ciclo (2s), serve scalare a 1 giorno
            # daily_vol = vol * sqrt(43200) ≈ vol * 207, ma usiamo direttamente
            # la stima realistica: ~1% daily vol per equity, ~2.5% per crypto
            is_crypto = any(k in symbol for k in ("BTC", "ETH", "SOL", "BNB", "UNI", "LINK", "AVAX"))
            daily_vol = 0.025 if is_crypto else 0.010  # deviazione std giornaliera realistica
            h4_vol    = daily_vol / 2.5                # 4h vol ≈ daily / 2.5

            p = price
            for _ in range(_BUF_1D):
                o = p
                drift = random.gauss(0.0002, daily_vol)  # leggero drift positivo + rumore
                c = max(o * (1 + drift), o * 0.01)
                h = max(o, c) * (1 + abs(random.gauss(0, daily_vol * 0.5)))
                l = min(o, c) * (1 - abs(random.gauss(0, daily_vol * 0.5)))
                v = random.uniform(*vr)
                candles_1d.append({"t": ts_1d, "o": round(o,4), "h": round(h,4),
                                    "l": round(l,4), "c": round(c,4), "v": round(v,4)})
                p = c
                ts_1d += 86400

            # 4h candles — ancorate al prezzo reale, non alla fine del daily walk
            p4 = price
            for _ in range(_BUF_4H):
                o = p4
                drift4 = random.gauss(0.0, h4_vol)
                c = max(o * (1 + drift4), o * 0.01)
                h = max(o, c) * (1 + abs(random.gauss(0, h4_vol * 0.4)))
                l = min(o, c) * (1 - abs(random.gauss(0, h4_vol * 0.4)))
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

    def _reanchor_sim_prices(self) -> None:
        """Re-anchor simulated last_price to real yfinance prices every N cycles.
        Prevents unbounded random-walk drift that causes unrealistic P&L."""
        self.logger.info("Re-anchoring sim prices to yfinance…")
        live_prices = _fetch_yfinance_prices(self._market_symbols)
        if not live_prices:
            self.logger.warning("yfinance unavailable — skipping re-anchor")
            return
        anchored = 0
        for symbol in self._market_symbols:
            real_price = live_prices.get(symbol)
            if real_price and real_price > 0 and symbol in self._sim_state:
                self._sim_state[symbol]["last_price"] = real_price
                # Also update last candle close so next candle opens at real price
                if self._sim_state[symbol]["candles_4h"]:
                    self._sim_state[symbol]["candles_4h"][-1]["c"] = round(real_price, 4)
                anchored += 1
        self.logger.info(f"Re-anchored {anchored}/{len(self._market_symbols)} symbols to live prices")

    def _update_sim_symbol(self, symbol: str) -> dict:
        state = self._sim_state[symbol]
        vr  = _SIM_VOLUME_RANGES.get(symbol, (1.0, 10.0))

        # Volatilità realistica per ciclo live (60s)
        # equity: ~1% daily → per ciclo 60s: 1% / sqrt(390 minuti di mercato / 1) ≈ 0.05%
        # crypto: ~2.5% daily → ~0.12% per ciclo
        is_crypto = any(k in symbol for k in ("BTC", "ETH", "SOL", "BNB", "UNI", "LINK", "AVAX"))
        cycle_vol = 0.0012 if is_crypto else 0.0005  # std per singolo ciclo

        # New 4h candle
        o = state["last_price"]
        drift = random.gauss(0.0, cycle_vol)
        c = max(o * (1 + drift), o * 0.01)
        h = max(o, c) * (1 + abs(random.gauss(0, cycle_vol * 0.3)))
        l = min(o, c) * (1 - abs(random.gauss(0, cycle_vol * 0.3)))
        v = random.uniform(*vr) / 6
        candle_4h = {"t": int(time.time()), "o": round(o,4), "h": round(h,4),
                     "l": round(l,4), "c": round(c,4), "v": round(v,4)}
        state["candles_4h"].append(candle_4h)
        if len(state["candles_4h"]) > _BUF_4H:
            state["candles_4h"].pop(0)

        # New daily candle (volatilità giornaliera, non per ciclo)
        daily_vol = 0.025 if is_crypto else 0.010
        c_d = max(o * (1 + random.gauss(0.0002, daily_vol)), o * 0.01)
        h_d = max(o, c_d) * (1 + abs(random.gauss(0, daily_vol * 0.5)))
        l_d = min(o, c_d) * (1 - abs(random.gauss(0, daily_vol * 0.5)))
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
            "vwap":       self._compute_vwap(state["candles_4h"]),
        }

    def _resolve_market_symbols(self) -> list[str]:
        scanner_cfg = self.config.get("scanner", {})
        base_assets = list(self.config.get("assets", []))
        if not scanner_cfg.get("enabled", False):
            return base_assets

        crypto_universe = scanner_cfg.get("crypto_universe", [])
        institutional_universe = scanner_cfg.get("institutional_universe", [])
        # Merge: core assets + crypto scanner + institutional equity/ETF universe
        return list(dict.fromkeys([*base_assets, *crypto_universe, *institutional_universe]))

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

    @staticmethod
    def _compute_vwap(candles_4h: list[dict]) -> float | None:
        """VWAP from 4h candles of the most recent trading day (last 6 bars = 24h)."""
        if not candles_4h:
            return None
        recent = candles_4h[-6:]  # last 6 × 4h = 24h window
        cum_vol = sum(c["v"] for c in recent)
        if cum_vol <= 0:
            return None
        cum_pv = sum(((c["h"] + c["l"] + c["c"]) / 3.0) * c["v"] for c in recent)
        return round(cum_pv / cum_vol, 6)

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
