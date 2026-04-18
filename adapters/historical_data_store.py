from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from agents.base_agent import BASE_DIR, DATA_DIR

# One file per symbol: backtest_data/sym_AAPL.json
# Format: {"symbol": "AAPL", "ohlcv_1d": [...], "ohlcv_4h": [...]}
# Backtest reads slices up to cutoff date on the fly — no per-day files.


def _yf_to_ohlcv(df, yf_sym: str) -> list[dict]:
    import pandas as pd
    out = []
    for ts, row in df.iterrows():
        try:
            out.append({
                "t": int(pd.Timestamp(ts).timestamp()),
                "o": round(float(row["Open"]), 6),
                "h": round(float(row["High"]), 6),
                "l": round(float(row["Low"]), 6),
                "c": round(float(row["Close"]), 6),
                "v": round(float(row["Volume"]), 2),
            })
        except Exception:
            continue
    return out


def fetch_real_snapshots(symbols: list[str], start: date, end: date, bt_dir: Path) -> None:
    """
    Download yfinance history for symbols and write one sym_<TICKER>.json per symbol.
    Much smaller than per-day files: ~50MB total vs 5GB.
    Already-existing symbol files are skipped (re-run safe).
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        raise RuntimeError("yfinance not installed")

    bt_dir.mkdir(parents=True, exist_ok=True)

    fetch_start = start - timedelta(days=400)  # extra history for 200d EMA warmup

    # Normalize: BTCUSDT → BTC-USD for yfinance
    yf_map: dict[str, str] = {}  # yf_sym → original_sym
    missing: list[str] = []
    for s in symbols:
        sym_file = bt_dir / f"sym_{s}.json"
        if sym_file.exists():
            continue  # already cached
        yf_sym = s.replace("USDT", "-USD") if s.endswith("USDT") else s
        yf_map[yf_sym] = s

    if not yf_map:
        print("[HistoricalStore] All symbols cached — skipping download.")
        return

    yf_syms = list(yf_map.keys())
    print(f"[HistoricalStore] Downloading {len(yf_syms)} symbols {fetch_start} → {end}…")

    # Daily bars — full history, no limit
    df_1d = yf.download(
        yf_syms,
        start=fetch_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=True, progress=True, threads=True,
    )

    # 1h bars — yfinance only provides last 730 days
    today = date.today()
    h1_start = max(fetch_start, today - timedelta(days=720))
    if h1_start <= end:
        df_1h = yf.download(
            yf_syms,
            start=h1_start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1h", auto_adjust=True, progress=False, threads=True,
        )
    else:
        df_1h = pd.DataFrame()

    def _extract(df, yf_sym: str, resample: str | None = None) -> list[dict]:
        try:
            if df is None or df.empty:
                return []
            if isinstance(df.columns, pd.MultiIndex):
                if yf_sym not in df.columns.get_level_values(1):
                    return []
                sub = df.xs(yf_sym, axis=1, level=1).dropna(how="all")
            else:
                sub = df.dropna(how="all")
            if resample:
                sub = sub.resample(resample).agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum",
                }).dropna()
            return _yf_to_ohlcv(sub, yf_sym)
        except Exception:
            return []

    written = 0
    for yf_sym, orig_sym in yf_map.items():
        candles_1d = _extract(df_1d, yf_sym)
        candles_4h = _extract(df_1h, yf_sym, resample="4h")
        if not candles_1d:
            continue
        payload = {
            "symbol": orig_sym,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ohlcv_1d": candles_1d,
            "ohlcv_4h": candles_4h if candles_4h else candles_1d,
        }
        sym_file = bt_dir / f"sym_{orig_sym}.json"
        sym_file.write_text(json.dumps(payload))
        written += 1

    print(f"[HistoricalStore] Done: {written} written, {len(symbols) - len(yf_map)} already cached")


class HistoricalDataStore:
    """
    Per-symbol store. Each symbol has one file: backtest_data/sym_<TICKER>.json
    Slices up to cutoff on read — no per-day files, no disk bloat.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.base_dir = BASE_DIR
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.bt_dir = self.base_dir / "backtest_data"
        self.bt_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}  # sym → loaded json (in-memory during backtest)

    def freeze_universe(self, universe: list[str]) -> list[str]:
        return list(dict.fromkeys(universe))

    def get_snapshot_at(self, day: date, universe: list[str]) -> dict:
        """Return market snapshot for *day* with no-lookahead candle slicing."""
        cutoff = int(datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc).timestamp())
        assets: dict = {}

        for sym in universe:
            data = self._load_sym(sym)
            if not data:
                continue
            c1 = [c for c in data["ohlcv_1d"] if c["t"] <= cutoff]
            c4 = [c for c in data["ohlcv_4h"] if c["t"] <= cutoff]
            if not c1:
                continue
            last_price = c1[-1]["c"]
            assets[sym] = {
                "last_price": last_price,
                "ohlcv_1d": c1,
                "ohlcv_4h": c4 if c4 else c1,
                "orderbook": {"bids": [], "asks": []},
                "volume_24h": c1[-1]["v"],
            }

        return {
            "timestamp": datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "data_source": "yfinance_historical",
            "assets": assets,
            "world_events": [],
        }

    def _load_sym(self, sym: str) -> dict | None:
        if sym in self._cache:
            return self._cache[sym]
        fp = self.bt_dir / f"sym_{sym}.json"
        if not fp.exists():
            return None
        try:
            data = json.loads(fp.read_text())
            self._cache[sym] = data
            return data
        except Exception:
            return None

    # ── Legacy random fallback (used only if sym file missing) ───────────

    def _generate_snapshot(self, day: date, universe: list[str]) -> dict:
        seed = int(day.strftime("%Y%m%d"))
        rng = random.Random(seed)
        live = self._read_json(self.data_dir / "market_data.json")
        live_assets = live.get("assets", {}) if live else {}
        assets: dict = {}
        for symbol in universe:
            base = live_assets.get(symbol, {})
            base_price = float(base.get("last_price", 100.0)) if base else 100.0
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
                "volume_24h": round(abs(price) * (5 + rng.random() * 20), 4),
            }
        return {
            "timestamp": datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "data_source": "historical_store_random",
            "assets": assets,
            "world_events": [],
        }

    def _build_ohlcv(self, day: date, last_price: float, bars: int, hours_per_bar: int, rng: random.Random) -> list[dict]:
        step = hours_per_bar * 3600
        end_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=23)
        start = int(end_dt.timestamp()) - bars * step
        price = last_price
        out = []
        for i in range(bars):
            o = price
            chg = rng.gauss(0, 0.004 if hours_per_bar == 4 else 0.008)
            c = max(o * (1 + chg), 0.0001)
            h = max(o, c) * (1 + abs(rng.gauss(0, 0.002)))
            l = min(o, c) * (1 - abs(rng.gauss(0, 0.002)))
            v = abs(rng.gauss(50, 20))
            out.append({"t": start + i * step, "o": round(o, 6), "h": round(h, 6),
                        "l": round(l, 6), "c": round(c, 6), "v": round(v, 4)})
            price = c
        return out

    def _orderbook(self, mid: float, rng: random.Random) -> dict:
        bids, asks = [], []
        for i in range(1, 11):
            s = 0.0005 * i
            bids.append([round(mid * (1 - s), 6), round(rng.uniform(0.1, 2.0), 4)])
            asks.append([round(mid * (1 + s), 6), round(rng.uniform(0.1, 2.0), 4)])
        return {"bids": bids, "asks": asks}

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
