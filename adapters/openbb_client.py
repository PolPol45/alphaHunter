"""
OpenBB Client Adapter
=====================
Wraps the OpenBB SDK (v4+) to provide OHLCV candle data in the internal
trading bot format: list of {"t": unix_ts, "o", "h", "l", "c", "v"}.

Provides two data streams:
  - get_ohlcv(symbol)      → daily candles (up to lookback_days)
  - get_ohlcv_4h(symbol)   → 4h candles via 1h fetch + pandas resample
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("openbb_client")

_SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
}

_CANDLE_LIMIT_1D  = 300
_CANDLE_LIMIT_4H  = 200


class OpenBBClient:
    def __init__(self, cfg: dict) -> None:
        self._provider: str        = cfg.get("provider", "yfinance")
        self._fallback_provider: str = cfg.get("fallback_provider", "yfinance")
        self._lookback_days: int   = int(cfg.get("lookback_days", 300))
        self._lookback_4h: int     = int(cfg.get("lookback_days_4h", 60))
        self._timeout: int         = int(cfg.get("timeout_seconds", 30))
        self._retries: int         = int(cfg.get("retry_attempts", 3))
        self._backoff: float       = float(cfg.get("retry_backoff_seconds", 2.0))
        self._cache_ttl: float     = float(cfg.get("cache_ttl_seconds", 300))

        self._obb: Any = None
        self._connected: bool = False
        self._last_error: str | None = None
        self._cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        if self._connected:
            return True
        try:
            from openbb import obb
            self._obb = obb
            self._connected = True
            logger.info(f"OpenBB initialised | provider={self._provider}")
            return True
        except ImportError as exc:
            self._last_error = f"openbb not installed: {exc}"
            logger.error(self._last_error)
            return False
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"OpenBB init failed: {exc}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------ #
    # Public data accessors                                                #
    # ------------------------------------------------------------------ #

    def get_ohlcv(self, symbol: str) -> list[dict]:
        """Daily candles for *symbol*."""
        return self._get_cached(f"1d:{symbol}", lambda: self._fetch_ohlcv(symbol, "1d", self._lookback_days, _CANDLE_LIMIT_1D))

    def get_ohlcv_4h(self, symbol: str) -> list[dict]:
        """4-hour candles via 1h fetch + resample. Falls back to daily on error."""
        cache_key = f"4h:{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]

        for attempt in range(1, self._retries + 1):
            try:
                candles_1h = self._fetch_ohlcv(symbol, "1h", self._lookback_4h, _CANDLE_LIMIT_4H * 4)
                candles_4h = _resample_4h(candles_1h)[-_CANDLE_LIMIT_4H:]
                self._cache[cache_key] = {"data": candles_4h, "ts": time.monotonic()}
                logger.info(f"[{symbol}] 4h: {len(candles_4h)} candles via 1h resample")
                return candles_4h
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning(f"[{symbol}] 4h attempt {attempt}/{self._retries} failed: {exc}")
                if attempt < self._retries:
                    time.sleep(self._backoff ** attempt)

        # Fallback: use daily candles as 4h proxy
        logger.warning(f"[{symbol}] 4h fetch exhausted — falling back to daily candles")
        return self.get_ohlcv(symbol)

    def get_last_price(self, symbol: str) -> float:
        candles = self.get_ohlcv(symbol)
        if not candles:
            raise RuntimeError(f"[{symbol}] no OHLCV data available")
        return float(candles[-1]["c"])

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_cached(self, key: str, fetcher) -> list[dict]:
        if self._is_cache_valid(key):
            return self._cache[key]["data"]
        symbol = key.split(":", 1)[1]
        for attempt in range(1, self._retries + 1):
            try:
                data = fetcher()
                self._cache[key] = {"data": data, "ts": time.monotonic()}
                return data
            except Exception as exc:
                self._last_error = str(exc)
                wait = self._backoff ** attempt
                logger.warning(f"[{symbol}] attempt {attempt}/{self._retries}: {exc} | retry in {wait:.1f}s")
                if attempt < self._retries:
                    time.sleep(wait)
        if key in self._cache:
            logger.warning(f"[{symbol}] using stale cache after all retries failed")
            return self._cache[key]["data"]
        raise RuntimeError(f"[{symbol}] all {self._retries} attempts failed: {self._last_error}")

    def _fetch_ohlcv(self, symbol: str, interval: str, lookback_days: int, limit: int) -> list[dict]:
        if not self._obb:
            raise RuntimeError("OpenBB not initialised — call connect() first")

        obb_sym = _to_provider_symbol(symbol)
        start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        for provider in _unique([self._provider, self._fallback_provider]):
            try:
                kwargs: dict = dict(symbol=obb_sym, start_date=start_date, provider=provider)
                if interval != "1d":
                    kwargs["interval"] = interval
                result = self._obb.crypto.price.historical(**kwargs)
                df = result.to_df()
                if df is not None and not df.empty:
                    break
            except Exception as exc:
                logger.warning(f"Provider '{provider}' failed ({interval}) for {obb_sym}: {exc}")
        else:
            raise RuntimeError(f"All providers exhausted for {obb_sym} interval={interval}")

        df.columns = [str(c).lower() for c in df.columns]
        if df.index.name in ("date", "datetime", "timestamp"):
            df = df.reset_index()

        candles: list[dict] = []
        for _, row in df.iterrows():
            date_val = row.get("date") or row.get("datetime") or row.get("timestamp")
            ts = _to_unix(date_val)
            if ts is None:
                continue
            candles.append({
                "t": ts,
                "o": round(float(row["open"]),  6),
                "h": round(float(row["high"]),  6),
                "l": round(float(row["low"]),   6),
                "c": round(float(row["close"]), 6),
                "v": round(float(row.get("volume", 0) or 0), 4),
            })

        if not candles:
            raise ValueError(f"Empty candle list for {obb_sym} interval={interval}")

        return candles[-limit:]

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        return (time.monotonic() - self._cache[key]["ts"]) < self._cache_ttl


# ------------------------------------------------------------------ #
# Module-level utilities                                               #
# ------------------------------------------------------------------ #

def _resample_4h(candles_1h: list[dict]) -> list[dict]:
    """Resample a list of 1h OHLCV dicts to 4h candles using pandas."""
    import pandas as pd

    if not candles_1h:
        return []

    df = pd.DataFrame(candles_1h)
    df["dt"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.set_index("dt").sort_index()

    agg = df.resample("4h").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    agg = agg.dropna(subset=["o", "c"])

    result: list[dict] = []
    for ts, row in agg.iterrows():
        result.append({
            "t": int(ts.timestamp()),
            "o": round(float(row["o"]), 6),
            "h": round(float(row["h"]), 6),
            "l": round(float(row["l"]), 6),
            "c": round(float(row["c"]), 6),
            "v": round(float(row["v"]), 4),
        })
    return result


def _to_unix(val: Any) -> int | None:
    if val is None:
        return None
    if hasattr(val, "timestamp"):
        return int(val.timestamp())
    try:
        return int(datetime.fromisoformat(str(val)).timestamp())
    except (ValueError, TypeError):
        return None


def _unique(seq: list) -> list:
    seen: set = set()
    return [x for x in seq if not (x in seen or seen.add(x))]  # type: ignore


def _to_provider_symbol(symbol: str) -> str:
    if symbol in _SYMBOL_MAP:
        return _SYMBOL_MAP[symbol]
    if symbol.endswith("USDT") and len(symbol) > 4:
        return f"{symbol[:-4]}-USD"
    return symbol
