"""
Binance Public REST Adapter
============================
Fetches OHLCV candle data for crypto pairs from the Binance public API.
No authentication required — uses only public market data endpoints.

Endpoints used:
  GET /api/v3/ping            — connectivity check
  GET /api/v3/klines          — OHLCV candlestick data
  GET /api/v3/ticker/24hr     — 24-hour rolling window price stats

Binance klines response per candle (index → field):
  0  open_time_ms   4  close      8  n_trades
  1  open           5  volume     9  taker_buy_base_vol
  2  high           6  close_ms  10  taker_buy_quote_vol
  3  low            7  quote_vol 11  unused

Symbol format: already matches internal format (BTCUSDT, ETHUSDT, …)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("binance_client")

_BASE_URL = "https://api.binance.com"
_MAX_KLINES = 1000   # Binance hard limit per request


class BinanceClient:
    """Synchronous Binance public REST client for OHLCV market data."""

    def __init__(self, cfg: dict) -> None:
        self._base_url: str   = cfg.get("base_url", _BASE_URL).rstrip("/")
        self._timeout: int    = int(cfg.get("timeout_seconds", 15))
        self._retries: int    = int(cfg.get("retry_attempts", 2))
        self._backoff: float  = float(cfg.get("retry_backoff_seconds", 1.0))
        self._limit_1d: int   = min(int(cfg.get("lookback_days", 300)),    _MAX_KLINES)
        self._limit_4h: int   = min(int(cfg.get("lookback_days_4h", 200)), _MAX_KLINES)
        self._connected: bool          = False
        self._last_error: str | None   = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        """Verify connectivity with a lightweight ping."""
        try:
            self._get("/api/v3/ping")
            self._connected = True
            logger.info(f"Binance REST connected | base={self._base_url}")
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"Binance connect failed: {exc}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected

    def ensure_connected(self) -> bool:
        if self._connected:
            return True
        return self.connect()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------ #
    # Market data                                                          #
    # ------------------------------------------------------------------ #

    def get_ohlcv(self, symbol: str, limit: int | None = None) -> list[dict]:
        """Fetch daily (1d) candles. Returns [{t, o, h, l, c, v}, …]."""
        return self._fetch_klines(symbol, "1d", limit or self._limit_1d)

    def get_ohlcv_4h(self, symbol: str, limit: int | None = None) -> list[dict]:
        """Fetch 4-hour candles. Returns [{t, o, h, l, c, v}, …]."""
        return self._fetch_klines(symbol, "4h", limit or self._limit_4h)

    def get_ticker_24h(self, symbol: str) -> dict:
        """Return 24-hour rolling stats for a single symbol."""
        data = self._get("/api/v3/ticker/24hr", {"symbol": symbol})
        return {
            "last_price": float(data.get("lastPrice",          0)),
            "volume_24h": float(data.get("quoteVolume",        0)),
            "high_24h":   float(data.get("highPrice",          0)),
            "low_24h":    float(data.get("lowPrice",           0)),
            "change_pct": float(data.get("priceChangePercent", 0)),
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list[dict]:
        raw: list[list] = self._get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        candles = []
        for row in raw:
            candles.append({
                "t": int(row[0]) // 1000,   # ms → unix seconds
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
                "v": float(row[5]),
            })
        return candles

    def _get(self, path: str, params: dict | None = None) -> Any:
        """HTTP GET with exponential-backoff retry. Returns parsed JSON."""
        qs  = ("?" + urllib.parse.urlencode(params)) if params else ""
        url = self._base_url + path + qs

        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "trading-bot/1.0", "Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode())

            except urllib.error.HTTPError as exc:
                last_exc = exc
                # 429 / 418 = rate-limited; back off hard
                if exc.code in (429, 418):
                    wait = self._backoff * 10 * (attempt + 1)
                    logger.warning(f"Binance rate-limited (HTTP {exc.code}) — sleeping {wait}s")
                    time.sleep(wait)
                    continue
                raise   # other 4xx are not retriable

            except Exception as exc:
                last_exc = exc
                if attempt < self._retries:
                    wait = self._backoff * (attempt + 1)
                    logger.warning(
                        f"Binance {path} error (attempt {attempt + 1}/{self._retries + 1}): "
                        f"{exc} — retry in {wait}s"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Binance {path} failed after {self._retries + 1} attempts: {last_exc}"
        )
