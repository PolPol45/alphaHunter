"""
Yahoo Finance News Adapter
==========================
Wraps yfinance headline fetching for a configurable watchlist.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("yfinance_news_client")


class YFinanceNewsClient:
    def __init__(self, cfg: dict) -> None:
        self._enabled = bool(cfg.get("enabled", True))
        self._watchlist = list(cfg.get("watchlist", []))
        self._news_per_symbol = int(cfg.get("news_per_symbol", 4))
        self._connected = False
        self._last_error: str | None = None
        self._yf = None

    def connect(self) -> bool:
        if not self._enabled:
            self._last_error = "yfinance news disabled"
            logger.info(self._last_error)
            return False
        try:
            import yfinance as yf

            self._yf = yf
            self._connected = True
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"yfinance import failed: {exc}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def get_news(self) -> list[dict]:
        if not self._connected or not self._yf:
            return []
        items: list[dict] = []
        for symbol in self._watchlist:
            try:
                ticker = self._yf.Ticker(symbol)
                raw_items = ticker.get_news(count=self._news_per_symbol) or []
            except Exception as exc:
                logger.debug(f"yfinance news fetch failed for {symbol}: {exc}")
                continue

            for item in raw_items[: self._news_per_symbol]:
                content = item.get("content") or {}
                pub = content.get("pubDate") or item.get("providerPublishTime")
                published_at = self._to_iso(pub)
                title = content.get("title") or item.get("title") or ""
                summary = content.get("summary") or item.get("summary") or ""
                canonical = content.get("canonicalUrl") or {}
                url = canonical.get("url") or item.get("link") or ""

                items.append(
                    {
                        "symbol": symbol,
                        "headline": title[:200],
                        "summary": summary[:600],
                        "published_at": published_at,
                        "url": url,
                        "source": f"yfinance/{symbol}",
                        "category": "equity" if "-" not in symbol else "crypto",
                    }
                )
        return items

    @staticmethod
    def _to_iso(raw) -> str:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=timezone.utc).isoformat()
        if isinstance(raw, str):
            for raw_value in (raw.replace("Z", "+00:00"), raw):
                try:
                    dt = datetime.fromisoformat(raw_value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc).isoformat()
                except ValueError:
                    continue
        return datetime.now(timezone.utc).isoformat()
