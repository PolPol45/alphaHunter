"""
OpenInsider Client Adapter
==========================
Parses recent insider filings and derives "cluster buy" signals.

The upstream site exposes an HTML table. This adapter keeps parsing simple and
dependency-light so tests can use static HTML fixtures without network access.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.request
from urllib.error import HTTPError, URLError
from collections import defaultdict
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("openinsider_client")

_DEFAULT_URL = (
    "https://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&"
    "xp=1&xs=1&vl=50&vh=&ocl=10000&och=&sic1=-1&sicl=100&sich=9999&"
    "isofficer=1&isdirector=1&istenpercent=1&iscfo=1&ceo=1&pres=1&"
    "owner=1&sic2=-1&sortcol=0&cnt=100&page=1"
)


class OpenInsiderClient:
    def __init__(self, cfg: dict) -> None:
        self._enabled = bool(cfg.get("enabled", True))
        self._timeout = int(cfg.get("timeout_seconds", 15))
        self._url = cfg.get("url", _DEFAULT_URL)
        self._limit = int(cfg.get("limit", 100))
        self._lookback_days = int(cfg.get("lookback_days", 14))
        self._min_cluster_filings = int(cfg.get("min_cluster_filings", 2))
        self._min_cluster_value = float(cfg.get("min_cluster_value", 250_000.0))
        self._retry_attempts = int(cfg.get("retry_attempts", 2))
        self._connected = False
        self._reachable = False
        self._state = "unknown"
        self._last_error: str | None = None
        self._last_success_at: str | None = None

    def connect(self) -> bool:
        if not self._enabled:
            self._last_error = "openinsider disabled"
            self._state = "down"
            logger.info(self._last_error)
            return False
        try:
            self._probe()
            self._connected = True
            self._reachable = True
            self._state = "connected"
            self._last_error = None
            return True
        except Exception as exc:
            self._connected = False
            self._reachable = False
            self._state = "down"
            self._last_error = str(exc)
            logger.warning(f"OpenInsider probe failed: {exc}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def reachable(self) -> bool:
        return self._reachable

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_success_at(self) -> str | None:
        return self._last_success_at

    def get_status(self) -> dict:
        return {
            "connected": self._connected,
            "reachable": self._reachable,
            "state": self._state,
            "last_error": self._last_error,
            "last_success_at": self._last_success_at,
        }

    def get_activity(self) -> dict:
        if not self._enabled:
            return {"recent_filings": [], "clusters": []}
        try:
            html_doc = self._fetch_html()
            filings = self.parse_filings(html_doc, lookback_days=self._lookback_days)
            filings = filings[: self._limit]
            clusters = self.detect_clusters(
                filings,
                min_filings=self._min_cluster_filings,
                min_total_value=self._min_cluster_value,
            )
            self._connected = True
            self._reachable = True
            self._state = "connected"
            self._last_error = None
            self._last_success_at = datetime.now(timezone.utc).isoformat()
            return {"recent_filings": filings, "clusters": clusters}
        except Exception as exc:
            self._connected = False
            self._reachable = False
            self._state = "down"
            self._last_error = str(exc)
            logger.warning(f"OpenInsider fetch failed: {exc}")
            return {"recent_filings": [], "clusters": []}

    def _fetch_html(self) -> str:
        last_exc = None
        for _ in range(max(self._retry_attempts, 1)):
            try:
                req = urllib.request.Request(
                    self._url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36"
                        ),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Cache-Control": "no-cache",
                        "Pragma": "no-cache",
                        "Referer": "https://openinsider.com/",
                    },
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_exc = exc
        if isinstance(last_exc, HTTPError):
            raise RuntimeError(f"HTTP {last_exc.code}") from last_exc
        if isinstance(last_exc, URLError):
            raise RuntimeError(f"network error: {last_exc.reason}") from last_exc
        if last_exc is not None:
            raise RuntimeError(str(last_exc)) from last_exc
        raise RuntimeError("unknown openinsider fetch failure")

    def _probe(self) -> None:
        req = urllib.request.Request(
            "https://openinsider.com/",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            resp.read(128)

    @staticmethod
    def parse_filings(html_doc: str, lookback_days: int = 14) -> list[dict]:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_doc, re.IGNORECASE | re.DOTALL)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        filings: list[dict] = []

        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)
            if len(cells) < 9:
                continue
            clean = [OpenInsiderClient._strip_html(cell) for cell in cells]

            filing_dt = OpenInsiderClient._parse_datetime(clean[1] if len(clean) > 1 else clean[0])
            symbol = clean[3] if len(clean) > 3 else ""
            insider = clean[5] if len(clean) > 5 else ""
            title = clean[6] if len(clean) > 6 else ""
            trade_type = clean[8] if len(clean) > 8 else ""
            price = OpenInsiderClient._parse_number(clean[9] if len(clean) > 9 else "0")
            qty = OpenInsiderClient._parse_number(clean[10] if len(clean) > 10 else "0")
            value = OpenInsiderClient._parse_number(clean[12] if len(clean) > 12 else "0")

            if not filing_dt or filing_dt < cutoff:
                continue
            if not symbol or "P - Purchase" not in trade_type:
                continue

            filings.append(
                {
                    "filed_at": filing_dt.isoformat(),
                    "symbol": symbol.upper(),
                    "insider": insider,
                    "title": title,
                    "trade_type": trade_type,
                    "price": price,
                    "quantity": qty,
                    "value_usd": value,
                }
            )

        filings.sort(key=lambda item: item["filed_at"], reverse=True)
        return filings

    @staticmethod
    def detect_clusters(
        filings: list[dict],
        min_filings: int = 2,
        min_total_value: float = 250_000.0,
    ) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for filing in filings:
            grouped[filing["symbol"]].append(filing)

        clusters: list[dict] = []
        for symbol, rows in grouped.items():
            total_value = sum(float(row.get("value_usd", 0.0)) for row in rows)
            insiders = sorted({row.get("insider", "") for row in rows if row.get("insider")})
            if len(rows) < min_filings and total_value < min_total_value:
                continue
            clusters.append(
                {
                    "symbol": symbol,
                    "filing_count": len(rows),
                    "unique_insiders": len(insiders),
                    "insiders": insiders,
                    "total_value_usd": round(total_value, 2),
                    "latest_filed_at": rows[0]["filed_at"],
                    "avg_price": round(
                        sum(float(row.get("price", 0.0)) for row in rows) / max(len(rows), 1),
                        4,
                    ),
                    "alert_type": "INSIDER_CLUSTER_BUY",
                }
            )

        clusters.sort(key=lambda item: (item["filing_count"], item["total_value_usd"]), reverse=True)
        return clusters

    @staticmethod
    def _strip_html(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = html.unescape(value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def _parse_number(raw: str) -> float:
        cleaned = raw.replace("$", "").replace(",", "").replace("+", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_datetime(raw: str) -> datetime | None:
        raw = raw.strip()
        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
        ]
        for pattern in patterns:
            try:
                dt = datetime.strptime(raw, pattern)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
