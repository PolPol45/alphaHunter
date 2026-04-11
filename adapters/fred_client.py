"""
FRED Client Adapter
===================
Fetches macro time series from the St. Louis Fed.

Design notes:
- Prefer the public `fredgraph.csv` endpoint so the adapter works even without
  an API key for the core snapshot use case.
- Keep the output minimal and normalised so higher-level agents can derive
  regime flags without depending on FRED response shape.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone

logger = logging.getLogger("fred_client")

_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_JSON_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredClient:
    def __init__(self, cfg: dict) -> None:
        self._enabled = bool(cfg.get("enabled", True))
        self._api_key = cfg.get("api_key", "").strip()
        self._timeout = int(cfg.get("timeout_seconds", 15))
        self._probe_series_id = str(cfg.get("probe_series_id", "FEDFUNDS"))
        self._connected = False
        self._state = "unknown"
        self._last_error: str | None = None

    def connect(self) -> bool:
        if not self._enabled:
            self._last_error = "fred disabled"
            self._state = "down"
            logger.info(self._last_error)
            return False
        try:
            if self._api_key:
                rows = self._fetch_json(self._probe_series_id)
            else:
                rows = self._fetch_csv(self._probe_series_id)
            if rows:
                self._connected = True
                self._state = "connected"
                self._last_error = None
                return True
            self._connected = False
            self._state = "degraded"
            self._last_error = f"probe series {self._probe_series_id} returned no valid rows"
            logger.warning(self._last_error)
            return False
        except Exception as exc:
            self._connected = False
            self._state = "down"
            self._last_error = str(exc)
            logger.warning(f"FRED connect failed: {exc}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def get_series(self, series_id: str, limit: int = 24) -> list[dict]:
        try:
            if self._api_key:
                rows = self._fetch_json(series_id)
            else:
                rows = self._fetch_csv(series_id)
            if rows:
                self._connected = True
                self._state = "connected"
                self._last_error = None
            else:
                self._connected = False
                self._state = "degraded"
                self._last_error = f"{series_id}: empty CSV or all values missing"
                logger.warning(self._last_error)
            return rows[-limit:]
        except Exception as exc:
            self._connected = False
            self._state = "down"
            self._last_error = str(exc)
            logger.warning(f"FRED fetch failed for {series_id}: {exc}")
            return []

    def _fetch_csv(self, series_id: str) -> list[dict]:
        params = urllib.parse.urlencode({"id": series_id})
        req = urllib.request.Request(
            f"{_CSV_URL}?{params}",
            headers={"User-Agent": "trading-bot/1.0", "Accept": "text/csv"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} for series {series_id}") from exc
        except URLError as exc:
            raise RuntimeError(f"network error for series {series_id}: {exc.reason}") from exc
        return self.parse_csv(text)

    @staticmethod
    def parse_csv(csv_text: str) -> list[dict]:
        rows: list[dict] = []
        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 2:
            return rows
        value_key = "VALUE" if "VALUE" in fieldnames else fieldnames[1]
        for row in reader:
            date_raw = str(row.get("DATE", "")).strip()
            value_raw = str(row.get(value_key, "")).strip()
            if not date_raw or not value_raw or value_raw == ".":
                continue
            try:
                dt = datetime.strptime(date_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                value = float(value_raw)
            except ValueError:
                continue
            rows.append({"date": dt.isoformat(), "value": value})
        return rows

    def _fetch_json(self, series_id: str) -> list[dict]:
        import json
        params = urllib.parse.urlencode({
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json"
        })
        req = urllib.request.Request(
            f"{_JSON_URL}?{params}",
            headers={"User-Agent": "trading-bot/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} for series {series_id}") from exc
        except URLError as exc:
            raise RuntimeError(f"network error for series {series_id}: {exc.reason}") from exc
            
        rows = []
        if "observations" not in data:
            return rows
            
        for obs in data["observations"]:
            date_raw = str(obs.get("date", "")).strip()
            value_raw = str(obs.get("value", "")).strip()
            if not date_raw or not value_raw or value_raw == ".":
                continue
            try:
                dt = datetime.strptime(date_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                value = float(value_raw)
                rows.append({"date": dt.isoformat(), "value": value})
            except ValueError:
                continue
                
        return rows
