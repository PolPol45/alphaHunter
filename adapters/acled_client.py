"""
ACLED Client Adapter
=====================
Fetches armed conflict events from the Armed Conflict Location & Event Data
Project (ACLED) public REST API.

Free access: register at https://acleddata.com/acleds-access/
Credentials:  api_key + email (both required in every request)
Rate limits:  no documented hard limit on free research access

Endpoint used:
  GET https://api.acleddata.com/acled/read
  Params: key, email, event_date, event_date_where, event_type, limit, fields

Sentiment mapping:
  Battles / Explosions       → strong bearish (−0.70 … −0.80)
  Strategic developments     → moderate bearish (−0.45)   ← troop movements etc.
  Violence vs civilians      → bearish (−0.60)
  Protests / Riots           → mild bearish (−0.25 … −0.35)
  Fatalities amplify score   → ×1.0 … ×1.25 depending on count

Market impact heuristic (by country/region):
  Middle East, Russia/Ukraine, Strait of Malacca area → affects BTC/ETH (risk-off)
  All events → at minimum soft bearish signal on BTC/ETH
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("acled_client")

_BASE_URL = "https://api.acleddata.com"

# ACLED event_type → base sentiment score
_EVENT_TYPE_SCORE: dict[str, float] = {
    "Battles":                       -0.75,
    "Explosions/Remote violence":    -0.80,
    "Violence against civilians":    -0.60,
    "Strategic developments":        -0.45,
    "Protests":                      -0.25,
    "Riots":                         -0.35,
}

# Fields we actually need (reduces response size)
_FIELDS = "event_date|event_type|country|fatalities|latitude|longitude|notes"

# Regions mapped to crypto symbols affected
_REGION_SYMBOLS: list[tuple[tuple[float, float, float, float], list[str]]] = [
    # (lat_min, lat_max, lon_min, lon_max) → symbols
    ((-5,  45,  25,  65), ["BTCUSDT", "ETHUSDT"]),          # Middle East
    ((44,  72,  22,  45), ["BTCUSDT", "ETHUSDT"]),          # Russia/Ukraine/Caucasus
    (( 0,  30,  95, 110), ["BTCUSDT", "ETHUSDT"]),          # Strait of Malacca
    ((30,  45, 115, 145), ["BTCUSDT", "ETHUSDT", "BNBUSDT"]),  # East Asia
]


class ACLEDClient:
    """Fetches recent armed conflict events from ACLED and converts to sentiment."""

    def __init__(self, cfg: dict) -> None:
        self._key: str          = cfg.get("api_key", "")
        self._email: str        = cfg.get("email", "")
        self._lookback: int     = int(cfg.get("lookback_days", 7))
        self._limit: int        = int(cfg.get("max_events", 100))
        self._min_fat: int      = int(cfg.get("min_fatalities", 0))
        self._timeout: int      = int(cfg.get("timeout_seconds", 15))
        self._event_types: list[str] = cfg.get(
            "event_types",
            ["Battles", "Explosions/Remote violence", "Strategic developments",
             "Violence against civilians"],
        )
        self._connected: bool       = False
        self._last_error: str | None = None

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        if not self._key or not self._email:
            self._last_error = "ACLED api_key and email are required"
            logger.warning(self._last_error)
            return False
        try:
            # Lightweight probe: fetch 1 event
            self._fetch_raw(limit=1)
            self._connected = True
            logger.info("ACLED connected")
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"ACLED connect failed: {exc}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get_events(self) -> list[dict]:
        """Return normalised conflict events as internal event dicts."""
        if not self._connected:
            return []
        try:
            raw_events = self._fetch_raw(limit=self._limit)
            return [self._normalise(e) for e in raw_events
                    if int(e.get("fatalities", 0)) >= self._min_fat]
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"ACLED fetch error: {exc}")
            return []

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _fetch_raw(self, limit: int) -> list[dict]:
        today      = datetime.now(timezone.utc).date()
        date_from  = (today - timedelta(days=self._lookback)).isoformat()
        date_to    = today.isoformat()
        types_str  = "|".join(self._event_types)

        params = {
            "key":              self._key,
            "email":            self._email,
            "event_date":       f"{date_from}|{date_to}",
            "event_date_where": "BETWEEN",
            "event_type":       types_str,
            "limit":            str(limit),
            "fields":           _FIELDS,
        }
        url = _BASE_URL + "/acled/read?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": "trading-bot/1.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body: dict = json.loads(resp.read().decode())

        if not body.get("success") and body.get("status") != 200:
            raise RuntimeError(f"ACLED API error: {body.get('error', body)}")

        return list(body.get("data", []))

    def _normalise(self, raw: dict) -> dict:
        import uuid as _uuid

        event_type  = str(raw.get("event_type", "Strategic developments"))
        fatalities  = int(raw.get("fatalities", 0))
        country     = str(raw.get("country", ""))
        notes       = str(raw.get("notes", ""))
        lat         = _safe_float(raw.get("latitude",  0))
        lon         = _safe_float(raw.get("longitude", 0))
        event_date  = str(raw.get("event_date", datetime.now(timezone.utc).date().isoformat()))

        base_score  = _EVENT_TYPE_SCORE.get(event_type, -0.40)
        # Fatalities amplifier: 0 → ×1.0, 10+ → ×1.15, 100+ → ×1.25
        if fatalities >= 100:
            base_score *= 1.25
        elif fatalities >= 10:
            base_score *= 1.15
        base_score = max(-1.0, base_score)

        severity = (
            "critical" if fatalities >= 50
            else "high"   if fatalities >= 10
            else "medium" if fatalities >= 1
            else "low"
        )

        return {
            "id":               str(_uuid.uuid4()),
            "title":            f"[ACLED] {event_type} — {country}",
            "category":         "conflict",
            "event_type":       event_type.lower().replace(" ", "_").replace("/", "_"),
            "severity":         severity,
            "timestamp":        f"{event_date}T00:00:00+00:00",
            "summary":          notes[:300] if notes else f"{event_type} in {country}. Fatalities: {fatalities}.",
            "symbols_affected": _region_to_symbols(lat, lon),
            "sentiment":        "bearish",
            "confidence":       min(0.50 + fatalities * 0.01, 0.95),
            "raw_signals":      [event_type.lower().replace(" ", "_").replace("/", "_")],
            "source":           "acled",
        }


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _region_to_symbols(lat: float, lon: float) -> list[str]:
    for (lat_min, lat_max, lon_min, lon_max), syms in _REGION_SYMBOLS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return syms
    return ["BTCUSDT", "ETHUSDT"]   # default: all global events affect major crypto
