"""
World Monitor Client Adapter
=============================
Connects to the WorldMonitor real-time global intelligence API.

Live API base: https://api.worldmonitor.app
Finance API:   https://finance.worldmonitor.app

Authentication: Bearer token in config["world_monitor"]["api_key"]
  Header: Authorization: Bearer wm_live_xxx

Known endpoints used by this adapter:
  GET /v1/intelligence/convergence?region=GLOBAL&time_window=6h
      → multi-signal convergence events (highest actionability)
  GET /api/conflict/v1/events
      → armed conflict / military / geopolitical events
  GET /api/economic/v1/events   (finance variant)
      → sanctions, trade routes, economic disruptions

Sentiment scoring (output: float −1.0 … +1.0):
  Bearish signals → negative score  (conflict, sanctions, outages)
  Bullish signals → positive score  (trade deals, growth data)
  Multiplied by event confidence when available.

Two modes:
  LIVE  — api_key set and stub_mode=false → real HTTP calls
  STUB  — no api_key or stub_mode=true   → rotating demo events (safe for dev)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("world_monitor_client")

# ── Sentiment classification tables ─────────────────────────────────────── #

# Individual signal names from convergence endpoint → directional weight
_SIGNAL_SENTIMENT: dict[str, float] = {
    # Bearish — geopolitical / military tension
    "military_flights":         -0.70,
    "ais_dark_ships":           -0.55,   # sanctions evasion
    "oref_sirens":              -0.80,   # active conflict alerts
    "missile_launch":           -0.90,
    "drone_activity":           -0.65,
    "naval_deployment":         -0.60,
    "military_buildup":         -0.65,
    "conflict_escalation":      -0.75,
    "sanctions_imposed":        -0.60,
    "trade_route_disruption":   -0.55,
    "infrastructure_outage":    -0.50,
    "pipeline_incident":        -0.60,
    "power_grid_attack":        -0.55,
    "cyber_attack":             -0.45,
    "earthquake":               -0.30,
    "extreme_weather":          -0.25,
    # Bullish — de-escalation / economic positive
    "ceasefire":                +0.50,
    "peace_agreement":          +0.55,
    "sanctions_lifted":         +0.65,
    "trade_deal":               +0.60,
    "economic_growth":          +0.45,
    "rate_cut":                 +0.55,
    "stimulus":                 +0.50,
    "trade_route_open":         +0.45,
}

# Convergence event type → baseline sentiment
_EVENT_TYPE_SENTIMENT: dict[str, float] = {
    "multi_signal_convergence": -0.65,   # by default bearish (conflict convergence)
    "conflict":                 -0.70,
    "military":                 -0.60,
    "sanctions":                -0.55,
    "cyber":                    -0.40,
    "disaster":                 -0.35,
    "economic_negative":        -0.50,
    "economic_positive":        +0.45,
    "diplomatic":               +0.30,
    "trade_agreement":          +0.55,
    "macro":                    0.0,     # neutral until classified
    "financial":                0.0,
}

# Category string → base sentiment (fallback)
_CATEGORY_SENTIMENT: dict[str, float] = {
    "geopolitical":  -0.50,
    "conflict":      -0.65,
    "military":      -0.55,
    "sanctions":     -0.50,
    "cyber":         -0.35,
    "disaster":      -0.30,
    "macro":          0.00,
    "financial":      0.00,
    "crypto":         0.10,   # crypto-specific news tends bullish for crypto
    "economic":       0.10,
}

# Severity → confidence multiplier
_SEVERITY_WEIGHT: dict[str, float] = {
    "critical": 1.00,
    "high":     0.80,
    "medium":   0.60,
    "low":      0.40,
    "info":     0.20,
}


def _score_event(event: dict) -> float:
    """Return a sentiment score in [−1, +1] for a single normalised event."""
    # 1. Try to score from raw signals list (convergence endpoint)
    signals: list[str] = event.get("raw_signals", [])
    if signals:
        raw_score = sum(_SIGNAL_SENTIMENT.get(s, 0.0) for s in signals) / max(len(signals), 1)
    else:
        # 2. Fall back to event type → category → neutral
        etype = str(event.get("event_type", "")).lower()
        cat   = str(event.get("category", "")).lower()
        sent  = str(event.get("sentiment", "neutral")).lower()

        if etype and etype in _EVENT_TYPE_SENTIMENT:
            raw_score = _EVENT_TYPE_SENTIMENT[etype]
        elif cat and cat in _CATEGORY_SENTIMENT:
            raw_score = _CATEGORY_SENTIMENT[cat]
        elif sent == "bullish":
            raw_score = +0.40
        elif sent == "bearish":
            raw_score = -0.40
        else:
            raw_score = 0.0

    # 3. Apply confidence / severity weight
    confidence = float(event.get("confidence", 0.0))
    if confidence > 0:
        weight = min(max(confidence, 0.0), 1.0)
    else:
        severity = str(event.get("severity", "medium")).lower()
        weight = _SEVERITY_WEIGHT.get(severity, 0.60)

    return max(-1.0, min(1.0, raw_score * weight))


def aggregate_sentiment(events: list[dict]) -> float:
    """
    Aggregate multiple event scores into a single market sentiment float.
    Weights recent events more heavily and caps extreme swings.
    Returns float in [−1, +1].  0.0 = neutral.
    """
    if not events:
        return 0.0

    scores = [_score_event(e) for e in events]
    # Simple mean — could be replaced with recency-weighted average
    avg = sum(scores) / len(scores)
    # Soft-cap: prevent a single event from dominating
    return max(-0.80, min(0.80, avg))


class WorldMonitorClient:
    def __init__(self, cfg: dict) -> None:
        self._enabled: bool        = bool(cfg.get("enabled", True))
        self._base_url: str        = cfg.get("base_url", "https://api.worldmonitor.app").rstrip("/")
        self._finance_url: str     = cfg.get("finance_base_url", "https://finance.worldmonitor.app").rstrip("/")
        self._api_key: str         = cfg.get("api_key", "")
        self._categories: list[str]= cfg.get(
            "categories", ["convergence", "conflict", "economic", "sanctions", "macro"]
        )
        self._max_events: int      = int(cfg.get("max_events", 30))
        self._time_window: str     = cfg.get("time_window", "6h")
        self._regions: list[str]   = cfg.get("regions", ["GLOBAL"])
        self._timeout: float       = float(cfg.get("timeout_seconds", 10))
        self._retries: int         = int(cfg.get("retry_attempts", 2))
        self._stub_mode: bool      = bool(cfg.get("stub_mode", True)) or not self._api_key

        self._http: Any            = None
        self._connected: bool      = False
        self._last_error: str | None = None

        # ── ACLED sub-client (conflict data) ─────────────────────────── #
        self._acled = None
        acled_cfg = cfg.get("acled", {})
        if acled_cfg.get("enabled", False) and acled_cfg.get("api_key"):
            try:
                from adapters.acled_client import ACLEDClient
                self._acled = ACLEDClient(acled_cfg)
            except ImportError:
                logger.warning("ACLEDClient import failed")

        # ── Finnhub sub-client (macro news + calendar) ───────────────── #
        self._finnhub = None
        finnhub_cfg = cfg.get("finnhub", {})
        if finnhub_cfg.get("enabled", False) and finnhub_cfg.get("api_key"):
            try:
                from adapters.finnhub_client import FinnhubClient
                self._finnhub = FinnhubClient(finnhub_cfg)
            except ImportError:
                logger.warning("FinnhubClient import failed")

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        if not self._enabled:
            logger.info("World Monitor disabled in config")
            return False

        if self._stub_mode:
            logger.info("World Monitor running in STUB mode (no api_key configured)")
            self._connected = True
            self._connect_subclients()
            return True

        try:
            import httpx

            headers: dict = {
                "Accept":     "application/json",
                "User-Agent": "crypto-trading-bot/1.0",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._http = httpx.Client(
                headers=headers,
                timeout=self._timeout,
                follow_redirects=True,
            )
            # Lightweight ping — try public endpoint first
            resp = self._http.get(
                f"{self._base_url}/v1/intelligence/convergence",
                params={"region": "GLOBAL", "time_window": "1h"},
            )
            resp.raise_for_status()
            self._connected = True
            logger.info(f"World Monitor connected (live) → {self._base_url}")
            return True

        except ImportError:
            self._last_error = "httpx not installed"
            logger.warning(f"{self._last_error} — stub mode")
            self._stub_mode = True
            self._connected = True
            return True

        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(f"World Monitor connect failed: {exc} — stub mode")
            self._stub_mode = True
            self._connected = True
            return True

        finally:
            self._connect_subclients()

    def _connect_subclients(self) -> None:
        """Connect ACLED and Finnhub sub-clients (idempotent)."""
        if self._acled and not self._acled.is_connected():
            if self._acled.connect():
                logger.info("ACLED sub-client ready")
            else:
                logger.warning(f"ACLED unavailable: {self._acled.last_error}")
                self._acled = None
        if self._finnhub and not self._finnhub.is_connected():
            if self._finnhub.connect():
                logger.info("Finnhub sub-client ready")
            else:
                logger.warning(f"Finnhub unavailable: {self._finnhub.last_error}")
                self._finnhub = None

    def disconnect(self) -> None:
        if self._http:
            try:
                self._http.close()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def mode(self) -> str:
        return "stub" if self._stub_mode else "live"

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get_events(self, limit: int | None = None) -> list[dict]:
        """Fetch normalised events. Returns [] on any error (non-blocking)."""
        if not self._enabled or not self._connected:
            return []

        n = limit or self._max_events
        if self._stub_mode:
            events = self._stub_events(n)
            # Even in stub mode, pull real data from sub-clients if connected
            events.extend(self._fetch_subclients())
            return events

        all_events: list[dict] = []
        for attempt in range(1, self._retries + 1):
            try:
                all_events = self._fetch_live(n)
                logger.info(
                    f"World Monitor: {len(all_events)} events | "
                    f"sentiment={aggregate_sentiment(all_events):+.2f}"
                )
                return all_events
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning(f"World Monitor attempt {attempt}/{self._retries}: {exc}")
                if attempt < self._retries:
                    time.sleep(2.0)

        logger.warning("World Monitor: all live attempts failed — returning []")
        return []

    def _fetch_subclients(self) -> list[dict]:
        """Fetch events from ACLED and Finnhub sub-clients (always, regardless of stub mode)."""
        events: list[dict] = []
        if self._acled:
            try:
                acled_events = self._acled.get_events()
                events.extend(acled_events)
                if acled_events:
                    logger.info(f"ACLED: {len(acled_events)} conflict events fetched")
            except Exception as exc:
                logger.warning(f"ACLED fetch error: {exc}")
        if self._finnhub:
            try:
                finnhub_events = self._finnhub.get_events()
                events.extend(finnhub_events)
                if finnhub_events:
                    logger.info(f"Finnhub: {len(finnhub_events)} macro/news events fetched")
            except Exception as exc:
                logger.warning(f"Finnhub fetch error: {exc}")
        return events

    def get_sentiment(self) -> float:
        """Convenience method: fetch events and return aggregate sentiment score."""
        events = self.get_events()
        return aggregate_sentiment(events)

    # ------------------------------------------------------------------ #
    # Live fetch — multiple endpoint strategy                             #
    # ------------------------------------------------------------------ #

    def _fetch_live(self, limit: int) -> list[dict]:
        events: list[dict] = []

        # 1. Intelligence convergence (highest value — multi-signal)
        if "convergence" in self._categories:
            for region in self._regions:
                try:
                    data = self._get(
                        self._base_url,
                        "/v1/intelligence/convergence",
                        {"region": region, "time_window": self._time_window},
                    )
                    for raw in (data.get("data") or []):
                        events.append(self._normalise_convergence(raw))
                except Exception as exc:
                    logger.debug(f"convergence/{region} skipped: {exc}")

        # 2. Conflict events
        if "conflict" in self._categories:
            try:
                data = self._get(self._base_url, "/api/conflict/v1/events", {"limit": str(limit)})
                for raw in (data.get("events") or data.get("data") or []):
                    events.append(self._normalise_generic(raw, "conflict"))
            except Exception as exc:
                logger.debug(f"conflict/events skipped: {exc}")

        # 3. Economic / sanctions (finance endpoint)
        if "economic" in self._categories or "sanctions" in self._categories:
            try:
                data = self._get(self._finance_url, "/api/economic/v1/events", {"limit": str(limit)})
                for raw in (data.get("events") or data.get("data") or []):
                    events.append(self._normalise_generic(raw, "economic"))
            except Exception as exc:
                logger.debug(f"economic/events skipped: {exc}")

        # 4. ACLED conflict data (if sub-client is active)
        if self._acled:
            try:
                acled_events = self._acled.get_events()
                events.extend(acled_events)
                if acled_events:
                    logger.info(f"ACLED: {len(acled_events)} conflict events fetched")
            except Exception as exc:
                logger.warning(f"ACLED fetch error: {exc}")

        # 5. Finnhub macro news + calendar (if sub-client is active)
        if self._finnhub:
            try:
                finnhub_events = self._finnhub.get_events()
                events.extend(finnhub_events)
                if finnhub_events:
                    logger.info(f"Finnhub: {len(finnhub_events)} macro/news events fetched")
            except Exception as exc:
                logger.warning(f"Finnhub fetch error: {exc}")

        # Deduplicate by id and return up to limit
        seen: set = set()
        unique: list[dict] = []
        for e in events:
            if e["id"] not in seen:
                seen.add(e["id"])
                unique.append(e)

        return unique[:limit]

    def _get(self, base: str, path: str, params: dict | None = None) -> dict:
        if not self._http:
            raise RuntimeError("HTTP client not initialised")
        resp = self._http.get(f"{base}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Normalisers                                                         #
    # ------------------------------------------------------------------ #

    def _normalise_convergence(self, raw: dict) -> dict:
        """Normalise /v1/intelligence/convergence response item."""
        signals: list[str] = list(raw.get("signals") or [])
        confidence: float  = float(raw.get("confidence", 0.7))
        loc: dict          = raw.get("location") or {}

        # Determine sentiment from signal names
        signal_scores = [_SIGNAL_SENTIMENT.get(s, -0.30) for s in signals]
        base_score    = (sum(signal_scores) / max(len(signal_scores), 1)) if signal_scores else -0.50
        sent_str      = "bearish" if base_score < -0.1 else ("bullish" if base_score > 0.1 else "neutral")

        return {
            "id":               str(raw.get("id") or uuid.uuid4()),
            "title":            f"Intelligence convergence: {', '.join(signals[:3])}",
            "category":         "geopolitical",
            "event_type":       str(raw.get("type", "multi_signal_convergence")),
            "severity":         "high" if confidence > 0.80 else "medium",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "summary":          (
                f"Multi-signal convergence detected. "
                f"Signals: {', '.join(signals)}. "
                f"Confidence: {confidence:.0%}. "
                f"Location: lat={loc.get('lat', '?')} lon={loc.get('lng', '?')}."
            ),
            "symbols_affected": self._infer_symbols(signals, loc),
            "sentiment":        sent_str,
            "confidence":       confidence,
            "raw_signals":      signals,
            "source":           "worldmonitor/convergence",
        }

    def _normalise_generic(self, raw: dict, default_category: str) -> dict:
        """Normalise generic event from conflict/economic endpoints."""
        sentiment_raw = str(raw.get("sentiment", "neutral")).lower()
        sent_str = (
            "bearish" if sentiment_raw in ("bearish", "negative", "bad")
            else ("bullish" if sentiment_raw in ("bullish", "positive", "good")
                  else "neutral")
        )
        return {
            "id":               str(raw.get("id") or uuid.uuid4()),
            "title":            str(raw.get("title") or raw.get("headline") or ""),
            "category":         str(raw.get("category") or raw.get("type") or default_category),
            "event_type":       str(raw.get("type") or default_category),
            "severity":         str(raw.get("severity") or raw.get("impact") or "medium"),
            "timestamp":        str(raw.get("timestamp") or raw.get("published_at")
                                    or datetime.now(timezone.utc).isoformat()),
            "summary":          str(raw.get("summary") or raw.get("body") or ""),
            "symbols_affected": list(raw.get("symbols_affected") or raw.get("tickers") or []),
            "sentiment":        sent_str,
            "confidence":       float(raw.get("confidence_score") or 0.60),
            "raw_signals":      [],
            "source":           str(raw.get("source") or self._base_url),
        }

    @staticmethod
    def _infer_symbols(signals: list[str], loc: dict) -> list[str]:
        """Heuristically map signals/location to affected trading symbols."""
        affected: set[str] = set()
        sig_str = " ".join(signals).lower()

        # Middle East / energy-sensitive regions → oil correlates
        lat = float(loc.get("lat", 0))
        lng = float(loc.get("lng", 0))
        is_mena   = -20 < lat < 45 and 25 < lng < 65
        is_russia = lat > 45 and 30 < lng < 180

        if is_mena or is_russia:
            affected.update(["BTCUSDT", "ETHUSDT"])   # risk-off → crypto down too

        if "ais" in sig_str or "maritime" in sig_str or "waterway" in sig_str:
            affected.update(["BTCUSDT", "ETHUSDT"])

        if "cyber" in sig_str:
            affected.update(["BTCUSDT", "ETHUSDT", "SOLUSDT"])

        if not affected:
            affected.update(["BTCUSDT", "ETHUSDT"])   # default: BTC/ETH react to all macro

        return sorted(affected)

    # ------------------------------------------------------------------ #
    # Stub / demo mode                                                     #
    # ------------------------------------------------------------------ #

    def _stub_events(self, limit: int) -> list[dict]:
        minute = datetime.now(timezone.utc).minute
        start  = minute % len(_STUB_EVENT_POOL)
        pool   = (_STUB_EVENT_POOL[start:] + _STUB_EVENT_POOL[:start])[:limit]
        now    = datetime.now(timezone.utc).isoformat()
        return [{**e, "timestamp": now, "id": str(uuid.uuid4())} for e in pool]


# ------------------------------------------------------------------ #
# Stub event pool                                                      #
# ------------------------------------------------------------------ #

_STUB_EVENT_POOL: list[dict] = [
    {
        "id": "stub-001",
        "title": "Multi-signal convergence: military flights + dark ships (Persian Gulf)",
        "category": "geopolitical", "event_type": "multi_signal_convergence",
        "severity": "high", "timestamp": "",
        "summary": "Convergence of military flight activity, AIS-dark vessels, and radar anomalies detected in the Persian Gulf region. Confidence 0.88.",
        "symbols_affected": ["BTCUSDT", "ETHUSDT"], "sentiment": "bearish",
        "confidence": 0.88, "raw_signals": ["military_flights", "ais_dark_ships"],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-002",
        "title": "Federal Reserve holds rates — data-dependent guidance",
        "category": "macro", "event_type": "macro",
        "severity": "high", "timestamp": "",
        "summary": "FOMC voted unanimously to hold. Chair signalled caution on cuts. Risk assets neutral.",
        "symbols_affected": ["BTCUSDT", "ETHUSDT"], "sentiment": "neutral",
        "confidence": 0.95, "raw_signals": [],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-003",
        "title": "US CPI below expectations — 2.9% YoY",
        "category": "macro", "event_type": "economic_positive",
        "severity": "high", "timestamp": "",
        "summary": "Consumer prices rose less than forecast. Rate cut expectations increased.",
        "symbols_affected": ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "sentiment": "bullish",
        "confidence": 0.92, "raw_signals": ["economic_growth"],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-004",
        "title": "New US sanctions on Iranian oil exports",
        "category": "sanctions", "event_type": "sanctions",
        "severity": "medium", "timestamp": "",
        "summary": "US Treasury designates additional Iranian oil trading entities. Energy supply risk elevated.",
        "symbols_affected": ["BTCUSDT", "BNBUSDT"], "sentiment": "bearish",
        "confidence": 0.75, "raw_signals": ["sanctions_imposed"],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-005",
        "title": "Strait of Hormuz: AIS-dark vessel cluster detected",
        "category": "geopolitical", "event_type": "multi_signal_convergence",
        "severity": "critical", "timestamp": "",
        "summary": "Multiple vessels went dark near the Strait of Hormuz. Energy supply route risk elevated.",
        "symbols_affected": ["BTCUSDT", "ETHUSDT"], "sentiment": "bearish",
        "confidence": 0.91, "raw_signals": ["ais_dark_ships", "naval_deployment"],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-006",
        "title": "US-China trade talks resume in Geneva",
        "category": "macro", "event_type": "diplomatic",
        "severity": "medium", "timestamp": "",
        "summary": "Senior negotiators met for the first time in 6 months. Markets interpreted as de-escalation.",
        "symbols_affected": ["BTCUSDT", "ETHUSDT"], "sentiment": "bullish",
        "confidence": 0.70, "raw_signals": ["peace_agreement"],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-007",
        "title": "SEC approves spot Bitcoin ETF options",
        "category": "crypto", "event_type": "economic_positive",
        "severity": "high", "timestamp": "",
        "summary": "Regulatory approval expands institutional access to Bitcoin derivatives. Bullish for crypto.",
        "symbols_affected": ["BTCUSDT", "ETHUSDT"], "sentiment": "bullish",
        "confidence": 0.98, "raw_signals": ["economic_growth"],
        "source": "worldmonitor/stub",
    },
    {
        "id": "stub-008",
        "title": "Power grid outages reported across Eastern Europe",
        "category": "infrastructure", "event_type": "infrastructure_outage",
        "severity": "medium", "timestamp": "",
        "summary": "Widespread outages linked to substation failures. Crypto mining operations in the region affected.",
        "symbols_affected": ["BTCUSDT"], "sentiment": "bearish",
        "confidence": 0.65, "raw_signals": ["infrastructure_outage"],
        "source": "worldmonitor/stub",
    },
]
