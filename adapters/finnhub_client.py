"""
Finnhub Client Adapter
=======================
Fetches macro news and economic calendar events from Finnhub.io.

Free tier: 60 calls/min, no credit card required
Register:  https://finnhub.io/register
Key format: alphanumeric string (e.g. "cv3abc123xyz")

Endpoints used (all free tier):
  GET /api/v1/news?category=general         — market news headlines
  GET /api/v1/news?category=crypto          — crypto-specific news
  GET /api/v1/calendar/economic             — scheduled macro events with actual vs estimate

Sentiment strategy:
  News headlines  → keyword-based scoring (no NLP required)
  Economic calendar → actual vs estimate deviation scoring
    • Positive surprise (actual > estimate for growth indicators) → bullish
    • Negative surprise (actual > estimate for inflation/rates)   → bearish
    • High-impact events get 2× weight

Keyword lists are intentionally broad and tunable via config.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("finnhub_client")

_BASE_URL = "https://finnhub.io/api/v1"

# ── Keyword sentiment dictionaries ─────────────────────────────────────── #

_BEARISH_KEYWORDS: list[str] = [
    "rate hike", "inflation surge", "inflation rises", "hot cpi",
    "war", "conflict", "attack", "sanction", "default", "recession",
    "crash", "plunge", "collapse", "contraction", "deficit", "debt ceiling",
    "bank failure", "liquidity crisis", "tightening", "hawkish",
    "supply shock", "trade war", "tariff", "embargo", "geopolitical",
    "selloff", "sell-off", "risk-off", "downgrade",
]

_BULLISH_KEYWORDS: list[str] = [
    "rate cut", "rate cuts", "dovish", "pivot", "stimulus",
    "beat expectations", "better than expected", "strong gdp",
    "jobs beat", "growth", "recovery", "rally", "surge",
    "approval", "deal", "agreement", "trade deal", "ceasefire",
    "soft landing", "disinflation", "inflation falls", "cpi cools",
    "etf approved", "institutional", "adoption", "upgrade",
    "bull", "bulls", "price target", "target", "inflows", "staking",
    "launch", "launches", "record high", "ath",
]

# Economic event name → True if higher actual = good (growth), False if bad (inflation)
_GROWTH_INDICATORS: set[str] = {
    "gdp", "non-farm payroll", "nonfarm payroll", "retail sales",
    "industrial production", "pmi", "consumer confidence", "employment",
    "jobs", "housing starts", "building permits",
}
_INFLATION_INDICATORS: set[str] = {
    "cpi", "pce", "inflation", "ppi", "core inflation",
    "wage", "hourly earnings",
}


def _score_headline(headline: str) -> float:
    """Keyword scan of a news headline. Returns score in [−0.6, +0.6]."""
    h = headline.lower()
    bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in h)
    bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in h)
    if "iran war" in h or ("war" in h and "inflation" in h):
        bear += 1
    if ("bitcoin" in h or "ethereum" in h or "crypto" in h) and (
        "target" in h or "bulls" in h or "inflows" in h or "staking" in h
    ):
        bull += 1
    net  = bull - bear
    # Clamp to ±2 matches max to avoid pile-on
    return max(-0.60, min(0.60, net * 0.30))


def _score_calendar_event(event: dict) -> float:
    """
    Score an economic calendar event by actual vs estimate deviation.
    Returns score in [−0.8, +0.8], 0.0 if data incomplete.
    """
    actual   = event.get("actual")
    estimate = event.get("estimate")
    name     = str(event.get("event", "")).lower()
    impact   = str(event.get("impact", "low")).lower()

    if actual is None or estimate is None:
        return 0.0   # not yet released

    try:
        actual   = float(actual)
        estimate = float(estimate)
    except (TypeError, ValueError):
        return 0.0

    if estimate == 0:
        return 0.0

    deviation = (actual - estimate) / abs(estimate)   # normalised surprise
    base = max(-0.80, min(0.80, deviation * 1.5))

    # Determine direction: inflation bad when above estimate, growth good when above
    is_growth    = any(k in name for k in _GROWTH_INDICATORS)
    is_inflation = any(k in name for k in _INFLATION_INDICATORS)

    if is_inflation:
        base = -base   # CPI above estimate → bearish
    elif not is_growth:
        base = base * 0.5   # unknown indicator → dampen

    # Impact multiplier
    mult = {"high": 1.0, "medium": 0.70, "low": 0.40}.get(impact, 0.50)
    return round(max(-0.80, min(0.80, base * mult)), 4)


class FinnhubClient:
    """Fetches macro news and economic calendar from Finnhub free tier."""

    def __init__(self, cfg: dict) -> None:
        self._token: str         = cfg.get("api_key", "")
        self._categories: list[str] = cfg.get("categories", ["general", "crypto"])
        self._max_news: int      = int(cfg.get("max_news", 20))
        self._calendar_ahead: int = int(cfg.get("calendar_days_ahead", 3))
        self._timeout: int       = int(cfg.get("timeout_seconds", 12))
        self._connected: bool    = False
        self._last_error: str | None = None

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        if not self._token:
            self._last_error = "Finnhub api_key is required"
            logger.warning(self._last_error)
            return False
        try:
            self._get("/quote", {"symbol": "AAPL"})   # lightweight probe
            self._connected = True
            logger.info("Finnhub connected")
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"Finnhub connect failed: {exc}")
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
        """Return normalised news + calendar events."""
        if not self._connected:
            return []
        events: list[dict] = []
        events.extend(self._fetch_news())
        events.extend(self._fetch_calendar())
        return events

    # ------------------------------------------------------------------ #
    # News                                                                 #
    # ------------------------------------------------------------------ #

    def _fetch_news(self) -> list[dict]:
        import uuid as _uuid

        events: list[dict] = []
        cutoff_ts = int(
            (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
        )

        for cat in self._categories:
            try:
                raw_list: list[dict] = self._get("/news", {"category": cat})
                # Filter last 24h and score
                for item in raw_list[:self._max_news]:
                    if int(item.get("datetime", 0)) < cutoff_ts:
                        continue
                    headline = str(item.get("headline", ""))
                    summary  = str(item.get("summary", ""))
                    score    = _score_headline(headline + " " + summary)
                    if abs(score) < 0.15:
                        continue   # skip low-signal noise

                    sent_str = "bearish" if score < 0 else "bullish"
                    ts_iso   = datetime.fromtimestamp(
                        int(item.get("datetime", 0)), tz=timezone.utc
                    ).isoformat()

                    events.append({
                        "id":               str(_uuid.uuid4()),
                        "title":            headline[:120],
                        "category":         cat,
                        "event_type":       "news_" + cat,
                        "severity":         "medium",
                        "timestamp":        ts_iso,
                        "summary":          summary[:300],
                        "symbols_affected": _news_symbols(headline, cat),
                        "sentiment":        sent_str,
                        "confidence":       min(0.40 + abs(score), 0.85),
                        "raw_signals":      [],
                        "source":           f"finnhub/{cat}",
                    })
            except Exception as exc:
                logger.debug(f"Finnhub news/{cat} error: {exc}")

        return events

    # ------------------------------------------------------------------ #
    # Economic calendar                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_calendar(self) -> list[dict]:
        import uuid as _uuid

        today    = datetime.now(timezone.utc).date()
        date_to  = (today + timedelta(days=self._calendar_ahead)).isoformat()
        date_from = (today - timedelta(days=1)).isoformat()

        try:
            body: dict = self._get(
                "/calendar/economic",
                {"from": date_from, "to": date_to},
            )
        except Exception as exc:
            logger.debug(f"Finnhub calendar error: {exc}")
            return []

        events: list[dict] = []
        for item in body.get("economicCalendar", []):
            impact = str(item.get("impact", "low")).lower()
            if impact not in ("high", "medium"):
                continue   # skip low-impact events

            score    = _score_calendar_event(item)
            sent_str = (
                "bearish" if score < -0.10
                else "bullish" if score > 0.10
                else "neutral"
            )
            event_name = str(item.get("event", "Economic event"))
            country    = str(item.get("country", ""))
            time_str   = str(item.get("time", today.isoformat()))

            events.append({
                "id":               str(_uuid.uuid4()),
                "title":            f"[Macro] {event_name} ({country})",
                "category":         "macro",
                "event_type":       "economic_calendar",
                "severity":         "high" if impact == "high" else "medium",
                "timestamp":        _parse_finnhub_time(time_str),
                "summary":          (
                    f"{event_name}: actual={item.get('actual', 'pending')} "
                    f"estimate={item.get('estimate', '?')} "
                    f"prev={item.get('prev', '?')}"
                ),
                "symbols_affected": ["BTCUSDT", "ETHUSDT"],
                "sentiment":        sent_str,
                "confidence":       0.85 if impact == "high" else 0.60,
                "raw_signals":      [],
                "source":           "finnhub/calendar",
            })

        return events

    # ------------------------------------------------------------------ #
    # HTTP helper                                                          #
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: dict | None = None) -> Any:
        p = dict(params or {})
        p["token"] = self._token
        url = _BASE_URL + path + "?" + urllib.parse.urlencode(p)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "trading-bot/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _news_symbols(headline: str, category: str) -> list[str]:
    h = headline.lower()
    syms: list[str] = []
    if "bitcoin" in h or "btc" in h:
        syms.append("BTCUSDT")
    if "ethereum" in h or "eth" in h:
        syms.append("ETHUSDT")
    if "solana" in h or "sol" in h:
        syms.append("SOLUSDT")
    if "bnb" in h or "binance" in h:
        syms.append("BNBUSDT")
    if not syms and category == "crypto":
        if any(token in h for token in ("crypto", "digital asset", "etf", "exchange", "stablecoin")):
            syms = ["BTCUSDT", "ETHUSDT"]
    return syms


def _parse_finnhub_time(time_str: str) -> str:
    """Parse Finnhub time string to ISO8601 UTC."""
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()
