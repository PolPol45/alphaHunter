"""
News Data Agent
===============
Aggregates real-world context from Finnhub, Yahoo Finance, OpenInsider, and
FRED into three canonical data products:

- data/news_feed.json
- data/macro_snapshot.json
- data/insider_activity.json
"""

from __future__ import annotations

import re
import hashlib
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone

from agents.base_agent import BaseAgent, DATA_DIR


class NewsDataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("news_data_agent")
        self._cfg = self.config.get("news_data", {})
        self._max_items = int(self._cfg.get("max_items", 40))
        self._top_alerts = int(self._cfg.get("top_alerts", 8))
        self._lookback_hours = int(self._cfg.get("lookback_hours", 48))
        self._high_impact_threshold = float(self._cfg.get("high_impact_threshold", 0.72))
        self._macro_adjustment_weight = float(self._cfg.get("macro_adjustment_weight", 0.12))
        self._news_adjustment_weight = float(self._cfg.get("news_adjustment_weight", 0.10))
        self._compatibility_allow_stub = bool(self._cfg.get("compatibility_allow_stub", False))
        self._market_symbols = list(dict.fromkeys([
            *self.config.get("assets", []),
            *self.config.get("scanner", {}).get("crypto_universe", []),
        ]))
        self._alpha_symbols = list(self.config.get("alpha_hunter", {}).get("universe", []))

        self._finnhub = None
        finnhub_cfg = self.config.get("world_monitor", {}).get("finnhub", {})
        if finnhub_cfg.get("enabled", False):
            from adapters.finnhub_client import FinnhubClient

            self._finnhub = FinnhubClient(finnhub_cfg)
            self._finnhub.connect()

        from adapters.fred_client import FredClient
        from adapters.openinsider_client import OpenInsiderClient
        from adapters.yfinance_news_client import YFinanceNewsClient

        self._fred = FredClient(self.config.get("fred", {}))
        self._fred.connect()
        self._openinsider = OpenInsiderClient(self.config.get("openinsider", {}))
        self._openinsider.connect()
        self._yfinance_news = YFinanceNewsClient(self._cfg.get("yfinance", {}))
        self._yfinance_news.connect()

    def run(self) -> bool:
        self.mark_running()
        try:
            market_doc = self.read_json(DATA_DIR / "market_data.json")
            world_events = market_doc.get("world_events", [])
            generated_at = datetime.now(timezone.utc).isoformat()

            finnhub_items, finnhub_status = self._collect_finnhub_items()
            yfinance_items, yfinance_status = self._collect_yfinance_items()
            insider_doc = self._build_insider_activity(generated_at)
            insider_items = self._cluster_items(insider_doc.get("clusters", []))
            fallback_items = self._fallback_world_events(world_events, generated_at)

            all_items = [
                *finnhub_items,
                *yfinance_items,
                *insider_items,
                *fallback_items,
            ]
            deduped = self._dedupe_items(all_items)
            for item in deduped:
                self._apply_scores(item)

            deduped.sort(key=lambda item: item["composite_score"], reverse=True)
            feed_items = deduped[: self._max_items]
            top_alerts = [
                item for item in feed_items
                if item["alert_type"] != "INFO"
            ][: self._top_alerts]

            macro_snapshot = self._build_macro_snapshot(generated_at)
            compatibility_items = [item for item in fallback_items if item.get("compatibility_source")]
            news_feed = {
                "generated_at": generated_at,
                "items": feed_items,
                "top_alerts": top_alerts,
                "source_status": {
                    "finnhub": finnhub_status,
                    "yfinance": yfinance_status,
                    "openinsider": insider_doc.get("source_status", {}),
                    "live_sources": {
                        "available": [
                            name for name, status in (
                                ("finnhub", finnhub_status),
                                ("yfinance", yfinance_status),
                                ("openinsider", insider_doc.get("source_status", {})),
                            )
                            if status.get("connected")
                        ],
                        "failed": [
                            name for name, status in (
                                ("finnhub", finnhub_status),
                                ("yfinance", yfinance_status),
                                ("openinsider", insider_doc.get("source_status", {})),
                            )
                            if not status.get("connected")
                        ],
                    },
                    "compatibility_sources": {
                        "world_monitor": {
                            "enabled": bool(compatibility_items),
                            "items": len(compatibility_items),
                            "mode": "stub_or_compatibility",
                        }
                    },
                    "fallback_world_monitor": {
                        "connected": False,
                        "items": len(compatibility_items),
                        "mode": "compatibility",
                    },
                },
            }

            self.write_json(DATA_DIR / "news_feed.json", news_feed)
            self.write_json(DATA_DIR / "macro_snapshot.json", macro_snapshot)
            self.write_json(DATA_DIR / "insider_activity.json", insider_doc)
            self.update_shared_state("data_freshness.news_feed", generated_at)
            self.update_shared_state("data_freshness.macro_snapshot", generated_at)
            self.update_shared_state("data_freshness.insider_activity", generated_at)

            self.logger.info(
                f"News feed: {len(feed_items)} items | alerts={len(top_alerts)} | "
                f"macro_flags={len(macro_snapshot.get('risk_flags', []))} | "
                f"clusters={len(insider_doc.get('clusters', []))}"
            )
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    def _collect_finnhub_items(self) -> tuple[list[dict], dict]:
        if not self._finnhub or not self._finnhub.is_connected():
            return [], {"connected": False, "items": 0, "last_error": getattr(self._finnhub, "last_error", None)}
        events = self._finnhub.get_events()
        items = [self._normalize_finnhub_event(event) for event in events]
        return items, {"connected": True, "items": len(items), "last_error": self._finnhub.last_error}

    def _collect_yfinance_items(self) -> tuple[list[dict], dict]:
        if not self._yfinance_news.is_connected():
            return [], {"connected": False, "items": 0, "last_error": self._yfinance_news.last_error}
        items = []
        for row in self._yfinance_news.get_news():
            items.append(
                self._base_item(
                    headline=row.get("headline", ""),
                    summary=row.get("summary", ""),
                    published_at=row.get("published_at"),
                    symbol=self._normalize_symbol(row.get("symbol")),
                    category=row.get("category", "equity"),
                    source=row.get("source", "yfinance"),
                    url=row.get("url", ""),
                    tags=["yfinance"],
                )
            )
        return items, {"connected": True, "items": len(items), "last_error": self._yfinance_news.last_error}

    def _build_insider_activity(self, generated_at: str) -> dict:
        activity = self._openinsider.get_activity()
        status = self._openinsider.get_status()
        return {
            "generated_at": generated_at,
            "clusters": activity.get("clusters", []),
            "recent_filings": activity.get("recent_filings", []),
            "source_status": {
                "connected": status.get("connected", False),
                "reachable": status.get("reachable", False),
                "state": status.get("state", "unknown"),
                "clusters": len(activity.get("clusters", [])),
                "filings": len(activity.get("recent_filings", [])),
                "last_error": status.get("last_error"),
                "last_success_at": status.get("last_success_at"),
            },
        }

    def _build_macro_snapshot(self, generated_at: str) -> dict:
        fred_cfg = self.config.get("fred", {})
        series_cfg = fred_cfg.get("series", {})
        series: dict[str, dict] = {}
        healthy_series = 0
        failed_series: list[str] = []
        for metric, spec in series_cfg.items():
            provider = spec.get("provider", "fred")
            if provider == "fred":
                rows = self._fred.get_series(spec.get("series_id", ""), limit=14)
                snapshot = self._series_snapshot_from_rows(metric, spec, rows, self._fred.state)
            else:
                snapshot = self._market_snapshot_from_yfinance(metric, spec)
            series[metric] = snapshot
            if snapshot.get("status") == "ok":
                healthy_series += 1
            else:
                failed_series.append(metric)

        risk_flags, market_bias = self._macro_risk_flags(series)
        return {
            "generated_at": generated_at,
            "series": series,
            "risk_flags": risk_flags,
            "market_bias": round(market_bias, 4),
            "source_status": {
                "fred": {
                    "connected": healthy_series > 0,
                    "state": self._fred.state,
                    "healthy_series": healthy_series,
                    "failed_series": failed_series,
                    "last_error": self._fred.last_error,
                },
                "computed_at": generated_at,
            },
        }

    def _series_snapshot_from_rows(self, metric: str, spec: dict, rows: list[dict], fred_state: str) -> dict:
        if not rows:
            return {
                "label": spec.get("label", metric),
                "provider": "fred",
                "series_id": spec.get("series_id"),
                "value": None,
                "raw_value": None,
                "previous_value": None,
                "change_pct": None,
                "unit": spec.get("unit", ""),
                "fetched_at": None,
                "status": "error" if fred_state == "down" else "missing",
            }
        latest = rows[-1] if rows else {"date": None, "value": None}
        previous = rows[-2] if len(rows) > 1 else latest
        value = latest.get("value")
        previous_value = previous.get("value")
        change_pct = 0.0
        if value is not None and previous_value not in (None, 0):
            change_pct = ((value - previous_value) / abs(previous_value)) * 100.0
        display_value = value
        if spec.get("transform") == "yoy_percent" and len(rows) >= 13:
            old = rows[-13]["value"]
            if old:
                display_value = ((value - old) / old) * 100.0
                previous_display = ((previous_value - rows[-14]["value"]) / rows[-14]["value"]) * 100.0 if len(rows) >= 14 and rows[-14]["value"] else display_value
                change_pct = display_value - previous_display

        return {
            "label": spec.get("label", metric),
            "provider": "fred",
            "series_id": spec.get("series_id"),
            "value": round(display_value, 4) if display_value is not None else None,
            "raw_value": round(value, 4) if value is not None else None,
            "previous_value": round(previous_value, 4) if previous_value is not None else None,
            "change_pct": round(change_pct, 4) if change_pct is not None else None,
            "unit": spec.get("unit", ""),
            "fetched_at": latest.get("date"),
            "status": "ok" if value is not None else "missing",
        }

    def _market_snapshot_from_yfinance(self, metric: str, spec: dict) -> dict:
        try:
            import yfinance as yf

            hist = yf.Ticker(spec.get("symbol", "")).history(period="10d", interval="1d", auto_adjust=True)
            if hist.empty:
                raise ValueError("empty history")
            closes = hist["Close"].tolist()
            latest = float(closes[-1])
            previous = float(closes[-2]) if len(closes) > 1 else latest
            change_pct = ((latest - previous) / abs(previous)) * 100.0 if previous else 0.0
            fetched_at = hist.index[-1].to_pydatetime().replace(tzinfo=timezone.utc).isoformat()
            return {
                "label": spec.get("label", metric),
                "provider": "yfinance",
                "symbol": spec.get("symbol"),
                "value": round(latest, 4),
                "previous_value": round(previous, 4),
                "change_pct": round(change_pct, 4),
                "unit": spec.get("unit", "index"),
                "fetched_at": fetched_at,
                "status": "ok",
            }
        except Exception as exc:
            return {
                "label": spec.get("label", metric),
                "provider": "yfinance",
                "symbol": spec.get("symbol"),
                "value": None,
                "previous_value": None,
                "change_pct": 0.0,
                "unit": spec.get("unit", "index"),
                "fetched_at": None,
                "status": f"error: {exc}",
            }

    def _macro_risk_flags(self, series: dict[str, dict]) -> tuple[list[dict], float]:
        flags: list[dict] = []
        bias = 0.0

        def value(key: str):
            row = series.get(key) or {}
            return row.get("value") if row.get("status") == "ok" else None

        fed = value("fed_funds")
        cpi = value("cpi_yoy")
        dxy = value("dxy")
        vix = value("vix")
        spx = value("sp500")
        nikkei = value("nikkei")
        stoxx = value("stoxx600")

        if fed is not None and fed >= 4.5:
            flags.append(self._risk_flag("FED_HAWKISH", "Fed restrittiva", "high", -0.18))
            bias -= 0.18
        elif fed is not None and fed <= 2.5:
            flags.append(self._risk_flag("FED_DOVISH", "Fed accomodante", "medium", 0.10))
            bias += 0.10

        if cpi is not None and cpi >= 3.5:
            flags.append(self._risk_flag("CPI_HOT", "Inflazione alta", "high", -0.14))
            bias -= 0.14
        elif cpi is not None and cpi <= 2.5:
            flags.append(self._risk_flag("CPI_COOLING", "Inflazione in raffreddamento", "medium", 0.08))
            bias += 0.08

        if dxy is not None and dxy >= 105:
            flags.append(self._risk_flag("DXY_STRONG", "Dollar index forte", "medium", -0.10))
            bias -= 0.10

        if vix is not None and vix >= 25:
            flags.append(self._risk_flag("VIX_SPIKE", "Volatilità elevata", "high", -0.16))
            bias -= 0.16

        for key, label, score in (
            ("sp500", "S&P 500", 0.06),
            ("stoxx600", "STOXX 600", 0.04),
            ("nikkei", "Nikkei", 0.04),
        ):
            change = (series.get(key) or {}).get("change_pct")
            if (series.get(key) or {}).get("status") != "ok" or change is None:
                continue
            if change >= 1.0:
                flags.append(self._risk_flag(f"{key.upper()}_UP", f"{label} in rialzo", "info", score))
                bias += score
            elif change <= -1.0:
                flags.append(self._risk_flag(f"{key.upper()}_DOWN", f"{label} in calo", "medium", -score))
                bias -= score

        return flags, max(-1.0, min(1.0, bias))

    @staticmethod
    def _risk_flag(code: str, label: str, severity: str, bias: float) -> dict:
        return {"code": code, "label": label, "severity": severity, "bias": round(bias, 4)}

    def _normalize_finnhub_event(self, event: dict) -> dict:
        headline = event.get("title", "")
        summary = event.get("summary", "")
        symbols = event.get("symbols_affected") or []
        category = event.get("category", "macro")
        source = event.get("source", "finnhub")
        sentiment = event.get("sentiment", "neutral")
        sentiment_score = 0.0
        if sentiment == "bullish":
            sentiment_score = 0.35
        elif sentiment == "bearish":
            sentiment_score = -0.35

        item = self._base_item(
            headline=headline,
            summary=summary,
            published_at=event.get("timestamp"),
            symbol=symbols[0] if len(symbols) == 1 else None,
            category=category,
            source=source,
            url="",
            tags=[category, event.get("event_type", "finnhub")],
        )
        item["symbols"] = symbols
        item["sentiment_score"] = round(sentiment_score, 4)
        item["market_impact_score"] = round(float(event.get("confidence", 0.0)), 4)
        if category == "macro" and (
            abs(sentiment_score) >= 0.25 or item["market_impact_score"] >= 0.75
        ):
            item["alert_type"] = "MACRO_SURPRISE"
        return item

    def _cluster_items(self, clusters: list[dict]) -> list[dict]:
        items = []
        for cluster in clusters:
            headline = (
                f"Cluster buy insider su {cluster['symbol']} — "
                f"{cluster['filing_count']} acquisti, ${cluster['total_value_usd']:,.0f}"
            )
            item = self._base_item(
                headline=headline,
                summary=f"Insider coinvolti: {', '.join(cluster.get('insiders', [])[:3]) or 'n/d'}",
                published_at=cluster.get("latest_filed_at"),
                symbol=cluster.get("symbol"),
                category="insider",
                source="openinsider",
                url="https://openinsider.com/",
                tags=["insider", "cluster_buy"],
            )
            item["is_insider_cluster"] = True
            item["alert_type"] = "INSIDER_CLUSTER_BUY"
            item["sentiment_score"] = 0.55
            item["relevance_score"] = 0.90
            item["composite_score"] = 0.88
            items.append(item)
        return items

    def _fallback_world_events(self, world_events: list[dict], generated_at: str) -> list[dict]:
        items = []
        for event in world_events[:10]:
            source = str(event.get("source", "world_monitor"))
            if "stub" in source.lower() and not self._compatibility_allow_stub:
                continue
            item = self._base_item(
                headline=event.get("title", event.get("type", "World event")),
                summary=event.get("summary", ""),
                published_at=event.get("timestamp", generated_at),
                symbol=(event.get("symbols_affected") or [None])[0],
                category=event.get("category", "macro"),
                source=source,
                url="",
                tags=["world_monitor"],
            )
            item["compatibility_source"] = True
            items.append(item)
        return items

    def _base_item(
        self,
        headline: str,
        summary: str,
        published_at: str | None,
        symbol: str | None,
        category: str,
        source: str,
        url: str,
        tags: list[str] | None = None,
    ) -> dict:
        clean_headline = " ".join(str(headline).split())[:220]
        return {
            "id": hashlib.sha1(f"{source}|{clean_headline}|{published_at}|{symbol}".encode("utf-8")).hexdigest(),
            "source": source,
            "published_at": published_at or datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "category": category,
            "headline": clean_headline,
            "summary": " ".join(str(summary).split())[:800],
            "url": url,
            "sentiment_score": 0.0,
            "market_impact_score": 0.0,
            "relevance_score": 0.0,
            "novelty_score": 1.0,
            "composite_score": 0.0,
            "tags": tags or [],
            "alert_type": "INFO",
            "is_insider_cluster": False,
        }

    def _apply_scores(self, item: dict) -> None:
        if item.get("composite_score", 0.0) > 0.0 and item.get("relevance_score", 0.0) > 0.0:
            return
        headline = f"{item.get('headline', '')} {item.get('summary', '')}"
        sentiment_score = item.get("sentiment_score", 0.0) or self._score_sentiment(headline)
        market_impact_score = item.get("market_impact_score", 0.0) or self._score_market_impact(
            headline, item.get("category", "")
        )
        relevance_score = self._score_relevance(
            headline,
            item.get("symbol"),
            item.get("category", ""),
            item.get("is_insider_cluster", False),
            item.get("symbols", []),
        )
        novelty_score = self._score_novelty(item)
        composite = (
            (abs(sentiment_score) * 0.24)
            + (market_impact_score * 0.36)
            + (relevance_score * 0.25)
            + (novelty_score * 0.15)
        )
        if item.get("is_insider_cluster"):
            composite = max(composite, 0.88)
        alert_type = item.get("alert_type", "INFO")
        if alert_type == "INFO":
            if item.get("is_insider_cluster"):
                alert_type = "INSIDER_CLUSTER_BUY"
            elif item.get("category") == "macro" and (
                abs(sentiment_score) >= 0.28 or market_impact_score >= 0.70
            ):
                alert_type = "MACRO_SURPRISE"
            elif (
                composite >= self._high_impact_threshold
                and relevance_score >= 0.55
                and market_impact_score >= 0.55
            ):
                alert_type = "HIGH_IMPACT_NEWS"

        item["sentiment_score"] = round(sentiment_score, 4)
        item["market_impact_score"] = round(market_impact_score, 4)
        item["relevance_score"] = round(relevance_score, 4)
        item["novelty_score"] = round(novelty_score, 4)
        item["composite_score"] = round(min(composite, 0.99), 4)
        item["alert_type"] = alert_type

    @staticmethod
    def _score_sentiment(text: str) -> float:
        lowered = text.lower()
        positive = [
            "beat", "upgrade", "approval", "adoption", "buyback", "accumulation",
            "cluster buy", "dovish", "rate cut", "cooling inflation", "surge",
            "bull", "bulls", "target", "inflows", "staking", "launch", "launches",
        ]
        negative = [
            "miss", "downgrade", "lawsuit", "default", "selloff", "hot inflation",
            "rate hike", "hawkish", "conflict", "attack", "bankruptcy",
            "war", "shipping disruption", "hormuz", "oil shock",
        ]
        pos = sum(1 for keyword in positive if keyword in lowered)
        neg = sum(1 for keyword in negative if keyword in lowered)
        return max(-1.0, min(1.0, (pos - neg) * 0.22))

    @staticmethod
    def _score_market_impact(text: str, category: str) -> float:
        lowered = text.lower()
        score = 0.12
        if category in {"macro", "insider"}:
            score += 0.18
        if any(keyword in lowered for keyword in ("fed", "cpi", "inflation", "vix", "dxy", "payroll", "fomc")):
            score += 0.32
        if any(keyword in lowered for keyword in ("war", "hormuz", "oil", "sanction", "shipping")):
            score += 0.16
        if any(keyword in lowered for keyword in ("etf", "inflows", "staking", "adoption", "exchange")):
            score += 0.14
        return min(score, 1.0)

    def _score_relevance(
        self,
        text: str,
        symbol: str | None,
        category: str,
        is_insider_cluster: bool,
        symbols: list[str] | None = None,
    ) -> float:
        lowered = text.lower()
        score = 0.08
        if category in {"macro", "insider"}:
            score += 0.20
        if is_insider_cluster:
            score += 0.35
        if symbol and symbol in set(self._market_symbols):
            score += 0.25
        elif symbol and symbol in set(self._alpha_symbols):
            score += 0.18
        elif symbols:
            if any(sym in set(self._market_symbols) for sym in symbols):
                score += 0.16
            if any(sym in set(self._alpha_symbols) for sym in symbols):
                score += 0.12
        if category == "crypto" and any(keyword in lowered for keyword in ("bitcoin", "ethereum", "solana", "bnb", "crypto", "etf")):
            score += 0.18
        if category == "general" and not any(
            keyword in lowered for keyword in ("bitcoin", "ethereum", "crypto", "fed", "cpi", "inflation", "vix", "dxy")
        ):
            score -= 0.12
        if any(keyword in lowered for keyword in ("fed", "cpi", "inflation", "vix", "dxy", "payroll", "etf", "staking")):
            score += 0.18
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _score_novelty(item: dict) -> float:
        published = NewsDataAgent._parse_iso(item.get("published_at"))
        age_hours = max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600.0)
        if age_hours <= 2:
            return 1.0
        if age_hours <= 8:
            return 0.85
        if age_hours <= 24:
            return 0.65
        return 0.45

    @staticmethod
    def _dedupe_items(items: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for item in items:
            replaced = False
            for idx, existing in enumerate(kept):
                if not NewsDataAgent._is_similar_item(existing, item):
                    continue
                existing_ts = NewsDataAgent._parse_iso(existing.get("published_at"))
                current_ts = NewsDataAgent._parse_iso(item.get("published_at"))
                if current_ts > existing_ts:
                    kept[idx] = item
                replaced = True
                break
            if not replaced:
                kept.append(item)
        return kept

    @staticmethod
    def _headline_key(headline: str) -> str:
        lowered = headline.lower()
        lowered = re.sub(r"\breuters\b", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        tokens = [tok for tok in lowered.split() if tok not in {"the", "and", "amid", "says", "say", "report", "reports"}]
        return " ".join(tokens[:8])

    @staticmethod
    def _is_similar_item(existing: dict, current: dict) -> bool:
        same_source = str(existing.get("source", "")).split("/")[0] == str(current.get("source", "")).split("/")[0]
        same_symbol = (existing.get("symbol") or "") == (current.get("symbol") or "")
        if not same_source or not same_symbol:
            return False
        key_a = NewsDataAgent._headline_key(existing.get("headline", ""))
        key_b = NewsDataAgent._headline_key(current.get("headline", ""))
        if key_a == key_b:
            return True
        return SequenceMatcher(None, key_a, key_b).ratio() >= 0.86

    @staticmethod
    def _parse_iso(raw: str | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        raw = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)

    def _normalize_symbol(self, symbol: str | None) -> str | None:
        if not symbol:
            return None
        symbol = symbol.upper()
        mapping = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT",
            "SOL-USD": "SOLUSDT",
            "BNB-USD": "BNBUSDT",
        }
        return mapping.get(symbol, symbol)
