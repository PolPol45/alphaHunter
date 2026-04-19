"""
Sentiment Analyzer Agent — Fase 5: Intelligenza NLP
====================================================
Legge news finanziarie da feed RSS gratuiti e le analizza con VADER
(lexicon-based NLP ottimizzato per testi finanziari).

Produce uno score normalizzato [-1.0, +1.0] per ticker o per mercato,
pronto per essere iniettato nel Feature Store come feature predittiva.

Architettura:
  1. Collector: raccoglie titoli da RSS feed gratuiti (Yahoo Finance, MarketWatch, CoinDesk)
  2. Scorer: VADER Sentiment Analysis (offline, zero API key)
  3. Aggregator: media pesata per recency → score finale per ticker/mercato
  4. (Futuro) LLM Upgrade: sostituire VADER con Groq/Claude Haiku per accuracy maggiore
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Feed RSS gratuiti, nessun API key richiesto
DEFAULT_RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "investing_com": "https://www.investing.com/rss/news.rss",
}

# Mapping di ticker / keyword per assegnare news a specifici asset
TICKER_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "satoshi", "halving"],
    "ETH": ["ethereum", "eth", "vitalik", "erc-20"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance", "bnb"],
    "XRP": ["ripple", "xrp"],
    "NVDA": ["nvidia", "nvda", "gpu", "cuda"],
    "AAPL": ["apple", "aapl", "iphone", "tim cook"],
    "MSFT": ["microsoft", "msft", "azure", "satya"],
    "TSLA": ["tesla", "tsla", "elon musk", "ev"],
    "META": ["meta", "facebook", "zuckerberg", "instagram"],
    "AMZN": ["amazon", "amzn", "aws", "bezos"],
    "GOOGL": ["google", "googl", "alphabet", "deepmind"],
    "AMD": ["amd", "advanced micro", "lisa su"],
    "COIN": ["coinbase"],
    "MSTR": ["microstrategy", "saylor"],
}

# Parole finanziarie con polarità forte che VADER da solo non cattura bene
FINANCIAL_LEXICON = {
    # Bullish
    "beat": 2.0, "beats": 2.0, "upgrade": 2.5, "upgraded": 2.5,
    "buyback": 2.0, "accumulation": 1.8, "dovish": 2.0, "rate cut": 2.5,
    "rate cuts": 2.5, "approval": 1.5, "approved": 1.5, "adoption": 1.5,
    "inflows": 1.8, "bullish": 2.5, "rally": 2.0, "surge": 2.0,
    "breakout": 1.8, "etf approved": 3.0, "partnership": 1.5,
    "all-time high": 2.5, "ath": 2.0, "moon": 1.0, "pump": 1.0,
    # Bearish
    "miss": -2.0, "misses": -2.0, "downgrade": -2.5, "downgraded": -2.5,
    "lawsuit": -2.0, "sued": -2.0, "default": -3.0, "defaulted": -3.0,
    "selloff": -2.5, "sell-off": -2.5, "hawkish": -2.0, "rate hike": -2.5,
    "rate hikes": -2.5, "bankruptcy": -3.5, "bankrupt": -3.5,
    "hack": -3.0, "hacked": -3.0, "exploit": -2.5, "rug pull": -3.5,
    "crash": -2.5, "plunge": -2.5, "dump": -2.0, "bearish": -2.5,
    "recession": -2.0, "layoffs": -1.8, "layoff": -1.8,
    "war": -2.0, "conflict": -1.8, "sanctions": -1.5, "tariff": -1.5,
    "tariffs": -1.5, "investigation": -1.5, "fraud": -3.0,
}


class SentimentAnalyzerAgent:
    """
    Agente NLP che analizza il sentiment di mercato da feed RSS gratuiti.
    Produce uno score [-1.0, +1.0] per ticker e per mercato aggregato.
    """

    def __init__(self, custom_feeds: Optional[Dict[str, str]] = None):
        self.logger = logging.getLogger("sentiment_analyzer_agent")
        self.feeds = custom_feeds or DEFAULT_RSS_FEEDS
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Inietta il lessico finanziario custom dentro VADER
        self.analyzer.lexicon.update(FINANCIAL_LEXICON)

    def collect_headlines(self, max_age_hours: int = 48) -> List[Dict]:
        """
        Raccoglie headline da tutti i feed RSS configurati.
        Ritorna una lista di dizionari con: title, summary, source, published, url.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        all_items = []

        for source_name, feed_url in self.feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:  # Max 30 per feed
                    published = self._parse_feed_date(entry)
                    if published and published < cutoff:
                        continue

                    item = {
                        "title": getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", "")[:500],
                        "source": source_name,
                        "published": published or datetime.now(timezone.utc),
                        "url": getattr(entry, "link", ""),
                    }
                    all_items.append(item)

            except Exception as e:
                self.logger.warning(f"Errore nel fetch di {source_name}: {e}")
                continue

        self.logger.info(f"Raccolte {len(all_items)} headline da {len(self.feeds)} feed.")
        return all_items

    def score_headline(self, text: str) -> Dict[str, float]:
        """
        Analizza il sentiment di un singolo testo con VADER potenziato.
        Ritorna: {neg, neu, pos, compound} dove compound è in [-1.0, +1.0].
        """
        clean = re.sub(r'<[^>]+>', '', text)  # Rimuovi HTML residuo
        clean = re.sub(r'http\S+', '', clean)  # Rimuovi URL
        scores = self.analyzer.polarity_scores(clean)
        return scores

    def analyze_all(self, max_age_hours: int = 48) -> Dict:
        """
        Pipeline completa: raccoglie, analizza e aggrega il sentiment.
        Ritorna un documento JSON-ready con:
          - market_sentiment: score aggregato del mercato [-1, +1]
          - ticker_sentiments: {TICKER: {score, count, headlines}}
          - raw_items: lista di tutti gli item analizzati
        """
        headlines = self.collect_headlines(max_age_hours=max_age_hours)

        if not headlines:
            return {
                "market_sentiment": 0.0,
                "ticker_sentiments": {},
                "raw_items": [],
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "total_headlines": 0,
            }

        analyzed = []
        ticker_scores: Dict[str, List[Tuple[float, datetime]]] = {}

        for item in headlines:
            text = f"{item['title']}. {item['summary']}"
            scores = self.score_headline(text)
            compound = scores["compound"]

            # Identifica i ticker menzionati
            matched_tickers = self._match_tickers(text)

            enriched = {
                **item,
                "published": item["published"].isoformat() if isinstance(item["published"], datetime) else item["published"],
                "sentiment_compound": round(compound, 4),
                "sentiment_label": self._label_from_compound(compound),
                "matched_tickers": matched_tickers,
            }
            analyzed.append(enriched)

            # Accumula per ticker
            pub_dt = item["published"] if isinstance(item["published"], datetime) else datetime.now(timezone.utc)
            for ticker in matched_tickers:
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = []
                ticker_scores[ticker].append((compound, pub_dt))

        # Calcola score aggregato per mercato (media pesata per recency)
        market_sentiment = self._weighted_aggregate(
            [(item["sentiment_compound"], item["published"]) for item in analyzed
             if isinstance(item.get("published"), str)]
        )

        # Calcola score per ticker
        ticker_sentiments = {}
        for ticker, scores_list in ticker_scores.items():
            agg_score = self._weighted_aggregate(scores_list)
            top_headlines = [
                item for item in analyzed
                if ticker in item.get("matched_tickers", [])
            ]
            top_headlines.sort(key=lambda x: abs(x["sentiment_compound"]), reverse=True)

            ticker_sentiments[ticker] = {
                "score": round(agg_score, 4),
                "label": self._label_from_compound(agg_score),
                "headline_count": len(scores_list),
                "top_headlines": [
                    {"title": h["title"], "score": h["sentiment_compound"], "source": h["source"]}
                    for h in top_headlines[:3]
                ],
            }

        return {
            "market_sentiment": round(market_sentiment, 4),
            "market_label": self._label_from_compound(market_sentiment),
            "ticker_sentiments": ticker_sentiments,
            "raw_items": analyzed,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "total_headlines": len(analyzed),
        }

    def get_feature_vector(self, max_age_hours: int = 48) -> Dict[str, float]:
        """
        Ritorna un dizionario chiave→valore pronto per il Feature Store.
        Esempio: {"sentiment_market": 0.23, "sentiment_BTC": -0.45, "sentiment_ETH": 0.12}
        Questo è il formato esatto iniettabile nello step ML.
        """
        result = self.analyze_all(max_age_hours=max_age_hours)
        features = {"sentiment_market": result["market_sentiment"]}

        for ticker, data in result.get("ticker_sentiments", {}).items():
            features[f"sentiment_{ticker}"] = data["score"]

        return features

    def _match_tickers(self, text: str) -> List[str]:
        """Identifica i ticker menzionati nel testo."""
        lowered = text.lower()
        matched = []
        for ticker, keywords in TICKER_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                matched.append(ticker)
        return matched

    def _weighted_aggregate(self, scores_with_dates: List) -> float:
        """
        Media pesata per recency: le news più recenti pesano di più.
        Peso = 1.0 per news di oggi, 0.5 per news di ieri, 0.25 per 2gg fa.
        """
        if not scores_with_dates:
            return 0.0

        now = datetime.now(timezone.utc)
        weighted_sum = 0.0
        total_weight = 0.0

        for item in scores_with_dates:
            if isinstance(item, tuple) and len(item) == 2:
                score, pub = item
            else:
                continue

            if isinstance(pub, str):
                try:
                    pub = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                except ValueError:
                    pub = now

            age_hours = max(0.0, (now - pub).total_seconds() / 3600.0)
            # Decadimento esponenziale: peso dimezza ogni 24 ore
            weight = 2.0 ** (-age_hours / 24.0)

            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _label_from_compound(compound: float) -> str:
        if compound >= 0.35:
            return "VERY_BULLISH"
        elif compound >= 0.15:
            return "BULLISH"
        elif compound > -0.15:
            return "NEUTRAL"
        elif compound > -0.35:
            return "BEARISH"
        else:
            return "VERY_BEARISH"

    @staticmethod
    def _parse_feed_date(entry) -> Optional[datetime]:
        """Prova a estrarre la data di pubblicazione da un feed RSS entry."""
        for attr in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    from time import mktime
                    return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
                except (ValueError, OverflowError):
                    continue
        return None

    def run(self) -> bool:
        """BaseAgent-compatible entry point. Runs analyze_all() and writes sentiment_data.json."""
        try:
            from pathlib import Path
            import json as _json
            # In backtest mode: skip live RSS fetch (would cause lookahead bias)
            # Return neutral sentiment so downstream agents don't crash
            ctx_path = Path(__file__).parent.parent / "data" / "backtest_context.json"
            try:
                ctx = _json.loads(ctx_path.read_text(encoding="utf-8"))
                if ctx.get("enabled"):
                    self.logger.info("Backtest mode: SentimentAnalyzer skipped (no lookahead)")
                    return True
            except Exception:
                pass
            result = self.analyze_all(max_age_hours=48)
            out_path = Path(__file__).parent.parent / "data" / "sentiment_data.json"
            out_path.write_text(_json.dumps(result, indent=2, default=str), encoding="utf-8")
            self.logger.info(
                f"Sentiment written | market={result['market_sentiment']:+.3f} "
                f"({result['market_label']}) | tickers={len(result['ticker_sentiments'])}"
            )
            return True
        except Exception as e:
            self.logger.error(f"SentimentAnalyzerAgent.run() failed: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    agent = SentimentAnalyzerAgent()

    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYZER — Test Live")
    print("=" * 60)

    result = agent.analyze_all(max_age_hours=72)

    print(f"\n📊 Market Sentiment: {result['market_sentiment']:+.3f} ({result['market_label']})")
    print(f"📰 Headlines analizzate: {result['total_headlines']}")

    if result["ticker_sentiments"]:
        print(f"\n🎯 Sentiment per Ticker:")
        for ticker, data in sorted(result["ticker_sentiments"].items(), key=lambda x: x[1]["score"], reverse=True):
            emoji = "🟢" if data["score"] > 0.15 else "🔴" if data["score"] < -0.15 else "⚪"
            print(f"  {emoji} {ticker:>6}: {data['score']:+.3f} ({data['label']}) — {data['headline_count']} news")
            for h in data["top_headlines"][:2]:
                print(f"         └─ [{h['score']:+.2f}] {h['title'][:80]}")

    print(f"\n🔗 Feature Vector (pronto per ML):")
    features = agent.get_feature_vector(max_age_hours=72)
    for k, v in features.items():
        print(f"  {k}: {v:+.4f}")
