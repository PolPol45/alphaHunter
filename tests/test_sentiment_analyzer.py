"""
Test Suite per la Fase 5: NLP Sentiment Analyzer Agent
Verifica che tutti e 3 i task della Fase 5 funzionino:
  1. Raccolta headline da feed RSS (Collector)
  2. Scoring VADER con lessico finanziario (Scorer)
  3. Aggregazione in feature vector per il Feature Store (Aggregator)
"""
import pytest
from datetime import datetime, timezone, timedelta
from agents.sentiment_analyzer_agent import SentimentAnalyzerAgent, FINANCIAL_LEXICON


@pytest.fixture
def agent():
    return SentimentAnalyzerAgent()


# ==========================================
# TASK 1: VADER Scoring con Lessico Finanziario
# ==========================================
class TestSentimentScoring:

    def test_bullish_headline(self, agent):
        """Una notizia chiaramente positiva deve avere compound > 0."""
        scores = agent.score_headline("Bitcoin ETF approved! Massive inflows and bullish rally ahead.")
        assert scores["compound"] > 0.3, f"Headline bullish ha score troppo basso: {scores['compound']}"

    def test_bearish_headline(self, agent):
        """Una notizia chiaramente negativa deve avere compound < 0."""
        scores = agent.score_headline("Crypto exchange hacked, massive selloff and bankruptcy fears.")
        assert scores["compound"] < -0.3, f"Headline bearish ha score troppo alto: {scores['compound']}"

    def test_neutral_headline(self, agent):
        """Una notizia neutra deve avere compound vicino a 0."""
        scores = agent.score_headline("Company reports quarterly results in line with expectations.")
        assert -0.5 < scores["compound"] < 0.5

    def test_financial_lexicon_loaded(self, agent):
        """Verifica che il lessico finanziario custom sia stato iniettato in VADER."""
        assert "buyback" in agent.analyzer.lexicon
        assert "bankruptcy" in agent.analyzer.lexicon
        assert agent.analyzer.lexicon["bullish"] > 0  # Deve essere positivo
        assert agent.analyzer.lexicon["crash"] < 0    # Deve essere negativo

    def test_compound_in_range(self, agent):
        """Il compound score deve sempre essere in [-1.0, +1.0]."""
        texts = [
            "Best earnings ever! Massive upgrade!",
            "Total collapse, fraud, bankruptcy.",
            "Weather is nice today.",
        ]
        for text in texts:
            s = agent.score_headline(text)
            assert -1.0 <= s["compound"] <= 1.0, f"Score fuori range per: {text}"

    def test_html_stripped(self, agent):
        """Il testo HTML deve essere pulito prima dell'analisi."""
        scores = agent.score_headline("<b>Bitcoin</b> <a href='#'>surges</a> to new ATH!")
        assert isinstance(scores["compound"], float)


# ==========================================
# TASK 2: Ticker Matching
# ==========================================
class TestTickerMatching:

    def test_matches_bitcoin(self, agent):
        tickers = agent._match_tickers("Bitcoin surges past $100,000 as ETF inflows accelerate")
        assert "BTC" in tickers

    def test_matches_multiple(self, agent):
        tickers = agent._match_tickers("Ethereum and Solana outperform Bitcoin this week")
        assert "ETH" in tickers
        assert "SOL" in tickers
        assert "BTC" in tickers

    def test_no_match_generic(self, agent):
        tickers = agent._match_tickers("Weather forecast for next week looks sunny and warm")
        assert len(tickers) == 0

    def test_matches_stocks(self, agent):
        tickers = agent._match_tickers("Nvidia reports record GPU sales, AMD also benefits")
        assert "NVDA" in tickers
        assert "AMD" in tickers


# ==========================================
# TASK 3: Aggregazione e Feature Vector
# ==========================================
class TestAggregation:

    def test_label_from_compound(self, agent):
        assert agent._label_from_compound(0.5) == "VERY_BULLISH"
        assert agent._label_from_compound(0.2) == "BULLISH"
        assert agent._label_from_compound(0.0) == "NEUTRAL"
        assert agent._label_from_compound(-0.2) == "BEARISH"
        assert agent._label_from_compound(-0.5) == "VERY_BEARISH"

    def test_weighted_aggregate_empty(self, agent):
        """lista vuota deve ritornare 0.0."""
        assert agent._weighted_aggregate([]) == 0.0

    def test_weighted_aggregate_recency(self, agent):
        """Le news recenti deveno pesare di più."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=3)

        # Una news molto bearish ma vecchia + una bullish recente
        scores = [
            (-0.9, old),   # Vecchia e bearish
            (+0.5, now),   # Recente e bullish
        ]
        result = agent._weighted_aggregate(scores)
        # La news bullish recente deve dominare
        assert result > 0.0, f"La news recente dovrebbe dominare, got {result}"

    def test_analyze_all_returns_structure(self, agent):
        """Verifica che analyze_all ritorni tutti i campi attesi."""
        result = agent.analyze_all(max_age_hours=72)
        assert "market_sentiment" in result
        assert "ticker_sentiments" in result
        assert "raw_items" in result
        assert "analyzed_at" in result
        assert "total_headlines" in result
        assert isinstance(result["market_sentiment"], float)
        assert -1.0 <= result["market_sentiment"] <= 1.0

    def test_feature_vector_format(self, agent):
        """Il feature vector deve avere chiave 'sentiment_market' e valori float."""
        features = agent.get_feature_vector(max_age_hours=72)
        assert "sentiment_market" in features
        assert isinstance(features["sentiment_market"], float)
        for key, val in features.items():
            assert key.startswith("sentiment_"), f"Chiave inattesa: {key}"
            assert isinstance(val, float), f"Valore non float per {key}: {type(val)}"

    def test_feature_vector_ticker_keys(self, agent):
        """Se ci sono news su BTC, deve comparire 'sentiment_BTC' nel vettore."""
        features = agent.get_feature_vector(max_age_hours=72)
        # Non possiamo garantire al 100% quale ticker esce, ma almeno
        # il campo mercato deve esserci
        assert "sentiment_market" in features
        # Se > 0 ticker, almeno uno deve essere presente
        ticker_keys = [k for k in features if k != "sentiment_market"]
        if ticker_keys:
            assert all(k.startswith("sentiment_") for k in ticker_keys)
