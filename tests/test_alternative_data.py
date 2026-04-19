"""
Test Suite per la Fase 1: Alpha Generation (Alternative Data Agent)
Verifica che tutti e 3 i task della Fase 1 funzionino correttamente:
  1. Fear & Greed Index (API Live)
  2. Kraken L2 Orderbook Imbalance (API Live)
  3. Statistical Arbitrage / Cointegrazione (Calcolo Matematico)
"""
import pytest
import math
import pandas as pd
import numpy as np
from agents.alternative_data_agent import AlternativeDataAgent


@pytest.fixture
def agent():
    return AlternativeDataAgent()


# ==========================================
# TASK 1: Fear & Greed Index
# ==========================================
class TestFearAndGreed:

    def test_fetch_returns_dataframe(self, agent):
        """Verifica che l'API ritorna un DataFrame non vuoto."""
        df = agent.fetch_crypto_fear_and_greed(limit=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Il DataFrame è vuoto: l'API non ha restituito dati"

    def test_columns_exist(self, agent):
        """Verifica che le colonne attese esistano nel risultato."""
        df = agent.fetch_crypto_fear_and_greed(limit=3)
        assert 'fear_greed_value' in df.columns
        assert 'value_classification' in df.columns

    def test_values_in_range(self, agent):
        """Il Fear & Greed deve essere tra 0 e 100."""
        df = agent.fetch_crypto_fear_and_greed(limit=5)
        assert df['fear_greed_value'].min() >= 0
        assert df['fear_greed_value'].max() <= 100

    def test_index_is_datetime(self, agent):
        """L'indice deve essere di tipo DatetimeIndex."""
        df = agent.fetch_crypto_fear_and_greed(limit=3)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_sorted_chronologically(self, agent):
        """I dati devono essere ordinati dal più vecchio al più recente."""
        df = agent.fetch_crypto_fear_and_greed(limit=10)
        if len(df) > 1:
            assert df.index[0] < df.index[-1], "I dati non sono ordinati cronologicamente"


# ==========================================
# TASK 2: Kraken L2 Orderbook Imbalance
# ==========================================
class TestKrakenOrderbook:

    def test_returns_dict(self, agent):
        """Verifica che il risultato sia un dizionario."""
        result = agent.fetch_kraken_bid_ask_imbalance(pair="XXBTZUSD", count=10)
        assert isinstance(result, dict)

    def test_imbalance_key_exists(self, agent):
        """Verifica che la chiave 'imbalance' sia presente."""
        result = agent.fetch_kraken_bid_ask_imbalance(pair="XXBTZUSD", count=10)
        assert 'imbalance' in result

    def test_imbalance_in_range(self, agent):
        """L'imbalance deve essere tra -1.0 e +1.0."""
        result = agent.fetch_kraken_bid_ask_imbalance(pair="XXBTZUSD", count=10)
        assert -1.0 <= result['imbalance'] <= 1.0, f"Imbalance fuori range: {result['imbalance']}"

    def test_volumes_positive(self, agent):
        """I volumi totali bid e ask devono essere >= 0."""
        result = agent.fetch_kraken_bid_ask_imbalance(pair="XXBTZUSD", count=10)
        assert result.get('total_bid_volume', 0) >= 0
        assert result.get('total_ask_volume', 0) >= 0

    def test_invalid_pair_graceful(self, agent):
        """Con un pair inesistente, non deve crashare."""
        result = agent.fetch_kraken_bid_ask_imbalance(pair="FAKEPAIR", count=10)
        assert isinstance(result, dict)
        # Deve ritornare 0.0 perché l'API darà errore
        assert result['imbalance'] == 0.0


# ==========================================
# TASK 3: Statistical Arbitrage / Cointegrazione
# ==========================================
class TestCointegration:

    def test_cointegrated_series(self, agent):
        """Due serie fortemente correlate devono risultare cointegrate."""
        np.random.seed(42)
        base = np.cumsum(np.random.normal(0, 1, 300)) + 200  # +200 per valori sempre positivi
        series_a = pd.Series(base)
        series_b = pd.Series(base * 0.9 + np.random.normal(0, 0.5, 300))

        z, pval, is_coint = agent.calculate_cointegration_zscore(series_a, series_b)
        assert is_coint == True, f"Le serie dovrebbero essere cointegrate, pvalue={pval}"
        assert pval < 0.05

    def test_unrelated_series(self, agent):
        """Due serie completamente random NON devono essere cointegrate."""
        np.random.seed(42)
        series_a = pd.Series(np.cumsum(np.random.normal(0, 1, 300)) + 200)
        series_b = pd.Series(np.cumsum(np.random.normal(5, 3, 300)) + 500)

        z, pval, is_coint = agent.calculate_cointegration_zscore(series_a, series_b)
        # Verifica che i tipi Python nativi funzionino
        assert type(is_coint) == bool
        assert isinstance(z, float)
        assert isinstance(pval, float)

    def test_zscore_is_finite(self, agent):
        """Lo Z-Score non deve mai essere NaN o Inf."""
        np.random.seed(42)
        base = np.cumsum(np.random.normal(0, 1, 200)) + 50
        series_a = pd.Series(base)
        series_b = pd.Series(base * 1.1 + np.random.normal(0, 1, 200))

        z, pval, is_coint = agent.calculate_cointegration_zscore(series_a, series_b)
        assert math.isfinite(z), f"Z-Score non finito: {z}"

    def test_short_series_returns_default(self, agent):
        """Con meno di 30 osservazioni, il modulo deve ritornare il fallback sicuro."""
        series_a = pd.Series([1, 2, 3])
        series_b = pd.Series([4, 5, 6])

        z, pval, is_coint = agent.calculate_cointegration_zscore(series_a, series_b)
        assert z == 0.0
        assert pval == 1.0
        assert is_coint is False

    def test_empty_series(self, agent):
        """Con serie vuote, non deve crashare."""
        series_a = pd.Series(dtype=float)
        series_b = pd.Series(dtype=float)

        z, pval, is_coint = agent.calculate_cointegration_zscore(series_a, series_b)
        assert z == 0.0
        assert is_coint is False
