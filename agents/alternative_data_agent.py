import json
import logging
import math
import pathlib
import requests
import pandas as pd
from typing import Tuple, Dict, List, Optional

# Assicurati di avere installato: pip install statsmodels
try:
    from statsmodels.tsa.stattools import coint
    import numpy as np
except ImportError:
    pass

# xStocks pairs: Kraken tokenized equities/ETFs (asset_class=tokenized_asset)
XSTOCKS_PAIRS = {
    "AAPL": "AAPLx/USD", "MSFT": "MSFTx/USD", "NVDA": "NVDAx/USD",
    "TSLA": "TSLAx/USD", "META": "METAx/USD", "AMZN": "AMZNx/USD",
    "GOOGL": "GOOGLx/USD", "AMD": "AMDx/USD", "COIN": "COINx/USD",
    "MSTR": "MSTRx/USD", "SPY": "SPYx/USD", "QQQ": "QQQx/USD",
    "IWM": "IWMx/USD",
}

# Kraken Futures perpetual symbols
FUTURES_SYMBOLS = {
    "BTC": "PI_XBTUSD",
    "ETH": "PI_ETHUSD",
    "SOL": "PI_SOLUSD",
    "XRP": "PI_XRPUSD",
}

_KRAKEN_XSTOCKS_ORDERBOOK_URL = "https://api.kraken.com/0/public/Depth"
_KRAKEN_FUTURES_BASE = "https://futures.kraken.com/derivatives/api/v3"
_GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"  # free tier, no key for some endpoints
_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


class AlternativeDataAgent:
    """
    Agente dedito alla raccolta di Alternative Data e Alpha Generation (Fase 1 completa).
    Fornisce Feature predittive come Fear&Greed, Bid/Ask Imbalance e Statistical Arbitrage.
    """
    def __init__(self):
        self.logger = logging.getLogger("alternative_data_agent")

    # 1. Analisi On-Chain & Sentiment (Fear & Greed)
    def fetch_crypto_fear_and_greed(self, limit: int = 365) -> pd.DataFrame:
        """
        Scarica il Fear & Greed Index delle crypto da alternative.me.
        Ritorna un DataFrame utile a capire se siamo in Euphoria (top) o Despair (bottom).
        """
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['value'] = pd.to_numeric(df['value'])
            df = df.rename(columns={'timestamp': 'date', 'value': 'fear_greed_value'})
            df = df[['date', 'fear_greed_value', 'value_classification']]
            return df.sort_values('date').set_index('date')
            
        except Exception as e:
            self.logger.error(f"Errore Fear & Greed: {e}")
            return pd.DataFrame()

    # 2. Order Flow Volume Profile (Kraken L2)
    def fetch_kraken_bid_ask_imbalance(self, pair: str = "XXBTZUSD", count: int = 50) -> Dict[str, float]:
        """
        Interroga il Book degli Ordini (L2 Snapshot) di Kraken.
        Calcola l'Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume).
        Se positivo (>0.2) c'è forte pressione di Buy. Se negativo c'è pressione Sell.
        """
        url = f"https://api.kraken.com/0/public/Depth?pair={pair}&count={count}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                self.logger.error(f"Errore Kraken API: {data['error']}")
                return {"imbalance": 0.0, "total_bid": 0.0, "total_ask": 0.0}

            # L'API torna i dati dentro la chiave del paio, es. data['result']['XXBTZUSD']
            book = list(data['result'].values())[0]
            
            # Format di kraken: [price, volume, timestamp]
            # Accumuliamo solo i volumi ai vari livelli di prezzo per capire la densità L2
            bid_vol = sum(float(level[1]) for level in book.get('bids', []))
            ask_vol = sum(float(level[1]) for level in book.get('asks', []))
            
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
            
            return {
                "imbalance": round(imbalance, 3), # +1.0 (all bids) to -1.0 (all asks)
                "total_bid_volume": round(bid_vol, 2),
                "total_ask_volume": round(ask_vol, 2)
            }
        except Exception as e:
            self.logger.error(f"Errore Book L2: {e}")
            return {"imbalance": 0.0, "total_bid": 0.0, "total_ask": 0.0}

    # 3. Statistical Arbitrage / Co-Integration
    def calculate_cointegration_zscore(self, series_a: pd.Series, series_b: pd.Series) -> Tuple[float, float, bool]:
        """
        Riceve due serie storiche di prezzi (es. BTC e ETH, oppure AAPL e MSFT).
        Utilizza l'Augmented Dickey-Fuller test (tramite 'coint') per stabilire 
        se i due asset si muovono matematicamente insieme nel tempo.
        Ritorna: Z-Score dello spread (se < -2 si compra A e vende B), P-Value, e Boolean is_cointegrated.
        """
        try:
            # Drop na and align
            df = pd.concat([series_a, series_b], axis=1).dropna()
            if len(df) < 30:
                return (0.0, 1.0, False)
            
            s1, s2 = df.iloc[:, 0], df.iloc[:, 1]
            
            # Test di Cointegrazione (Statistiche ADF)
            score, pvalue, _ = coint(s1, s2)
            
            # Se pvalue < 0.05, siamo confidenti al 95% che le due serie siano cointegrate
            is_coint = bool(pvalue < 0.05)
            
            # Calcolo Z-Score dello spread per direttività del trading
            # Spread = log(A) - n * log(B), per semplicità usiamo un rapporto base
            spread = np.log(s1) - np.log(s2)
            zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
            
            return (round(zscore, 3), round(pvalue, 4), is_coint)
        except Exception as e:
            self.logger.error(f"Errore calcolo cointegrazione: {e}")
            return (0.0, 1.0, False)

    def fetch_xstocks_bid_ask_imbalance(self, symbols: List[str] | None = None) -> Dict[str, float]:
        """
        Fetch L2 orderbook imbalance for Kraken xStocks (tokenized equities).
        Returns {symbol: imbalance} where imbalance in [-1.0, +1.0].
        Positive = buy pressure, negative = sell pressure.
        """
        targets = symbols or list(XSTOCKS_PAIRS.keys())
        result: Dict[str, float] = {}
        for sym in targets:
            pair = XSTOCKS_PAIRS.get(sym)
            if not pair:
                continue
            try:
                resp = requests.get(
                    _KRAKEN_XSTOCKS_ORDERBOOK_URL,
                    params={"pair": pair, "count": 20},
                    timeout=8,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("error"):
                    continue
                book = list(data["result"].values())[0]
                bid_vol = sum(float(l[1]) for l in book.get("bids", []))
                ask_vol = sum(float(l[1]) for l in book.get("asks", []))
                total = bid_vol + ask_vol
                result[sym] = round((bid_vol - ask_vol) / total, 3) if total > 0 else 0.0
            except Exception as e:
                self.logger.debug(f"xStocks L2 {sym}: {e}")
                result[sym] = 0.0
        return result

    # ------------------------------------------------------------------ #
    # 4. Kraken Futures — Funding Rate                                    #
    # ------------------------------------------------------------------ #

    def fetch_funding_rates(self) -> Dict[str, Dict]:
        """
        Fetch current + predicted funding rates from Kraken Futures (public, no key).
        Returns {symbol: {funding_rate, predicted_funding_rate, next_funding_time, mark_price, spot_basis_pct}}
        Positive funding = longs pay shorts = market is overleveraged long → contrarian bearish signal.
        """
        result = {}
        try:
            resp = requests.get(f"{_KRAKEN_FUTURES_BASE}/tickers", timeout=10)
            resp.raise_for_status()
            tickers = resp.json().get("tickers", [])
            ticker_map = {t["symbol"]: t for t in tickers}

            for sym, perp in FUTURES_SYMBOLS.items():
                t = ticker_map.get(perp)
                if not t:
                    continue
                funding = float(t.get("fundingRate", 0) or 0)
                pred_funding = float(t.get("fundingRatePrediction", 0) or 0)
                mark = float(t.get("markPrice", 0) or 0)
                last = float(t.get("last", 0) or 0)
                basis_pct = round((mark - last) / last * 100, 4) if last > 0 else 0.0
                result[sym] = {
                    "funding_rate": round(funding, 6),
                    "predicted_funding_rate": round(pred_funding, 6),
                    "mark_price": mark,
                    "last_price": last,
                    "basis_pct": basis_pct,
                    "funding_signal": (
                        "OVERLONG" if funding > 0.0005 else
                        "OVERSHORT" if funding < -0.0005 else
                        "NEUTRAL"
                    ),
                }
        except Exception as e:
            self.logger.warning(f"fetch_funding_rates: {e}")
        return result

    # ------------------------------------------------------------------ #
    # 5. Kraken Futures — Open Interest                                   #
    # ------------------------------------------------------------------ #

    def fetch_open_interest(self) -> Dict[str, Dict]:
        """
        Fetch open interest for BTC/ETH/SOL perpetuals from Kraken Futures.
        OI rising + price flat = divergence → potential reversal.
        """
        result = {}
        try:
            resp = requests.get(f"{_KRAKEN_FUTURES_BASE}/openinterests", timeout=10)
            resp.raise_for_status()
            items = resp.json().get("openInterests", [])
            oi_map = {i["symbol"]: i for i in items}

            for sym, perp in FUTURES_SYMBOLS.items():
                item = oi_map.get(perp)
                if not item:
                    continue
                result[sym] = {
                    "open_interest_usd": float(item.get("openInterest", 0) or 0),
                }
        except Exception as e:
            self.logger.warning(f"fetch_open_interest: {e}")
        return result

    # ------------------------------------------------------------------ #
    # 6. Kraken Futures — Historical Funding (7d rolling)                 #
    # ------------------------------------------------------------------ #

    def fetch_historical_funding(self, symbol: str = "BTC", periods: int = 56) -> Dict:
        """
        Fetch historical funding rates (8h intervals) from Kraken Futures.
        Returns 7d rolling mean and current z-score vs 30d history.
        Used as ML feature: extreme funding → mean reversion edge.
        """
        perp = FUTURES_SYMBOLS.get(symbol, "PI_XBTUSD")
        try:
            resp = requests.get(
                f"{_KRAKEN_FUTURES_BASE}/historicalfundingrates",
                params={"symbol": perp},
                timeout=10,
            )
            resp.raise_for_status()
            rates_raw = resp.json().get("rates", [])
            if not rates_raw:
                return {}

            rates = [float(r.get("fundingRate", 0) or 0) for r in rates_raw[-periods:]]
            if len(rates) < 2:
                return {}

            mean_7d = round(sum(rates[-21:]) / len(rates[-21:]), 6)   # 21 × 8h = 7d
            mean_30d = round(sum(rates) / len(rates), 6)
            std_30d = round((sum((r - mean_30d) ** 2 for r in rates) / len(rates)) ** 0.5, 6)
            zscore = round((rates[-1] - mean_30d) / std_30d, 3) if std_30d > 0 else 0.0

            return {
                "symbol": symbol,
                "current_funding": rates[-1],
                "mean_7d": mean_7d,
                "mean_30d": mean_30d,
                "zscore_30d": zscore,
                "extremity": (
                    "EXTREME_LONG" if zscore > 2.0 else
                    "EXTREME_SHORT" if zscore < -2.0 else
                    "NORMAL"
                ),
            }
        except Exception as e:
            self.logger.warning(f"fetch_historical_funding {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------ #
    # 7. On-Chain — BTC Exchange Netflow (CryptoQuant public)             #
    # ------------------------------------------------------------------ #

    def fetch_btc_exchange_netflow(self) -> Dict:
        """
        Fetch BTC exchange netflow from CryptoQuant free RSS/public endpoint.
        Positive netflow = coins moving TO exchanges = sell pressure.
        Negative netflow = coins leaving exchanges = accumulation.
        Falls back to 0 if unavailable (no API key in free tier).
        """
        try:
            # Glassnode free tier: exchange net position change (BTC)
            resp = requests.get(
                f"{_GLASSNODE_BASE}/transactions/transfers_volume_to_exchanges_sum",
                params={"a": "BTC", "i": "24h", "f": "JSON"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list):
                latest = data[-1]
                inflow = float(latest.get("v", 0) or 0)
                # Fetch outflow separately
                resp2 = requests.get(
                    f"{_GLASSNODE_BASE}/transactions/transfers_volume_from_exchanges_sum",
                    params={"a": "BTC", "i": "24h", "f": "JSON"},
                    timeout=10,
                )
                resp2.raise_for_status()
                data2 = resp2.json()
                outflow = float(data2[-1].get("v", 0) or 0) if data2 else 0.0
                netflow = round(inflow - outflow, 2)
                return {
                    "netflow_btc": netflow,
                    "inflow_btc": round(inflow, 2),
                    "outflow_btc": round(outflow, 2),
                    "signal": "SELL_PRESSURE" if netflow > 1000 else "ACCUMULATION" if netflow < -1000 else "NEUTRAL",
                }
        except Exception as e:
            self.logger.debug(f"fetch_btc_exchange_netflow: {e}")
        return {"netflow_btc": 0.0, "inflow_btc": 0.0, "outflow_btc": 0.0, "signal": "NEUTRAL"}

    # ------------------------------------------------------------------ #
    # 8. Options — Put/Call Ratio (Deribit public)                        #
    # ------------------------------------------------------------------ #

    def fetch_options_put_call_ratio(self, currency: str = "BTC") -> Dict:
        """
        Fetch BTC/ETH options Put/Call ratio from Deribit (free public API).
        PCR > 1.2 = heavy put buying = fear/hedging = contrarian bullish signal.
        PCR < 0.6 = call dominance = complacency = potential top.
        """
        try:
            resp = requests.get(
                f"{_DERIBIT_BASE}/get_book_summary_by_currency",
                params={"currency": currency, "kind": "option"},
                timeout=10,
            )
            resp.raise_for_status()
            instruments = resp.json().get("result", [])

            put_oi = sum(float(i.get("open_interest", 0) or 0) for i in instruments if i.get("instrument_name", "").endswith("-P"))
            call_oi = sum(float(i.get("open_interest", 0) or 0) for i in instruments if i.get("instrument_name", "").endswith("-C"))
            pcr = round(put_oi / call_oi, 3) if call_oi > 0 else 1.0

            return {
                "currency": currency,
                "put_oi": round(put_oi, 2),
                "call_oi": round(call_oi, 2),
                "put_call_ratio": pcr,
                "signal": (
                    "CONTRARIAN_BULLISH" if pcr > 1.2 else
                    "COMPLACENCY_TOP" if pcr < 0.6 else
                    "NEUTRAL"
                ),
            }
        except Exception as e:
            self.logger.warning(f"fetch_options_put_call_ratio {currency}: {e}")
            return {"currency": currency, "put_oi": 0, "call_oi": 0, "put_call_ratio": 1.0, "signal": "NEUTRAL"}

    def run(self) -> bool:
        """
        BaseAgent-compatible entry point.
        Fetches Fear&Greed + xStocks bid/ask imbalance and writes
        data/alternative_data.json for downstream agents (RiskAgent, TA).
        Skips external calls in backtest mode.
        """
        try:
            data_dir = pathlib.Path(__file__).parent.parent / "data"
            ctx_path = data_dir / "backtest_context.json"
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                if ctx.get("enabled"):
                    self.logger.info("Backtest mode: AlternativeDataAgent skipped")
                    return True
            except Exception:
                pass

            out: Dict = {}

            # Fear & Greed
            fg_df = self.fetch_crypto_fear_and_greed(limit=1)
            if not fg_df.empty:
                row = fg_df.iloc[-1]
                out["fear_greed"] = {
                    "value": int(row["fear_greed_value"]),
                    "label": str(row.get("value_classification", "")),
                }
            else:
                out["fear_greed"] = {"value": 50, "label": "Neutral"}

            # xStocks orderbook imbalance
            out["xstocks_imbalance"] = self.fetch_xstocks_bid_ask_imbalance()

            # Crypto BTC L2 spot orderbook
            out["btc_orderbook"] = self.fetch_kraken_bid_ask_imbalance(pair="XXBTZUSD", count=50)

            # Kraken Futures — funding rates + mark price + basis
            out["funding_rates"] = self.fetch_funding_rates()

            # Kraken Futures — open interest
            out["open_interest"] = self.fetch_open_interest()

            # Kraken Futures — historical funding 7d rolling + z-score
            out["historical_funding"] = {
                sym: self.fetch_historical_funding(sym)
                for sym in ["BTC", "ETH"]
            }

            # On-chain — BTC exchange netflow
            out["btc_exchange_netflow"] = self.fetch_btc_exchange_netflow()

            # Options — Put/Call ratio BTC + ETH
            out["options_pcr"] = {
                "BTC": self.fetch_options_put_call_ratio("BTC"),
                "ETH": self.fetch_options_put_call_ratio("ETH"),
            }

            out_path = data_dir / "alternative_data.json"
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

            fr_btc = out["funding_rates"].get("BTC", {}).get("funding_signal", "N/A")
            oi_btc = out["open_interest"].get("BTC", {}).get("open_interest_usd", 0)
            pcr_btc = out["options_pcr"]["BTC"].get("put_call_ratio", 0)
            netflow = out["btc_exchange_netflow"].get("netflow_btc", 0)
            self.logger.info(
                f"AlternativeDataAgent done | F&G={out['fear_greed']['value']} "
                f"| BTC funding={fr_btc} | OI=${oi_btc:,.0f} "
                f"| PCR={pcr_btc:.2f} | netflow={netflow:+.0f} BTC"
            )
            return True
        except Exception as e:
            self.logger.error(f"AlternativeDataAgent.run() failed: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = AlternativeDataAgent()
    
    # 1. Test Fear & Greed
    print("\n--- 1. FEAR & GREED INDEX ---")
    df_fgi = agent.fetch_crypto_fear_and_greed(limit=5)
    print(df_fgi)
    
    # 2. Test Orderbook L2 su Bitcoin
    print("\n--- 2. KRAKEN L2 ORDERBOOK IMBALANCE (BTC/USD) ---")
    imbalance_data = agent.fetch_kraken_bid_ask_imbalance(pair="XXBTZUSD", count=50) # Primi 50 livelli
    print(f"Risultato: {imbalance_data}")
    if imbalance_data['imbalance'] > 0.1:
         print("- forte pressione BUY sul supporto")
    elif imbalance_data['imbalance'] < -0.1:
         print("- forte pressione SELL sulla resistenza")
    else:
         print("- Orderbook bilanciato.")
         
    # 3. Mock Test per Co-integration
    print("\n--- 3. STATISTICAL ARBITRAGE (Cointegrazione) ---")
    # Generiamo due serie fake co-integrate (Una è l'altra shiftata e con un po' di rumore)
    dummy_a = pd.Series(np.random.normal(100, 5, 200) + np.linspace(0, 50, 200))
    dummy_b = dummy_a * 0.8 + np.random.normal(0, 2, 200)
    
    z, pval, is_c = agent.calculate_cointegration_zscore(dummy_a, dummy_b)
    print(f"Asset Fake A e Fake B -> Cointegrati? {is_c} (P-Value: {pval}) | Z-Score Spread: {z}")
    if is_c:
        if z < -2.0:
            print("- Segnale: LONG A, SHORT B (Spread troppo stretto)")
        elif z > 2.0:
            print("- Segnale: SHORT A, LONG B (Spread troppo largo)")
        else:
            print("- Segnale: HOLD (Spread normale)")
