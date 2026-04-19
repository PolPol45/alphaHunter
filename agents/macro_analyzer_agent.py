import json
import logging
import time
import uuid
import yfinance as yf
from datetime import datetime, timezone

import requests
import feedparser
from agents.base_agent import BaseAgent, DATA_DIR
from adapters.fred_client import FredClient

class MacroAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__("macro_analyzer_agent")
        self.fred_client = FredClient(self.config.get("fred", {}))
        # Ensure FRED client connects on initialization or gracefully degrades
        self.fred_client.connect()
        
    def run(self) -> bool:
        self.mark_running()
        try:
            # In backtest: skip all live FRED/yfinance calls — regime already in market_regime.json
            _backtest_mode = self.config.get("orchestrator", {}).get("mode") == "backtest"
            if _backtest_mode:
                existing = self.read_json(DATA_DIR / "macro_snapshot.json") or {}
                if existing:
                    self.logger.info("Backtest mode — reusing cached macro_snapshot, skip live fetch")
                    self.mark_done()
                    return True

            self.logger.info("Fetching macro data from FRED...")
            fed_funds_hist = self._fetch_fred_history("FEDFUNDS")
            fed_funds = fed_funds_hist[-1] if fed_funds_hist else None
            fed_funds_trend = (fed_funds_hist[-1] - fed_funds_hist[0]) if len(fed_funds_hist) > 1 else 0

            cpi_hist = self._fetch_fred_history("CPIAUCSL")
            cpi = cpi_hist[-1] if cpi_hist else None

            yield_spread_hist = self._fetch_fred_history("T10Y2Y", limit=5)
            yield_spread = yield_spread_hist[-1] if yield_spread_hist else None
            
            qe_qt_hist = self._fetch_fred_history("WALCL")
            qe_qt_proxy = qe_qt_hist[-1] if qe_qt_hist else None
            qe_qt_trend = (qe_qt_hist[-1] - qe_qt_hist[0]) if len(qe_qt_hist) > 1 else 0
            
            self.logger.info("Fetching macro data from yfinance...")
            dxy_hist = self._fetch_yf_history("DX-Y.NYB", 60)
            dxy = dxy_hist[-1] if dxy_hist else None
            dxy_ma50 = sum(dxy_hist[-50:])/min(len(dxy_hist), 50) if len(dxy_hist) >= 50 else None

            sp500_hist = self._fetch_yf_history("^GSPC", 60)
            sp500 = sp500_hist[-1] if sp500_hist else None
            sp500_ma50 = sum(sp500_hist[-50:])/min(len(sp500_hist), 50) if len(sp500_hist) >= 50 else None

            stoxx600 = self._fetch_yf_close("EXSA.DE")
            nikkei = self._fetch_yf_close("^N225")
            vix = self._fetch_yf_close("^VIX")
            eem_close = self._fetch_yf_close("EEM")
            spy_close = self._fetch_yf_close("SPY")
            
            if fed_funds is None:
                # Fallback to 13-week T-Bill yield as proxy
                irx = self._fetch_yf_close("^IRX")
                if irx is not None:
                    fed_funds = round(irx, 2)
            
            eem_vs_spy = None
            if eem_close and spy_close:
                eem_vs_spy = eem_close / spy_close
                
            put_call_ratio = self._calculate_spy_put_call_ratio()
            
            correlations = self._calculate_regional_correlations()
            
            regime, score = self._determine_market_regime(
                vix=vix,
                sp500=sp500,
                dxy=dxy,
                fed_funds=fed_funds,
                qe_qt_proxy=qe_qt_proxy,
                put_call_ratio=put_call_ratio,
                dxy_ma50=dxy_ma50,
                qe_qt_trend=qe_qt_trend,
                fed_funds_trend=fed_funds_trend,
                sp500_ma50=sp500_ma50,
                yield_spread=yield_spread
            )
            
            volatility_sentiment = {
                "vix_regime": "HIGH" if (vix and vix > 20) else "NORMAL",
                "fear_greed_proxy": round((100 - vix) / 100.0, 2) if vix else 0.5,
                "news_sentiment_score": self._fetch_news_sentiment()
            }
            
            # --- FASE 14: NLP Macro Intelligence (LLM Integration) ---
            llm_enabled = self.config.get("llm_nlp_enabled", False)
            llm_result = {"sentiment_bias": 0.0, "narrative": "LLM NLP disabled."}
            
            if llm_enabled:
                self.logger.info("Fase 14: NLP Intelligence starting. Fetching news feeds...")
                news_text = self._fetch_rss_news()
                if news_text:
                    self.logger.info("Asking LLM to analyze sentiment...")
                    llm_result = self._analyze_sentiment_llm(news_text)
                    if llm_result and "sentiment_bias" in llm_result:
                        llm_bias = float(llm_result["sentiment_bias"])
                        # Iniezione Quantitativa: Modifichiamo pesantemente lo SCORE Macro
                        # Se l'LLM fischia panico puro, crolla lo score
                        if llm_bias < -0.6:
                            self.logger.critical(f"⚠️ LLM MACRO PANIC DETECTED! Narrative: {llm_result.get('narrative')}")
                            score = min(score, llm_bias)
                        else:
                            # Mixiamo 70% quantitativo, 30% qualitativo
                            score = round((score * 0.7) + (llm_bias * 0.3), 2)
                        
                        regime = "RISK_ON" if score > 0.1 else ("RISK_OFF" if score < -0.1 else "NEUTRAL")
            # -------------------------------------------------------------
            
            risk_flags = self._generate_risk_flags(vix, dxy)
            
            # Confidence: HIGH if >=3 data sources available, LOW if <=1
            available_sources = sum(1 for x in [vix, dxy, fed_funds, yield_spread, sp500] if x is not None)
            confidence = "HIGH" if available_sources >= 3 else ("MEDIUM" if available_sources == 2 else "LOW")

            now_iso = datetime.now(timezone.utc).isoformat()
            macro_state = {
                "timestamp": now_iso,
                "generated_at": now_iso,
                "regime": regime,
                "score": score,
                "confidence": confidence,
                "macro_factors": {
                    "fed_proxy": fed_funds,
                    "inflation_proxy": cpi,
                    "dxy": dxy,
                    "yield_spread_10y2y": yield_spread,
                    "sp500": sp500,
                    "stoxx600": stoxx600,
                    "nikkei": nikkei,
                    "eem_vs_spy": eem_vs_spy,
                    "vix": vix,
                    "put_call_ratio": put_call_ratio,
                    "qe_qt_proxy": qe_qt_proxy
                },
                "regional_correlations": correlations,
                "volatility_sentiment": volatility_sentiment,
                "risk_flags": risk_flags,
                "summary": {
                    "headline": f"Market remains in {regime} mode",
                    "explanation": f"Calculated score of {score}. VIX is at {vix or 'unknown'}, indicating {'high' if (vix and vix > 20) else 'stable'} volatility.",
                    "llm_narrative": llm_result.get("narrative", "")
                }
            }
            
            self.write_json(DATA_DIR / "market_regime.json", macro_state)
            self.update_shared_state("data_freshness.macro_snapshot", macro_state["generated_at"])
            
            self.logger.info(f"Macro regime generated: {regime} (Score: {score})")
            self.mark_done()
            return True
            
        except Exception as e:
            self.mark_error(e)
            return False

    def _fetch_fred(self, series_id: str) -> float | None:
        try:
            series = self.fred_client.get_series(series_id, limit=1)
            if series and len(series) > 0:
                return round(float(series[-1]["value"]), 4)
        except Exception as e:
            self.logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        return None

    def _fetch_fred_history(self, series_id: str, limit: int = 12) -> list[float]:
        try:
            series = self.fred_client.get_series(series_id, limit=limit)
            if series and len(series) > 0:
                return [round(float(s["value"]), 4) for s in series]
        except Exception as e:
            self.logger.warning(f"Failed to fetch FRED series history {series_id}: {e}")
        return []
        
    def _fetch_yf_close(self, ticker: str) -> float | None:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5d")
            if not df.empty:
                return round(float(df["Close"].iloc[-1]), 4)
        except Exception as e:
            self.logger.warning(f"Failed to fetch yfinance series {ticker}: {e}")
        return None

    def _fetch_yf_history(self, ticker: str, days: int = 60) -> list[float]:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{days}d")
            if not df.empty:
                return [round(float(x), 4) for x in df["Close"].tolist()]
        except Exception as e:
            self.logger.warning(f"Failed to fetch yfinance history {ticker}: {e}")
        return []

    def _calculate_spy_put_call_ratio(self) -> float | None:
        try:
            spy = yf.Ticker("SPY")
            expirations = spy.options
            if expirations:
                # Use nearest expiration for sentiment
                chain = spy.option_chain(expirations[0])
                calls_oi = chain.calls["openInterest"].sum()
                puts_oi = chain.puts["openInterest"].sum()
                if calls_oi > 0:
                    return round(float(puts_oi / calls_oi), 3)
        except Exception as e:
            self.logger.warning(f"Failed to calculate SPY Put/Call ratio: {e}")
        return None

    def _calculate_regional_correlations(self) -> dict:
        try:
            ticks = yf.download(["^GSPC", "EXSA.DE", "^N225", "EEM"], period="1mo", progress=False)
            if ticks is not None and not ticks.empty and "Close" in ticks:
                closes = ticks["Close"]
                corr = closes.corr()
                return {
                    "spx_vs_stoxx_20d": round(float(corr.loc["^GSPC", "EXSA.DE"]), 2) if "EXSA.DE" in corr.columns and "^GSPC" in corr.index else None,
                    "spx_vs_nikkei_20d": round(float(corr.loc["^GSPC", "^N225"]), 2) if "^N225" in corr.columns and "^GSPC" in corr.index else None,
                    "spx_vs_em_20d": round(float(corr.loc["^GSPC", "EEM"]), 2) if "EEM" in corr.columns and "^GSPC" in corr.index else None
                }
        except Exception as e:
            self.logger.warning(f"Failed to calculate regional correlations: {e}")
            
        return {
            "spx_vs_stoxx_20d": None,
            "spx_vs_nikkei_20d": None,
            "spx_vs_em_20d": None
        }
        
    def _determine_market_regime(
        self,
        vix: float | None,
        sp500: float | None,
        dxy: float | None,
        fed_funds: float | None,
        qe_qt_proxy: float | None,
        put_call_ratio: float | None,
        dxy_ma50: float | None = None,
        qe_qt_trend: float | None = None,
        fed_funds_trend: float | None = None,
        sp500_ma50: float | None = None,
        yield_spread: float | None = None,
    ) -> tuple[str, float]:
        score = 0.0
        
        # --- 40% VIX & Trend del Mercato ---
        market_score = 0.0
        if vix is not None:
            if vix < 15: market_score += 0.2
            elif vix > 25: market_score -= 0.3
            
        if sp500 is not None and sp500_ma50 is not None:
            if sp500 > sp500_ma50:
                market_score += 0.2
            else:
                market_score -= 0.1
                
        if put_call_ratio is not None:
            if put_call_ratio < 0.7:  # greesh / bullish
                market_score += 0.1
            elif put_call_ratio > 1.0: # bearish
                market_score -= 0.1
                
        # --- 30% Dollaro (DXY relative index) ---
        dxy_score = 0.0
        if dxy is not None and dxy_ma50 is not None:
            if dxy > dxy_ma50:
                dxy_score -= 0.3  # Forte dollaro = bearish risk assets
            else:
                dxy_score += 0.3
        elif dxy is not None: # fallback
            if dxy < 100: dxy_score += 0.15
            elif dxy > 105: dxy_score -= 0.3

        # --- Yield spread (10Y-2Y): positive = normal curve = bullish ---
        if yield_spread is not None:
            if yield_spread > 0.5:
                score += 0.1
            elif yield_spread < 0:
                score -= 0.1  # inverted curve = warning

        # --- 30% Variabili FED (Liquidità QE/QT, e Tassi d'interesse) ---
        fed_score = 0.0
        if fed_funds_trend is not None:
            if fed_funds_trend > 0:
                fed_score -= 0.15  # tassi salgono
            elif fed_funds_trend < 0:
                fed_score += 0.15
        
        if qe_qt_trend is not None:
            if qe_qt_trend > 0:
                fed_score += 0.15  # liquidità aumenta
            elif qe_qt_trend < 0:
                fed_score -= 0.15  # liquidità diminuisce
                
        score = market_score + dxy_score + fed_score
        
        # Override se estremo RISK_OFF (tassi salgono + DXY forte + liquidità scende)
        if (fed_funds_trend and fed_funds_trend > 0) and (dxy and dxy_ma50 and dxy > dxy_ma50) and (qe_qt_trend and qe_qt_trend < 0):
            self.logger.warning("EXTREME RISK OFF TRIGGERED! Rates Up + DXY Up + Liquidity Down")
            score = -1.0
            return "RISK_OFF", score
            
        score = max(-1.0, min(1.0, score))
        
        if score > 0.3:
            return "RISK_ON", round(score, 2)
        elif score < -0.3:
            return "RISK_OFF", round(score, 2)
        else:
            return "NEUTRAL", round(score, 2)

    def _fetch_news_sentiment(self) -> float:
        news_file = DATA_DIR / "news_feed.json"
        news_data = self.read_json(news_file)
        if news_data and "summary_metrics" in news_data:
            return round(float(news_data["summary_metrics"].get("average_sentiment", 0.0)), 2)
        return 0.0

    # --- FASE 14 METODI NLP ---
    
    def _fetch_rss_news(self) -> str:
        """Fetch e consolida le testate giornalistiche finanziarie."""
        feeds = [
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", # Finance
            "https://cointelegraph.com/rss" # Crypto
        ]
        
        consolidated = []
        try:
            for url in feeds:
                d = feedparser.parse(url)
                for entry in d.entries[:5]: # Take top 5 per feed
                    consolidated.append(f"- {entry.title}: {entry.get('summary', '')}")
            return "\\n".join(consolidated)
        except Exception as e:
            self.logger.warning(f"Failed to fetch RSS: {e}")
            return ""

    def _analyze_sentiment_llm(self, news_text: str) -> dict:
        """Invia i testi ad un LLM via OpenRouter o Ollama locale."""
        llm_config = self.config.get("llm_nlp", {})
        provider = llm_config.get("provider", "openrouter")
        api_key = llm_config.get("api_key", "")
        model = llm_config.get("model", "meta-llama/llama-3-8b-instruct:free")
        
        prompt = (
            "Sei un Chief Economist per un Hedge Fund Quantitativo.\\n"
            "Leggi le seguenti breaking news e quantifica rigorosamente il sentiment Macro.\\n"
            "Rispondi ESCLUSIVAMENTE con un JSON valido strutturato così:\\n"
            '{"sentiment_bias": float da -1.0 a +1.0, "narrative": "Tua spiegazione testuale concisa"}\\n\\n'
            f"NEWS:\\n{news_text}"
        )

        try:
            if provider == "openrouter" and api_key:
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "Sei un bot JSON NLP specializzato in finanza."},
                        {"role": "user", "content": prompt}
                    ]
                }
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    content = resp.json()
                    json_str = content['choices'][0]['message']['content']
                    return json.loads(json_str)
                    
            elif provider == "ollama":
                endpoint = llm_config.get("endpoint", "http://localhost:11434/api/generate")
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }
                resp = requests.post(endpoint, json=payload, timeout=30)
                if resp.status_code == 200:
                    json_str = resp.json().get("response", "{}")
                    return json.loads(json_str)
                    
        except Exception as e:
            self.logger.warning(f"Error during LLM sentiment analysis: {e}")
            
        return {"sentiment_bias": 0.0, "narrative": "LLM failed or skipped"}
    
    # -------------------------

    def _from_snapshot(self, snapshot: dict) -> dict:
        """Build a market regime dict from a macro_snapshot.json-style document."""
        score = float(snapshot.get("market_bias", 0.0) or 0.0)
        regime = "RISK_ON" if score > 0.1 else ("RISK_OFF" if score < -0.1 else "NEUTRAL")

        series = snapshot.get("series", {})
        fed_value = None
        for key, entry in series.items():
            if isinstance(entry, dict) and entry.get("status") == "ok":
                if key in {"fed_funds", "fedfunds"}:
                    fed_value = entry.get("value")
                    break
        if fed_value is None:
            # fallback: first available series value
            for entry in series.values():
                if isinstance(entry, dict) and entry.get("status") == "ok":
                    fed_value = entry.get("value")
                    break

        advanced = snapshot.get("advanced_macro", {})
        liquidity = snapshot.get("liquidity", {})
        risk_flags = snapshot.get("risk_flags", [])

        return {
            "regime": regime,
            "score": score,
            "macro_factors": {
                "fed_proxy": fed_value,
                "advanced_macro": advanced,
                "liquidity": liquidity,
                "risk_flags": risk_flags,
            },
            "source": "macro_snapshot",
            "generated_at": snapshot.get("generated_at"),
        }

    def _generate_risk_flags(self, vix: float | None, dxy: float | None) -> list[dict]:
        flags = []
        if vix is not None and vix > 25:
            flags.append({
                "code": "VOL_HIGH",
                "label": "High Volatility (VIX > 25)",
                "severity": "critical" if vix > 30 else "warning",
                "bias": -0.8
            })
        if dxy is not None and dxy > 105:
            flags.append({
                "code": "DXY_HIGH",
                "label": "Strong Dollar Pressure",
                "severity": "warning",
                "bias": -0.4
            })
        if not flags:
            flags.append({
                "code": "R_NORMAL",
                "label": "No extraordinary macro risks detected",
                "severity": "info",
                "bias": 0.0
            })
        return flags
