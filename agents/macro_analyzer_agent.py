import json
import logging
import time
import uuid
import yfinance as yf
from datetime import datetime, timezone

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
            self.logger.info("Fetching macro data from FRED...")
            fed_funds = self._fetch_fred("FEDFUNDS")
            cpi = self._fetch_fred("CPIAUCSL")
            qe_qt_proxy = self._fetch_fred("WALCL")
            
            self.logger.info("Fetching macro data from yfinance...")
            dxy = self._fetch_yf_close("DX-Y.NYB")
            sp500 = self._fetch_yf_close("^GSPC")
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
            
            regime, score = self._determine_market_regime(vix, sp500, dxy)
            
            volatility_sentiment = {
                "vix_regime": "HIGH" if (vix and vix > 20) else "NORMAL",
                "fear_greed_proxy": round((100 - vix) / 100.0, 2) if vix else 0.5,
                "news_sentiment_score": self._fetch_news_sentiment()
            }
            
            risk_flags = self._generate_risk_flags(vix, dxy)
            
            macro_state = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "regime": regime,
                "score": score,
                "macro_factors": {
                    "fed_proxy": fed_funds,
                    "inflation_proxy": cpi,
                    "dxy": dxy,
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
                    "explanation": f"Calculated score of {score}. VIX is at {vix or 'unknown'}, indicating {'high' if (vix and vix > 20) else 'stable'} volatility."
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
        
    def _fetch_yf_close(self, ticker: str) -> float | None:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5d")
            if not df.empty:
                return round(float(df["Close"].iloc[-1]), 4)
        except Exception as e:
            self.logger.warning(f"Failed to fetch yfinance series {ticker}: {e}")
        return None

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
        
    def _determine_market_regime(self, vix: float | None, sp500: float | None, dxy: float | None) -> tuple[str, float]:
        score = 0.0
        
        if vix is not None:
            if vix < 15: score += 0.3
            elif vix > 25: score -= 0.5
            
        if dxy is not None:
            if dxy < 100: score += 0.2
            elif dxy > 105: score -= 0.3
            
        if score > 0.3:
            return "RISK_ON", round(min(score, 1.0), 2)
        elif score < -0.3:
            return "RISK_OFF", round(max(score, -1.0), 2)
        else:
            return "NEUTRAL", round(score, 2)

    def _fetch_news_sentiment(self) -> float:
        news_file = DATA_DIR / "news_feed.json"
        news_data = self.read_json(news_file)
        if news_data and "summary_metrics" in news_data:
            return round(float(news_data["summary_metrics"].get("average_sentiment", 0.0)), 2)
        return 0.0

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
