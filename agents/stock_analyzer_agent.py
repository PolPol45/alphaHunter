import json
import logging
import os
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.base_agent import BaseAgent, DATA_DIR

STATE_FILE = DATA_DIR / "stock_analyzer_state.json"
CHUNK_SIZE = 50

class StockAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__("stock_analyzer_agent")
        
        # Parse all symbols from master_universe
        self.all_symbols = []
        universe = self.config.get("master_universe", {})
        
        # Ensure we skip strictly crypto pools to avoid yfinance throwing weird exceptions occasionally
        # if the user hasn't mapped them right, but the config has 'crypto_core' and 'crypto_defi_bridge'.
        for category, symbols in universe.items():
            if not category.startswith("_") and "crypto" not in category.lower():
                for sym in symbols:
                    if sym not in self.all_symbols:
                        self.all_symbols.append(sym)
                        
        # We might also want to hard-check the explicit sector_map to map known benchmarks
        self.sector_map = self.config.get("sector_map", {})
        self.known_symbols = {}
        for sector, cfg in self.sector_map.items():
            for sym in cfg.get("members", []):
                self.known_symbols[sym] = {"sector": sector, "benchmark": cfg.get("benchmark", "SPY")}

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self.all_symbols:
                self.logger.warning("No equities found in master_universe. Skipped.")
                self.mark_done()
                return True
                
            total = len(self.all_symbols)
            
            # 1. Read state
            current_index = 0
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, "r") as f:
                        state = json.load(f)
                        current_index = state.get("current_index", 0)
                except:
                    current_index = 0
                    
            if current_index >= total:
                current_index = 0
                
            # 2. Slice Chunk
            chunk_symbols = self.all_symbols[current_index : current_index + CHUNK_SIZE]
            self.logger.info(f"Processing chunk {current_index}-{current_index + len(chunk_symbols)} of {total} symbols...")
            
            # 3. Read existing scores
            existing_scores = {}
            score_file = DATA_DIR / "stock_scores.json"
            if score_file.exists():
                try:
                    with open(score_file, "r") as f:
                        data = json.load(f)
                        for item in data.get("scores", []):
                            existing_scores[item["symbol"]] = item
                except:
                    pass
            
            # 4. Fetch and compute metrics for chunk
            prices = self._fetch_bulk_history(chunk_symbols)
            
            # Track failures slightly
            for symbol in chunk_symbols:
                df = None
                if prices is not None:
                    if len(chunk_symbols) == 1:
                        df = prices
                    elif symbol in prices.columns.levels[0]:
                        df = prices[symbol]
                
                technical_data = self._calculate_technicals(df)
                fundamental_data = self._fetch_fundamentals(symbol)
                
                tech_score = self._compute_technical_score(technical_data)
                fund_score = self._compute_fundamental_score(fundamental_data)
                composite = round((tech_score * 0.6) + (fund_score * 0.4), 2)
                
                alerts = []
                if composite > 0.7: alerts.append("STRONG_BUY_SIGNAL")
                if fundamental_data.get("roe") and fundamental_data["roe"] > 0.2: alerts.append("HIGH_PROFITABILITY")
                if technical_data.get("rsi") and technical_data["rsi"] < 30: alerts.append("OVERSOLD_BOUNCE")
                if technical_data.get("rsi") and technical_data["rsi"] > 70: alerts.append("OVERBOUGHT")

                # Resolve sector/benchmark
                if symbol in self.known_symbols:
                    sect = self.known_symbols[symbol]["sector"]
                    bench = self.known_symbols[symbol]["benchmark"]
                else:
                    sect = fundamental_data.get("sector", "Unknown_Sector")
                    bench = "SPY"

                existing_scores[symbol] = {
                    "symbol": symbol,
                    "sector": sect,
                    "benchmark": bench,
                    "composite_score": composite,
                    "technical_score": tech_score,
                    "fundamental_score": fund_score,
                    "news_score": 0.0,
                    "beta": fundamental_data.get("beta"),
                    "price_target_mean": fundamental_data.get("target_mean"),
                    "recommendation_mean": fundamental_data.get("recommendation_mean"),
                    "analyst_count": fundamental_data.get("analyst_count"),
                    "roa": fundamental_data.get("roa"),
                    "roe": fundamental_data.get("roe"),
                    "debt_to_equity": fundamental_data.get("debt_to_equity"),
                    "free_cash_flow": fundamental_data.get("fcf"),
                    "current_ratio": fundamental_data.get("current_ratio"),
                    "ema50_above_ema200": technical_data.get("ema50_above_200", False),
                    "rsi": technical_data.get("rsi"),
                    "macd_state": technical_data.get("macd_state", "NEUTRAL"),
                    "momentum_7d": technical_data.get("momentum_7d"),
                    "momentum_30d": technical_data.get("momentum_30d"),
                    "correlation_to_sector": None,
                    "alerts": alerts
                }
                
            # 5. Flat and sort all known scores
            scores_list = list(existing_scores.values())
            scores_list.sort(key=lambda x: x["composite_score"], reverse=True)
            
            output = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scores": scores_list
            }
            
            self.write_json(score_file, output)
            self.update_shared_state("data_freshness.stock_scores", output["generated_at"])
            
            # 6. Save new state
            next_index = current_index + CHUNK_SIZE
            if next_index >= total:
                next_index = 0
            with open(STATE_FILE, "w") as f:
                json.dump({"current_index": next_index}, f)
            
            top_scorer = scores_list[0]['symbol'] if scores_list else "None"
            self.logger.info(f"Saved {len(scores_list)} total evaluated stocks. Global Leader: {top_scorer}")
            
            self.mark_done()
            return True

        except Exception as e:
            self.logger.error(f"StockAnalyzerAgent failed: {e}", exc_info=True)
            self.mark_error(e)
            return False

    def _fetch_bulk_history(self, symbols: list[str]) -> pd.DataFrame | None:
        try:
            return yf.download(symbols, period="1y", group_by='ticker', progress=False)
        except Exception as e:
            self.logger.warning(f"Failed bulk download: {e}")
            return None

    def _fetch_fundamentals(self, symbol: str) -> dict:
        try:
            t = yf.Ticker(symbol)
            info = t.info
            
            return {
                "sector": info.get("sector", "Unknown_Sector"),
                "beta": self._safe_float(info.get("beta")),
                "target_mean": self._safe_float(info.get("targetMeanPrice")),
                "recommendation_mean": self._safe_float(info.get("recommendationMean")),
                "analyst_count": int(info.get("numberOfAnalystOpinions")) if info.get("numberOfAnalystOpinions") is not None else None,
                "roa": self._safe_float(info.get("returnOnAssets")),
                "roe": self._safe_float(info.get("returnOnEquity")),
                "debt_to_equity": self._safe_float(info.get("debtToEquity")),
                "fcf": self._safe_float(info.get("freeCashflow")),
                "current_ratio": self._safe_float(info.get("currentRatio"))
            }
        except Exception as e:
            self.logger.debug(f"Failed to fetch fundamentals for {symbol}: {e}")
            return {}

    def _calculate_technicals(self, df: pd.DataFrame | None) -> dict:
        data = {
            "ema50_above_200": False,
            "rsi": None,
            "macd_state": "NEUTRAL",
            "momentum_7d": None,
            "momentum_30d": None
        }
        
        if df is None or df.empty or 'Close' not in df.columns:
            return data
            
        try:
            closes = df['Close'].dropna()
            if len(closes) < 35:
                return data
                
            current = float(closes.iloc[-1])
            
            # Momentum
            if len(closes) >= 7:
                data["momentum_7d"] = round((current / float(closes.iloc[-7]) - 1.0), 4)
            if len(closes) >= 30:
                data["momentum_30d"] = round((current / float(closes.iloc[-30]) - 1.0), 4)
                
            # EMA calculations
            if len(closes) >= 200:
                ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
                ema200 = closes.ewm(span=200, adjust=False).mean().iloc[-1]
                data["ema50_above_200"] = bool(ema50 > ema200)
                
            # RSI 14
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            data["rsi"] = round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else None
            
            # MACD (12, 26, 9)
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            macd_cur = float(macd.iloc[-1])
            sig_cur = float(signal.iloc[-1])
            if macd_cur > sig_cur and macd_cur > 0:
                data["macd_state"] = "BULLISH"
            elif macd_cur < sig_cur and macd_cur < 0:
                data["macd_state"] = "BEARISH"
                
            return data
            
        except Exception as e:
            self.logger.debug(f"Error in technical calculation: {e}", exc_info=True)
            return data

    def _compute_technical_score(self, tech: dict) -> float:
        score = 0.5 
        
        if tech.get("ema50_above_200"): score += 0.2
        else: score -= 0.1
        
        macd = tech.get("macd_state", "NEUTRAL")
        if macd == "BULLISH": score += 0.15
        elif macd == "BEARISH": score -= 0.15
            
        mom = tech.get("momentum_30d")
        if mom is not None:
            if mom > 0.05: score += 0.15
            elif mom < -0.05: score -= 0.15
            
        return round(max(0.0, min(1.0, score)), 2)
        
    def _compute_fundamental_score(self, fund: dict) -> float:
        score = 0.5
        
        roe = fund.get("roe")
        if roe is not None:
            if roe > 0.15: score += 0.2
            elif roe < 0: score -= 0.2
            
        roa = fund.get("roa")
        if roa is not None:
            if roa > 0.05: score += 0.1
            elif roa < 0: score -= 0.1
            
        rec = fund.get("recommendation_mean")
        if rec is not None:
            if rec <= 2.0: score += 0.2
            elif rec >= 3.5: score -= 0.2
            
        return round(max(0.0, min(1.0, score)), 2)

    def _safe_float(self, val) -> float | None:
        if val is None: return None
        try:
            return round(float(val), 4)
        except (ValueError, TypeError):
            return None
