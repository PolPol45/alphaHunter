import json
import logging
import time
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR

_CACHE_FILE = DATA_DIR / "sector_price_cache.json"

class SectorAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__("sector_analyzer_agent")
        self.sector_map = self.config.get("sector_map", {})
        sa_cfg = self.config.get("sector_analyzer", {})
        self._cache_ttl = float(sa_cfg.get("cache_ttl_seconds", 3600))
        
    def run(self) -> bool:
        self.mark_running()
        try:
            if self.config.get("orchestrator", {}).get("mode") == "backtest":
                existing = self.read_json(DATA_DIR / "sector_scorecard.json") or {}
                if existing:
                    self.logger.info("Backtest mode — reusing sector_scorecard.json, skip live fetch")
                    self.mark_done()
                    return True

            if not self.sector_map:
                self.logger.warning("No sector_map defined in config.json. Skipping SectorAnalyzerAgent.")
                self.mark_done()
                return True
                
            self.logger.info(f"Analyzing {len(self.sector_map)} sectors defined in config...")
            
            # Retrieve Macro layer context
            macro_state = self.read_json(DATA_DIR / "market_regime.json") or {}
            regime = macro_state.get("regime", "UNKNOWN")

            # Invalidate cache when macro confidence is LOW (stale/uncertain data)
            force_refresh = macro_state.get("confidence", "LOW") == "LOW"
            if force_refresh:
                self.logger.info("Macro confidence=LOW — forcing cache refresh for sector data")

            sectors_processed = []

            spy_data = self._fetch_history_cached("SPY", lookback_days=100, force=force_refresh)
            
            for sector_name, config in self.sector_map.items():
                benchmark = config.get("benchmark")
                members = config.get("members", [])
                
                self.logger.info(f"Processing sector {sector_name} ({benchmark}) with {len(members)} members")
                
                # Fetch Benchmark Data
                benchmark_data = self._fetch_history_cached(benchmark, lookback_days=100, force=force_refresh)

                # Metrics
                ret_5d, ret_20d, ret_60d = self._calculate_returns(benchmark_data)
                beta = self._calculate_beta(benchmark_data, spy_data)
                rs_20d = self._calculate_relative_strength(benchmark_data, spy_data, days=20)

                # Breadth calculation
                breadth = self._calculate_breadth_cached(members, force=force_refresh)
                
                # Scoring synthetic
                raw_score = 0.0
                if rs_20d is not None:
                    raw_score += rs_20d * 2.0  # Momentum component
                if breadth is not None:
                    raw_score += (breadth - 0.5)  # Breadth component: above 50% is positive
                
                score = round(max(0.0, min(1.0, (raw_score + 1.0) / 2.0)), 2) # normalize 0-1
                
                driver_type = "UNKNOWN"
                if beta is not None and beta > 1.2:
                    driver_type = "MARKET_BETA"
                elif rs_20d is not None and rs_20d > 0.05 and beta is not None and beta < 1.1:
                    driver_type = "IDIOSYNCRATIC"
                else:
                    driver_type = "MIXED"

                signal = "STRONG" if score > 0.7 else ("WEAK" if score < 0.4 else "NEUTRAL")

                sectors_processed.append({
                    "name": sector_name,
                    "benchmark": benchmark,
                    "members": members,
                    "score": score,
                    "signal": signal,
                    "rank": 0,  # Will calculate after processing all
                    "rs_rank": 0,  # Will calculate after sorting by RS
                    "return_5d": ret_5d,
                    "return_20d": ret_20d,
                    "return_60d": ret_60d,
                    "relative_strength_20d": rs_20d,
                    "breadth_50d": breadth,
                    "breadth_above_ema50": breadth,  # legacy alias
                    "avg_volume_trend": None,
                    "news_count": 0,
                    "news_sentiment_score": 0.0,
                    "beta_to_spy": beta,
                    "driver_type": driver_type,
                    "alerts": []
                })
                
            # Rank by score descending
            sectors_processed.sort(key=lambda s: s["score"], reverse=True)
            for i, s in enumerate(sectors_processed):
                s["rank"] = i + 1
                if i == 0:
                    s["alerts"].append("SECTOR_LEADER")
                elif i == len(sectors_processed) - 1:
                    s["alerts"].append("SECTOR_LAGGARD")

            # RS rank: separate ranking by relative_strength_20d
            rs_sorted = sorted(
                sectors_processed,
                key=lambda s: (s["relative_strength_20d"] or -999),
                reverse=True
            )
            for i, s in enumerate(rs_sorted):
                s["rs_rank"] = i + 1

            scorecard = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "market_regime_ref": regime,
                "sectors": sectors_processed
            }
            
            self.write_json(DATA_DIR / "sector_scorecard.json", scorecard)
            self.update_shared_state("data_freshness.sector_scorecard", scorecard["generated_at"])
            
            self.logger.info(f"Sector analysis complete. Leader: {sectors_processed[0]['name']}")
            self.mark_done()
            return True

        except Exception as e:
            self.logger.error(f"SectorAnalyzerAgent failed: {e}", exc_info=True)
            self.mark_error(e)
            return False

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        try:
            if _CACHE_FILE.exists():
                return json.loads(_CACHE_FILE.read_text())
        except Exception:
            pass
        return {}

    def _save_cache(self, cache: dict) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps(cache))
        except Exception as e:
            self.logger.debug(f"Cache write failed: {e}")

    def _fetch_history_cached(self, symbol: str, lookback_days: int, force: bool = False) -> pd.DataFrame | None:
        cache = self._load_cache()
        key = f"{symbol}_{lookback_days}d"
        entry = cache.get(key, {})
        age = time.time() - entry.get("ts", 0)
        if not force and age < self._cache_ttl and entry.get("records"):
            try:
                df = pd.DataFrame(entry["records"])
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                self.logger.debug(f"Cache hit: {key} (age={age:.0f}s)")
                return df
            except Exception:
                pass  # corrupt entry → fall through to fresh fetch
        df = self._fetch_history(symbol, lookback_days)
        if df is not None:
            reset = df.reset_index()
            # index column may be named "Date", "Datetime", or something else
            idx_col = reset.columns[0]
            reset[idx_col] = reset[idx_col].astype(str)
            reset = reset.rename(columns={idx_col: "Date"})
            cache[key] = {"ts": time.time(), "records": reset.to_dict("records")}
            self._save_cache(cache)
        return df

    def _calculate_breadth_cached(self, members: list[str], force: bool = False) -> float | None:
        if not members:
            return None
        key = "breadth_" + hashlib.md5(",".join(sorted(members)).encode()).hexdigest()[:8]
        cache = self._load_cache()
        entry = cache.get(key, {})
        age = time.time() - entry.get("ts", 0)
        if not force and age < self._cache_ttl and "value" in entry:
            self.logger.debug(f"Cache hit: {key} (age={age:.0f}s)")
            return entry["value"]
        value = self._calculate_breadth(members)
        cache[key] = {"ts": time.time(), "value": value}
        self._save_cache(cache)
        return value

    # ── yfinance fetchers (no caching — called only on cache miss) ─────────────

    def _fetch_history(self, symbol: str, lookback_days: int) -> pd.DataFrame | None:
        try:
            t = yf.Ticker(symbol)
            # Fetch a bit more to allow calculations like 60d returns safely
            df = t.history(period=f"{lookback_days}d")
            if df.empty:
                return None
            return df
        except Exception as e:
            self.logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _calculate_returns(self, df: pd.DataFrame | None) -> tuple[float|None, float|None, float|None]:
        if df is None or df.empty or len(df) < 5:
            return None, None, None
            
        try:
            closes = df['Close']
            current = float(closes.iloc[-1])
            
            ret_5d = round((current / float(closes.iloc[-5]) - 1.0), 4) if len(closes) >= 5 else None
            ret_20d = round((current / float(closes.iloc[-20]) - 1.0), 4) if len(closes) >= 20 else None
            ret_60d = round((current / float(closes.iloc[-60]) - 1.0), 4) if len(closes) >= 60 else None
            
            return ret_5d, ret_20d, ret_60d
        except Exception as e:
            self.logger.warning(f"Error calculating returns: {e}")
            return None, None, None

    def _calculate_beta(self, asset_df: pd.DataFrame | None, spy_df: pd.DataFrame | None) -> float | None:
        if asset_df is None or spy_df is None or len(asset_df) < 60 or len(spy_df) < 60:
            return None
        
        try:
            # Align dates
            df = pd.DataFrame({
                "Asset": asset_df['Close'],
                "SPY": spy_df['Close']
            }).dropna()
            
            if len(df) < 20: 
                return None
                
            # Log returns
            returns = np.log(df / df.shift(1)).dropna()
            cov = np.cov(returns['Asset'], returns['SPY'])[0][1]
            var = np.var(returns['SPY'])
            
            if var == 0:
                return None
                
            return round(float(cov / var), 2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating beta: {e}")
            return None

    def _calculate_relative_strength(self, asset_df: pd.DataFrame | None, spy_df: pd.DataFrame | None, days: int = 20) -> float | None:
        if asset_df is None or spy_df is None or len(asset_df) < days or len(spy_df) < days:
            return None
            
        try:
            asset_ret = (float(asset_df['Close'].iloc[-1]) / float(asset_df['Close'].iloc[-days])) - 1.0
            spy_ret = (float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-days])) - 1.0
            return round(asset_ret - spy_ret, 4)
        except Exception as e:
            self.logger.warning(f"Error calculating relative strength: {e}")
            return None

    def _calculate_breadth(self, members: list[str]) -> float | None:
        if not members:
            return None
            
        above_50 = 0
        valid = 0
        
        try:
            data = yf.download(members, period="100d", group_by='ticker', progress=False)
        except Exception as e:
            self.logger.warning(f"Bulk download failed for breadth calculation: {e}")
            return None
            
        for ticker in members:
            try:
                # Handle single ticker edge case vs multi-ticker
                if len(members) == 1:
                    df = data
                else:
                    df = data[ticker] if ticker in data.columns.levels[0] else None
                    
                if df is None or df.empty or 'Close' not in df.columns:
                    continue
                    
                closes = df['Close'].dropna()
                if len(closes) < 50:
                    continue
                    
                ema50 = closes.ewm(span=50, adjust=False).mean()
                current_price = float(closes.iloc[-1])
                current_ema = float(ema50.iloc[-1])
                
                if current_price > current_ema:
                    above_50 += 1
                valid += 1
                
            except Exception as e:
                self.logger.debug(f"Failed to process breadth for {ticker} - {e}")
                
        if valid == 0:
            return None
            
        return round(float(above_50 / valid), 3)
