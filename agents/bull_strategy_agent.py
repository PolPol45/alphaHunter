"""
BullStrategyAgent — Analyst Bot Top-Down Mode
=============================================
Consuma `macro_snapshot.json` e `stock_scores.json`.

Levels of analysis:
1. Legge JSON scores pre-calcolati (istantaneo).
2. Controlla macro environment per assegnare moltiplicatori di sicurezza.
3. Filtra la creme de la creme degli ETF, Large e Small cap usando
   "composite_score", momentum e fondamentali puri.
4. Genera data/bull_signals.json istantaneamente, libero da fetch di rete.
"""

from __future__ import annotations
from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR

class BullStrategyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("bull_strategy_agent")
        self._cfg = self.config.get("bull_strategy", {})
        mu = self.config.get("master_universe", {})
        self.etf_universe        = mu.get("etf_long",            [])
        self.large_cap_universe  = mu.get("equities_large_cap",  [])
        self.small_cap_universe  = mu.get("equities_small_cap",  [])

    def run(self) -> bool:
        self.mark_running()
        try:
            # 1. Macro Analysis
            macro_bias, macro_multiplier = self._analyze_macro()

            # 2. Read centralized stock scores
            scores_data = self.read_json(DATA_DIR / "stock_scores.json")
            all_scores = scores_data.get("scores", [])
            
            if not all_scores:
                self.logger.warning("No pre-calculated stock_scores.json found. Engine skipped.")
                self.mark_done()
                return True

            etf_candidates = []
            large_candidates = []
            small_candidates = []
            
            # Base threshold adjusted by macro
            threshold = self._cfg.get("signal_threshold", 0.60) * (2.0 - macro_multiplier) 

            for item in all_scores:
                sym = item["symbol"]
                comp = item.get("composite_score", 0)
                
                # Minimum viable filter: positive composite, non-crashing momentum
                if comp < threshold: 
                    continue
                    
                mom30 = item.get("momentum_30d")
                if mom30 is not None and mom30 < -0.05:
                    continue # Ignore dipping trends
                
                # Tech filter check (we want EMA50 > EMA200 for Bull picks)
                healthy = item.get("ema50_above_ema200", True)
                if not healthy:
                    # Se non è in golden cross, pretendiamo un gran punteggio o RSI over-sold
                    rsi = item.get("rsi", 50)
                    if comp < (threshold + 0.1) and rsi >= 40:
                        continue
                
                # Check mapping
                if sym in self.etf_universe:
                    etf_candidates.append(item)
                elif sym in self.large_cap_universe:
                    large_candidates.append(item)
                elif sym in self.small_cap_universe:
                    small_candidates.append(item)

            # Sort and allocate target lists
            target_large = 4 # 3 to 5
            target_small = 10 # 10 to 20
            
            etf_picks = sorted(etf_candidates, key=lambda x: x["composite_score"], reverse=True)
            large_picks = sorted(large_candidates, key=lambda x: x["composite_score"], reverse=True)[:target_large]
            small_picks = sorted(small_candidates, key=lambda x: x["composite_score"], reverse=True)[:target_small]

            output = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "macro_multiplier": round(macro_multiplier, 3),
                "allocations": {
                    "etf_50_pct": etf_picks,
                    "large_cap_30_pct": large_picks,
                    "small_cap_20_pct": small_picks
                },
                "summary": {
                    "etf_count": len(etf_picks),
                    "large_cap_count": len(large_picks),
                    "small_cap_count": len(small_picks)
                }
            }

            self.write_json(DATA_DIR / "bull_signals.json", output)
            self.logger.info(f"Bull strategy scan complete. Found {len(etf_picks)} ETFs, {len(large_picks)} Large Caps, {len(small_picks)} Small Caps.")

            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    def _analyze_macro(self) -> tuple[float, float]:
        """Livello 1: Ritorna bias (-1.0 to 1.0) e un PE threshold multiplier."""
        macro = self.read_json(DATA_DIR / "macro_snapshot.json")
        series = macro.get("series", {})
        
        fed = series.get("fed_funds", {}).get("value", 4.0) or 4.0
        cpi = series.get("cpi_yoy", {}).get("value", 3.0) or 3.0
        dxy = series.get("dxy", {}).get("value", 100.0) or 100.0
        
        bias = 0.0
        if fed > 4.5: bias -= 0.3
        elif fed < 3.0: bias += 0.3
        
        if cpi > 3.5: bias -= 0.3
        elif cpi < 2.5: bias += 0.3
        
        if dxy > 105.0: bias -= 0.2
        elif dxy < 100.0: bias += 0.2
        
        multiplier = 1.0 + (bias * 0.3) 
        return bias, multiplier
