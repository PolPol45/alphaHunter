"""
BearStrategyAgent — Short & Hedging Strategy (Top-Down Analyst — Bear Mode)
=============================================================================
Implementa il processo Top-Down INVERSO consumando istantaneamente:
- `macro_snapshot.json`
- `stock_scores.json`

Strategie supportate:
  A) SHORT CANDIDATES  — Titoli con fondamentali deteriorati + breakdown tecnico
  B) HEDGE ETFs        — ETF inversi e volatilità per protezione portafoglio
  C) BANKRUPTCY RISK   — Utilizza euristiche fondamentaliste "di collasso" da stock_scores.json.
"""

from __future__ import annotations
from datetime import datetime, timezone
from agents.base_agent import BaseAgent, DATA_DIR

class BearStrategyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("bear_strategy_agent")
        cfg = self.config.get("bear_strategy", {})
        self.mode = cfg.get("mode", "full")
        mu = self.config.get("master_universe", {})
        
        self.etf_hedge       = mu.get("etf_hedge",            [])
        large_cap            = mu.get("equities_large_cap",   [])
        small_cap            = mu.get("equities_small_cap",   [])
        distressed           = mu.get("equities_distressed",  [])

        self.short_universe  = large_cap + small_cap + distressed
        self.altman_universe = small_cap + distressed

        self.max_shorts:       int   = cfg.get("max_shorts",        15)
        self.max_hedges:       int   = cfg.get("max_hedges",        5)
        self.max_bankruptcy:   int   = cfg.get("max_bankruptcy",    10)

    def run(self) -> bool:
        self.mark_running()
        try:
            # 1. Macro Analysis
            macro_bias, risk_off = self._analyze_macro()
            self.logger.info(f"Macro bias: {macro_bias:+.2f} | Risk-off environment: {'YES' if risk_off else 'NO'}")

            # 2. Read centralized stock scores
            scores_data = self.read_json(DATA_DIR / "stock_scores.json")
            all_scores = scores_data.get("scores", [])
            
            if not all_scores:
                self.logger.warning("No pre-calculated stock_scores.json found. Engine skipped.")
                self.mark_done()
                return True

            hedge_results = []
            short_results = []
            altman_results = []

            for item in all_scores:
                sym = item["symbol"]
                comp = item.get("composite_score", 0.50)
                fund = item.get("fundamental_score", 0.50)
                
                # Check ETF Hedge
                if sym in self.etf_hedge:
                    # Se il mercato crolla, l'ETF inverso sale (uptrend)
                    mom30 = item.get("momentum_30d")
                    if (mom30 is not None and mom30 > 0.02) or risk_off:
                        hedge_results.append(item)

                # Check Short Candidates (comp < 0.4, bad tech, bad mom)
                elif sym in self.short_universe:
                    mom30 = item.get("momentum_30d", 0)
                    rsi = item.get("rsi", 50)
                    healthy = item.get("ema50_above_ema200", True)
                    
                    if comp < 0.40 and not healthy and (mom30 is not None and mom30 < 0):
                        # Non shortiamo titoli in oversold estremo, vogliamo venderli nei rimbalzi morti
                        if rsi > 35:
                            short_results.append(item)
                            
                # Check Bankruptcy proxy (high debt, negative ROE, bad fundamentally)
                if sym in self.altman_universe:
                    debt = item.get("debt_to_equity")
                    roe = item.get("roe")
                    fcf = item.get("free_cash_flow")
                    
                    bad_flags = 0
                    if debt and debt > 200: bad_flags += 1
                    if roe and roe < -0.15: bad_flags += 1
                    if fcf and fcf < 0:     bad_flags += 1
                    if fund < 0.20:         bad_flags += 1
                    
                    if bad_flags >= 2:
                        altman_results.append(item)

            # PR 3: Se in modalità hedge_only, annulla i segnali direzionali short
            if self.mode == "hedge_only":
                self.logger.info("Modalità hedge_only attiva: filtraggio segnali short e bankruptcy")
                short_results = []
                altman_results = []

            # Sort and allocate
            # Hedging: highest composite_score (we want the hedge to be performing well in Bear)
            hedge_picks = sorted(hedge_results, key=lambda x: x.get("composite_score", 0), reverse=True)[:self.max_hedges]
            # Shorts: lowest composite score (we want the worst companies)
            short_picks = sorted(short_results, key=lambda x: x.get("composite_score", 1.0))[:self.max_shorts]
            # Bankruptcy: worst fundamentals
            bankrupt_picks = sorted(altman_results, key=lambda x: x.get("fundamental_score", 1.0))[:self.max_bankruptcy]

            output = {
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "macro_bias":   round(macro_bias, 3),
                "risk_off":     risk_off,
                "allocations": {
                    "hedge_etfs":        hedge_picks,      
                    "short_candidates":  short_picks,     
                    "bankruptcy_risk":   bankrupt_picks,  
                },
                "summary": {
                    "hedge_count":      len(hedge_picks),
                    "short_count":      len(short_picks),
                    "bankruptcy_count": len(bankrupt_picks),
                },
            }

            self.write_json(DATA_DIR / "bear_signals.json", output)
            self.logger.info(f"Bear strategy scan complete. Hedges: {len(hedge_picks)} | Shorts: {len(short_picks)} | Bankrupt: {len(bankrupt_picks)}")

            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    def _analyze_macro(self) -> tuple[float, bool]:
        macro = self.read_json(DATA_DIR / "macro_snapshot.json")
        series = macro.get("series", {})
        
        fed = series.get("fed_funds", {}).get("value", 4.0) or 4.0
        cpi = series.get("cpi_yoy", {}).get("value", 3.0) or 3.0
        dxy = series.get("dxy", {}).get("value", 100.0) or 100.0
        vix = series.get("vix", {}).get("value", 18.0) or 18.0
        
        bias = 0.0
        if fed > 5.0: bias -= 0.4
        elif fed > 4.0: bias -= 0.2
        elif fed < 2.5: bias += 0.3
        
        if cpi > 4.0: bias -= 0.4
        elif cpi > 3.0: bias -= 0.2
        elif cpi < 2.0: bias += 0.2
        
        if dxy > 108.0: bias -= 0.3
        elif dxy > 104.0: bias -= 0.1
        elif dxy < 98.0: bias += 0.2
        
        if vix > 30.0: bias -= 0.4
        elif vix > 22.0: bias -= 0.2
        elif vix < 15.0: bias += 0.1
        
        bias = max(-1.0, min(1.0, bias))
        risk_off = bias < -0.3
        return bias, risk_off
