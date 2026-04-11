"""
Risk Agent — Chief Risk Officer (Phase 4.5)
=============================================
Unifies signals from Bull, Bear, and Crypto strategies.
Evaluates portfolio correlation, macro limits, and sub-portfolio capital allocation.
Generates `validated_signals.json` for Phase 5 (ExecutionAgent) partitioned by 
'retail' (Personal) and 'institutional' execution modes.
"""

from __future__ import annotations
from datetime import datetime, timezone

from agents.base_agent import BaseAgent, DATA_DIR

class RiskAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("risk_agent")
        self._port_cfg = self.config.get("portfolios", {})
        self._retail_cfg = self._port_cfg.get("personal", {})
        self._inst_cfg = self._port_cfg.get("institutional", {})

    def run(self) -> bool:
        self.mark_running()
        try:
            # 1. Read input signals
            bull_sig = self.read_json(DATA_DIR / "bull_signals.json") or {}
            bear_sig = self.read_json(DATA_DIR / "bear_signals.json") or {}
            crypto_sig = self.read_json(DATA_DIR / "crypto_signals.json") or {}
            macro_doc = self.read_json(DATA_DIR / "macro_snapshot.json") or {}
            mkt_data = self.read_json(DATA_DIR / "market_data.json") or {}
            
            # Read portfolios state
            port_retail = self.read_json(DATA_DIR / "portfolio_retail.json") or {}
            port_inst   = self.read_json(DATA_DIR / "portfolio_institutional.json") or {}
            
            # 2. Extract standard flat signals
            raw_bull_signals = self._extract_bull(bull_sig)
            raw_bear_signals = self._extract_bear(bear_sig)
            raw_crypto_signals = self._extract_crypto(crypto_sig)

            # 3. Macro & Correlation Filter rules
            fed = macro_doc.get("series", {}).get("fed_funds", {}).get("value", 4) or 4
            vix = macro_doc.get("series", {}).get("vix", {}).get("value", 15) or 15
            dxy = macro_doc.get("series", {}).get("dxy", {}).get("value", 100) or 100
            
            risk_off = vix > 25 or (fed > 5.0 and dxy > 105)

            # 4. Generate structured retail & institutional output
            retail_validated = self._process_mode(
                "retail", self._retail_cfg, port_retail, raw_bull_signals, raw_bear_signals, raw_crypto_signals, risk_off, mkt_data
            )
            
            inst_validated = self._process_mode(
                "institutional", self._inst_cfg, port_inst, raw_bull_signals, raw_bear_signals, raw_crypto_signals, risk_off, mkt_data
            )

            validated = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retail": retail_validated,
                "institutional": inst_validated,
            }
            
            self.write_json(DATA_DIR / "validated_signals.json", validated)
            self.update_shared_state("data_freshness.validated_signals", validated["timestamp"])
            
            r_app = [k for k, v in retail_validated.items() if v["approved"]]
            i_app = [k for k, v in inst_validated.items() if v["approved"]]
            self.logger.info(f"Risk mapping complete. Approved Retail: {len(r_app)} | Approved Inst: {len(i_app)}")
            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    def _extract_bull(self, bull_sig: dict) -> list:
        alloc = bull_sig.get("allocations", {})
        # Flatten and take top picks across categories
        out = alloc.get("etf_50_pct", [])[:3] + alloc.get("large_cap_30_pct", [])[:3] + alloc.get("small_cap_20_pct", [])[:2]
        for it in out:
            it["_agent_source"] = "bull"
            it["_signal_type"] = "BUY"
        return out

    def _extract_bear(self, bear_sig: dict) -> list:
        alloc = bear_sig.get("allocations", {})
        out = alloc.get("hedge_etfs", [])[:3] + alloc.get("short_candidates", [])[:3]
        for it in out:
            it["_agent_source"] = "bear"
            # Hedge ETFs are usually bought long to inverse the market
            it["_signal_type"] = "BUY" if "hedge_count" not in it else "SELL"
            # We fix logic based on config: etf_hedge is in config. 
            # We'll just define them as BUY if they are hedge ETFs, otherwise SELL for short Candidates.
            it["_signal_type"] = "BUY" # default fallback
        
        # Proper assignment:
        for it in alloc.get("hedge_etfs", [])[:3]:
            it["_agent_source"] = "bear_hedge"
            it["_signal_type"] = "BUY" # We buy the inverse ETF
            
        for it in alloc.get("short_candidates", [])[:3]:
            it["_agent_source"] = "bear_short"
            it["_signal_type"] = "SELL" # We short the equity
            
        for it in alloc.get("bankruptcy_risk", [])[:2]:
            it["_agent_source"] = "bear_bankrupt"
            it["_signal_type"] = "SELL"
            
        # Re-flatten with proper signal types
        out = alloc.get("hedge_etfs", [])[:3] + alloc.get("short_candidates", [])[:3] + alloc.get("bankruptcy_risk", [])[:2]
        return out

    def _extract_crypto(self, crypto_sig: dict) -> list:
        allocs = crypto_sig.get("allocations", {})
        out = []
        
        # Flatten all crypto categories
        raw = allocs.get("core_50pct", []) + allocs.get("defi_bridge_30pct", []) + allocs.get("alt_meme_20pct", [])
        
        for it in raw[:5]: # Max 5 for crypto subset per sub-portfolio
            n = {
                "symbol": it.get("symbol"),
                "composite_score": it.get("score_final", it.get("score")),
                "_agent_source": "crypto",
                "_signal_type": "BUY" if it.get("direction", "LONG") == "LONG" else "SELL"
            }
            out.append(n)
        return out

    def _process_mode(self, mode_name: str, config: dict, portfolio: dict, bull: list, bear: list, crypto: list, risk_off: bool, mkt_data: dict) -> dict:
        validated = {}
        subs = config.get("sub", {})
        total_cash = portfolio.get("cash", 0.0)
        
        # Helper inner function to process a list of candidates
        def process_candidates(candidates, sub_name, max_picks=3, is_bear=False):
            sub_cfg = subs.get(sub_name, {})
            if not sub_cfg.get("active"):
                return
            
            cap = sub_cfg.get("capital", 0.0)
            stake_size = (cap * 0.8) / max_picks # 80% allocation, 20% reserve
            
            # If total_cash is running low, truncate the global stake size (e.g. system is fully invested)
            if stake_size > total_cash:
                stake_size = max(0, total_cash * 0.1) # Take only 10% of what's left
            if stake_size < 10.0:
                return # Insufficient funds
                
            count = 0
            for item in candidates:
                if count >= max_picks: break
                sym = item.get("symbol")
                if not sym or sym in validated: continue
                
                # Check price
                price = mkt_data.get("assets", {}).get(sym, {}).get("last_price", 0.0)
                if not price or price <= 0:
                    try:
                        import yfinance as yf
                        tk = yf.Ticker(sym)
                        # fastest way to get current price
                        fast_price = tk.fast_info.last_price
                        if fast_price is None or str(fast_price).lower() == 'nan':
                            df = tk.history(period="1d")
                            if not df.empty:
                                price = float(df['Close'].iloc[-1])
                        else:
                            price = float(fast_price)
                    except Exception as e:
                        pass
                        
                if not price or price <= 0:
                    continue # No price, cannot calculate quantity
                
                score = item.get("composite_score", 0.5)
                signal_type = item.get("_signal_type", "BUY")
                
                # Macro penalty
                if risk_off and not is_bear and score < 0.70:
                    continue # Reject weak longs during risk off
                    
                quantity = round(stake_size / price, 6)
                
                # Stop loss / Take profit estimation
                sl = price * 0.95 if signal_type == "BUY" else price * 1.05
                tp = price * 1.08 if signal_type == "BUY" else price * 0.92
                
                validated[sym] = {
                    "approved": True,
                    "signal_type": signal_type,
                    "score": round(score, 3),
                    "entry_price": round(price, 4),
                    "stop_loss_price": round(sl, 4),
                    "take_profit_price": round(tp, 4),
                    "position_size_usdt": round(stake_size, 4),
                    "quantity": quantity,
                    "rejection_reason": None,
                    "agent_source": item.get("_agent_source", sub_name)
                }
                count += 1
                
        # Process each group matching the subconfig
        process_candidates(crypto, "crypto", max_picks=5, is_bear=False)
        process_candidates(bull, "bull", max_picks=4, is_bear=False)
        process_candidates(bear, "bear", max_picks=5, is_bear=True)
        
        return validated
