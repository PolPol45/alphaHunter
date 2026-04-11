"""
Risk Agent — Chief Risk Officer (Phase 4.5)
=============================================
Unifies signals from Bull, Bear, and Crypto strategies.
Evaluates portfolio correlation, macro limits, and sub-portfolio capital allocation.
Generates `validated_signals.json` for Phase 5 (ExecutionAgent) partitioned by 
'retail' (Personal) and 'institutional' execution modes.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone, timedelta

from agents.base_agent import BaseAgent, DATA_DIR

class RiskAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("risk_agent")
        self._port_cfg = self.config.get("portfolios", {})
        self._retail_cfg = self._port_cfg.get("personal", {})
        self._inst_cfg = self._port_cfg.get("institutional", {})
        self._risk_cfg = self.config.get("risk_agent", {})

        # PR2 defaults (used when config keys are missing)
        self._corr_threshold = float(self._risk_cfg.get("correlation_threshold", 0.85))
        self._covariance_penalty = float(self._risk_cfg.get("covariance_penalty", 0.35))
        self._target_vol_daily = float(self._risk_cfg.get("target_vol_daily", 0.02))
        self._min_vol_floor = float(self._risk_cfg.get("min_vol_floor", 0.003))
        self._max_asset_exposure_pct = float(self._risk_cfg.get("max_asset_exposure_pct", 0.08))
        self._max_sub_exposure_pct = float(self._risk_cfg.get("max_sub_exposure_pct", 0.45))
        self._min_score_by_bucket = self._risk_cfg.get(
            "min_score_by_bucket",
            {
                "crypto": 0.58,
                "bull": 0.62,
                "bear_hedge": 0.55,
                "bear_short": 0.68,
                "bear_bankrupt": 0.66,
            },
        )
        self._market_vol_extreme_vix = float(self._risk_cfg.get("market_vol_extreme_vix", 28.0))
        self._rolling_window_days = int(self._risk_cfg.get("strategy_weight_window_days", 21))
        self._strategy_weight_min = float(self._risk_cfg.get("strategy_weight_min", 0.75))
        self._strategy_weight_max = float(self._risk_cfg.get("strategy_weight_max", 1.25))

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
            returns_map = self._build_returns_map(mkt_data)

            # 3. Macro & Correlation Filter rules
            fed = macro_doc.get("series", {}).get("fed_funds", {}).get("value", 4) or 4
            vix = macro_doc.get("series", {}).get("vix", {}).get("value", 15) or 15
            dxy = macro_doc.get("series", {}).get("dxy", {}).get("value", 100) or 100
            market_bias = float(macro_doc.get("market_bias", 0.0) or 0.0)
            
            risk_off = vix > 25 or (fed > 5.0 and dxy > 105)
            market_vol_extreme = vix >= self._market_vol_extreme_vix

            # 4. Generate structured retail & institutional output
            retail_validated = self._process_mode(
                "retail", self._retail_cfg, port_retail, raw_bull_signals, raw_bear_signals, raw_crypto_signals,
                risk_off, market_bias, market_vol_extreme, mkt_data, returns_map
            )
            
            inst_validated = self._process_mode(
                "institutional", self._inst_cfg, port_inst, raw_bull_signals, raw_bear_signals, raw_crypto_signals,
                risk_off, market_bias, market_vol_extreme, mkt_data, returns_map
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
        mode = str(self.config.get("bear_strategy", {}).get("mode", "full")).strip().lower()
        hedge_only = mode in {"bear_hedge_only", "hedge_only"}

        if hedge_only:
            out = alloc.get("hedge_etfs", [])[:5]
            for it in out:
                it["_agent_source"] = "bear_hedge"
                it["_signal_type"] = "BUY"
            return out

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

    def _process_mode(
        self,
        mode_name: str,
        config: dict,
        portfolio: dict,
        bull: list,
        bear: list,
        crypto: list,
        risk_off: bool,
        market_bias: float,
        market_vol_extreme: bool,
        mkt_data: dict,
        returns_map: dict,
    ) -> dict:
        validated = {}
        subs = config.get("sub", {})
        total_cash = portfolio.get("cash", 0.0)
        mode_equity = float(portfolio.get("total_equity") or portfolio.get("initial_capital") or total_cash or 0.0)
        selected_symbols = []
        sub_exposure = {"crypto": 0.0, "bull": 0.0, "bear": 0.0}

        # Helper inner function to process a list of candidates
        def process_candidates(candidates, sub_name, max_picks=3, is_bear=False):
            sub_cfg = subs.get(sub_name, {})
            if not sub_cfg.get("active"):
                return

            cap = sub_cfg.get("capital", 0.0)
            # Base budget for this bucket (reserve preserved)
            base_stake = (cap * 0.8) / max_picks
            base_stake *= self._strategy_weight_multiplier(sub_name, portfolio)

            # If total_cash is running low, truncate the global stake size (e.g. system is fully invested)
            if base_stake > total_cash:
                base_stake = max(0, total_cash * 0.1) # Take only 10% of what's left
            if base_stake < 10.0:
                return # Insufficient funds

            # Noise reduction: sort by quality descending
            candidates_sorted = sorted(
                candidates,
                key=lambda x: float(x.get("composite_score", 0.0)),
                reverse=True,
            )

            count = 0
            for item in candidates_sorted:
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

                score = float(item.get("composite_score", 0.5) or 0.5)
                signal_type = item.get("_signal_type", "BUY")
                source = item.get("_agent_source", sub_name)

                # Signal noise filter (bucket-specific minimum)
                score_floor = float(self._min_score_by_bucket.get(source, self._min_score_by_bucket.get(sub_name, 0.55)))
                if score < score_floor:
                    continue

                # Macro penalty
                if risk_off and not is_bear and score < 0.70:
                    continue # Reject weak longs during risk off

                # Regime filter: only trade pro-risk longs in positive bias and non-extreme vol
                if signal_type == "BUY" and source in {"bull", "crypto"}:
                    if market_bias <= 0.0 or market_vol_extreme:
                        continue

                # Bear short filter: only in true risk-off
                if source in {"bear_short", "bear_bankrupt"} and signal_type == "SELL" and not risk_off:
                    continue

                # Multi-timeframe confirmation (1d + 4h trend alignment)
                if not self._multi_tf_confirm(sym, signal_type, mkt_data):
                    continue

                # Correlation clustering / covariance cap gate
                if not self._passes_correlation_gate(sym, selected_symbols, returns_map):
                    continue

                # Volatility targeting + covariance penalty on size
                vol_pct = self._estimate_volatility_pct(sym, mkt_data)
                corr_penalty = self._correlation_penalty(sym, selected_symbols, returns_map)
                sizing_mult = self._sizing_multiplier(score, vol_pct, corr_penalty)
                stake_size = base_stake * sizing_mult

                # Max exposure per asset (global) and per sub-bucket
                max_asset_abs = mode_equity * self._max_asset_exposure_pct
                if signal_type == "BUY":
                    stake_size = min(stake_size, max_asset_abs)
                sub_cap_limit = cap * self._max_sub_exposure_pct
                allowed_sub_remaining = max(0.0, sub_cap_limit - sub_exposure.get(sub_name, 0.0))
                stake_size = min(stake_size, allowed_sub_remaining)
                stake_size = min(stake_size, total_cash)
                if stake_size < 10.0:
                    continue

                quantity = round(stake_size / price, 6)
                if quantity <= 0:
                    continue

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
                    "agent_source": source,
                    "strategy_bucket": (
                        "crypto"
                        if sub_name == "crypto"
                        else ("bull" if sub_name == "bull" else "bear")
                    ),
                    "volatility_pct": round(vol_pct, 6),
                    "corr_penalty": round(corr_penalty, 6),
                }
                selected_symbols.append(sym)
                sub_exposure[sub_name] = sub_exposure.get(sub_name, 0.0) + stake_size
                count += 1

        # Process each group matching the subconfig
        process_candidates(crypto, "crypto", max_picks=5, is_bear=False)
        process_candidates(bull, "bull", max_picks=4, is_bear=False)
        process_candidates(bear, "bear", max_picks=5, is_bear=True)

        return validated

    def _strategy_weight_multiplier(self, sub_name: str, portfolio: dict) -> float:
        trades = portfolio.get("trades", []) or []
        if not trades:
            return 1.0
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._rolling_window_days)
        rows = []
        for t in trades:
            bucket = str(t.get("strategy_bucket", "")).lower()
            if bucket != sub_name:
                continue
            ts = t.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt < cutoff:
                continue
            rows.append(t)
        if len(rows) < 8:
            return 1.0

        realized = [float(t.get("realized_pnl", 0.0)) for t in rows if t.get("realized_pnl") is not None]
        if not realized:
            return 1.0
        gross = sum(abs(float(t.get("notional", 0.0))) for t in rows)
        if gross <= 0:
            return 1.0
        edge = sum(realized) / gross  # normalized pnl
        # map edge roughly into multiplier band
        raw = 1.0 + (edge * 10.0)
        return max(self._strategy_weight_min, min(self._strategy_weight_max, raw))

    @staticmethod
    def _multi_tf_confirm(symbol: str, signal_type: str, mkt_data: dict) -> bool:
        asset = (mkt_data.get("assets", {}) or {}).get(symbol, {})
        d1 = asset.get("ohlcv_1d", [])[-6:]
        h4 = asset.get("ohlcv_4h", [])[-12:]
        if len(d1) < 3 or len(h4) < 6:
            return True

        d1_first = float(d1[0].get("c", 0.0))
        d1_last = float(d1[-1].get("c", 0.0))
        h4_first = float(h4[0].get("c", 0.0))
        h4_last = float(h4[-1].get("c", 0.0))
        if min(d1_first, d1_last, h4_first, h4_last) <= 0:
            return True

        d1_up = d1_last > d1_first
        h4_up = h4_last > h4_first
        d1_down = d1_last < d1_first
        h4_down = h4_last < h4_first

        if signal_type == "BUY":
            return d1_up and h4_up
        if signal_type == "SELL":
            return d1_down and h4_down
        return True

    @staticmethod
    def _rolling_returns(closes: list[float]) -> list[float]:
        out = []
        for i in range(1, len(closes)):
            prev = float(closes[i - 1])
            cur = float(closes[i])
            if prev > 0:
                out.append((cur / prev) - 1.0)
        return out

    def _build_returns_map(self, mkt_data: dict) -> dict:
        out = {}
        assets = mkt_data.get("assets", {})
        for sym, info in assets.items():
            candles = info.get("ohlcv_4h") or info.get("ohlcv_1d") or []
            closes = [float(c.get("c", 0.0)) for c in candles[-80:] if c.get("c")]
            if len(closes) >= 20:
                out[sym] = self._rolling_returns(closes)
        return out

    def _passes_correlation_gate(self, symbol: str, selected_symbols: list[str], returns_map: dict) -> bool:
        if not selected_symbols:
            return True
        cand = returns_map.get(symbol)
        if not cand:
            return True
        for peer in selected_symbols:
            p = returns_map.get(peer)
            if not p:
                continue
            corr = self._pearson_corr(cand, p)
            if corr is not None and abs(corr) >= self._corr_threshold:
                return False
        return True

    def _correlation_penalty(self, symbol: str, selected_symbols: list[str], returns_map: dict) -> float:
        if not selected_symbols:
            return 0.0
        cand = returns_map.get(symbol)
        if not cand:
            return 0.0
        corrs = []
        for peer in selected_symbols:
            p = returns_map.get(peer)
            if not p:
                continue
            corr = self._pearson_corr(cand, p)
            if corr is not None:
                corrs.append(abs(corr))
        if not corrs:
            return 0.0
        return sum(corrs) / len(corrs)

    def _estimate_volatility_pct(self, symbol: str, mkt_data: dict) -> float:
        info = mkt_data.get("assets", {}).get(symbol, {})
        candles = info.get("ohlcv_4h") or info.get("ohlcv_1d") or []
        closes = [float(c.get("c", 0.0)) for c in candles[-80:] if c.get("c")]
        rets = self._rolling_returns(closes)
        if len(rets) < 5:
            return self._target_vol_daily
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
        vol = math.sqrt(max(var, 0.0))
        return max(vol, self._min_vol_floor)

    def _sizing_multiplier(self, score: float, vol_pct: float, corr_penalty: float) -> float:
        # score in [0,1] → multiplier in [0.7, 1.3]
        score_mult = 0.7 + max(0.0, min(1.0, score)) * 0.6
        # target-vol sizing, clipped
        vol_mult_raw = self._target_vol_daily / max(vol_pct, self._min_vol_floor)
        vol_mult = max(0.5, min(1.5, vol_mult_raw))
        # covariance/correlation penalty
        corr_mult = 1.0 - min(0.9, corr_penalty * self._covariance_penalty)
        return max(0.35, min(1.6, score_mult * vol_mult * corr_mult))

    @staticmethod
    def _pearson_corr(a: list[float], b: list[float]) -> float | None:
        n = min(len(a), len(b))
        if n < 5:
            return None
        x = a[-n:]
        y = b[-n:]
        mx = sum(x) / n
        my = sum(y) / n
        vx = sum((i - mx) ** 2 for i in x)
        vy = sum((j - my) ** 2 for j in y)
        if vx <= 0 or vy <= 0:
            return None
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        return cov / math.sqrt(vx * vy)
