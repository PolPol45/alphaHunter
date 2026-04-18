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

        # Defaults statici (sovrascrivibili da adaptive_params.json ogni run)
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
                "ta_crypto": 0.65,   # TA signals need higher bar — no fundamental filter
                "ta_retail": 0.65,
            },
        )
        self._market_vol_extreme_vix = float(self._risk_cfg.get("market_vol_extreme_vix", 28.0))
        self._rolling_window_days = int(self._risk_cfg.get("strategy_weight_window_days", 21))
        self._strategy_weight_min = float(self._risk_cfg.get("strategy_weight_min", 0.75))
        self._strategy_weight_max = float(self._risk_cfg.get("strategy_weight_max", 1.25))
        # Pesi strategia da AdaptiveLearner (default 1.0 = neutro)
        self._adaptive_strategy_weights: dict = {"bull": 1.0, "bear": 1.0, "crypto": 1.0, "ta": 1.0}

    def _load_adaptive_params(self) -> None:
        """Sovrascrive parametri di rischio con quelli appresi dall'AdaptiveLearner."""
        ap = self.read_json(DATA_DIR / "adaptive_params.json") or {}
        if not ap:
            return
        if "min_score_by_bucket" in ap:
            self._min_score_by_bucket = ap["min_score_by_bucket"]
        if "correlation_threshold" in ap:
            self._corr_threshold = float(ap["correlation_threshold"])
        if "covariance_penalty" in ap:
            self._covariance_penalty = float(ap["covariance_penalty"])
        if "target_vol_daily" in ap:
            self._target_vol_daily = float(ap["target_vol_daily"])
        if "max_asset_exposure_pct" in ap:
            self._max_asset_exposure_pct = float(ap["max_asset_exposure_pct"])
        if "strategy_weights" in ap:
            self._adaptive_strategy_weights = ap["strategy_weights"]

    def run(self) -> bool:
        self.mark_running()
        try:
            # 0. Carica parametri adattativi dall'AdaptiveLearner (sovrascrive defaults)
            self._load_adaptive_params()

            # 1. Read input signals
            bull_sig    = self.read_json(DATA_DIR / "bull_signals.json") or {}
            bear_sig    = self.read_json(DATA_DIR / "bear_signals.json") or {}
            crypto_sig  = self.read_json(DATA_DIR / "crypto_signals.json") or {}
            ta_sig      = self.read_json(DATA_DIR / "signals.json") or {}
            macro_doc    = self.read_json(DATA_DIR / "macro_snapshot.json") or {}
            market_regime = self.read_json(DATA_DIR / "market_regime.json") or {}
            mkt_data    = self.read_json(DATA_DIR / "market_data.json") or {}
            ml_signals  = self.read_json(DATA_DIR / "ml_signals.json") or {}
            news_doc    = self.read_json(DATA_DIR / "news_feed.json") or {}
            sector_doc  = self.read_json(DATA_DIR / "sector_scorecard.json") or {}

            # Read portfolios state
            port_retail = self.read_json(DATA_DIR / "portfolio_retail.json") or {}
            port_inst   = self.read_json(DATA_DIR / "portfolio_institutional.json") or {}

            # 2. Extract standard flat signals
            raw_bull_signals   = self._extract_bull(bull_sig)
            raw_bear_signals   = self._extract_bear(bear_sig)
            raw_crypto_signals = self._extract_crypto(crypto_sig)
            raw_ta_signals     = self._extract_ta(ta_sig)
            returns_map        = self._build_returns_map(mkt_data)

            # 2b. Apply ML score boost where available (skip if stale)
            _ml_max_stale = int(self._risk_cfg.get("ml_max_staleness_seconds", 10800))  # 3h default
            _ml_generated = ml_signals.get("generated_at") or ml_signals.get("updated_at")
            _ml_fresh = False
            if _ml_generated:
                try:
                    from datetime import datetime, timezone
                    _ml_age = (datetime.now(timezone.utc) - datetime.fromisoformat(_ml_generated)).total_seconds()
                    _ml_fresh = _ml_age <= _ml_max_stale
                    if not _ml_fresh:
                        self.logger.warning(f"ML signals stale ({_ml_age/3600:.1f}h > {_ml_max_stale/3600:.1f}h) — boost skipped")
                except Exception:
                    pass
            ml_boost_map = self._build_ml_boost_map(ml_signals) if _ml_fresh else {}
            if ml_boost_map:
                self._apply_ml_boost(raw_bull_signals, ml_boost_map)
                self._apply_ml_boost(raw_bear_signals, ml_boost_map)
                self._apply_ml_boost(raw_crypto_signals, ml_boost_map)
                self._apply_ml_boost(raw_ta_signals, ml_boost_map)
                self.logger.info("ML boost applied to %d symbols", len(ml_boost_map))

            # 2c. Apply news sentiment boost
            news_boost_map = self._build_news_boost_map(news_doc)
            if news_boost_map:
                self._apply_ml_boost(raw_bull_signals, news_boost_map)
                self._apply_ml_boost(raw_bear_signals, news_boost_map)
                self._apply_ml_boost(raw_ta_signals, news_boost_map)

            # 2d. Apply sector momentum boost
            sector_boost_map = self._build_sector_boost_map(sector_doc)
            if sector_boost_map:
                self._apply_ml_boost(raw_bull_signals, sector_boost_map)

            # 3. Macro & Correlation Filter rules
            # Single source of truth: market_regime.json (issue #176)
            # risk_off and market_vol_extreme both derived from market_regime.json
            # to prevent dual-source disagreement (e.g. VIX=24 but score-based RISK_OFF)
            regime_label      = market_regime.get("regime", "NEUTRAL")
            regime_confidence = market_regime.get("confidence", "LOW")
            regime_vix        = float(
                (market_regime.get("macro_factors") or {}).get("vix") or 15
            )

            risk_off           = (regime_label == "RISK_OFF")
            market_vol_extreme = regime_vix >= self._market_vol_extreme_vix

            # market_bias kept from macro_snapshot for granular bull/crypto filters
            # (continuous score used for threshold comparisons, not binary gate)
            market_bias = float(macro_doc.get("market_bias", 0.0) or 0.0)

            if risk_off:
                self.logger.warning(
                    f"Regime=RISK_OFF (confidence={regime_confidence}, VIX={regime_vix:.1f}) "
                    "— weak BUY signals will be blocked"
                )

            # 4. Generate structured retail & institutional output
            retail_enabled = bool(self._retail_cfg.get("enabled", True))
            retail_validated = self._process_mode(
                "retail", self._retail_cfg, port_retail, raw_bull_signals, raw_bear_signals, raw_crypto_signals,
                risk_off, market_bias, market_vol_extreme, mkt_data, returns_map, raw_ta_signals,
                regime_label=regime_label,
                regime_vix=regime_vix,
            ) if retail_enabled else {}

            inst_validated = self._process_mode(
                "institutional", self._inst_cfg, port_inst, raw_bull_signals, raw_bear_signals, raw_crypto_signals,
                risk_off, market_bias, market_vol_extreme, mkt_data, returns_map, raw_ta_signals,
                regime_label=regime_label,
                regime_vix=regime_vix,
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

    def _extract_ta(self, ta_sig: dict) -> list:
        """Convert signals.json retail BUY/SELL entries into the common candidate format.

        Uses the top-ranked scanner candidates (pre-sorted by TA score) to cap the
        list size and avoid flooding the risk pipeline with 50 identical-looking
        crypto signals. Falls back to scanning all symbols if scanner key is absent.
        """
        out = []
        top_candidates = ta_sig.get("scanner", {}).get("retail_top_candidates", [])
        if not top_candidates:
            # Fallback: pull directly from retail dict, BUY only, sorted by score
            retail = ta_sig.get("retail", {})
            if not isinstance(retail, dict):
                return out
            top_candidates = sorted(
                [
                    {"symbol": sym, "signal_type": sig.get("signal_type"), "score": sig.get("score", 0.0)}
                    for sym, sig in retail.items()
                    if sig.get("signal_type") in {"BUY", "SELL"}
                ],
                key=lambda x: x.get("score", 0.0),
                reverse=True,
            )[:10]

        retail_signals = ta_sig.get("retail", {})
        for row in top_candidates[:10]:
            sym = row.get("symbol")
            if not sym:
                continue
            sig = retail_signals.get(sym, {})
            signal_type = sig.get("signal_type") or row.get("signal_type", "BUY")
            if signal_type not in {"BUY", "SELL"}:
                continue
            score = float(sig.get("score") or row.get("score") or 0.0)
            out.append({
                "symbol": sym,
                "composite_score": score,
                "_agent_source": "ta_crypto",
                "_signal_type": signal_type,
                # Pass TA-computed SL/TP so RiskAgent can use them directly
                "_ta_stop_loss": sig.get("stop_loss"),
                "_ta_take_profit": sig.get("take_profit"),
                "_ta_atr": sig.get("atr"),
                # Pass last_price so price lookup doesn't fail for non-OHLCV symbols
                "_ta_last_price": sig.get("last_price") or row.get("last_price"),
                # Pass noise bands computed by TechnicalAnalysisAgent (ISSUE-002)
                "_noise_bands": sig.get("noise_bands"),
            })
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
        ta: list | None = None,
        regime_label: str = "NEUTRAL",
        regime_vix: float = 15.0,
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
            # Applica peso adattivo da AdaptiveLearner (learned_strategy_weights)
            adaptive_key = "crypto" if "crypto" in sub_name else ("bear" if "bear" in sub_name else "bull")
            base_stake *= float(self._adaptive_strategy_weights.get(adaptive_key, 1.0))

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
                
                # Check price — priority: mkt_data → TA embedded price → yfinance
                price = mkt_data.get("assets", {}).get(sym, {}).get("last_price", 0.0)
                if not price or price <= 0:
                    ta_price = item.get("_ta_last_price")
                    if ta_price:
                        price = float(ta_price)
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

                # Regime filter: bull blocked only in clear risk-off, crypto more permissive
                if signal_type == "BUY" and source == "bull":
                    if market_bias < -0.15 or market_vol_extreme:
                        continue
                if signal_type == "BUY" and source == "crypto":
                    if market_bias < -0.25 or market_vol_extreme:
                        continue

                # Bear hedge always allowed; bear short/bankrupt only in risk-off
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
                # VIX-regime multiplier (ISSUE-003 — Beat-the-Market §4.1)
                sizing_mult *= self._vix_size_multiplier(regime_vix)
                # Stock-tier multiplier — blue chip gets larger size, small cap smaller
                tier_size_mult = float(item.get("_tier_size_mult", 1.0))
                sizing_mult *= tier_size_mult
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

                # Noise Area Dynamic Breakout Filter (ISSUE-002)
                # Skip for bear/hedge signals — they have different logic
                if not is_bear and not source.startswith("bear"):
                    noise_bands = item.get("_noise_bands")
                    if not self._noise_area_check(sym, signal_type, price, noise_bands):
                        continue

                # Stop loss / Take profit: prefer TA-computed values when present
                ta_sl = item.get("_ta_stop_loss")
                ta_tp = item.get("_ta_take_profit")
                ta_atr = item.get("_ta_atr")
                if ta_sl and ta_tp and ta_atr:
                    sl = float(ta_sl)
                    tp = float(ta_tp)
                    atr_daily = float(ta_atr)
                else:
                    # Fallback: ATR-based, modulated by stock tier
                    # BLUE_CHIP: sl=1.8×, tp=2.5× | LARGE_CAP: sl=1.5×, tp=2.2×
                    # SMALL_CAP: sl=1.2×, tp=1.8× — tighter because higher base vol
                    tier_sl_mult = float(item.get("_tier_sl_mult", 1.5))
                    tier_tp_mult = float(item.get("_tier_tp_mult", 2.2))
                    atr_daily = vol_pct * price
                    atr_daily = max(atr_daily, price * 0.005)
                    sl_dist = tier_sl_mult * atr_daily
                    tp_dist = tier_tp_mult * atr_daily
                    if signal_type == "BUY":
                        sl = price - sl_dist
                        tp = price + tp_dist
                    else:
                        sl = price + sl_dist
                        tp = price - tp_dist
                
                # Derived fields for issue #173 schema
                sl_pct = round(abs(price - sl) / price, 6) if price > 0 else 0.0
                pos_size_pct = round(stake_size / mode_equity, 6) if mode_equity > 0 else 0.0
                vol_20d_ann = round(vol_pct * (252 ** 0.5), 6)  # annualize daily vol
                max_corr = round(corr_penalty, 6)  # avg abs corr with portfolio

                asset_mkt = mkt_data.get("assets", {}).get(sym, {})
                validated[sym] = {
                    "approved": True,
                    "signal_type": signal_type,
                    "score": round(score, 3),
                    "entry_price": round(price, 4),
                    "stop_loss_price": round(sl, 4),
                    "stop_loss_pct": sl_pct,
                    "take_profit_price": round(tp, 4),
                    "position_size_usdt": round(stake_size, 4),
                    "position_size_pct": pos_size_pct,
                    "quantity": quantity,
                    "rejection_reason": None,
                    "max_corr_with_portfolio": max_corr,
                    "vol_20d_annualized": vol_20d_ann,
                    "regime_at_validation": regime_label,
                    "agent_source": source,
                    "strategy_bucket": (
                        "crypto"
                        if sub_name == "crypto"
                        else ("bull" if sub_name == "bull" else "bear")
                    ),
                    "volatility_pct": round(vol_pct, 6),
                    "atr": round(atr_daily, 6),
                    "corr_penalty": round(corr_penalty, 6),
                    "vwap": asset_mkt.get("vwap"),
                    "noise_upper_band": (item.get("_noise_bands") or {}).get("upper"),
                    "noise_lower_band": (item.get("_noise_bands") or {}).get("lower"),
                }
                selected_symbols.append(sym)
                sub_exposure[sub_name] = sub_exposure.get(sub_name, 0.0) + stake_size
                count += 1

        # Process each group matching the subconfig
        process_candidates(crypto, "crypto", max_picks=5, is_bear=False)
        process_candidates(bull, "bull", max_picks=4, is_bear=False)
        process_candidates(bear, "bear", max_picks=5, is_bear=True)
        # TA signals: use "crypto" sub-portfolio capital for crypto symbols,
        # capped at 3 additional picks so they don't crowd out specialist strategies
        if ta:
            process_candidates(ta, "crypto", max_picks=3, is_bear=False)

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

    @staticmethod
    def _vix_size_multiplier(vix: float) -> float:
        """Scale position size with VIX regime (ISSUE-003 — Beat-the-Market §4.1).
        High VIX → stronger momentum → larger size. Low VIX → weak momentum → smaller.
        Paper: Sharpe 1.5 at VIX>6, 3.5 at VIX>40. Capped to avoid extreme leverage."""
        if vix < 6:
            return 0.50   # very low vol — momentum signals weak
        elif vix < 12:
            return 0.75
        elif vix < 20:
            return 1.00   # baseline
        elif vix < 30:
            return 1.20
        elif vix < 40:
            return 1.35
        else:
            return 1.50   # extreme vol — strong momentum but capped

    def _sizing_multiplier(self, score: float, vol_pct: float, corr_penalty: float) -> float:
        # score in [0,1] → multiplier in [0.7, 1.3]
        score_mult = 0.7 + max(0.0, min(1.0, score)) * 0.6
        # target-vol sizing, clipped
        vol_mult_raw = self._target_vol_daily / max(vol_pct, self._min_vol_floor)
        vol_mult = max(0.5, min(1.5, vol_mult_raw))
        # covariance/correlation penalty — stronger in enforce mode
        if self._corr_gate_is_active():
            # enforce: full penalty weight applied
            corr_mult = 1.0 - min(0.9, corr_penalty * self._covariance_penalty)
        else:
            # off: reduced penalty (half weight) so low_corr > high_corr still holds
            corr_mult = 1.0 - min(0.45, corr_penalty * self._covariance_penalty * 0.5)
        return max(0.35, min(1.6, score_mult * vol_mult * corr_mult))

    def _build_ml_boost_map(self, ml_signals: dict) -> dict[str, float]:
        """
        Builds a symbol → score_delta map from the latest ML ranking.
        Top-decile longs get a positive boost, bottom-decile shorts get a negative boost.
        Boost is capped at score_boost_cap (default 0.12) from config.
        """
        cap = float(self.config.get("ml_strategy", {}).get("score_boost_cap", 0.12))
        ranking = ml_signals.get("latest_ranking") or {}
        boost: dict[str, float] = {}

        for entry in ranking.get("top_decile_long", []):
            sym = entry.get("symbol")
            pred = float(entry.get("predicted_excess_return", 0.0))
            if sym:
                boost[sym] = min(cap, max(0.0, pred * 2.0))

        for entry in ranking.get("bottom_decile_short", []):
            sym = entry.get("symbol")
            pred = float(entry.get("predicted_excess_return", 0.0))
            if sym:
                boost[sym] = max(-cap, min(0.0, pred * 2.0))

        return boost

    @staticmethod
    def _apply_ml_boost(signals: list[dict], boost_map: dict[str, float]) -> None:
        """Adds ML predicted excess return as a score adjustment (in-place)."""
        for item in signals:
            sym = item.get("symbol")
            if not sym or sym not in boost_map:
                continue
            delta = boost_map[sym]
            current = float(item.get("composite_score", 0.5) or 0.5)
            item["composite_score"] = round(min(1.0, max(0.0, current + delta)), 4)
            item["ml_boost"] = round(delta, 4)

    def _build_news_boost_map(self, news_doc: dict) -> dict[str, float]:
        """Estrae sentiment da news_feed.json e lo converte in boost score per simbolo.
        Sentiment positivo → piccolo boost positivo (max ±0.05).
        """
        boost: dict[str, float] = {}
        articles = news_doc.get("articles", news_doc.get("items", []))
        if not articles:
            return boost
        sentiment_sum: dict[str, float] = {}
        sentiment_cnt: dict[str, int] = {}
        for art in articles:
            sym = art.get("symbol") or art.get("ticker")
            score = art.get("sentiment_score") or art.get("sentiment")
            if not sym or score is None:
                continue
            try:
                s = float(score)
            except (TypeError, ValueError):
                continue
            sentiment_sum[sym] = sentiment_sum.get(sym, 0.0) + s
            sentiment_cnt[sym] = sentiment_cnt.get(sym, 0) + 1
        cap = self.config.get("ml_strategy", {}).get("score_boost_cap", 0.05)
        for sym, total in sentiment_sum.items():
            n = sentiment_cnt[sym]
            avg = total / n
            # Normalizza in [-cap, +cap]
            boost[sym] = round(max(-cap, min(cap, avg * cap)), 5)
        return boost

    def _build_sector_boost_map(self, sector_doc: dict) -> dict[str, float]:
        """Usa sector_scorecard per boost proporzionale al momentum settoriale.
        sector_scorecard.json ha sectors come lista di {name, members, score, ...}
        """
        boost: dict[str, float] = {}
        sectors = sector_doc.get("sectors", [])
        if not sectors:
            return boost
        if isinstance(sectors, dict):
            sectors = list(sectors.values())
        for data in sectors:
            if not isinstance(data, dict):
                continue
            momentum = data.get("score") or data.get("momentum") or 0.0
            try:
                m = float(momentum)
            except (TypeError, ValueError):
                continue
            delta = round(max(-0.04, min(0.04, (m - 0.5) * 0.08)), 5)
            for sym in data.get("members", data.get("symbols", [])):
                boost[sym] = boost.get(sym, 0.0) + delta
        return boost

    # ── Noise Area Dynamic Breakout Filter (ISSUE-002) ────────────────────────

    def _noise_area_check(self, sym: str, signal_type: str, price: float, noise_bands: dict | None) -> bool:
        """Return True if price has broken outside noise area (valid entry).

        BUY valid only if price > bands["upper"].
        SELL valid only if price < bands["lower"].
        If no bands → return True (conservative fallback: allow).
        Logs blocked signals with reason "noise_gate".
        """
        if not noise_bands:
            return True

        if signal_type == "BUY" and price <= noise_bands["upper"]:
            self.logger.info(
                f"noise_gate blocked {sym}: BUY price {price:.4f} <= upper {noise_bands['upper']:.4f} "
                f"(sigma={noise_bands.get('sigma', '?')})"
            )
            return False

        if signal_type == "SELL" and price >= noise_bands["lower"]:
            self.logger.info(
                f"noise_gate blocked {sym}: SELL price {price:.4f} >= lower {noise_bands['lower']:.4f} "
                f"(sigma={noise_bands.get('sigma', '?')})"
            )
            return False

        return True

    # ── Bear rollout controls ──────────────────────────────────────────────────

    def _bear_short_enabled(self, risk_off: bool = False) -> bool:
        """Return True only when the bear rollout stage permits short selling AND risk_off is True."""
        cfg = getattr(self, "_bear_rollout_cfg", None) or self.config.get("bear_strategy", {}).get("rollout", {})
        stage = str(cfg.get("stage", "hedge_only")).strip().lower()
        if stage == "hedge_only":
            return False
        if stage == "cautious_short":
            return risk_off
        # full / unrestricted stage
        return True

    # ── Controls rollout ───────────────────────────────────────────────────────

    def _controls_mode_name(self) -> str:
        """Return the current controls rollout mode string."""
        cfg = getattr(self, "_controls_rollout_cfg", None) or self.config.get("controls_rollout", {})
        return str(cfg.get("mode", "off")).strip().lower()

    def _corr_gate_is_active(self) -> bool:
        """Return True when controls mode is 'enforce'."""
        mode = getattr(self, "_controls_mode", None)
        if mode is None:
            mode = self._controls_mode_name()
        return mode == "enforce"

    # ── ML helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rank_overlap_ratio(set_a: set, set_b: set) -> float:
        """Overlap coefficient: |intersection| / max(|A|, |B|)."""
        denom = max(len(set_a), len(set_b))
        if denom == 0:
            return 0.0
        return len(set_a & set_b) / denom

    def _extract_ml(self, payload: dict) -> list[dict]:
        """Extract ML trade candidates from an ml_signals payload.

        In 'informative_only' mode returns an empty list (telemetry only).
        In active modes (e.g. 'long_short_parallel') returns candidate dicts.
        """
        mode = str(payload.get("runtime_mode", "informative_only")).strip().lower()
        if mode == "informative_only":
            return []
        ranking = payload.get("latest_ranking", {})
        candidates: list[dict] = []
        for entry in ranking.get("top_decile_long", []):
            candidates.append({
                "symbol": entry.get("symbol"),
                "composite_score": float(entry.get("predicted_excess_return", 0.01)),
                "_agent_source": "ml_long",
                "_signal_type": "BUY",
                **entry,
            })
        for entry in ranking.get("bottom_decile_short", []):
            candidates.append({
                "symbol": entry.get("symbol"),
                "composite_score": abs(float(entry.get("predicted_excess_return", 0.01))),
                "_agent_source": "ml_short",
                "_signal_type": "SELL",
                **entry,
            })
        return candidates

    @staticmethod
    def _ml_telemetry(payload: dict) -> dict:
        """Return a telemetry-only dict summarising the ML payload (never traded)."""
        ranking = payload.get("latest_ranking", {})
        longs = ranking.get("top_decile_long", [])
        shorts = ranking.get("bottom_decile_short", [])
        return {
            "ingested_for_telemetry_only": True,
            "runtime_mode": payload.get("runtime_mode", "informative_only"),
            "latest_date": payload.get("latest_date"),
            "long_count": len(longs),
            "short_count": len(shorts),
            "long_symbols": [e.get("symbol") for e in longs],
            "short_symbols": [e.get("symbol") for e in shorts],
        }

    @staticmethod
    def _select_candidates(
        signals: dict,
        positions: dict,
        limit: int,
        threshold: float,
        allow_short: bool,
    ) -> list[str]:
        """Filter and rank signals by score, respecting allow_short and threshold.

        Symbols already held in positions are excluded.
        Returns an ordered list of approved symbol strings (highest score first),
        up to `limit` entries.
        """
        held = set(positions.keys())
        eligible = []
        for sym, sig in signals.items():
            if sym in held:
                continue
            signal_type = sig.get("signal_type", "BUY")
            if signal_type == "SELL" and not allow_short:
                continue
            score = float(sig.get("score", 0.0))
            if score < threshold:
                continue
            eligible.append((sym, score))
        eligible.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in eligible[:limit]]

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
