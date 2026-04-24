"""
Technical Analysis Agent — Dual Strategy
=========================================
Runs two independent signal engines per cycle:

  Retail (crypto_momentum_bot_v2 logic):
    - EMA cross (9/21) + EMA-50 daily trend filter  [FIX #8]
    - RSI-14 oversold=35 / overbought=65
    - MACD histogram cross
    - ADX-14 (trend strength amplifier)
    - Volume spike 1.5x MA-20
    - ROC-10 momentum
    → BUY if buy_score > 0.55 | SELL if sell_score > 0.55

  Institutional (tighter thresholds):
    - Same indicators
    - RSI oversold=40 / overbought=60
    → BUY/SELL threshold 0.60

Multi-timeframe:
  ohlcv_4h  — primary signal (EMA cross, RSI, MACD, ADX, Volume, ROC)
  ohlcv_1d  — daily trend confirmation (EMA-50 filter)

Reads:  data/market_data.json
Writes: data/signals.json  →  {"retail": {symbol: signal}, "institutional": {symbol: signal}}
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, DATA_DIR


class TechnicalAnalysisAgent(BaseAgent):
    PATTERN_MULTIPLIERS = {
        "NR4":            1.15,
        "NR7":            1.10,
        "TRIANGLE":       1.10,
        "ID":             1.03,
        "BIG_TAIL_BULL":  1.08,
        "BIG_TAIL_BEAR":  1.08,
        "STRONG_CLOSE":   1.08,
        "WEAK_CLOSE":     1.08,
        "TREND_DAY_BULL": 0.85,
        "TREND_DAY_BEAR": 0.85,
    }

    _DOW_MULTIPLIERS = {0: 0.88, 1: 0.95, 2: 1.05, 3: 1.07, 4: 1.05}

    def __init__(self) -> None:
        super().__init__("technical_analysis_agent")
        self._timeout = self.config["orchestrator"]["agent_timeout_seconds"]
        self._core_assets = set(self.config.get("assets", []))
        self._scanner_cfg = self.config.get("scanner", {})
        # Institutional analyzes equity/ETF universe, not just core crypto assets
        inst_universe = self._scanner_cfg.get("institutional_universe", [])
        self._institutional_assets = set(inst_universe) if inst_universe else self._core_assets

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        self.mark_running()
        try:
            market = self.read_json(DATA_DIR / "market_data.json")
            news_feed = self.read_json(DATA_DIR / "news_feed.json")
            macro_snapshot = self.read_json(DATA_DIR / "macro_snapshot.json")
            if not market:
                raise ValueError("market_data.json is empty or missing")
            self._validate_freshness(market.get("timestamp"))

            # Extract current date for DOW multiplier
            from datetime import date as _date
            _ts_str = market.get("timestamp", "")
            try:
                current_date = _date.fromisoformat(_ts_str[:10])
            except Exception:
                current_date = _date.today()
            dow_mult = self._dow_multiplier(current_date)
            self.logger.info(
                f"DOW adjustment: {current_date.strftime('%A')} × {dow_mult:.2f}"
            )

            # World Monitor sentiment — modifies final scores
            world_events   = market.get("world_events", [])
            sentiment_score, sentiment_label = self._compute_sentiment(world_events)
            if world_events:
                self.logger.info(
                    f"World sentiment: {sentiment_score:+.2f} ({sentiment_label}) "
                    f"from {len(world_events)} events"
                )

            # Regime gate: suppress BUY signals if RISK_OFF
            macro_regime = self.read_json(DATA_DIR / "market_regime.json")
            regime = macro_regime.get("regime", "NEUTRAL") if macro_regime else "NEUTRAL"
            regime_blocks_buy = (regime == "RISK_OFF")
            if regime_blocks_buy:
                self.logger.warning("Regime=RISK_OFF — BUY signals suppressed this cycle")

            # Stock score filter: prefer BUY only for composite_score > 0.60
            stock_scores_data = self.read_json(DATA_DIR / "stock_scores.json")
            stock_min_scores: dict[str, float] = {}
            if stock_scores_data:
                for sym, info in stock_scores_data.get("stocks", {}).items():
                    stock_min_scores[sym] = float(info.get("composite_score", 0.0))

            retail_signals:        dict = {}
            institutional_signals: dict = {}

            for symbol, data in market.get("assets", {}).items():
                try:
                    df_4h = self._to_df(data.get("ohlcv_4h") or data.get("ohlcv_1d", []))
                    df_1d = self._to_df(data.get("ohlcv_1d", []))
                    if len(df_4h) < 30 or len(df_1d) < 30:
                        raise ValueError(f"Not enough candles: 4h={len(df_4h)} 1d={len(df_1d)}")

                    df_4h = self._compute_indicators(df_4h)
                    df_1d = self._compute_indicators(df_1d)
                    last_price = float(data["last_price"])

                    # Filter events that affect this symbol
                    symbol_events = [
                        e for e in world_events
                        if not e.get("symbols_affected")
                        or symbol in e.get("symbols_affected", [])
                    ]
                    sym_sentiment, _ = self._compute_sentiment(symbol_events)
                    news_context = self._news_context_for_symbol(symbol, news_feed, macro_snapshot)

                    # None = not in stock_scores (crypto) → skip filter entirely
                    # float < 0.60 = equity below threshold → downgrade BUY → HOLD
                    # None = not in stock_scores (crypto) → skip filter entirely
                    # float < 0.60 = equity below threshold → downgrade BUY → HOLD
                    stock_composite = stock_min_scores.get(symbol, None)
                    # Compute noise bands from raw daily candles (before DataFrame conversion)
                    candles_1d = data.get("ohlcv_1d", [])
                    noise_bands = self._compute_noise_bands(candles_1d)

                    # ISSUE-003: Realized volatility 14d per asset
                    realized_vol_14d = self._compute_realized_vol_14d(candles_1d)

                    # Detect daily patterns from 1d candles
                    patterns = self._detect_daily_patterns(candles_1d)

                    retail_sig = self._generate_signal(
                        df_4h, df_1d, last_price, self.config["retail"], "retail",
                        sentiment=sym_sentiment,
                        context=news_context,
                        regime_blocks_buy=regime_blocks_buy,
                        stock_composite_score=stock_composite,
                        symbol=symbol,
                    )
                    retail_sig["noise_bands"] = noise_bands
                    retail_sig["realized_vol_14d"] = realized_vol_14d
                    # Apply pattern boost and DOW multiplier to retail score
                    r_score = self._apply_pattern_boost(retail_sig["score"], retail_sig["signal_type"], patterns)
                    r_score = min(1.0, r_score * dow_mult)
                    retail_sig["score"] = round(r_score, 3)
                    retail_sig["daily_patterns"] = list(patterns.keys())
                    retail_sig["pattern_boost"] = round(r_score, 3)
                    retail_signals[symbol] = retail_sig
                    if symbol in self._institutional_assets:
                        inst_sig = self._generate_signal(
                            df_4h, df_1d, last_price, self.config["institutional"], "institutional",
                            sentiment=sym_sentiment,
                            context=news_context,
                            regime_blocks_buy=regime_blocks_buy,
                            stock_composite_score=stock_composite,
                            symbol=symbol,
                        )
                        inst_sig["noise_bands"] = noise_bands
                        inst_sig["realized_vol_14d"] = realized_vol_14d
                        # Apply pattern boost and DOW multiplier to institutional score
                        i_score = self._apply_pattern_boost(inst_sig["score"], inst_sig["signal_type"], patterns)
                        i_score = min(1.0, i_score * dow_mult)
                        inst_sig["score"] = round(i_score, 3)
                        inst_sig["daily_patterns"] = list(patterns.keys())
                        inst_sig["pattern_boost"] = round(i_score, 3)
                        institutional_signals[symbol] = inst_sig

                except Exception as e:
                    self.logger.warning(f"Could not analyze {symbol}: {e}")
                    fallback = self._hold_signal(data.get("last_price", 0))
                    retail_signals[symbol] = fallback
                    if symbol in self._institutional_assets:
                        institutional_signals[symbol] = fallback

            retail_top = self._rank_candidates(
                retail_signals,
                self._scanner_cfg.get("retail_execute_candidates", self.config["retail"]["max_open_trades"]),
            )
            institutional_top = self._rank_candidates(
                institutional_signals,
                self._scanner_cfg.get("institutional_execute_candidates", self.config["institutional"]["max_open_trades"]),
            )

            signals_doc = {
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "retail":        retail_signals,
                "institutional": institutional_signals,
                "scanner": {
                    "retail_scanned": len(retail_signals),
                    "institutional_scanned": len(institutional_signals),
                    "retail_top_candidates": retail_top,
                    "institutional_top_candidates": institutional_top,
                },
                "context": {
                    "news_generated_at": news_feed.get("generated_at"),
                    "macro_generated_at": macro_snapshot.get("generated_at"),
                    "macro_bias": macro_snapshot.get("market_bias", 0.0),
                    "top_alerts": news_feed.get("top_alerts", [])[:5],
                },
            }
            self.require_write(DATA_DIR / "signals.json", signals_doc)
            self.update_shared_state("data_freshness.signals", signals_doc["timestamp"])

            r_summary = {s: d["signal_type"] for s, d in retail_signals.items()}
            i_summary = {s: d["signal_type"] for s, d in institutional_signals.items()}
            self.logger.info(f"Retail signals:        {r_summary}")
            self.logger.info(f"Institutional signals: {i_summary}")
            self.logger.info(
                "Retail scanner top: "
                + (", ".join(f"{row['symbol']}:{row['signal_type']}:{row['score']:.3f}" for row in retail_top) or "none")
            )
            self.mark_done()
            return True

        except Exception as e:
            self.mark_error(e)
            return False

    # ------------------------------------------------------------------ #
    # Data helpers                                                         #
    # ------------------------------------------------------------------ #

    def _validate_freshness(self, ts_str: str | None) -> None:
        if not ts_str:
            raise ValueError("No timestamp in market data")
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        backtest_mode = self.config.get("orchestrator", {}).get("mode") == "backtest"
        if not backtest_mode and age > self._timeout:
            raise ValueError(f"Market data is {age:.0f}s old (timeout={self._timeout}s)")

    @staticmethod
    def _to_df(ohlcv: list[dict]) -> pd.DataFrame:
        """Convert list of {t,o,h,l,c,v} dicts to DataFrame."""
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        df = df.sort_values("t").reset_index(drop=True)
        return df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})

    # ------------------------------------------------------------------ #
    # Indicator computation                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).ewm(com=period-1, adjust=False).mean()
        loss  = (-delta.where(delta < 0, 0.0)).ewm(com=period-1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        dm_p = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
        dm_m = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)
        atr   = tr.ewm(span=period, adjust=False).mean()
        di_p  = 100 * dm_p.ewm(span=period, adjust=False).mean() / atr
        di_m  = 100 * dm_m.ewm(span=period, adjust=False).mean() / atr
        dx    = (di_p - di_m).abs() / (di_p + di_m + 1e-10) * 100
        return dx.ewm(span=period, adjust=False).mean()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # EMA for signals
        d["ema_fast"]  = self._ema(d["close"], 9)
        d["ema_slow"]  = self._ema(d["close"], 21)
        d["ema_trend"] = self._ema(d["close"], 50)
        # RSI
        d["rsi"] = self._rsi(d["close"], 14)
        d["rsi5"] = self._rsi(d["close"], 5)
        # MACD
        ema12 = self._ema(d["close"], 12)
        ema26 = self._ema(d["close"], 26)
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        d["macd_hist"] = macd - sig
        # ADX
        d["adx"] = self._adx(d, 14)
        # Volume ratio
        d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
        # ATR (for SL/TP)
        tr = pd.concat([
            d["high"] - d["low"],
            (d["high"] - d["close"].shift()).abs(),
            (d["low"]  - d["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        d["atr"] = tr.ewm(span=14, adjust=False).mean()
        # Rate of Change
        d["roc10"] = d["close"].pct_change(10) * 100
        return d

    # ------------------------------------------------------------------ #
    # World Monitor sentiment                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_sentiment(events: list[dict]) -> tuple[float, str]:
        """
        Aggregate world_events into a single sentiment score [-1, +1].
        Returns (score, label) where label ∈ {'bearish','neutral','bullish'}.
        """
        try:
            from adapters.world_monitor_client import aggregate_sentiment
            score = aggregate_sentiment(events)
        except Exception:
            score = 0.0
        if score <= -0.15:
            label = "bearish"
        elif score >= 0.15:
            label = "bullish"
        else:
            label = "neutral"
        return round(score, 3), label

    # ------------------------------------------------------------------ #
    # Signal generation (ported from crypto_momentum_bot_v2 SignalEngine) #
    # ------------------------------------------------------------------ #

    def _generate_signal(
        self,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        last_price: float,
        cfg: dict,
        mode: str,
        sentiment: float = 0.0,
        context: dict | None = None,
        regime_blocks_buy: bool = False,
        stock_composite_score: float | None = None,
        symbol: str | None = None,
    ) -> dict:
        curr = df_4h.iloc[-1]
        prev = df_4h.iloc[-2]
        last_d = df_1d.iloc[-1]

        buy_score  = 0.0
        sell_score = 0.0

        # 1. EMA cross 9/21 (weight 35% cross, 20% bull alignment)
        ema_cross_up  = curr["ema_fast"] > curr["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]
        ema_cross_dn  = curr["ema_fast"] < curr["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]
        ema_bull      = curr["ema_fast"] > curr["ema_slow"]
        ema_bear      = curr["ema_fast"] < curr["ema_slow"]

        if ema_cross_up:
            buy_score  += 0.35
        elif ema_bull:
            buy_score  += 0.20
        if ema_cross_dn:
            sell_score += 0.35
        elif ema_bear:
            sell_score += 0.20

        # 2. RSI (weight 25% extreme, 15% neutral zone)
        rsi         = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0
        rsi_over    = cfg["rsi_oversold"]
        rsi_overbgt = cfg["rsi_overbought"]

        if rsi < rsi_over:
            buy_score  += 0.25
        elif rsi_over <= rsi <= rsi_overbgt:
            buy_score  += 0.15
        if rsi > rsi_overbgt:
            sell_score += 0.25

        # 3. MACD histogram (weight 25% cross, 12% continuation)
        hist_curr = float(curr["macd_hist"]) if not np.isnan(curr["macd_hist"]) else 0.0
        hist_prev = float(prev["macd_hist"]) if not np.isnan(prev["macd_hist"]) else 0.0

        if hist_curr > 0 and hist_prev <= 0:
            buy_score  += 0.25
        elif hist_curr > 0:
            buy_score  += 0.12
        if hist_curr < 0 and hist_prev >= 0:
            sell_score += 0.25
        elif hist_curr < 0:
            sell_score += 0.12

        # 4. ADX trend strength amplifier (> 25 = strong trend)
        adx = float(curr["adx"]) if not np.isnan(curr["adx"]) else 0.0
        if adx > 25.0:
            buy_score  *= 1.1 if ema_bull else 1.0
            sell_score *= 1.1 if ema_bear else 1.0
        else:
            buy_score  *= 0.7
            sell_score *= 0.7

        # 5. Volume spike (> 1.5x MA-20, weight 0.10)
        vr = float(curr["vol_ratio"]) if not np.isnan(curr["vol_ratio"]) else 1.0
        if vr > 1.5:
            buy_score += 0.10

        # 6. Daily trend filter — price vs EMA-50 daily  [FIX #8]
        daily_ema50 = float(last_d["ema_trend"]) if not np.isnan(last_d["ema_trend"]) else last_price
        if last_price > daily_ema50:
            buy_score  += 0.08
        else:
            sell_score += 0.10

        # 7. ROC-10 momentum
        roc = float(curr["roc10"]) if not np.isnan(curr["roc10"]) else 0.0
        if roc > 3.0:
            buy_score  += 0.08
        elif roc < -3.0:
            sell_score += 0.08

        # 8. RSI-5 gamma imbalance proxy (Beat-the-Market §5.3)
        # High RSI-5 → dealers long gamma → momentum smorzato → penalizza BUY
        # Low RSI-5  → dealers short gamma → momentum amplificato → boost BUY
        rsi5 = float(curr["rsi5"]) if not np.isnan(curr["rsi5"]) else 50.0
        if rsi5 > 75:
            buy_score  = max(0.0, buy_score  - 0.08)  # gamma smorzamento
        elif rsi5 < 25:
            buy_score  = min(1.0, buy_score  + 0.06)  # gamma amplificazione
        if rsi5 < 25:
            sell_score = max(0.0, sell_score - 0.06)  # short gamma: meno vendita efficace
        elif rsi5 > 75:
            sell_score = min(1.0, sell_score + 0.04)

        buy_score  = min(buy_score,  1.0)
        sell_score = min(sell_score, 1.0)

        # ── 8. World Monitor sentiment modifier ──────────────────────── #
        # sentiment in [-1, +1]: negative → amplify sell / dampen buy
        #                        positive → amplify buy  / dampen sell
        # Max adjustment: ±15% of the score
        adjustment_breakdown = {
            "world_sentiment": round(sentiment, 4),
            "news_sentiment": 0.0,
            "macro_bias": 0.0,
            "net_adjustment": 0.0,
            "drivers": [],
        }
        if abs(sentiment) >= 0.15:
            adj = sentiment * 0.15
            buy_score  = min(max(round(buy_score  + adj, 4), 0.0), 1.0)
            sell_score = min(max(round(sell_score - adj, 4), 0.0), 1.0)
            adjustment_breakdown["net_adjustment"] += round(adj, 4)

        if context:
            news_sentiment = float(context.get("news_sentiment", 0.0))
            macro_bias = float(context.get("macro_bias", 0.0))
            drivers = list(context.get("drivers", []))
            news_adj = news_sentiment * self.config.get("news_data", {}).get("news_adjustment_weight", 0.10)
            macro_adj = macro_bias * self.config.get("news_data", {}).get("macro_adjustment_weight", 0.12)
            total_adj = round(news_adj + macro_adj, 4)
            if abs(total_adj) > 0:
                buy_score = min(max(round(buy_score + total_adj, 4), 0.0), 1.0)
                sell_score = min(max(round(sell_score - total_adj, 4), 0.0), 1.0)
            adjustment_breakdown["news_sentiment"] = round(news_sentiment, 4)
            adjustment_breakdown["macro_bias"] = round(macro_bias, 4)
            adjustment_breakdown["net_adjustment"] = round(
                adjustment_breakdown["net_adjustment"] + total_adj, 4
            )
            adjustment_breakdown["drivers"] = drivers[:4]

        threshold = cfg["signal_threshold"]

        # EMA 9/21 cross booleans (for issue #172 schema)
        ema9_cross_up = bool(curr["ema_fast"] > curr["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"])
        ema9_cross_dn = bool(curr["ema_fast"] < curr["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"])

        # Stop-loss and take-profit
        atr_val  = float(curr["atr"]) if not np.isnan(curr["atr"]) else last_price * 0.02
        sl_buy   = max(last_price - 1.5 * atr_val, last_price * (1 - cfg["stop_loss_pct"]))
        tp_buy   = min(last_price + 3.0 * atr_val, last_price * (1 + cfg["take_profit_pct"]))
        sl_sell  = min(last_price + 1.5 * atr_val, last_price * (1 + cfg["stop_loss_pct"]))
        tp_sell  = max(last_price - 3.0 * atr_val, last_price * (1 - cfg["take_profit_pct"]))
        suggested_sl_pct = round(1.5 * atr_val / last_price, 4) if last_price > 0 else cfg["stop_loss_pct"]

        if buy_score > threshold and buy_score > sell_score:
            signal_type = "BUY"
            score       = round(buy_score, 4)
            stop_loss   = round(sl_buy,  4)
            take_profit = round(tp_buy,  4)
        elif sell_score > threshold and sell_score > buy_score:
            signal_type = "SELL"
            score       = round(sell_score, 4)
            stop_loss   = round(sl_sell, 4)
            take_profit = round(tp_sell, 4)
        else:
            signal_type = "HOLD"
            score       = round(max(buy_score, sell_score), 4)
            stop_loss   = round(last_price * (1 - cfg["stop_loss_pct"]), 4)
            take_profit = round(last_price * (1 + cfg["take_profit_pct"]), 4)

        # Regime gate: downgrade BUY → HOLD if RISK_OFF
        if signal_type == "BUY" and regime_blocks_buy:
            signal_type = "HOLD"
            score = round(buy_score, 4)

        # Stock composite filter: only applied for equities (score is not None)
        # Crypto symbols have score=None → filter skipped intentionally
        stock_filter_applied = (
            stock_composite_score is not None and stock_composite_score < 0.60
        )

        # Crypto strict temporal confirmation: require persistent trend + ADX strength.
        asset_role = self.get_asset_role(symbol)
        is_crypto_asset = asset_role in {"crypto_native", "crypto_proxy"} or symbol.upper().endswith("USDT")
        crypto_temporal_confirm = bool(
            curr["ema_fast"] > curr["ema_slow"]
            and prev["ema_fast"] > prev["ema_slow"]
            and adx >= 25.0
        )
        if signal_type == "BUY" and is_crypto_asset and not crypto_temporal_confirm:
            signal_type = "HOLD"
            score = round(buy_score, 4)

        if signal_type == "BUY" and stock_filter_applied:
            signal_type = "HOLD"
            score = round(buy_score, 4)

        return {
            "signal_type": signal_type,
            "score":       score,
            "buy_score":   round(buy_score,  4),
            "sell_score":  round(sell_score, 4),
            "last_price":  round(last_price, 4),
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "suggested_stop_loss_pct": suggested_sl_pct,
            "ema9_cross":  ema9_cross_up,
            "ema9_cross_dn": ema9_cross_dn,
            "atr":         round(atr_val, 4),
            "rsi":         round(rsi, 2),
            "rsi5":        round(rsi5, 2),
            "adx":         round(adx, 2),
            "ema_trend":   round(daily_ema50, 4),
            "regime_filter_applied": regime_blocks_buy,
            "stock_score_filter_applied": stock_filter_applied,
            "mode":        mode,
            "context_adjustment": adjustment_breakdown,
            "asset_role":  self.get_asset_role(symbol),
        }

    @staticmethod
    def _hold_signal(last_price: float) -> dict:
        return {
            "signal_type": "HOLD",
            "score": 0.0,
            "buy_score": 0.0,
            "sell_score": 0.0,
            "last_price": float(last_price),
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "suggested_stop_loss_pct": 0.0,
            "ema9_cross": False,
            "ema9_cross_dn": False,
            "atr": 0.0,
            "rsi": 50.0,
            "rsi5": 50.0,
            "adx": 0.0,
            "ema_trend": 0.0,
            "regime_filter_applied": False,
            "stock_score_filter_applied": False,
            "mode": "error",
            "noise_bands": None,
            "context_adjustment": {
                "world_sentiment": 0.0,
                "news_sentiment": 0.0,
                "macro_bias": 0.0,
                "net_adjustment": 0.0,
                "drivers": [],
            },
        }

    def _news_context_for_symbol(self, symbol: str, news_feed: dict, macro_snapshot: dict) -> dict:
        items = news_feed.get("items", [])
        symbol_items = []
        for item in items:
            symbols = item.get("symbols", [])
            if item.get("symbol") == symbol or symbol in symbols:
                symbol_items.append(item)
            elif item.get("category") in {"macro", "insider"} and item.get("symbol") in (None, symbol):
                symbol_items.append(item)

        symbol_items = sorted(
            symbol_items,
            key=lambda row: float(row.get("composite_score", 0.0)),
            reverse=True,
        )[:5]

        if symbol_items:
            weighted = 0.0
            total_weight = 0.0
            for row in symbol_items:
                weight = max(float(row.get("relevance_score", 0.0)), 0.1)
                weighted += float(row.get("sentiment_score", 0.0)) * weight
                total_weight += weight
            news_sentiment = weighted / total_weight if total_weight else 0.0
        else:
            news_sentiment = 0.0

        drivers = [
            {
                "headline": row.get("headline"),
                "alert_type": row.get("alert_type"),
                "score": row.get("composite_score"),
            }
            for row in symbol_items[:3]
        ]

        return {
            "news_sentiment": round(news_sentiment, 4),
            "macro_bias": float(macro_snapshot.get("market_bias", 0.0)),
            "drivers": drivers,
        }

    @staticmethod
    def _compute_noise_bands(candles_1d: list[dict]) -> dict | None:
        """
        Returns upper/lower noise band for today using 14-day avg move from open.
        Uses daily candles (no intraday needed — adapted for daily backtest).
        Requires ≥15 daily candles.
        """
        if len(candles_1d) < 15:
            return None

        recent = candles_1d[-15:-1]  # 14 days before today
        today = candles_1d[-1]

        # avg absolute move from open over last 14 days
        moves = [abs(c["c"] / c["o"] - 1) for c in recent if c.get("o", 0) > 0]
        if not moves:
            return None
        sigma = sum(moves) / len(moves)

        # anchor: max/min of today open and yesterday close
        prev_close = candles_1d[-2]["c"]
        upper_anchor = max(today["o"], prev_close)
        lower_anchor = min(today["o"], prev_close)

        return {
            "upper": round(upper_anchor * (1 + sigma), 6),
            "lower": round(lower_anchor * (1 - sigma), 6),
            "sigma": round(sigma, 6),
        }

    @staticmethod
    def _compute_realized_vol_14d(candles_1d: list[dict]) -> float | None:
        """ISSUE-003: 14-day realized volatility (annualised daily std dev of log-returns).
        Used by RiskAgent to scale position size via vol-targeting.
        Returns annualised float (e.g. 0.25 = 25% annual vol) or None if insufficient data."""
        if len(candles_1d) < 16:
            return None
        import math
        closes = [float(c["c"]) for c in candles_1d[-15:] if c.get("c") and float(c["c"]) > 0]
        if len(closes) < 15:
            return None
        log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        n = len(log_rets)
        mu = sum(log_rets) / n
        variance = sum((r - mu) ** 2 for r in log_rets) / max(1, n - 1)
        daily_vol = math.sqrt(max(variance, 0.0))
        return round(daily_vol * math.sqrt(252), 6)  # annualise

    @staticmethod
    def _detect_daily_patterns(candles_1d: list[dict]) -> dict:
        """Detect daily price structure patterns (NR4, NR7, ID, Triangle, etc.)."""
        if len(candles_1d) < 8:
            return {}
        today = candles_1d[-1]
        prev = candles_1d[-2]
        t_range = today["h"] - today["l"]
        ranges = [c["h"] - c["l"] for c in candles_1d]
        patterns = {}
        # NR4: smallest range of last 4 days
        if t_range > 0 and all(t_range <= r for r in ranges[-5:-1]):
            patterns["NR4"] = True
        # NR7: smallest range of last 7 days
        if len(candles_1d) >= 8 and t_range > 0 and all(t_range <= r for r in ranges[-8:-1]):
            patterns["NR7"] = True
        # Inside Day
        if today["h"] < prev["h"] and today["l"] > prev["l"]:
            patterns["ID"] = True
        # Triangle
        if len(candles_1d) >= 3:
            prev2_h = max(candles_1d[-3]["h"], candles_1d[-2]["h"])
            prev2_l = min(candles_1d[-3]["l"], candles_1d[-2]["l"])
            if today["h"] < prev2_h and today["l"] > prev2_l:
                patterns["TRIANGLE"] = True
        # Big Tail
        if t_range > 0:
            body_low = min(today["o"], today["c"])
            body_high = max(today["o"], today["c"])
            if (body_low - today["l"]) / t_range >= 0.75:
                patterns["BIG_TAIL_BULL"] = True
            if (body_high - today["l"]) / t_range <= 0.25:
                patterns["BIG_TAIL_BEAR"] = True
        # Strong/Weak Closure
        if t_range > 0:
            close_pct = (today["c"] - today["l"]) / t_range
            if close_pct >= 0.90:
                patterns["STRONG_CLOSE"] = True
            elif close_pct <= 0.10:
                patterns["WEAK_CLOSE"] = True
        # Trend Day
        if t_range > 0:
            open_pct = (today["o"] - today["l"]) / t_range
            close_pct2 = (today["c"] - today["l"]) / t_range
            if open_pct <= 0.15 and close_pct2 >= 0.85:
                patterns["TREND_DAY_BULL"] = True
            elif open_pct >= 0.85 and close_pct2 <= 0.15:
                patterns["TREND_DAY_BEAR"] = True
        return patterns

    def _apply_pattern_boost(self, score: float, signal: str, patterns: dict) -> float:
        """Apply pattern-based score multipliers, capped at 1.0."""
        mult = 1.0
        for p, active in patterns.items():
            if not active:
                continue
            if p == "BIG_TAIL_BULL" and signal != "BUY":
                continue
            if p == "BIG_TAIL_BEAR" and signal != "SELL":
                continue
            if p == "STRONG_CLOSE" and signal != "BUY":
                continue
            if p == "WEAK_CLOSE" and signal != "SELL":
                continue
            mult *= self.PATTERN_MULTIPLIERS.get(p, 1.0)
        return min(1.0, score * mult)

    @staticmethod
    def _dow_multiplier(current_date) -> float:
        """Return day-of-week score multiplier (Mon=0.88 … Thu=1.07)."""
        from datetime import date
        if isinstance(current_date, str):
            try:
                current_date = date.fromisoformat(current_date[:10])
            except Exception:
                return 1.0
        return {0: 0.88, 1: 0.95, 2: 1.05, 3: 1.07, 4: 1.05}.get(current_date.weekday(), 1.0)

    @staticmethod
    def _rank_candidates(signals: dict, limit: int) -> list[dict]:
        ranked = []
        for symbol, signal in signals.items():
            signal_type = signal.get("signal_type", "HOLD")
            if signal_type not in {"BUY", "SELL"}:
                continue
            ranked.append({
                "symbol": symbol,
                "signal_type": signal_type,
                "score": float(signal.get("score", 0.0)),
                "buy_score": float(signal.get("buy_score", 0.0)),
                "sell_score": float(signal.get("sell_score", 0.0)),
                "last_price": float(signal.get("last_price", 0.0)),
            })
        ranked.sort(key=lambda row: (row["score"], max(row["buy_score"], row["sell_score"])), reverse=True)
        return ranked[: max(0, int(limit))]
