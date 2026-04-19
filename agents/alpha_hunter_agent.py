"""
Alpha Hunter Agent — Multi-Asset Mispricing Strategy
=====================================================
Budget  : $100,000 (sub-portfolio of institutional fund)
Universe: 10 large-cap US equities  + 4 existing crypto assets
Approach: 3-factor composite signal per asset

  FACTOR 1 — IV/HV Mispricing (40 %)
    Options market implied volatility vs 30-day realised volatility.
    • IV/HV < 0.85  → options cheap → underlying often coiled for a move → BUY amplifier
    • IV/HV > 1.40  → options rich  → fear premium, mean-reversion expected  → SELL pressure
    • IV data sourced from nearest ~30-DTE call option chain (yfinance)

  FACTOR 2 — Momentum (35 %)
    RSI-14 + EMA alignment (20/50/200) + MACD cross

  FACTOR 3 — Value / Mean Reversion (25 %)
    52-week range z-score + price vs EMA-200 deviation + forward P/E vs sector median

Signal threshold: 0.58 (configurable in config.json → alpha_hunter.signal_threshold)
  BUY  if composite_buy  > threshold AND composite_buy  > composite_sell
  SELL if composite_sell > threshold AND composite_sell > composite_buy
  HOLD otherwise

Position sizing: alpha_hunter.stake_pct × available_cash, max alpha_hunter.max_open_trades

Reads:  (nothing — fetches live market data via yfinance)
Writes: data/alpha_signals.json
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from agents.base_agent import BaseAgent, DATA_DIR


# ─── Strategy constants (can be overridden via config.json alpha_hunter section) ─ #
_DEFAULT_CFG = {
    "enabled":                      True,
    "universe":                     ["AAPL", "MSFT", "NVDA", "META", "GOOGL",
                                     "TSLA", "AMZN", "AMD",  "SPY",  "QQQ",
                                     "NFLX", "COST", "JPM",  "XOM",  "UNH",
                                     "SMH",  "IWM",  "XLF",  "GLD",  "TLT"],
    "allow_short":                  False,
    "top_candidates":               8,
    "capital":                      100_000.0,
    "max_open_trades":              5,
    "stake_pct":                    0.20,          # 20 % of available cash per position
    "cash_reserve_pct":             0.10,
    "min_trade_usd":                500.0,
    "signal_threshold":             0.58,
    "stop_loss_pct":                0.03,          # 3 % hard stop
    "take_profit_pct":              0.08,          # 8 % initial take-profit
    "trailing_stop_pct":            0.02,
    "trailing_stop_activation_pct": 0.015,
    "max_drawdown_pct":             0.10,
    "max_daily_loss_pct":           0.04,
    "fee_taker":                    0.0005,        # 0.05 % commission
    "slippage_pct":                 0.0002,
    # IV/HV thresholds
    "iv_hv_cheap":                  0.85,
    "iv_hv_expensive":              1.40,
    # Lookback for HV calculation (days)
    "hv_window":                    30,
    # Options DTE target
    "options_dte_target":           30,
}

# Approximate sector median forward P/E ratios (Q1 2026 consensus)
_SECTOR_PE: dict[str, float] = {
    "Technology":            28.0,
    "Consumer Cyclical":     22.0,
    "Healthcare":            20.0,
    "Financial Services":    14.0,
    "Communication Services":18.0,
    "Consumer Defensive":    22.0,
    "Energy":                12.0,
    "Basic Materials":       15.0,
    "Industrials":           22.0,
    "Real Estate":           30.0,
    "Utilities":             18.0,
}


class AlphaHunterAgent(BaseAgent):
    """
    Generates BUY / SELL / HOLD signals for a universe of US equities using
    a 3-factor composite: IV/HV mispricing + momentum + value/mean-reversion.
    """

    def __init__(self) -> None:
        super().__init__("alpha_hunter_agent")
        # Merge default config with anything supplied in config.json
        self._cfg: dict = {**_DEFAULT_CFG, **self.config.get("alpha_hunter", {})}
        self._universe: list[str] = self._cfg["universe"]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> bool:
        self.mark_running()
        try:
            import yfinance as yf  # lazy import — not needed by other agents

            _backtest_mode = self.config.get("orchestrator", {}).get("mode") == "backtest"

            # In backtest: restrict universe to symbols available in market_data.json
            # to avoid 275 live yfinance calls (9 min per cycle → unusable for backtest)
            if _backtest_mode:
                mkt = self.read_json(DATA_DIR / "market_data.json") or {}
                available = set(mkt.get("assets", {}).keys())
                original_universe = self._universe
                self._universe = [s for s in self._universe if s in available]
                if not self._universe:
                    self.logger.info("AlphaHunter: nessun simbolo con dati backtest — skip")
                    self.mark_done()
                    return True
                self.logger.info(
                    f"AlphaHunter backtest: {len(self._universe)}/{len(original_universe)} "
                    f"simboli da market_data ({', '.join(self._universe[:5])}...)"
                )

            # In backtest mode: load price cache to avoid live yfinance calls per symbol
            _price_cache: dict = {}
            if _backtest_mode:
                cache_doc = self.read_json(DATA_DIR / "sector_price_cache.json") or {}
                # sector_price_cache format: {"{SYM}_100d": {"ts": ..., "records": [{Date,Open,High,Low,Close,...}]}}
                for key, val in cache_doc.items():
                    if not key.endswith("_100d") or not isinstance(val, dict):
                        continue
                    sym = key.replace("_100d", "")
                    records = val.get("records", [])
                    if records:
                        _price_cache[sym] = {
                            "closes": [float(r["Close"]) for r in records],
                            "highs":  [float(r["High"])  for r in records],
                            "lows":   [float(r["Low"])   for r in records],
                        }
                # Also pull from market_data.json for crypto/main assets
                mkt = self.read_json(DATA_DIR / "market_data.json") or {}
                for sym, info in mkt.get("assets", {}).items():
                    candles = info.get("ohlcv_1d", [])
                    if candles:
                        _price_cache[sym] = {
                            "closes": [float(c["c"]) for c in candles],
                            "highs":  [float(c["h"]) for c in candles],
                            "lows":   [float(c["l"]) for c in candles],
                        }

            signals: dict[str, dict] = {}
            for symbol in self._universe:
                try:
                    sig = self._analyze(yf, symbol, _price_cache if _backtest_mode else None)
                    signals[symbol] = sig
                    self.logger.info(
                        f"{symbol:6s}: {sig['signal_type']:4s} "
                        f"buy={sig['buy_score']:.3f}  sell={sig['sell_score']:.3f}  "
                        f"iv/hv={sig['iv_hv_ratio']:.2f}  rsi={sig['rsi']:.1f}  "
                        f"52w%={sig['w52_position']*100:.0f}%"
                    )
                except Exception as exc:
                    self.logger.warning(f"{symbol} analysis error: {exc}")
                    signals[symbol] = self._hold_signal(symbol)

            buy_signals  = [s for s in signals.values() if s["signal_type"] == "BUY"]
            sell_signals = [s for s in signals.values() if s["signal_type"] == "SELL"]
            top_candidates = self._rank_candidates(
                signals,
                self._cfg.get("top_candidates", 8),
            )
            self.logger.info(
                f"Scan complete — BUY: {len(buy_signals)}  "
                f"SELL: {len(sell_signals)}  "
                f"HOLD: {len(signals) - len(buy_signals) - len(sell_signals)}"
            )

            output = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signals":   signals,
                "universe":  self._universe,
                "config_snapshot": {
                    "signal_threshold": self._cfg["signal_threshold"],
                    "iv_hv_cheap":      self._cfg["iv_hv_cheap"],
                    "iv_hv_expensive":  self._cfg["iv_hv_expensive"],
                    "stop_loss_pct":    self._cfg["stop_loss_pct"],
                    "take_profit_pct":  self._cfg["take_profit_pct"],
                    "allow_short":      self._cfg["allow_short"],
                    "top_candidates":   self._cfg["top_candidates"],
                },
                "scanner": {
                    "top_candidates": top_candidates,
                },
            }
            self.write_json(DATA_DIR / "alpha_signals.json", output)
            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    # ------------------------------------------------------------------ #
    # Per-symbol analysis pipeline                                        #
    # ------------------------------------------------------------------ #

    def _analyze(self, yf, symbol: str, price_cache: dict | None = None) -> dict:
        # ── 1. Price history — cache-first in backtest mode ───────────── #
        closes = highs = lows = None

        if price_cache and symbol in price_cache:
            cached = price_cache[symbol]
            if isinstance(cached, dict) and "closes" in cached:
                closes = np.array(cached["closes"], dtype=float)
                highs  = np.array(cached.get("highs",  cached["closes"]), dtype=float)
                lows   = np.array(cached.get("lows",   cached["closes"]), dtype=float)
            elif isinstance(cached, list) and len(cached) > 0:
                # flat list of closes
                closes = np.array(cached, dtype=float)
                highs  = closes.copy()
                lows   = closes.copy()

        if closes is None or len(closes) < 60:
            # Fallback: live yfinance (only in live mode or cache miss)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d", auto_adjust=True)
            if len(hist) < 60:
                return self._hold_signal(symbol)
            closes = hist["Close"].values.astype(float)
            highs  = hist["High"].values.astype(float)
            lows   = hist["Low"].values.astype(float)

        last = float(closes[-1])

        # ── 2. ATR-14 ─────────────────────────────────────────────────── #
        atr = self._atr(highs, lows, closes, 14)

        # ── 3. Historical Volatility (annualised, 30-day window) ──────── #
        hw = self._cfg["hv_window"]
        log_ret = np.diff(np.log(closes[-(hw + 1):]))
        hv = float(np.std(log_ret) * math.sqrt(252)) if len(log_ret) >= hw else 0.20
        hv = max(hv, 0.005)

        # ── 4. RSI-14 ─────────────────────────────────────────────────── #
        rsi = self._rsi(closes, 14)

        # ── 5. EMA alignment (20 / 50 / 200) ─────────────────────────── #
        ema20  = self._ema(closes, 20)[-1]
        ema50  = self._ema(closes, 50)[-1]
        ema200 = self._ema(closes, 200)[-1]
        ema_bull = last > ema20 > ema50       # price above rising short/mid EMA
        ema_bear = last < ema20 < ema50

        # ── 6. MACD (12 / 26 / 9) ────────────────────────────────────── #
        ema12    = self._ema(closes, 12)
        ema26    = self._ema(closes, 26)
        macd_l   = ema12 - ema26
        sig_l    = self._ema(macd_l, 9)
        macd_bull = float(macd_l[-1]) > float(sig_l[-1]) and float(macd_l[-2]) <= float(sig_l[-2])
        macd_bear = float(macd_l[-1]) < float(sig_l[-1]) and float(macd_l[-2]) >= float(sig_l[-2])

        # ── 7. 52-week range position (0 = at low, 1 = at high) ──────── #
        w52_hi  = float(np.max(closes[-252:]))
        w52_lo  = float(np.min(closes[-252:]))
        w52_rng = w52_hi - w52_lo
        w52_pos = (last - w52_lo) / w52_rng if w52_rng > 0 else 0.50

        # ── 8. Forward P/E vs sector median ──────────────────────────── #
        pe_buy = pe_sell = 0.0
        try:
            info   = ticker.fast_info
            # fast_info may not have PE; fall back to .info (slower)
            pe = getattr(info, "pe_ratio", None)
            if pe is None:
                full = ticker.info
                pe   = full.get("forwardPE") or full.get("trailingPE")
                sect = full.get("sector", "")
            else:
                full = ticker.info
                sect = full.get("sector", "")
            med_pe = _SECTOR_PE.get(sect)
            if pe and med_pe and pe > 0:
                pe_z = (pe - med_pe) / (med_pe * 0.30)   # normalised deviation
                pe_sell = max(0.0, min(1.0,  pe_z / 2.0))
                pe_buy  = max(0.0, min(1.0, -pe_z / 2.0))
        except Exception:
            pass

        # ── 9. IV / HV ratio ─────────────────────────────────────────── #
        iv_hv = 1.0
        try:
            iv_hv = self._iv_hv_ratio(ticker, last, hv)
        except Exception:
            pass

        # ── 10. Composite score ──────────────────────────────────────── #
        buy_s = sell_s = 0.0
        thr   = self._cfg["signal_threshold"]
        cheap  = self._cfg["iv_hv_cheap"]
        rich   = self._cfg["iv_hv_expensive"]

        # — Momentum (35 %) —
        if   rsi < 30:  buy_s  += 0.20
        elif rsi < 40:  buy_s  += 0.12
        elif rsi < 45:  buy_s  += 0.06
        elif rsi > 70:  sell_s += 0.20
        elif rsi > 60:  sell_s += 0.12
        elif rsi > 55:  sell_s += 0.06

        if   ema_bull:  buy_s  += 0.10
        elif ema_bear:  sell_s += 0.10

        if   macd_bull: buy_s  += 0.05
        elif macd_bear: sell_s += 0.05

        # — IV/HV factor (40 %) —
        if iv_hv < cheap:
            # Cheap options → potential coiled spring; amplify directional signal
            amp = (cheap - iv_hv) / cheap          # 0→1 as IV/HV → 0
            buy_s  += min(0.25, amp * 0.45)
        elif iv_hv > rich:
            # Expensive options → fear premium / overvalued asset
            amp = (iv_hv - rich) / rich             # 0→1 as IV/HV → ∞
            sell_s += min(0.22, amp * 0.38)
            # Contrarian: if price is also near 52w low, it may bounce
            if w52_pos < 0.20:
                buy_s += min(0.08, amp * 0.18)

        # — Value / Mean Reversion (25 %) —
        if   w52_pos < 0.10: buy_s  += 0.15
        elif w52_pos < 0.25: buy_s  += 0.09
        elif w52_pos < 0.35: buy_s  += 0.04
        elif w52_pos > 0.90: sell_s += 0.13
        elif w52_pos > 0.75: sell_s += 0.07
        elif w52_pos > 0.65: sell_s += 0.03

        buy_s  += pe_buy  * 0.10
        sell_s += pe_sell * 0.10

        if ema200 > 0:
            dev = (last - ema200) / ema200
            if   dev < -0.12: buy_s  += 0.08   # >12 % below 200d EMA
            elif dev < -0.06: buy_s  += 0.04
            elif dev >  0.25: sell_s += 0.08   # >25 % above 200d EMA
            elif dev >  0.12: sell_s += 0.04

        buy_s  = min(round(buy_s,  4), 1.0)
        sell_s = min(round(sell_s, 4), 1.0)

        if   buy_s  > thr and buy_s  > sell_s: signal = "BUY";  score = buy_s
        elif sell_s > thr and sell_s > buy_s:  signal = "SELL"; score = sell_s
        else:                                   signal = "HOLD"; score = max(buy_s, sell_s)

        # ── Stop / TP based on ATR and configured pcts ───────────────── #
        sl_pct = self._cfg["stop_loss_pct"]
        tp_pct = self._cfg["take_profit_pct"]
        # Use larger of ATR-based or pct-based to avoid getting stopped out on noise
        atr_sl = max(sl_pct * last, 2.0 * atr)
        atr_tp = max(tp_pct * last, 3.0 * atr)
        if signal == "SELL":
            stop_loss = round(last + atr_sl, 4)
            take_profit = round(max(last - atr_tp, 0.01), 4)
        else:
            stop_loss = round(last - atr_sl, 4)
            take_profit = round(last + atr_tp, 4)

        return {
            "signal_type":  signal,
            "score":        round(score, 4),
            "buy_score":    buy_s,
            "sell_score":   sell_s,
            "last_price":   round(last,    4),
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "atr":          round(atr,     4),
            "rsi":          round(rsi,     2),
            "ema200":       round(ema200,  4),
            "w52_position": round(w52_pos, 4),
            "iv_hv_ratio":  round(iv_hv,   4),
            "hv_30d":       round(hv,      4),
            "pe_buy":       round(pe_buy,  4),
            "pe_sell":      round(pe_sell, 4),
            "asset_type":   "equity",
            "mode":         "alpha",
        }

    @staticmethod
    def _rank_candidates(signals: dict[str, dict], limit: int) -> list[dict]:
        ranked: list[dict] = []
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

        ranked.sort(
            key=lambda row: (
                row["score"],
                max(row["buy_score"], row["sell_score"]),
            ),
            reverse=True,
        )
        return ranked[: max(0, int(limit))]

    # ------------------------------------------------------------------ #
    # IV / HV ratio via yfinance options chain                            #
    # ------------------------------------------------------------------ #

    def _iv_hv_ratio(self, ticker, last_price: float, hv: float) -> float:
        """
        Fetch the ATM call IV from the options expiry closest to DTE target.
        Returns IV / HV.  Falls back to 1.0 if options data unavailable.
        """
        exp_dates = getattr(ticker, "options", None)
        if not exp_dates:
            return 1.0

        dte_target = self._cfg["options_dte_target"]
        today  = datetime.now(timezone.utc).date()
        target = today + timedelta(days=dte_target)

        best_exp = min(
            exp_dates,
            key=lambda d: abs(
                (datetime.strptime(d, "%Y-%m-%d").date() - target).days
            ),
        )
        chain = ticker.option_chain(best_exp)
        calls = chain.calls.copy()
        if calls.empty:
            return 1.0

        calls = calls.dropna(subset=["impliedVolatility"])
        calls = calls[calls["impliedVolatility"] > 0.005]
        if calls.empty:
            return 1.0

        calls["_dist"] = (calls["strike"] - last_price).abs()
        atm = calls.nsmallest(3, "_dist")  # average top-3 closest strikes
        iv  = float(atm["impliedVolatility"].mean())

        if iv < 0.005 or hv < 0.005:
            return 1.0
        return round(iv / hv, 4)

    # ------------------------------------------------------------------ #
    # Technical helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        out  = np.zeros(len(data))
        mult = 2.0 / (period + 1.0)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = data[i] * mult + out[i - 1] * (1.0 - mult)
        return out

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 2:
            return 50.0
        deltas = np.diff(closes[-(period + 20):])
        gains  = np.where(deltas > 0,  deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        ag, al = np.mean(gains[:period]), np.mean(losses[:period])
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i])  / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0
        return float(100.0 - 100.0 / (1.0 + ag / al))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 14) -> float:
        if len(closes) < period + 1:
            return float(closes[-1]) * 0.01
        h = highs[-(period + 1):]
        l = lows[ -(period + 1):]
        c = closes[-(period + 1):]
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]),
                        np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr))

    def _blend_with_ml(
        self,
        base_buy: float,
        base_sell: float,
        ml_weight: float,
    ) -> tuple[float, float, float]:
        """Blend base buy/sell scores with an ML weight contribution.

        Args:
            base_buy:  base buy composite score [0, 1]
            base_sell: base sell composite score [0, 1]
            ml_weight: ML predicted excess return weight (positive → bullish signal)

        Returns:
            (blended_buy, blended_sell, ml_contribution)
        """
        if not self._cfg.get("ml_blend_enabled", False):
            return base_buy, base_sell, 0.0

        blend_w = float(self._cfg.get("ml_blend_weight", 0.2))
        contrib = ml_weight * blend_w
        if contrib > 0.0:
            blended_buy = min(1.0, base_buy + contrib)
            blended_sell = base_sell
        else:
            blended_buy = base_buy
            blended_sell = min(1.0, base_sell + abs(contrib))
        return blended_buy, blended_sell, abs(contrib)

    # ------------------------------------------------------------------ #
    # Fallback signal                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hold_signal(symbol: str) -> dict:
        return {
            "signal_type":  "HOLD",
            "score":         0.0,
            "buy_score":     0.0,
            "sell_score":    0.0,
            "last_price":    0.0,
            "stop_loss":     0.0,
            "take_profit":   0.0,
            "atr":           0.0,
            "rsi":          50.0,
            "ema200":        0.0,
            "w52_position":  0.5,
            "iv_hv_ratio":   1.0,
            "hv_30d":        0.0,
            "pe_buy":        0.0,
            "pe_sell":       0.0,
            "asset_type":   "equity",
            "mode":         "alpha",
        }
