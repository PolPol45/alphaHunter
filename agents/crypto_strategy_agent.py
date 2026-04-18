"""
CryptoStrategyAgent — Hybrid Crypto Analyst (Bull + Bear)
==========================================================
Implementa la strategia crypto ibrida descritta nel goal.pdf.
Guadagna sia in mercati rialzisti che ribassisti tramite
allocazioni dinamiche su tre livelli:

  50% → Core BTC/ETH          —  blue chip crypto, massima liquidità
  30% → DeFi/Bridge Layer     —  DEX token, crypto ETF, MSTR proxy
  20% → Altcoin + Memecoin    —  alta volatilità, alto upside potenziale

La modalità cambia in base al regime macro:
  BULL MODE → overweight altcoin/memecoin, leverage BTC upside
  BEAR MODE → rifugio su BTC/ETH core, rotate verso stablecoin proxy e MSTR puts

Livelli di analisi:
1. Crypto Macro  : Analisi BTC dominance proxy + macro globale (FRED) + Fear-like score
2. Core Analysis : BTC/ETH – tecnica multi-TF (1D, 4H simulato) + momentum 30/7/1 giorno
3. DeFi/Bridge   : On-chain proxy via prezzo + TVL momentum + correlazione BTC
4. Alt/Meme      : Momentum puro + RSI mean-reversion + volume surge + social proxy

Output: data/crypto_signals.json
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from agents.base_agent import BaseAgent, DATA_DIR


class CryptoStrategyAgent(BaseAgent):
    """
    Agente ibrido per la strategia crypto.
    Lavora su 4 bucket (core, defi, alt, meme) e produce
    un output strutturato con allocazione 50/30/20.
    """

    def __init__(self) -> None:
        super().__init__("crypto_strategy_agent")
        cfg = self.config.get("crypto_strategy", {})
        mu  = self.config.get("master_universe", {})

        # Legge dal master_universe
        self.core_universe:   list[str] = mu.get("crypto_core",         ["BTC-USD", "ETH-USD"])
        self.bridge_universe: list[str] = mu.get("crypto_defi_bridge",  [])
        self.alt_universe:    list[str] = mu.get("crypto_altcoin",      [])
        self.meme_universe:   list[str] = mu.get("crypto_memecoin",     [])

        # Config
        self.signal_threshold: float = cfg.get("signal_threshold", 0.50)  # core/defi
        self.alt_threshold:    float = cfg.get("alt_threshold",    0.43)  # altcoin/meme più aggressive
        self.max_core_picks:   int   = cfg.get("max_core_picks",   2)
        self.max_defi_picks:   int   = cfg.get("max_defi_picks",   5)
        self.max_alt_picks:    int   = cfg.get("max_alt_picks",    12)

    # ─────────────────────────────────────────────────────────────────────────
    # Main
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> bool:
        self.mark_running()
        try:
            import yfinance as yf

            # 1. Macro crypto regime
            regime, macro_score = self._analyze_crypto_macro(yf)
            self.logger.info(
                f"Crypto regime: {regime} | macro_score: {macro_score:+.2f}"
            )

            # 2. Core: BTC + ETH (sempre analizzati, peso 50%)
            core_results = self._analyze_bucket(
                yf, self.core_universe, "CORE", macro_score, is_meme=False
            )

            # 3. DeFi/Bridge layer (30%): DEX + ETF + MSTR proxy
            bridge_results = self._analyze_bucket(
                yf, self.bridge_universe, "DEFI_BRIDGE", macro_score, is_meme=False
            )

            # 4. Altcoin (15%) + Memecoin (5%)
            alt_results  = self._analyze_bucket(
                yf, self.alt_universe, "ALTCOIN", macro_score, is_meme=False
            )
            meme_results = self._analyze_bucket(
                yf, self.meme_universe, "MEMECOIN", macro_score, is_meme=True
            )
            alt_and_meme = alt_results + meme_results

            # 5. Rank e filtraggio
            # Core: sempre tutti i 2 (BTC + ETH) inclusi
            core_picks   = sorted(core_results, key=lambda x: x["score"], reverse=True)[: self.max_core_picks]

            # DeFi/Bridge: soglia standard
            bridge_picks = sorted(
                [r for r in bridge_results if r["score"] >= self.signal_threshold],
                key=lambda x: x["score"], reverse=True
            )[: self.max_defi_picks]

            # Alt + Meme: soglia più bassa perché sono asset più volatili
            altmeme_picks = sorted(
                [r for r in alt_and_meme if r["score"] >= self.alt_threshold],
                key=lambda x: x["score"], reverse=True
            )[: self.max_alt_picks]

            # 6. Modalità ibrida BEAR: riduce altcoin, mantiene BTC core + ETF hedge crypto
            if regime == "BEAR":
                altmeme_picks = [p for p in altmeme_picks if p["direction"] == "SHORT"][:5]
                bridge_picks  = [b for b in bridge_picks
                                 if b["symbol"] in {"MSTR", "GBTC", "IBIT", "FBTC", "ETHE"}]
                self.logger.info("BEAR mode: solo asset SHORT e bridge ETF selezionati.")

            # 7. Output
            output = {
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "regime":      regime,
                "macro_score": round(macro_score, 3),
                "allocations": {
                    "core_50pct":     core_picks,    # 50% BTC + ETH
                    "defi_bridge_30pct": bridge_picks,  # 30% DeFi/ETF/MSTR
                    "alt_meme_20pct": altmeme_picks,    # 20% Altcoin + Meme
                },
                "summary": {
                    "core_count":       len(core_picks),
                    "defi_count":       len(bridge_picks),
                    "alt_meme_count":   len(altmeme_picks),
                    "total_picks":      len(core_picks) + len(bridge_picks) + len(altmeme_picks),
                },
            }
            self.write_json(DATA_DIR / "crypto_signals.json", output)
            self.logger.info(
                f"Crypto scan complete [{regime}] | "
                f"Core: {len(core_picks)} | "
                f"DeFi/Bridge: {len(bridge_picks)} | "
                f"Alt+Meme: {len(altmeme_picks)}"
            )

            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Level 1: Crypto Macro Regime
    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_crypto_macro(self, yf) -> tuple[str, float]:
        """
        Determina il regime crypto (BULL / NEUTRAL / BEAR) basandosi su:
          - BTC momentum e RSI (proxy dominance)
          - Macro snapshot FRED (tassi, inflazione, DXY)
          - Sentiment proxy (notizie già estratte dal news_data_agent)
        """
        score = 0.0

        # ── BTC come barometro dell'intero mercato crypto ──────────────────── #
        try:
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(period="1y", interval="1d")
            if len(hist) >= 200:
                closes = hist["Close"].values.astype(float)
                ema50  = self._ema(closes, 50)[-1]
                ema200 = self._ema(closes, 200)[-1]
                rsi    = self._rsi(closes)
                last   = float(closes[-1])

                # BTC trend
                if last > ema200: score += 0.30
                if last > ema50:  score += 0.15
                if ema50 > ema200: score += 0.15  # golden cross

                # RSI
                if rsi > 60:  score += 0.10
                elif rsi < 35: score -= 0.20

                # Momentum 30gg
                mom_30d = (closes[-1] - closes[-30]) / closes[-30]
                if mom_30d > 0.15:   score += 0.15
                elif mom_30d > 0.05: score += 0.08
                elif mom_30d < -0.15: score -= 0.20
                elif mom_30d < -0.05: score -= 0.10

                # Volume surge proxy (ultimi 7gg vs precedenti 23gg)
                volumes = hist["Volume"].values.astype(float)
                vol_recent = float(np.mean(volumes[-7:]))
                vol_baseline = float(np.mean(volumes[-30:-7]))
                if vol_baseline > 0:
                    vol_ratio = vol_recent / vol_baseline
                    if vol_ratio > 1.5:   score += 0.10
                    elif vol_ratio < 0.7: score -= 0.05

        except Exception as e:
            self.logger.warning(f"BTC macro probe failed: {e}")

        # ── Macro FRED ────────────────────────────────────────────────────────── #
        try:
            macro  = self.read_json(DATA_DIR / "macro_snapshot.json")
            series = macro.get("series", {})
            fed = series.get("fed_funds", {}).get("value") or 4.0
            cpi = series.get("cpi_yoy",   {}).get("value") or 3.0
            dxy = series.get("dxy",       {}).get("value") or 100.0

            if fed > 5.0:    score -= 0.20
            elif fed < 3.0:  score += 0.15

            if cpi > 4.0:    score -= 0.15
            elif cpi < 2.5:  score += 0.10

            if dxy > 107.0:  score -= 0.15
            elif dxy < 100.0: score += 0.10

            # Risk flags da news agent
            risk_flags = macro.get("risk_flags", [])
            score -= len(risk_flags) * 0.05
        except Exception:
            pass

        # ── News sentiment ─────────────────────────────────────────────────────── #
        try:
            news = self.read_json(DATA_DIR / "news_data.json")
            macro_flags = news.get("macro_flags", 0)
            score -= macro_flags * 0.05
        except Exception:
            pass

        score = max(-1.0, min(1.0, score))

        if score > 0.30:   regime = "BULL"
        elif score < -0.20: regime = "BEAR"
        else:              regime = "NEUTRAL"

        return regime, score

    # ─────────────────────────────────────────────────────────────────────────
    # Level 2–4: Analisi per bucket
    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_bucket(
        self, yf, universe: list[str], bucket: str,
        macro_score: float, is_meme: bool
    ) -> list[dict]:
        results = []
        for symbol in universe:
            try:
                res = self._analyze_crypto_asset(yf, symbol, bucket, macro_score, is_meme)
                if res:
                    results.append(res)
                    self.logger.debug(
                        f"{symbol:12s} [{bucket}] score: {res['score']:.2f} | " 
                        f"dir: {res['direction']:5s} | "
                        f"rsi: {res['rsi']:.1f} | mom7d: {res['momentum_7d']:+.1f}%"
                    )
            except Exception as e:
                self.logger.warning(f"{symbol} [{bucket}] error: {e}")
        return results

    def _analyze_crypto_asset(
        self, yf, symbol: str, bucket: str,
        macro_score: float, is_meme: bool
    ) -> dict | None:

        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="1y", interval="1d")
        if len(hist) < 30:
            return None

        closes  = hist["Close"].values.astype(float)
        volumes = hist["Volume"].values.astype(float)
        last    = float(closes[-1])

        # ── Indicatori tecnici ─────────────────────────────────────────────── #
        ema50  = self._ema(closes, 50)[-1]
        ema200 = self._ema(closes, 200)[-1] if len(closes) >= 200 else ema50
        rsi    = self._rsi(closes)
        macd_bull, macd_bear = self._macd_signal(closes)

        if last <= 0:
            return None    # Token con prezzo zero o corrotto → salta

        # Momentum multi-timeframe (protetto da divisione per zero)
        mom_7d  = (closes[-1] - closes[-8])  / closes[-8]  if len(closes) >= 8  and closes[-8]  > 0 else 0.0
        mom_30d = (closes[-1] - closes[-31]) / closes[-31] if len(closes) >= 31 and closes[-31] > 0 else mom_7d
        mom_1d  = (closes[-1] - closes[-2])  / closes[-2]  if len(closes) >= 2  and closes[-2]  > 0 else 0.0

        # Volume surge (7d vs 30d baseline)
        vol_recent   = float(np.mean(volumes[-7:]))
        vol_baseline = float(np.mean(volumes[-30:-7])) if len(volumes) >= 30 else vol_recent
        vol_ratio    = vol_recent / vol_baseline if vol_baseline > 0 else 1.0

        # 52w position
        w52_hi  = float(np.max(closes))
        w52_lo  = float(np.min(closes))
        w52_rng = w52_hi - w52_lo
        w52_pos = (last - w52_lo) / w52_rng if w52_rng > 0 else 0.5

        # ── Score building ─────────────────────────────────────────────────── #
        if is_meme:
            # Memecoins: puro momentum + volume surge + RSI mean-reversion
            score = 0.40
            if mom_7d  > 0.30:  score += 0.25
            elif mom_7d > 0.10:  score += 0.15
            elif mom_7d < -0.20: score -= 0.20

            if mom_1d > 0.05:   score += 0.10
            if vol_ratio > 2.0:  score += 0.20
            elif vol_ratio > 1.5: score += 0.10

            if rsi < 35:  score += 0.15   # Oversold → rimbalzo
            elif rsi > 80: score -= 0.15  # Già esausto

            if macd_bull: score += 0.10
            if macd_bear: score -= 0.15

        elif bucket in ("DEFI_BRIDGE",):
            # DeFi e Bridge (MSTR, GBTC, ETF proxy): fondamentali + correlazione BTC
            score = 0.40

            # Trend primario
            if last > ema200: score += 0.20
            if last > ema50:  score += 0.10
            if ema50 > ema200: score += 0.10

            # RSI zone
            if rsi < 40:  score += 0.15
            elif rsi > 70: score -= 0.15

            # Amplificato dal regime macro
            score += macro_score * 0.15

            if macd_bull: score += 0.10
            if macd_bear: score -= 0.10

            if mom_30d > 0.10:  score += 0.10
            elif mom_30d < -0.10: score -= 0.10

        else:
            # Core (BTC/ETH) e Altcoin: analisi tecnica completa
            score = 0.40

            # Trend primario multi EMA
            if last > ema200: score += 0.20
            elif last < ema200: score -= 0.10
            if last > ema50:  score += 0.10
            if ema50 > ema200: score += 0.10   # golden cross

            # RSI
            if rsi < 35:  score += 0.20   # Oversold → opportunità
            elif rsi < 50: score += 0.05
            elif rsi > 70: score -= 0.15

            # MACD
            if macd_bull: score += 0.10
            if macd_bear: score -= 0.10

            # Momentum
            if mom_30d > 0.15:   score += 0.15
            elif mom_30d > 0.05: score += 0.08
            elif mom_30d < -0.15: score -= 0.15

            # Volume conferma
            if vol_ratio > 1.5: score += 0.08

            # Amplificatore macro
            score += macro_score * 0.10

        score = round(max(0.0, min(1.0, score)), 3)

        # ── Livelli entry/SL/TP ───────────────────────────────────────────── #
        recent_lo = float(np.min(closes[-14:]))
        recent_hi = float(np.max(closes[-14:]))
        # ── Determina direzione (LONG o SHORT) ── #
        # SHORT se: RSI overbought + momentum negativo a 7d + MACD bear + sotto EMA50
        short_signals = 0
        if rsi > 70:       short_signals += 1
        if mom_7d < -0.05: short_signals += 1
        if not macd_bull and macd_bear: short_signals += 1
        if last < ema50:   short_signals += 1

        # LONG se: momentum positivo + RSI sano + sopra EMA50
        long_signals = 0
        if mom_7d > 0.03:  long_signals += 1
        if 35 < rsi < 68:  long_signals += 1
        if last > ema50:   long_signals += 1
        if macd_bull:      long_signals += 1

        direction = "SHORT" if short_signals >= 3 else "LONG"

        # Adatta punteggio al regime di direzione
        if direction == "SHORT":
            score = round(max(0.0, min(1.0, 1.0 - score + 0.15)), 3)  # inverti score per short

        stop_loss   = round(recent_lo * 0.97, 6)             # SL: 3% sotto il low recente
        take_profit = round(last + (last - stop_loss) * 2.5, 6)  # TP: Risk/Reward 1:2.5

        return {
            "symbol":       symbol,
            "bucket":       bucket,
            "direction":    direction,
            "score":        score,
            "last_price":   round(last, 6),
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "rsi":          round(rsi, 1),
            "momentum_1d":  round(mom_1d * 100, 2),
            "momentum_7d":  round(mom_7d * 100, 2),
            "momentum_30d": round(mom_30d * 100, 2),
            "volume_ratio": round(vol_ratio, 2),
            "w52_position": round(w52_pos, 3),
            "golden_cross": bool(ema50 > ema200),
            "macd_bull":    bool(macd_bull),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Technical helpers
    # ─────────────────────────────────────────────────────────────────────────

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
        ag = np.mean(gains[:period])
        al = np.mean(losses[:period])
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i])  / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0
        return float(100.0 - 100.0 / (1.0 + ag / al))

    def _passes_cost_guard(self, metrics: dict) -> bool:
        """Return True when asset liquidity and spread metrics meet minimum thresholds.

        Thresholds (configurable via config.json → crypto_strategy → cost_guard):
          min_dollar_volume_30d : 1_000_000  (default)
          max_spread_proxy      : 0.05       (default)
        """
        cg = self.config.get("crypto_strategy", {}).get("cost_guard", {})
        min_volume = float(cg.get("min_dollar_volume_30d", 1_000_000.0))
        max_spread = float(cg.get("max_spread_proxy", 0.05))
        vol = float(metrics.get("median_dollar_volume_30d", 0.0))
        spread = float(metrics.get("spread_proxy", 1.0))
        return vol >= min_volume and spread <= max_spread

    def _macd_signal(self, closes: np.ndarray) -> tuple[bool, bool]:
        if len(closes) < 35:
            return False, False
        ema12  = self._ema(closes, 12)
        ema26  = self._ema(closes, 26)
        macd   = ema12 - ema26
        signal = self._ema(macd, 9)
        bull = (float(macd[-1]) > float(signal[-1]) and float(macd[-2]) <= float(signal[-2]))
        bear = (float(macd[-1]) < float(signal[-1]) and float(macd[-2]) >= float(signal[-2]))
        return bull, bear
