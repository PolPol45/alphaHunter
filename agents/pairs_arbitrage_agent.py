"""
PairsArbitrageAgent
===================
Identifica divergenze su coppie cointegrate hardcoded.
Segnale: z-score dello spread > soglia → long leg cheap / short leg expensive.

Coppie monitorate:
  Energy:    XLE/CVX, XOM/CVX, XOP/XLE
  Tech:      QQQ/SPY, SOXX/SMH, NVDA/AMD
  Crypto:    COIN/MSTR, IBIT/GBTC (proxy via MSTR)
  Commodities: GLD/IAU, SLV/GLD
  Finance:   JPM/BAC, MS/GS

Logica:
  spread = log(price_A) - hedge_ratio * log(price_B)
  z_score = (spread - mean_spread_20d) / std_spread_20d
  z > +2.0  → A cheap vs B  → BUY A / SELL B
  z < -2.0  → A expensive   → SELL A / BUY B
  |z| < 0.5 → convergenza   → segnale chiusura

Scrive: data/pairs_signals.json
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from agents.base_agent import BaseAgent, DATA_DIR


PAIRS: list[dict] = [
    # (leg_a, leg_b, hedge_ratio, sector)
    {"a": "XLE",  "b": "CVX",  "hr": 1.0, "sector": "energy"},
    {"a": "XOM",  "b": "CVX",  "hr": 1.0, "sector": "energy"},
    {"a": "XOP",  "b": "XLE",  "hr": 1.0, "sector": "energy"},
    {"a": "QQQ",  "b": "SPY",  "hr": 1.2, "sector": "index"},
    {"a": "SOXX", "b": "SMH",  "hr": 1.0, "sector": "semis"},
    {"a": "NVDA", "b": "AMD",  "hr": 1.0, "sector": "semis"},
    {"a": "COIN", "b": "MSTR", "hr": 0.8, "sector": "crypto_equity"},
    {"a": "GLD",  "b": "IAU",  "hr": 1.0, "sector": "gold"},
    {"a": "JPM",  "b": "BAC",  "hr": 1.0, "sector": "finance"},
    {"a": "MS",   "b": "GS",   "hr": 1.0, "sector": "finance"},
    {"a": "XOM",  "b": "XLE",  "hr": 1.0, "sector": "energy"},
    {"a": "IBIT", "b": "MSTR", "hr": 0.5, "sector": "crypto_equity"},
]

ZSCORE_ENTRY  = 2.0   # entra quando spread diverge oltre 2σ
ZSCORE_EXIT   = 0.5   # esci quando spread converge sotto 0.5σ
LOOKBACK      = 20    # giorni per calcolo mean/std spread


class PairsArbitrageAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__("pairs_arbitrage_agent")
        cfg = self.config.get("pairs_arbitrage", {})
        self._zscore_entry  = float(cfg.get("zscore_entry",  ZSCORE_ENTRY))
        self._zscore_exit   = float(cfg.get("zscore_exit",   ZSCORE_EXIT))
        self._lookback      = int(cfg.get("lookback_days",   LOOKBACK))
        self._enabled       = bool(cfg.get("enabled",        True))
        self._max_pairs     = int(cfg.get("max_active_pairs", 4))

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self._enabled:
                self.mark_done()
                return True

            mkt = self.read_json(DATA_DIR / "market_data.json") or {}
            assets = mkt.get("assets", {})
            regime = (self.read_json(DATA_DIR / "market_regime.json") or {}).get("regime", "NEUTRAL")
            forbidden_combos = self._ik.get("pairs_policy", {}).get("forbidden_role_combos", [])

            signals: list[dict] = []
            for pair in PAIRS:
                try:
                    # Block cross-role pairs with incompatible macro behavior
                    role_a = self.get_asset_role(pair["a"])
                    role_b = self.get_asset_role(pair["b"])
                    if [role_a, role_b] in forbidden_combos or [role_b, role_a] in forbidden_combos:
                        self.logger.debug(f"Pair {pair['a']}/{pair['b']} blocked: incompatible roles {role_a}/{role_b}")
                        continue

                    sig = self._analyze_pair(pair, assets)
                    if sig:
                        # Regime gate: block leg actions that violate role rules
                        for leg, action, role in [(pair["a"], sig["action_a"], role_a), (pair["b"], sig["action_b"], role_b)]:
                            if action in ("BUY", "SELL"):
                                _rcfg = self._ik.get("roles", {}).get(role, {})
                                _rrules = _rcfg.get("regime_rules", {}).get(regime, {})
                                if action == "BUY" and not _rrules.get("allow_long", True):
                                    sig = None; break
                                if action == "SELL" and not _rrules.get("allow_short", True):
                                    sig = None; break
                        if sig is None:
                            continue

                        # Same-role bonus: tighter cointegration expected
                        if role_a == role_b:
                            sig["score"] = round(min(sig["score"] + 0.10, 1.0), 4)
                            sig["same_role"] = True
                        signals.append(sig)
                        self.logger.info(
                            f"Pair {sig['leg_a']}/{sig['leg_b']} z={sig['zscore']:+.2f} "
                            f"→ {sig['action_a']} {sig['leg_a']} / {sig['action_b']} {sig['leg_b']}"
                        )
                except Exception as e:
                    self.logger.debug(f"Pair {pair['a']}/{pair['b']} skip: {e}")

            # Rank by |z-score| descending, cap at max_pairs
            signals.sort(key=lambda x: abs(x["zscore"]), reverse=True)
            active = signals[:self._max_pairs]

            output = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "active_pairs": len(active),
                "signals": active,
                "all_pairs_scanned": len(PAIRS),
            }
            self.write_json(DATA_DIR / "pairs_signals.json", output)
            self.logger.info(f"Pairs scan: {len(active)} active signals from {len(PAIRS)} pairs")
            self.mark_done()
            return True

        except Exception as exc:
            self.mark_error(exc)
            return False

    def _analyze_pair(self, pair: dict, assets: dict) -> Optional[dict]:
        a, b, hr = pair["a"], pair["b"], pair["hr"]

        closes_a = self._get_closes(a, assets)
        closes_b = self._get_closes(b, assets)
        if closes_a is None or closes_b is None:
            return None

        min_len = min(len(closes_a), len(closes_b))
        if min_len < self._lookback + 5:
            return None

        closes_a = closes_a[-min_len:]
        closes_b = closes_b[-min_len:]

        # Log-spread
        spread = np.log(closes_a) - hr * np.log(closes_b)

        # Rolling z-score on last lookback days
        window = spread[-self._lookback:]
        mean_s = float(np.mean(window))
        std_s  = float(np.std(window))
        if std_s < 1e-8:
            return None

        z = (float(spread[-1]) - mean_s) / std_s

        price_a = float(closes_a[-1])
        price_b = float(closes_b[-1])

        # Determine signal direction
        if z > self._zscore_entry:
            # A expensive vs B → SELL A / BUY B
            action_a, action_b = "SELL", "BUY"
        elif z < -self._zscore_entry:
            # A cheap vs B → BUY A / SELL B
            action_a, action_b = "BUY", "SELL"
        elif abs(z) < self._zscore_exit:
            # Convergence signal — mark for exit
            action_a, action_b = "CLOSE", "CLOSE"
        else:
            return None  # no signal

        return {
            "leg_a":       a,
            "leg_b":       b,
            "hedge_ratio": hr,
            "sector":      pair["sector"],
            "zscore":      round(z, 4),
            "spread":      round(float(spread[-1]), 6),
            "spread_mean": round(mean_s, 6),
            "spread_std":  round(std_s, 6),
            "price_a":     round(price_a, 4),
            "price_b":     round(price_b, 4),
            "action_a":    action_a,
            "action_b":    action_b,
            "signal_type": "PAIRS",
            "score":       round(min(abs(z) / 4.0, 1.0), 4),  # normalize to [0,1]
        }

    @staticmethod
    def _get_closes(symbol: str, assets: dict) -> Optional[np.ndarray]:
        info = assets.get(symbol, {})
        candles = info.get("ohlcv_1d") or info.get("ohlcv_4h") or []
        if len(candles) < 10:
            return None
        closes = np.array([float(c["c"]) for c in candles if c.get("c") and float(c["c"]) > 0])
        return closes if len(closes) >= 10 else None
