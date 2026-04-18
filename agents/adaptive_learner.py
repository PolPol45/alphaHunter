"""AdaptiveLearner — Feedback loop centralizzato per il trading bot.

Ogni ciclo legge i trade storici e aggiorna in data/adaptive_params.json:

  min_score_by_bucket     — alzato se win rate bucket < target, abbassato se > target
  correlation_threshold   — alzato se realized corr < expected, abbassato se > expected
  strategy_weights        — integra learned_strategy_weights.json (auto-ML output)
  target_vol_daily        — ridotto se drawdown > soglia, aumentato se equity in crescita
  covariance_penalty      — adattato su realized vs expected portfolio vol

Il RiskAgent legge adaptive_params.json ogni ciclo e sovrascrive i valori di config.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from agents.base_agent import BaseAgent, DATA_DIR


ADAPTIVE_PARAMS_PATH = DATA_DIR / "adaptive_params.json"

# Parametri di default (punto di partenza)
_DEFAULTS: dict = {
    "min_score_by_bucket": {
        "crypto": 0.58,
        "bull": 0.62,
        "bear_hedge": 0.55,
        "bear_short": 0.68,
        "bear_bankrupt": 0.66,
    },
    "correlation_threshold": 0.75,
    "covariance_penalty": 0.35,
    "target_vol_daily": 0.02,
    "strategy_weights": {
        "bull": 1.0,
        "bear": 1.0,
        "crypto": 1.0,
    },
    "max_asset_exposure_pct": 0.08,
    "last_updated": None,
    "cycle_count": 0,
    "metrics": {},
}

# Bounds assoluti per evitare valori estremi
_BOUNDS = {
    "min_score_by_bucket": {"min": 0.40, "max": 0.85},
    "correlation_threshold": {"min": 0.50, "max": 0.95},
    "covariance_penalty": {"min": 0.10, "max": 0.70},
    "target_vol_daily": {"min": 0.005, "max": 0.04},
    "max_asset_exposure_pct": {"min": 0.03, "max": 0.15},
    "strategy_weights": {"min": 0.5, "max": 2.0},
}

# Conservation layer: se Sharpe recente > soglia, congela i parametri
_FREEZE_SHARPE_THRESHOLD = 1.5   # Sharpe annualizzato: solo se davvero buono congela
_FREEZE_MIN_TRADES = 10          # almeno N trade prima di valutare il freeze

# t-test: aggiorna un parametro solo se il segnale è statisticamente significativo
_TTEST_MIN_SAMPLES = 8           # minimo trade per bucket per fare il t-test
_TTEST_ALPHA = 0.10              # p-value threshold (10% = più permissivo del classico 5%)

# Regularizzazione pesi: pull verso 1.0 ogni ciclo
_WEIGHT_REGULARIZATION = 0.02   # 2% di pull verso neutro per ciclo

# Velocità di aggiornamento (learning rate)
_LR = {
    "min_score": 0.01,        # ±0.01 per ciclo
    "corr_threshold": 0.02,
    "vol_target": 0.001,
    "covariance_penalty": 0.02,
    "exposure": 0.005,
    "strategy_weight": 0.05,
}

# Target win rate per bucket
_WIN_RATE_TARGET = {
    "bull": 0.52,
    "bear_hedge": 0.50,
    "bear_short": 0.50,
    "bear_bankrupt": 0.50,
    "crypto": 0.50,
}

# Lookback per calcoli
_LOOKBACK_DAYS = 21


class AdaptiveLearner(BaseAgent):
    """Aggiornamento parametri adattativi basato su performance reale."""

    def __init__(self) -> None:
        super().__init__("adaptive_learner")

    def run(self) -> bool:
        self.mark_running()
        try:
            # Carica stato corrente (o default)
            params = self.read_json(ADAPTIVE_PARAMS_PATH) or {}
            params = _merge_defaults(params)

            # Carica dati necessari
            port_retail = self.read_json(DATA_DIR / "portfolio_retail.json") or {}
            port_inst   = self.read_json(DATA_DIR / "portfolio_institutional.json") or {}
            learned_w   = self.read_json(DATA_DIR / "learned_strategy_weights.json") or {}
            macro_doc   = self.read_json(DATA_DIR / "macro_snapshot.json") or {}
            mkt_data    = self.read_json(DATA_DIR / "market_data.json") or {}

            trades_retail = port_retail.get("trades", [])
            trades_inst   = port_inst.get("trades", [])
            all_trades    = trades_retail + trades_inst

            # Filtra ultimi N giorni
            cutoff = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)
            recent = [t for t in all_trades if _parse_ts(t.get("timestamp")) >= cutoff]

            metrics = {}

            # ── 0. Conservation layer: calcola Sharpe corrente ───────────────
            # Se il sistema sta performando bene, NON toccare i parametri core.
            # Adatta solo se c'è evidenza statistica di deterioramento.
            sharpe_recent = self._rolling_sharpe(recent)
            metrics["sharpe_recent"] = sharpe_recent
            frozen = self._should_freeze(params, sharpe_recent)
            metrics["frozen"] = frozen
            if frozen:
                self.logger.info(
                    "AdaptiveLearner: FROZEN (Sharpe=%.2f > %.2f) — parametri conservati",
                    sharpe_recent, _FREEZE_SHARPE_THRESHOLD,
                )
                # Aggiorna solo il ciclo e le metriche, non i parametri
                params["cycle_count"] = params.get("cycle_count", 0) + 1
                params["last_updated"] = datetime.now(timezone.utc).isoformat()
                params["metrics"] = metrics
                self.write_json(ADAPTIVE_PARAMS_PATH, params)
                self.mark_done()
                return True

            # ── 1. Win rate per bucket → aggiusta min_score (solo se t-test OK)
            bucket_stats = self._bucket_stats(recent)
            metrics["bucket_stats"] = bucket_stats
            params = self._update_min_scores(params, bucket_stats)

            # ── 2. Drawdown → aggiusta target_vol e max_exposure ─────────────
            dd_retail = float(port_retail.get("drawdown_pct", 0.0) or 0.0)
            dd_inst   = float(port_inst.get("drawdown_pct", 0.0) or 0.0)
            max_dd    = max(dd_retail, dd_inst)
            metrics["max_drawdown"] = max_dd
            params = self._update_vol_target(params, max_dd)
            params = self._update_exposure(params, max_dd)

            # ── 3. Realized correlation → aggiusta correlation_threshold ──────
            corr_stats = self._realized_correlation(recent, mkt_data)
            metrics["realized_correlation"] = corr_stats
            params = self._update_corr_threshold(params, corr_stats)

            # ── 4. Strategy weights da AutoML + regularizzazione ──────────────
            params = self._merge_learned_weights(params, learned_w)
            params = self._regularize_weights(params)  # pull verso 1.0

            # ── 5. Portfolio covariance → aggiusta covariance_penalty ─────────
            cov_stats = self._portfolio_covariance(recent, mkt_data)
            metrics["portfolio_vol_realized"] = cov_stats.get("realized_vol")
            params = self._update_covariance_penalty(params, cov_stats, params["target_vol_daily"])

            # ── 6. Macro regime → override conservativo ───────────────────────
            params = self._macro_override(params, macro_doc)

            # ── Finalizza ─────────────────────────────────────────────────────
            params["cycle_count"] = params.get("cycle_count", 0) + 1
            params["last_updated"] = datetime.now(timezone.utc).isoformat()
            params["metrics"] = metrics

            self.write_json(ADAPTIVE_PARAMS_PATH, params)

            self.logger.info(
                "AdaptiveLearner cycle=%d | DD=%.2f%% | vol_target=%.4f | "
                "corr_thr=%.2f | bull_score=%.2f | bull_w=%.2f",
                params["cycle_count"],
                max_dd * 100,
                params["target_vol_daily"],
                params["correlation_threshold"],
                params["min_score_by_bucket"].get("bull", 0.62),
                params["strategy_weights"].get("bull", 1.0),
            )
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    # ── Bucket stats ──────────────────────────────────────────────────────────

    # ── Conservation layer ────────────────────────────────────────────────────

    def _rolling_sharpe(self, trades: List[dict]) -> float:
        """Sharpe ratio annualizzato sui trade recenti (PnL / std)."""
        pnls = [float(t["realized_pnl"]) for t in trades
                if t.get("realized_pnl") is not None]
        if len(pnls) < _FREEZE_MIN_TRADES:
            return 0.0
        mean = sum(pnls) / len(pnls)
        if mean <= 0:
            return 0.0
        var = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(var) if var > 0 else 1e-9
        # Annualizza assumendo ~252 trade/anno (approssimazione)
        return round((mean / std) * math.sqrt(252), 4)

    def _should_freeze(self, params: dict, sharpe: float) -> bool:
        """Congela i parametri se il sistema sta andando bene."""
        if sharpe >= _FREEZE_SHARPE_THRESHOLD:
            return True
        return False

    # ── Bucket stats + t-test ─────────────────────────────────────────────────

    def _bucket_stats(self, trades: List[dict]) -> dict:
        """Win rate e PnL medio per bucket, con t-test di significatività."""
        buckets: dict[str, dict] = {}
        for t in trades:
            pnl = t.get("realized_pnl")
            if pnl is None:
                continue
            bucket = _infer_bucket(t.get("strategy_bucket") or t.get("sub_portfolio", ""))
            if bucket not in buckets:
                buckets[bucket] = {"wins": 0, "losses": 0, "total_pnl": 0.0,
                                   "count": 0, "pnls": []}
            b = buckets[bucket]
            b["count"] += 1
            b["total_pnl"] += float(pnl)
            b["pnls"].append(float(pnl))
            if float(pnl) > 0:
                b["wins"] += 1
            else:
                b["losses"] += 1
        for b in buckets.values():
            n = b["wins"] + b["losses"]
            b["win_rate"] = b["wins"] / n if n > 0 else None
            b["avg_pnl"] = b["total_pnl"] / b["count"] if b["count"] > 0 else 0.0
            # t-test: PnL medio significativamente diverso da 0?
            b["pnl_significant"] = _ttest_nonzero(b["pnls"])
        return buckets

    def _update_min_scores(self, params: dict, bucket_stats: dict) -> dict:
        scores = params["min_score_by_bucket"]
        for bucket, target_wr in _WIN_RATE_TARGET.items():
            stats = bucket_stats.get(bucket) or bucket_stats.get(bucket.split("_")[0], {})
            if not stats:
                continue
            wr = stats.get("win_rate")
            n = stats.get("count", 0)
            # Requisiti minimi: abbastanza trade E segnale statisticamente significativo
            if wr is None or n < _TTEST_MIN_SAMPLES:
                continue
            if not stats.get("pnl_significant", False) and abs(wr - target_wr) < 0.10:
                # Differenza non significativa → non adattare (evita inseguire rumore)
                continue
            current = float(scores.get(bucket, 0.60))
            if wr < target_wr - 0.05:
                new_val = current + _LR["min_score"]
            elif wr > target_wr + 0.10:
                new_val = current - _LR["min_score"]
            else:
                new_val = current
            b = _BOUNDS["min_score_by_bucket"]
            scores[bucket] = round(max(b["min"], min(b["max"], new_val)), 4)
        params["min_score_by_bucket"] = scores
        return params

    # ── Regularizzazione pesi ─────────────────────────────────────────────────

    def _regularize_weights(self, params: dict) -> dict:
        """Pull dei strategy_weights verso 1.0 (neutro) ogni ciclo.
        Evita che un bucket venga azzerato per perdite temporanee.
        L = 2% pull per ciclo → ci vogliono ~35 cicli per tornare a neutro
        anche se AutoML spinge forte verso 0.
        """
        w = params["strategy_weights"]
        b = _BOUNDS["strategy_weights"]
        for k in list(w.keys()):
            current = float(w[k])
            # Mean-reversion: sposta il 2% verso 1.0
            new_val = current + _WEIGHT_REGULARIZATION * (1.0 - current)
            w[k] = round(max(b["min"], min(b["max"], new_val)), 4)
        params["strategy_weights"] = w
        return params

    # ── Drawdown → vol target e exposure ──────────────────────────────────────

    def _update_vol_target(self, params: dict, max_dd: float) -> dict:
        current = float(params["target_vol_daily"])
        if max_dd > 0.10:
            # DD > 10% → riduci vol target aggressivamente
            new_val = current - _LR["vol_target"] * 2
        elif max_dd > 0.06:
            # DD > 6% → riduci leggermente
            new_val = current - _LR["vol_target"]
        elif max_dd < 0.02:
            # DD basso → puoi alzare leggermente il target
            new_val = current + _LR["vol_target"] * 0.5
        else:
            new_val = current
        b = _BOUNDS["target_vol_daily"]
        params["target_vol_daily"] = round(max(b["min"], min(b["max"], new_val)), 5)
        return params

    def _update_exposure(self, params: dict, max_dd: float) -> dict:
        current = float(params["max_asset_exposure_pct"])
        if max_dd > 0.10:
            new_val = current - _LR["exposure"] * 2
        elif max_dd > 0.06:
            new_val = current - _LR["exposure"]
        elif max_dd < 0.02:
            new_val = current + _LR["exposure"] * 0.5
        else:
            new_val = current
        b = _BOUNDS["max_asset_exposure_pct"]
        params["max_asset_exposure_pct"] = round(max(b["min"], min(b["max"], new_val)), 4)
        return params

    # ── Realized correlation ──────────────────────────────────────────────────

    def _realized_correlation(self, trades: List[dict], mkt_data: dict) -> dict:
        """Stima correlazione realizzata tra posizioni aperte."""
        candles_map = {sym: c for sym, c in mkt_data.get("candles", {}).items()}
        symbols = list({t.get("symbol") for t in trades if t.get("symbol")})
        if len(symbols) < 2:
            return {"avg_pairwise_corr": None, "count": 0}

        returns: dict[str, list] = {}
        for sym in symbols:
            candles = candles_map.get(sym, [])
            closes = [float(c.get("c", 0)) for c in candles[-30:] if c.get("c")]
            if len(closes) < 10:
                continue
            rets = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
            returns[sym] = rets

        pairs = [(s1, s2) for i, s1 in enumerate(list(returns)) for s2 in list(returns)[i+1:]]
        if not pairs:
            return {"avg_pairwise_corr": None, "count": 0}

        corrs = []
        for s1, s2 in pairs:
            r1, r2 = returns[s1], returns[s2]
            n = min(len(r1), len(r2))
            if n < 5:
                continue
            r1, r2 = r1[-n:], r2[-n:]
            c = _pearson(r1, r2)
            if c is not None:
                corrs.append(c)

        if not corrs:
            return {"avg_pairwise_corr": None, "count": 0}
        return {"avg_pairwise_corr": round(sum(corrs) / len(corrs), 4), "count": len(corrs)}

    def _update_corr_threshold(self, params: dict, corr_stats: dict) -> dict:
        avg_corr = corr_stats.get("avg_pairwise_corr")
        if avg_corr is None or corr_stats.get("count", 0) < 3:
            return params
        current = float(params["correlation_threshold"])
        # Se la correlazione realizzata è alta, abbassa il threshold (più restrittivo)
        if avg_corr > 0.60:
            new_val = current - _LR["corr_threshold"]
        elif avg_corr < 0.30:
            new_val = current + _LR["corr_threshold"]
        else:
            new_val = current
        b = _BOUNDS["correlation_threshold"]
        params["correlation_threshold"] = round(max(b["min"], min(b["max"], new_val)), 3)
        return params

    # ── Learned strategy weights (da AutoML) ─────────────────────────────────

    def _merge_learned_weights(self, params: dict, learned: dict) -> dict:
        """Integra gradualmente i pesi da AutoML nei pesi correnti."""
        if not learned:
            return params
        current_w = params["strategy_weights"]
        b = _BOUNDS["strategy_weights"]
        for strategy in ["bull", "bear", "crypto"]:
            learned_val = float(learned.get(strategy, current_w.get(strategy, 1.0)))
            current_val = float(current_w.get(strategy, 1.0))
            # Integrazione graduale: 10% del delta per ciclo (EWM)
            new_val = current_val + _LR["strategy_weight"] * (learned_val - current_val)
            current_w[strategy] = round(max(b["min"], min(b["max"], new_val)), 4)
        params["strategy_weights"] = current_w
        return params

    # ── Portfolio covariance ──────────────────────────────────────────────────

    def _portfolio_covariance(self, trades: List[dict], mkt_data: dict) -> dict:
        """Stima volatilità realizzata del portfolio."""
        candles_map = {sym: c for sym, c in mkt_data.get("candles", {}).items()}
        # Usa top 10 simboli per trades recenti
        sym_count: dict[str, int] = {}
        for t in trades:
            s = t.get("symbol")
            if s:
                sym_count[s] = sym_count.get(s, 0) + 1
        top_symbols = sorted(sym_count, key=sym_count.get, reverse=True)[:10]

        all_rets: list[list] = []
        for sym in top_symbols:
            candles = candles_map.get(sym, [])
            closes = [float(c.get("c", 0)) for c in candles[-30:] if c.get("c")]
            if len(closes) < 10:
                continue
            rets = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
            all_rets.append(rets)

        if not all_rets:
            return {"realized_vol": None}

        # Portfolio equally weighted vol (approssimazione)
        n = min(len(r) for r in all_rets)
        port_rets = [sum(r[i] for r in all_rets) / len(all_rets) for i in range(n)]
        if len(port_rets) < 5:
            return {"realized_vol": None}

        mean = sum(port_rets) / len(port_rets)
        variance = sum((r - mean) ** 2 for r in port_rets) / len(port_rets)
        realized_vol = math.sqrt(variance)
        return {"realized_vol": round(realized_vol, 6)}

    def _update_covariance_penalty(self, params: dict, cov_stats: dict, vol_target: float) -> dict:
        realized_vol = cov_stats.get("realized_vol")
        if realized_vol is None:
            return params
        current = float(params["covariance_penalty"])
        if realized_vol > vol_target * 1.5:
            # Portfolio troppo volatile → aumenta penalty
            new_val = current + _LR["covariance_penalty"]
        elif realized_vol < vol_target * 0.5:
            # Portfolio troppo conservativo → riduci penalty
            new_val = current - _LR["covariance_penalty"]
        else:
            new_val = current
        b = _BOUNDS["covariance_penalty"]
        params["covariance_penalty"] = round(max(b["min"], min(b["max"], new_val)), 4)
        return params

    # ── Macro override ────────────────────────────────────────────────────────

    def _macro_override(self, params: dict, macro_doc: dict) -> dict:
        """In regime risk-off, forza parametri conservativi."""
        series = macro_doc.get("series", {})
        vix = (series.get("vix", {}) or {}).get("value") or 15
        market_bias = float(macro_doc.get("market_bias", 0.0) or 0.0)

        if float(vix) > 30 or market_bias < -0.30:
            # Regime di crisi: massima prudenza
            scores = params["min_score_by_bucket"]
            b = _BOUNDS["min_score_by_bucket"]
            for k in scores:
                scores[k] = min(b["max"], scores[k] + 0.03)
            params["min_score_by_bucket"] = scores
            params["target_vol_daily"] = max(_BOUNDS["target_vol_daily"]["min"],
                                             params["target_vol_daily"] - 0.003)
            self.logger.warning("Macro override: RISK-OFF regime (VIX=%.1f, bias=%.2f)", vix, market_bias)
        return params


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_defaults(params: dict) -> dict:
    result = dict(_DEFAULTS)
    result.update(params)
    # Assicura sotto-dict presenti
    for key in ["min_score_by_bucket", "strategy_weights"]:
        if key not in result or not isinstance(result[key], dict):
            result[key] = dict(_DEFAULTS[key])
        else:
            merged = dict(_DEFAULTS[key])
            merged.update(result[key])
            result[key] = merged
    return result


def _infer_bucket(tag: str) -> str:
    tag = (tag or "").lower()
    if "crypto" in tag:
        return "crypto"
    if "bear_hedge" in tag or "hedge" in tag:
        return "bear_hedge"
    if "bear_short" in tag or "short" in tag:
        return "bear_short"
    if "bear_bankrupt" in tag or "bankrupt" in tag:
        return "bear_bankrupt"
    if "bear" in tag:
        return "bear_hedge"
    return "bull"


def _parse_ts(ts: str | None) -> datetime:
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _ttest_nonzero(pnls: list) -> bool:
    """t-test one-sample: H0=PnL medio=0. Ritorna True se rifiutiamo H0 (segnale reale).
    Implementazione senza scipy: usa distribuzione t approssimata.
    """
    n = len(pnls)
    if n < _TTEST_MIN_SAMPLES:
        return False
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0
    if std == 0:
        return mean != 0
    t_stat = mean / (std / math.sqrt(n))
    # Approssimazione: per df > 8, t > 1.86 corrisponde a p < 0.10 (one-tailed)
    # t > 1.40 corrisponde a p < 0.10 (two-tailed) per df=8
    # Usiamo soglia conservativa: |t| > 1.5
    return abs(t_stat) > 1.5


def _pearson(x: list, y: list) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in x))
    dy = math.sqrt(sum((v - my) ** 2 for v in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)
