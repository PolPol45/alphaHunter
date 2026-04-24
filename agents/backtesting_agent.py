from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import shutil

from agents.base_agent import BaseAgent, BASE_DIR, DATA_DIR, REPORTS_DIR
from agents.execution_agent import ExecutionAgent
from agents.risk_agent import RiskAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from adapters.historical_data_store import HistoricalDataStore, fetch_real_snapshots

# Separate directory for backtest working files — never touches live DATA_DIR files
BT_RUN_DIR = BASE_DIR / "data" / "bt_run"
OVERRIDES_PATH = DATA_DIR / "backtest_overrides.json"
from backtesting.backtest_metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)


class BacktestingAgent(BaseAgent):
    """End-to-end historical replay agent.

    This agent emulates the pipeline with anti-bias constraints:
    - Look-ahead: only candles timestamp <= simulated day end are used.
    - Survivorship: universe frozen at start of run.
    - Slippage/fees: explicit cost model via backtesting config and execution hook.
    """

    def __init__(self) -> None:
        super().__init__("backtesting_agent")
        self._cfg = self.config.get("backtesting", {})
        self._store = HistoricalDataStore()
        self._risk = RiskAgent()
        self._execution = ExecutionAgent()
        self._ta = TechnicalAnalysisAgent()
        self._ta._timeout = float("inf")  # backtest data is always "stale" by design
        self._overrides: dict = {}
        self._overrides_mtime: float = 0.0

    def run(self) -> bool:
        self.mark_running()
        try:
            start_s = self._cfg.get("start_date")
            end_s = self._cfg.get("end_date")
            if not start_s or not end_s:
                raise ValueError("backtesting.start_date/end_date are required")

            start = datetime.fromisoformat(start_s).date()
            end = datetime.fromisoformat(end_s).date()
            if end < start:
                raise ValueError("backtesting.end_date must be >= start_date")

            universe = self._freeze_universe()
            self.logger.info(f"Backtest start: {start} -> {end} | universe={len(universe)}")

            # Pre-download real historical data from yfinance (skips already-cached days)
            use_real_data = self._cfg.get("use_real_data", True)
            if use_real_data:
                self.logger.info("Pre-downloading real historical data from yfinance…")
                # Include macro symbols for regime/macro snapshot
                macro_syms = ["SPY", "^VIX", "TLT", "GLD", "DX-Y.NYB"]
                fetch_real_snapshots(
                    list(dict.fromkeys([*universe, *macro_syms])),
                    start, end,
                    self._store.bt_dir,
                )
                self.logger.info("Historical data ready.")

            # Reset runtime state for clean replay.
            self._reset_runtime_state()

            # Snapshot live files we will temporarily overwrite, restore them after.
            # Snapshot live files we overwrite (except market_data.json — too large).
            # market_data.json is written to a separate bt path and swapped temporarily.
            _LIVE_FILES = [
                "macro_snapshot.json", "market_regime.json",
                "sector_scorecard.json", "stock_scores.json", "bull_signals.json",
                "bear_signals.json", "crypto_signals.json", "news_feed.json",
                "signals.json", "alpha_signals.json",
            ]
            live_snapshots: dict[str, bytes | None] = {}
            for fname in _LIVE_FILES:
                p = DATA_DIR / fname
                live_snapshots[fname] = p.read_bytes() if p.exists() else None

            # For market_data.json: save original, use bt-specific file during loop
            _md_live = DATA_DIR / "market_data.json"
            _md_live_backup = DATA_DIR / "market_data_live_backup.json"
            if _md_live.exists() and _md_live.stat().st_size < 5 * 1024 * 1024:
                shutil.copy2(str(_md_live), str(_md_live_backup))

            equity_curve: list[float] = []
            equity_dates: list[str] = []
            returns: list[float] = []
            prev_equity: float | None = None

            day = start
            steps = 0
            try:
                while day <= end:
                    steps += 1
                    self._load_overrides()
                    self._set_backtest_context(day, universe)

                    market = self._store.get_snapshot_at(day, universe)
                    market = self._filter_no_lookahead(market, day)
                    self.write_json(DATA_DIR / "market_data.json", market)

                    macro = self._build_macro_snapshot_real(day, market)
                    self.write_json(DATA_DIR / "macro_snapshot.json", macro)
                    self.write_json(DATA_DIR / "market_regime.json", self._build_market_regime(day, macro))

                    sector = self._build_sector_scorecard(day)
                    self.write_json(DATA_DIR / "sector_scorecard.json", sector)

                    # Placeholder stock_scores — TA agent will compute real signals below
                    self.write_json(DATA_DIR / "stock_scores.json", {"generated_at": self._iso(day), "scores": [], "stocks": {}})

                    # Placeholder news/alpha — no real news data in backtest
                    self.write_json(DATA_DIR / "news_feed.json", {"generated_at": self._iso(day), "items": [], "top_alerts": []})
                    self.write_json(DATA_DIR / "alpha_signals.json", {"timestamp": self._iso(day), "signals": {}, "scanner": {"top_candidates": []}})

                    # Run real TechnicalAnalysisAgent on actual historical candles
                    self._ta.run()

                    # Build strategy signals from TA output
                    signals = self.read_json(DATA_DIR / "signals.json") or {}
                    stock_scores = self._build_stock_scores_from_signals(signals, universe)
                    self.write_json(DATA_DIR / "stock_scores.json", stock_scores)

                    bull = self._build_bull_signals(stock_scores, macro)
                    # ISSUE-006: merge intraday momentum SPY/QQQ/IWM picks into bull ETF bucket
                    intraday_picks = self._build_intraday_momentum_signals(day, market, macro)
                    if intraday_picks:
                        bull.setdefault("allocations", {}).setdefault("etf_50_pct", [])
                        bull["allocations"]["etf_50_pct"] = intraday_picks + bull["allocations"]["etf_50_pct"]
                    bear = self._build_bear_signals(stock_scores, macro)
                    crypto = self._build_crypto_signals(day, market)
                    self.write_json(DATA_DIR / "bull_signals.json", bull)
                    self.write_json(DATA_DIR / "bear_signals.json", bear)
                    self.write_json(DATA_DIR / "crypto_signals.json", crypto)

                    # Apply hot overrides to adaptive_params so RiskAgent reads them
                    if self._overrides:
                        ap_path = DATA_DIR / "adaptive_params.json"
                        try:
                            ap = json.loads(ap_path.read_text()) if ap_path.exists() else {}
                            changed = False
                            if "bucket_weights" in self._overrides:
                                ap.setdefault("strategy_weights", {}).update(self._overrides["bucket_weights"])
                                changed = True
                            if "max_drawdown_halt" in self._overrides:
                                ap["max_drawdown_pct"] = float(self._overrides["max_drawdown_halt"])
                                changed = True
                            if "target_vol_daily" in self._overrides:
                                ap["target_vol_daily"] = float(self._overrides["target_vol_daily"])
                                changed = True
                            if "max_asset_exposure_pct" in self._overrides:
                                ap["max_asset_exposure_pct"] = float(self._overrides["max_asset_exposure_pct"])
                                changed = True
                            if changed:
                                ap_path.write_text(json.dumps(ap, indent=2))
                        except Exception as exc:
                            self.logger.warning(f"Failed to apply overrides to adaptive_params: {exc}")

                    if not self._risk.run():
                        raise RuntimeError(f"RiskAgent failed at {day}")
                    if not self._execution.run():
                        raise RuntimeError(f"ExecutionAgent failed at {day}")

                    eq = self._current_total_equity()
                    equity_curve.append(eq)
                    equity_dates.append(day.isoformat())
                    if prev_equity and prev_equity > 0:
                        returns.append((eq - prev_equity) / prev_equity)
                    prev_equity = eq

                    if steps % 20 == 0:
                        self.logger.info(f"Backtest progress: {day} | equity={eq:.2f}")

                    day += timedelta(days=1)
            finally:
                # Restore market_data.json from backup if we saved one
                if _md_live_backup.exists():
                    shutil.copy2(str(_md_live_backup), str(_md_live))
                    _md_live_backup.unlink(missing_ok=True)
                else:
                    # No backup — delete the backtest-generated market_data so live agent recreates it
                    _md_live.unlink(missing_ok=True)
                # Restore all other live data files
                for fname, content in live_snapshots.items():
                    p = DATA_DIR / fname
                    if content is not None:
                        p.write_bytes(content)
                    elif p.exists():
                        p.unlink(missing_ok=True)

            report = self._build_report(start, end, universe, equity_curve, equity_dates, returns)
            report_tag = str(self._cfg.get("report_tag", "")).strip()
            if report_tag:
                report_path = REPORTS_DIR / f"backtest_report_{report_tag}_{start.isoformat()}_{end.isoformat()}.json"
            else:
                report_path = REPORTS_DIR / f"backtest_report_{start.isoformat()}_{end.isoformat()}.json"
            self.write_json(report_path, report)
            self.logger.info(f"Backtest completed | steps={steps} | report={report_path.name}")

            # Cleanup context marker.
            self._clear_backtest_context()
            self.mark_done()
            return True
        except Exception as exc:
            self._clear_backtest_context()
            self.mark_error(exc)
            return False

    def _load_overrides(self) -> None:
        """Reload backtest_overrides.json if it changed since last check."""
        try:
            mtime = OVERRIDES_PATH.stat().st_mtime if OVERRIDES_PATH.exists() else 0.0
            if mtime != self._overrides_mtime:
                self._overrides = json.loads(OVERRIDES_PATH.read_text()) if OVERRIDES_PATH.exists() else {}
                self._overrides_mtime = mtime
                if self._overrides:
                    self.logger.info(f"backtest_overrides reloaded: {list(self._overrides.keys())}")
        except Exception as exc:
            self.logger.warning(f"Failed to load backtest_overrides.json: {exc}")

    def _freeze_universe(self) -> list[str]:
        requested = self._cfg.get("universe_snapshot") or self.config.get("scanner", {}).get("crypto_universe", [])
        # Include distressed and small caps to avoid survivorship bias.
        mu = self.config.get("master_universe", {})
        requested = [
            *requested,
            *mu.get("equities_distressed", []),
            *mu.get("equities_small_cap", []),
            *mu.get("equities_large_cap", []),
            *mu.get("etf_long", []),
            *mu.get("etf_hedge", []),
        ]
        return self._store.freeze_universe(requested)

    def _filter_no_lookahead(self, market: dict, day: date) -> dict:
        cutoff = int(datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc).timestamp())
        out_assets: dict = {}
        for sym, data in market.get("assets", {}).items():
            c1 = [c for c in data.get("ohlcv_1d", []) if int(c.get("t", 0)) <= cutoff]
            c4 = [c for c in data.get("ohlcv_4h", []) if int(c.get("t", 0)) <= cutoff]
            if not c1 and not c4:
                continue
            last_price = float((c1[-1] if c1 else c4[-1]).get("c", data.get("last_price", 0.0)))
            c4_or_c1 = c4 or c1
            # Compute VWAP from last 6 × 4h bars (24h window) — needed for VWAP trail exit
            recent = c4_or_c1[-6:]
            cum_vol = sum(float(c.get("v", 0)) for c in recent)
            if cum_vol > 0:
                vwap = round(sum(((float(c["h"]) + float(c["l"]) + float(c["c"])) / 3.0) * float(c.get("v", 0)) for c in recent) / cum_vol, 6)
            else:
                vwap = last_price  # fallback: use last price so trail still works
            out_assets[sym] = {
                "last_price": last_price,
                "ohlcv_1d": c1,
                "ohlcv_4h": c4_or_c1,
                "orderbook": data.get("orderbook", {"bids": [], "asks": []}),
                "volume_24h": float(data.get("volume_24h", 0.0)),
                "vwap": vwap,
            }
        market["assets"] = out_assets
        market["timestamp"] = self._iso(day)
        market["data_source"] = "backtest_historical"
        return market

    def _build_macro_snapshot_real(self, day: date, market: dict) -> dict:
        """Build macro snapshot from real historical prices (SPY momentum + VIX proxy)."""
        # SPY momentum proxy for regime
        spy = market.get("assets", {}).get("SPY", {})
        spy_closes = [float(c["c"]) for c in spy.get("ohlcv_1d", [])[-31:]]
        mom30 = 0.0
        if len(spy_closes) >= 30 and spy_closes[0] > 0:
            mom30 = (spy_closes[-1] - spy_closes[0]) / spy_closes[0]

        # VIX proxy: use ^VIX if available, else estimate from SPY daily returns std
        vix_asset = market.get("assets", {}).get("^VIX", {})
        vix_closes = [float(c["c"]) for c in vix_asset.get("ohlcv_1d", [])[-5:]]
        if vix_closes:
            vix_val = round(vix_closes[-1], 2)
        elif len(spy_closes) >= 20:
            import numpy as np
            rets = np.diff(spy_closes[-21:]) / spy_closes[-21:-1]
            vix_val = round(float(np.std(rets)) * 100 * (252 ** 0.5), 2)
        else:
            vix_val = 20.0

        # TLT for rates context
        tlt = market.get("assets", {}).get("TLT", {})
        tlt_closes = [float(c["c"]) for c in tlt.get("ohlcv_1d", [])[-6:]]
        tlt_mom = 0.0
        if len(tlt_closes) >= 5 and tlt_closes[0] > 0:
            tlt_mom = (tlt_closes[-1] - tlt_closes[0]) / tlt_closes[0]

        bias = max(-1.0, min(1.0, mom30 * 3.0 - (vix_val - 20) * 0.01))
        risk_flags = []
        if vix_val > 30:
            risk_flags.append({"code": "HIGH_VIX", "severity": "warning", "vix": vix_val})
        if bias < -0.2:
            risk_flags.append({"code": "RISK_OFF", "severity": "warning", "bias": round(bias, 3)})

        return {
            "generated_at": self._iso(day),
            "series": {
                "fed_funds": {"value": 4.5, "status": "ok"},
                "cpi_yoy":   {"value": 3.0, "status": "ok"},
                "dxy":       {"value": 101.0, "status": "ok"},
                "vix":       {"value": vix_val, "status": "ok"},
                "tlt_mom5d": {"value": round(tlt_mom, 4), "status": "ok"},
                "spy_mom30d":{"value": round(mom30, 4), "status": "ok"},
            },
            "risk_flags": risk_flags,
            "market_bias": round(bias, 4),
            "source_status": {"backtest": True, "real_prices": True},
        }

    # ── Stock classifier ────────────────────────────────────────────── #
    # Classifies each equity by size/quality tier using price, volume and
    # realised volatility derived from historical candles. No external data
    # needed — everything comes from market_data.json / HistoricalDataStore.
    #
    # Tiers and their behavioural rules:
    #   BLUE_CHIP   price≥50  vol20d_avg≥5M   realised_vol≤0.025  → lowest score bar, widest SL
    #   LARGE_CAP   price≥10  vol20d_avg≥1M   realised_vol≤0.040  → standard
    #   SMALL_CAP   price≥5   vol20d_avg≥200K realised_vol≤0.060  → higher score bar, tighter size
    #   SPECULATIVE price<5 OR vol<200K OR vol>0.06                → blocked entirely in bull
    @staticmethod
    def _classify_stock(sym: str, candles_1d: list[dict]) -> dict:
        """Return tier + derived metrics for a single equity."""
        if len(candles_1d) < 10:
            return {"tier": "UNKNOWN", "avg_vol_20d": 0, "realised_vol": 0.10, "last_price": 0}

        recent = candles_1d[-20:]
        closes = [float(c["c"]) for c in recent if c.get("c")]
        volumes = [float(c["v"]) for c in recent if c.get("v")]
        last_price = closes[-1] if closes else 0.0
        avg_vol = sum(volumes) / len(volumes) if volumes else 0.0

        # Daily log-return volatility (annualised not needed — compare vs thresholds below)
        import math as _math
        rets = [_math.log(closes[i] / closes[i-1]) for i in range(1, len(closes)) if closes[i-1] > 0]
        if len(rets) >= 5:
            mean_r = sum(rets) / len(rets)
            var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
            realised_vol = _math.sqrt(max(var_r, 0.0))
        else:
            realised_vol = 0.10

        # ATR relative to price (avg true range / price)
        atr_vals = []
        for i in range(1, len(recent)):
            h = float(recent[i].get("h", 0))
            l = float(recent[i].get("l", 0))
            prev_c = float(recent[i-1].get("c", 0))
            if h > 0 and l > 0 and prev_c > 0:
                atr_vals.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
        atr_pct = (sum(atr_vals) / len(atr_vals) / last_price) if atr_vals and last_price > 0 else 0.10

        if last_price >= 50 and avg_vol >= 5_000_000 and realised_vol <= 0.020:
            tier = "BLUE_CHIP"
        elif last_price >= 10 and avg_vol >= 1_000_000 and realised_vol <= 0.028:
            tier = "LARGE_CAP"
        elif last_price >= 5 and avg_vol >= 200_000 and realised_vol <= 0.035:
            tier = "SMALL_CAP"
        else:
            tier = "SPECULATIVE"

        # Volume spike: compare last day vs 20d avg (momentum breakout signal)
        last_vol = float(candles_1d[-1].get("v", 0)) if candles_1d else 0.0
        vol_spike_ratio = (last_vol / avg_vol) if avg_vol > 0 else 0.0

        # 20-day high (for new high breakout check)
        high_20d = max((float(c.get("h", 0)) for c in recent if c.get("h")), default=0.0)

        return {
            "tier": tier,
            "avg_vol_20d": round(avg_vol),
            "realised_vol": round(realised_vol, 5),
            "atr_pct": round(atr_pct, 5),
            "last_price": round(last_price, 4),
            "vol_spike_ratio": round(vol_spike_ratio, 3),
            "high_20d": round(high_20d, 4),
        }

    # Score floor and size multiplier by tier — used in _build_stock_scores_from_signals
    _TIER_RULES: dict = {
        # tier                  : (min_composite_score, size_mult, sl_mult, tp_mult)
        "BLUE_CHIP"             : (0.68, 1.20, 1.8, 2.5),   # easier entry, larger size
        "LARGE_CAP"             : (0.73, 1.00, 1.5, 2.2),   # standard
        "SMALL_CAP"             : (0.78, 0.60, 1.2, 1.8),   # harder entry, smaller size
        "SPECULATIVE_MOMENTUM"  : (0.82, 0.20, 0.8, 3.0),   # breakout only, tiny size, aggressive R/R
        "SPECULATIVE"           : (99.0, 0.00, 0.0, 0.0),   # blocked
        "UNKNOWN"               : (0.80, 0.50, 1.0, 1.5),
    }

    def _build_stock_scores_from_signals(self, signals: dict, universe: list[str]) -> dict:
        """Derive stock_scores from real TA signals, enriched with stock-tier classification."""
        retail = signals.get("retail", {})
        market_doc = self.read_json(DATA_DIR / "market_data.json") or {}
        rows = []
        for sym in universe:
            r_sig = retail.get(sym, {})
            sig_label = r_sig.get("signal", "HOLD")
            buy_score  = float(r_sig.get("buy_score",  0.5))
            sell_score = float(r_sig.get("sell_score", 0.5))
            composite = round(buy_score * 0.6 + (1 - sell_score) * 0.4, 3)

            # Classify stock tier from live market_data candles
            candles_1d = (market_doc.get("assets", {}).get(sym, {}).get("ohlcv_1d") or [])
            stock_info = self._classify_stock(sym, candles_1d)
            tier = stock_info["tier"]
            tier_rules = self._TIER_RULES.get(tier, self._TIER_RULES["UNKNOWN"])
            min_score, size_mult, sl_mult, tp_mult = tier_rules

            # Speculative: allow only on confirmed momentum breakout
            if tier == "SPECULATIVE":
                rsi = float(r_sig.get("rsi") or 0)
                vol_spike = stock_info.get("vol_spike_ratio", 0.0)
                last_px = stock_info["last_price"]
                high_20d = stock_info.get("high_20d", 0.0)
                is_momentum_breakout = (
                    vol_spike >= 3.0          # volume 3× la media 20gg
                    and rsi >= 65             # momentum confermato
                    and last_px >= high_20d * 0.995  # prezzo vicino/sopra 20d high
                    and composite >= 0.82     # score alto
                )
                if not is_momentum_breakout:
                    continue
                tier = "SPECULATIVE_MOMENTUM"
                tier_rules = self._TIER_RULES["SPECULATIVE_MOMENTUM"]
                min_score, size_mult, sl_mult, tp_mult = tier_rules

            # Apply hot override for min_score if set
            ov_scores = self._overrides.get("min_score_by_tier", {})
            if tier in ov_scores:
                min_score = float(ov_scores[tier])
            # Apply global min_score_override (bucket-level from adaptive params)
            ov_bucket_scores = self._overrides.get("min_score_override", {})
            if "bull" in ov_bucket_scores:
                # bull bucket maps to stock score floors — apply as global floor
                global_floor = float(ov_bucket_scores["bull"])
                min_score = max(min_score, global_floor)

            # Enforce per-tier score floor
            if composite < min_score:
                continue

            rows.append({
                "symbol": sym,
                "sector": "Unknown_Sector",
                "benchmark": "SPY",
                "composite_score": composite,
                "technical_score": round(buy_score, 3),
                "fundamental_score": 0.5,
                "sector_tailwind": 0.85,
                "action": "BUY_CANDIDATE" if sig_label == "BUY" else ("AVOID" if sig_label == "SELL" else "WATCH"),
                "news_score": 0.0,
                "beta": None,
                "ema50_above_ema200": r_sig.get("ema50_above_200", False),
                "rsi": r_sig.get("rsi"),
                "macd_state": "BULLISH" if buy_score > 0.55 else ("BEARISH" if sell_score > 0.55 else "NEUTRAL"),
                "momentum_7d": None,
                "momentum_30d": None,
                "alerts": ["STRONG_BUY_SIGNAL"] if composite > 0.7 else [],
                # Stock-tier metadata passed downstream to RiskAgent via _noise_bands field
                "_stock_tier": tier,
                "_tier_size_mult": size_mult,
                "_tier_sl_mult": sl_mult,
                "_tier_tp_mult": tp_mult,
                "_avg_vol_20d": stock_info["avg_vol_20d"],
                "_realised_vol": stock_info["realised_vol"],
                "_atr_pct": stock_info["atr_pct"],
            })
        rows.sort(key=lambda x: x["composite_score"], reverse=True)
        stocks_dict = {r["symbol"]: {k: r[k] for k in ("composite_score","technical_score","fundamental_score","sector_tailwind","action")} for r in rows}
        return {"generated_at": self._iso(date.today()), "scores": rows, "stocks": stocks_dict}

    def _build_macro_snapshot(self, day: date, market: dict) -> dict:
        btc = market.get("assets", {}).get("BTCUSDT", {})
        closes = [float(c["c"]) for c in btc.get("ohlcv_1d", [])[-31:]]
        mom30 = 0.0
        if len(closes) >= 30 and closes[0] > 0:
            mom30 = (closes[-1] - closes[0]) / closes[0]
        bias = max(-1.0, min(1.0, mom30 * 3.0))
        risk_flags = [] if bias >= -0.2 else [{"code": "RISK_OFF", "severity": "warning", "bias": -0.4}]
        return {
            "generated_at": self._iso(day),
            "series": {
                "fed_funds": {"value": 4.0, "status": "ok"},
                "cpi_yoy": {"value": 3.0, "status": "ok"},
                "dxy": {"value": 101.0, "status": "ok"},
                "vix": {"value": 18.0, "status": "ok"},
            },
            "risk_flags": risk_flags,
            "market_bias": round(bias, 4),
            "source_status": {"backtest": True},
        }

    def _build_market_regime(self, day: date, macro: dict) -> dict:
        bias = float(macro.get("market_bias", 0.0))
        vix_val = float((macro.get("series") or {}).get("vix", {}).get("value") or 20.0)
        # RISK_OFF only when truly stressed: bias very negative OR VIX>30
        regime = "RISK_ON" if bias > 0.25 else ("RISK_OFF" if (bias < -0.40 or vix_val > 30) else "NEUTRAL")
        return {
            "generated_at": self._iso(day),
            "regime": regime,
            "score": round(bias, 4),
            "macro_factors": {"vix": vix_val},
            "regional_correlations": {},
            "volatility_sentiment": {},
            "risk_flags": macro.get("risk_flags", []),
            "summary": {"headline": f"Backtest regime {regime}", "explanation": "Derived from BTC momentum proxy"},
        }

    def _build_sector_scorecard(self, day: date) -> dict:
        # Lightweight deterministic placeholder for backtest path.
        sectors = []
        smap = self.config.get("sector_map", {})
        rank = 1
        for name, cfg in smap.items():
            score = round(max(0.0, min(1.0, 0.5 + ((abs(hash(name + day.isoformat())) % 30) - 15) / 100.0)), 2)
            sectors.append({
                "name": name,
                "benchmark": cfg.get("benchmark", "SPY"),
                "members": cfg.get("members", []),
                "score": score,
                "rank": rank,
                "return_5d": None,
                "return_20d": None,
                "return_60d": None,
                "relative_strength_20d": None,
                "breadth_above_ema50": None,
                "avg_volume_trend": None,
                "news_count": 0,
                "news_sentiment_score": 0.0,
                "beta_to_spy": None,
                "driver_type": "BACKTEST",
                "alerts": [],
            })
            rank += 1
        sectors.sort(key=lambda x: x["score"], reverse=True)
        for i, s in enumerate(sectors):
            s["rank"] = i + 1
            s["alerts"] = ["SECTOR_LEADER"] if i == 0 else (["SECTOR_LAGGARD"] if i == len(sectors) - 1 else [])
        return {"generated_at": self._iso(day), "market_regime_ref": "BACKTEST", "sectors": sectors}

    def _build_stock_scores(self, day: date, universe: list[str]) -> dict:
        rows = []
        for sym in universe:
            h = abs(hash(sym + day.isoformat()))
            technical = round((h % 101) / 100.0, 2)
            fundamental = round(((h // 101) % 101) / 100.0, 2)
            composite = round(technical * 0.6 + fundamental * 0.4, 2)
            rows.append({
                "symbol": sym,
                "sector": "Unknown_Sector",
                "benchmark": "SPY",
                "composite_score": composite,
                "technical_score": technical,
                "fundamental_score": fundamental,
                "news_score": 0.0,
                "beta": None,
                "price_target_mean": None,
                "recommendation_mean": None,
                "analyst_count": None,
                "roa": None,
                "roe": None,
                "debt_to_equity": None,
                "free_cash_flow": None,
                "current_ratio": None,
                "ema50_above_ema200": technical >= 0.5,
                "rsi": round(20 + (h % 60), 2),
                "macd_state": "BULLISH" if technical > 0.55 else ("BEARISH" if technical < 0.45 else "NEUTRAL"),
                "momentum_7d": round((technical - 0.5) / 5, 4),
                "momentum_30d": round((technical - 0.5) / 2, 4),
                "correlation_to_sector": None,
                "alerts": [],
            })
        rows.sort(key=lambda x: x["composite_score"], reverse=True)
        return {"generated_at": self._iso(day), "scores": rows}

    def _build_bull_signals(self, stock_scores: dict, macro: dict) -> dict:
        scores = stock_scores.get("scores", [])
        mu = self.config.get("master_universe", {})
        etf = [s for s in scores if s["symbol"] in set(mu.get("etf_long", [])) and s["composite_score"] >= 0.6][:35]
        large = [s for s in scores if s["symbol"] in set(mu.get("equities_large_cap", [])) and s["composite_score"] >= 0.6][:4]
        small = [s for s in scores if s["symbol"] in set(mu.get("equities_small_cap", [])) and s["composite_score"] >= 0.58][:10]
        bias = float(macro.get("market_bias", 0.0))
        return {
            "timestamp": macro.get("generated_at"),
            "macro_multiplier": round(1 + bias * 0.3, 3),
            "allocations": {"etf_50_pct": etf, "large_cap_30_pct": large, "small_cap_20_pct": small},
            "summary": {"etf_count": len(etf), "large_cap_count": len(large), "small_cap_count": len(small)},
        }

    def _build_bear_signals(self, stock_scores: dict, macro: dict) -> dict:
        scores = stock_scores.get("scores", [])
        mu = self.config.get("master_universe", {})
        hedge = [s for s in scores if s["symbol"] in set(mu.get("etf_hedge", []))][:5]
        shorts = [s for s in scores if s["symbol"] in set(mu.get("equities_distressed", [])) and s["composite_score"] < 0.4][:10]
        bankrupt = [s for s in scores if s["symbol"] in set(mu.get("equities_distressed", [])) and s["fundamental_score"] < 0.25][:10]
        return {
            "timestamp": macro.get("generated_at"),
            "macro_bias": round(float(macro.get("market_bias", 0.0)), 3),
            "risk_off": bool(float(macro.get("market_bias", 0.0)) < -0.3),
            "allocations": {"hedge_etfs": hedge, "short_candidates": shorts, "bankruptcy_risk": bankrupt},
            "summary": {"hedge_count": len(hedge), "short_count": len(shorts), "bankruptcy_count": len(bankrupt)},
        }

    def _build_crypto_signals(self, day: date, market: dict) -> dict:
        assets = market.get("assets", {})
        picks = []
        # Supporta sia formato yfinance (BTC-USD) sia formato Bybit (BTCUSDT)
        CORE_SYMS = {"BTCUSDT", "ETHUSDT", "BTC-USD", "ETH-USD"}
        candidates = [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
            "UNI-USD", "AAVE-USD", "MKR-USD", "LDO-USD", "DYDXUSDT", "LDOUSDT",
        ]
        seen_base: set[str] = set()
        for sym in candidates:
            if sym not in assets:
                continue
            base = sym.replace("-USD", "").replace("USDT", "")
            if base in seen_base:
                continue
            seen_base.add(base)
            c4 = assets[sym].get("ohlcv_4h", [])
            if len(c4) < 8:
                continue
            last = float(c4[-1]["c"])
            prev = float(c4[-8]["c"])
            mom7 = (last - prev) / prev if prev > 0 else 0.0
            direction = "SHORT" if mom7 < -0.03 else "LONG"
            score = round(min(1.0, max(0.0, 0.5 + mom7 * 3 + (0.12 if direction == "SHORT" else 0.0))), 3)
            picks.append({
                "symbol": sym,
                "bucket": "CORE" if sym in CORE_SYMS else "DEFI_BRIDGE",
                "direction": direction,
                "score": score,
                "last_price": round(last, 6),
                "stop_loss": round(last * (0.97 if direction == "LONG" else 1.03), 6),
                "take_profit": round(last * (1.08 if direction == "LONG" else 0.92), 6),
                "rsi": 50.0,
                "momentum_1d": round(mom7 * 30, 2),
                "momentum_7d": round(mom7 * 100, 2),
                "momentum_30d": round(mom7 * 150, 2),
                "volume_ratio": 1.0,
                "w52_position": 0.5,
                "golden_cross": True,
                "macd_bull": direction == "LONG",
            })

        core = [p for p in picks if p["bucket"] == "CORE"][:2]
        bridge = [p for p in picks if p["bucket"] == "DEFI_BRIDGE"][:5]
        altmeme = []
        regime = "NEUTRAL"
        macro_score = 0.0
        return {
            "timestamp": self._iso(day),
            "regime": regime,
            "macro_score": macro_score,
            "allocations": {"core_50pct": core, "defi_bridge_30pct": bridge, "alt_meme_20pct": altmeme},
            "summary": {
                "core_count": len(core),
                "defi_count": len(bridge),
                "alt_meme_count": len(altmeme),
                "total_picks": len(core) + len(bridge),
            },
        }

    def _build_intraday_momentum_signals(self, day: date, market: dict, macro: dict) -> list[dict]:
        """ISSUE-006: Intraday Momentum SPY/QQQ/IWM overlay (Beat-the-Market §6).
        Computes short-term momentum breakout for broad market ETFs.
        High score → market ETF BUY, sized at max 5% capital via noise band breakout.
        Uses 20-period band (mu ± 1.5σ) as noise filter, same as ISSUE-002."""
        ETF_SYMS = ["SPY", "QQQ", "IWM"]
        assets = market.get("assets", {})
        picks = []
        for sym in ETF_SYMS:
            data = assets.get(sym, {})
            ohlcv = data.get("ohlcv_4h") or data.get("ohlcv_1d") or []
            closes = [float(c["c"]) for c in ohlcv if c.get("c")]
            if len(closes) < 22:
                continue
            last = closes[-1]
            window = closes[-21:-1]
            mu = sum(window) / len(window)
            sigma = (sum((x - mu) ** 2 for x in window) / len(window)) ** 0.5
            upper = mu + 1.5 * sigma
            lower = mu - 1.5 * sigma
            mom_1d = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0
            mom_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0
            # Breakout above upper band → strong BUY
            if last > upper and mom_5d > 0.005:
                score = round(min(1.0, 0.70 + mom_5d * 5), 3)
                market_bias = float(macro.get("market_bias", 0.0))
                if market_bias < -0.30:
                    continue  # Suppress during strong risk-off
                picks.append({
                    "symbol": sym,
                    "composite_score": score,
                    "_signal_type": "BUY",
                    "_agent_source": "bull",
                    "_ta_last_price": round(last, 4),
                    "_noise_bands": {"upper": round(upper, 4), "lower": round(lower, 4), "sigma": round(sigma, 4)},
                    "_tier_size_mult": 0.5,  # Max 5% capital cap per paper spec
                    "_tier_sl_mult": 1.5,
                    "_tier_tp_mult": 2.0,
                    "momentum_1d": round(mom_1d * 100, 2),
                    "momentum_5d": round(mom_5d * 100, 2),
                    "source": "intraday_momentum_spy",
                })
        return picks

    def _current_total_equity(self) -> float:
        retail = self.read_json(DATA_DIR / "portfolio_retail.json")
        inst = self.read_json(DATA_DIR / "portfolio_institutional.json")
        return float(retail.get("total_equity", 0.0)) + float(inst.get("total_equity", 0.0))

    def _build_report(self, start: date, end: date, universe: list[str], equity: list[float], equity_dates: list[str], returns: list[float]) -> dict:
        retail = self.read_json(DATA_DIR / "portfolio_retail.json")
        inst = self.read_json(DATA_DIR / "portfolio_institutional.json")
        all_trades = retail.get("trades", []) + inst.get("trades", [])
        avg_eq = sum(equity) / len(equity) if equity else 0.0
        strategy_breakdown = self._strategy_breakdown(all_trades, avg_eq)
        return {
            "report_type": "backtest",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window": {"start_date": start.isoformat(), "end_date": end.isoformat()},
            "constraints": {
                "lookahead_guard": True,
                "survivorship_guard": True,
                "frozen_universe_size": len(universe),
            },
            "metrics": {
                "sharpe": round(sharpe_ratio(returns), 6),
                "sortino": round(sortino_ratio(returns), 6),
                "max_drawdown": round(max_drawdown(equity), 6),
                "calmar": round(calmar_ratio(equity), 6),
                "win_rate": round(win_rate(all_trades), 6),
                "turnover": round(turnover(all_trades, avg_eq), 6),
            },
            "equity_curve": [round(x, 4) for x in equity],
            "equity_dates": equity_dates,
            "summary": {
                "trades": len(all_trades),
                "final_equity": round(equity[-1], 4) if equity else 0.0,
                "start_equity": round(equity[0], 4) if equity else 0.0,
            },
            "strategy_breakdown": strategy_breakdown,
            "trades": all_trades,
        }

    def _strategy_breakdown(self, trades: list[dict], avg_equity: float) -> dict:
        buckets = {"bull": [], "bear": [], "crypto": [], "unknown": []}
        for t in trades:
            bucket = str(t.get("strategy_bucket") or "unknown").lower()
            if bucket not in buckets:
                bucket = "unknown"
            buckets[bucket].append(t)

        out = {}
        for name, rows in buckets.items():
            closed = [r for r in rows if r.get("realized_pnl") is not None]
            realized_pnl = sum(float(r.get("realized_pnl", 0.0)) for r in closed)
            gross_notional = sum(abs(float(r.get("notional", 0.0))) for r in rows)
            out[name] = {
                "trades": len(rows),
                "closed_trades": len(closed),
                "realized_pnl": round(realized_pnl, 4),
                "win_rate": round(win_rate(rows), 6),
                "turnover": round(gross_notional / avg_equity, 6) if avg_equity > 0 else 0.0,
            }
        return out

    def _reset_runtime_state(self) -> None:
        # Backtest starts from a clean portfolio state.
        for fp in [
            DATA_DIR / "portfolio_retail.json",
            DATA_DIR / "portfolio_institutional.json",
            DATA_DIR / "portfolio_alpha.json",
            DATA_DIR / "stop_orders_retail.json",
            DATA_DIR / "stop_orders_institutional.json",
            DATA_DIR / "validated_signals.json",
        ]:
            if fp.exists():
                fp.unlink(missing_ok=True)
        # Write empty sentinel files so execution_agent never reads stale state.
        # Retail sentinel uses capital=0 when disabled so it doesn't inflate equity.
        retail_enabled = bool(self.config.get("retail", {}).get("enabled", True))
        for mode in ["retail", "institutional"]:
            cfg = self.config.get(mode, {})
            if mode == "retail" and not retail_enabled:
                capital = 0.0
            else:
                capital = float(cfg.get("capital", 10000.0))
            self.write_json(DATA_DIR / f"portfolio_{mode}.json", {
                "mode": mode,
                "cash": capital,
                "total_equity": capital,
                "peak_equity": capital,
                "daily_start_equity": capital,
                "drawdown_pct": 0.0,
                "initial_capital": capital,
                "realized_pnl": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "positions": {},
                "trades": [],
            })
            self.write_json(DATA_DIR / f"stop_orders_{mode}.json", {})

    def _set_backtest_context(self, day: date, universe: list[str]) -> None:
        ctx = {
            "enabled": True,
            "simulated_now": self._iso(day),
            "lookahead_cutoff": self._iso(day),
            "frozen_universe": universe,
            "fees": self._cfg.get("fees", {}),
            "slippage_model": self._cfg.get("slippage_model", {"default_bps": 10}),
        }
        self.write_json(DATA_DIR / "backtest_context.json", ctx)

    def _clear_backtest_context(self) -> None:
        fp = DATA_DIR / "backtest_context.json"
        # Scrive sempre enabled:false invece di cancellare il file,
        # così market_data_agent trova sempre un JSON valido e non entra mai in modalità storica
        self.write_json(fp, {"enabled": False})

    @staticmethod
    def _iso(day: date) -> str:
        return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat()

