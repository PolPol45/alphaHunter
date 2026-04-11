from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from agents.base_agent import BaseAgent, DATA_DIR, REPORTS_DIR
from agents.execution_agent import ExecutionAgent
from agents.risk_agent import RiskAgent
from adapters.historical_data_store import HistoricalDataStore
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

            # Reset runtime state for clean replay.
            self._reset_runtime_state()

            equity_curve: list[float] = []
            returns: list[float] = []
            prev_equity: float | None = None

            day = start
            steps = 0
            while day <= end:
                steps += 1
                self._set_backtest_context(day, universe)

                market = self._store.get_snapshot_at(day, universe)
                market = self._filter_no_lookahead(market, day)
                self.write_json(DATA_DIR / "market_data.json", market)

                macro = self._build_macro_snapshot(day, market)
                self.write_json(DATA_DIR / "macro_snapshot.json", macro)
                self.write_json(DATA_DIR / "market_regime.json", self._build_market_regime(day, macro))

                sector = self._build_sector_scorecard(day)
                self.write_json(DATA_DIR / "sector_scorecard.json", sector)

                stock_scores = self._build_stock_scores(day, universe)
                self.write_json(DATA_DIR / "stock_scores.json", stock_scores)

                bull = self._build_bull_signals(stock_scores, macro)
                bear = self._build_bear_signals(stock_scores, macro)
                crypto = self._build_crypto_signals(day, market)
                self.write_json(DATA_DIR / "bull_signals.json", bull)
                self.write_json(DATA_DIR / "bear_signals.json", bear)
                self.write_json(DATA_DIR / "crypto_signals.json", crypto)

                # Feed placeholders used by downstream agents expecting these files.
                self.write_json(DATA_DIR / "news_feed.json", {"generated_at": self._iso(day), "items": [], "top_alerts": []})
                self.write_json(DATA_DIR / "signals.json", {
                    "timestamp": self._iso(day),
                    "retail": {},
                    "institutional": {},
                    "scanner": {"retail_top_candidates": [], "institutional_top_candidates": []},
                    "context": {"macro_bias": macro.get("market_bias", 0.0)},
                })
                self.write_json(DATA_DIR / "alpha_signals.json", {"timestamp": self._iso(day), "signals": {}, "scanner": {"top_candidates": []}})

                if not self._risk.run():
                    raise RuntimeError(f"RiskAgent failed at {day}")
                if not self._execution.run():
                    raise RuntimeError(f"ExecutionAgent failed at {day}")

                eq = self._current_total_equity()
                equity_curve.append(eq)
                if prev_equity and prev_equity > 0:
                    returns.append((eq - prev_equity) / prev_equity)
                prev_equity = eq

                if steps % 20 == 0:
                    self.logger.info(f"Backtest progress: {day} | equity={eq:.2f}")

                day += timedelta(days=1)

            report = self._build_report(start, end, universe, equity_curve, returns)
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
            out_assets[sym] = {
                "last_price": last_price,
                "ohlcv_1d": c1,
                "ohlcv_4h": c4 or c1,
                "orderbook": data.get("orderbook", {"bids": [], "asks": []}),
                "volume_24h": float(data.get("volume_24h", 0.0)),
            }
        market["assets"] = out_assets
        market["timestamp"] = self._iso(day)
        market["data_source"] = "backtest_historical"
        return market

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
        regime = "RISK_ON" if bias > 0.25 else ("RISK_OFF" if bias < -0.25 else "NEUTRAL")
        return {
            "generated_at": self._iso(day),
            "regime": regime,
            "score": round(bias, 4),
            "macro_factors": {},
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
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "MKRUSDT", "DYDXUSDT", "LDOUSDT"]:
            if sym not in assets:
                continue
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
                "bucket": "CORE" if sym in {"BTCUSDT", "ETHUSDT"} else "DEFI_BRIDGE",
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

    def _current_total_equity(self) -> float:
        retail = self.read_json(DATA_DIR / "portfolio_retail.json")
        inst = self.read_json(DATA_DIR / "portfolio_institutional.json")
        return float(retail.get("total_equity", 0.0)) + float(inst.get("total_equity", 0.0))

    def _build_report(self, start: date, end: date, universe: list[str], equity: list[float], returns: list[float]) -> dict:
        retail = self.read_json(DATA_DIR / "portfolio_retail.json")
        inst = self.read_json(DATA_DIR / "portfolio_institutional.json")
        all_trades = retail.get("trades", []) + inst.get("trades", [])
        avg_eq = sum(equity) / len(equity) if equity else 0.0
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
            "summary": {
                "trades": len(all_trades),
                "final_equity": round(equity[-1], 4) if equity else 0.0,
                "start_equity": round(equity[0], 4) if equity else 0.0,
            },
        }

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
                fp.unlink()

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
        if fp.exists():
            fp.unlink()

    @staticmethod
    def _iso(day: date) -> str:
        return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat()
