from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent, DATA_DIR


class FeatureStoreAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("feature_store_agent")
        cfg = self.config.get("feature_store", {})
        self._enabled = bool(cfg.get("enabled", True))
        self._lookback_period = str(cfg.get("lookback_period", "1y"))
        self._output_format = str(cfg.get("output_format", "jsonl"))
        self._output_path = DATA_DIR / str(cfg.get("output_path", "cross_sectional_features.jsonl"))
        self._history_path = DATA_DIR / str(cfg.get("history_output_path", "cross_sectional_features_history.jsonl"))

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self._enabled:
                self.logger.info("Feature store disabled by config")
                self.mark_done()
                return True

            scores_doc = self.read_json(DATA_DIR / "stock_scores.json")
            macro_doc = self.read_json(DATA_DIR / "macro_snapshot.json")
            symbols = [row.get("symbol") for row in scores_doc.get("scores", []) if row.get("symbol")]

            if not symbols:
                curated = self.read_json(DATA_DIR / "universe_snapshot_curated.json")
                symbols = curated.get("symbols", [])

            if not symbols:
                self.logger.warning("No symbols available for feature store")
                self.mark_done()
                return True

            score_map = {row.get("symbol"): row for row in scores_doc.get("scores", [])}
            # macro_snapshot.json may not exist in live mode; use market_regime.json as fallback
            if not macro_doc:
                macro_doc = self.read_json(DATA_DIR / "market_regime.json") or {}
            macro_features = self._macro_features(macro_doc)

            rows: list[dict] = []
            as_of = datetime.now(timezone.utc).date().isoformat()

            for symbol in symbols:
                history = self._fetch_history(symbol)
                if history is None or history.empty or "Close" not in history.columns:
                    continue
                base = self._market_features(history)
                if not base:
                    continue

                score_row = score_map.get(symbol, {})
                record = {
                    "date": as_of,
                    "symbol": symbol,
                    **base,
                    "roe": score_row.get("roe"),
                    "roa": score_row.get("roa"),
                    "debt_to_equity": score_row.get("debt_to_equity"),
                    "current_ratio": score_row.get("current_ratio"),
                    "analyst_count": score_row.get("analyst_count"),
                    "recommendation_mean": score_row.get("recommendation_mean"),
                    "beta": score_row.get("beta"),
                    "sector": score_row.get("sector", "Unknown_Sector"),
                    "benchmark": score_row.get("benchmark", "SPY"),
                    "sector_dummy": self._sector_dummy(score_row.get("sector", "Unknown_Sector")),
                    **macro_features,
                }
                rows.append(record)

            self._write_rows(rows)
            self._append_history(rows)

            meta = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(rows),
                "symbols_requested": len(symbols),
                "output_format": self._output_format,
                "output_path": str(self._output_path.name),
            }
            self.write_json(DATA_DIR / "cross_sectional_features.meta.json", meta)
            self.update_shared_state("data_freshness.cross_sectional_features", meta["generated_at"])
            self.logger.info("Feature store generated | rows=%s", len(rows))
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    def _fetch_history(self, symbol: str) -> pd.DataFrame | None:
        try:
            return yf.Ticker(symbol).history(period=self._lookback_period, interval="1d", auto_adjust=True)
        except Exception as exc:
            self.logger.debug("Feature history fetch failed for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _market_features(history: pd.DataFrame) -> dict:
        closes = history["Close"].dropna()
        volumes = history.get("Volume", pd.Series(index=history.index, dtype=float)).fillna(0)
        if len(closes) < 40:
            return {}

        returns = closes.pct_change().dropna()
        latest = float(closes.iloc[-1])

        def momentum(days: int) -> float | None:
            if len(closes) <= days:
                return None
            prev = float(closes.iloc[-1 - days])
            if prev == 0:
                return None
            return round((latest / prev) - 1.0, 6)

        rolling_max_90 = closes.tail(90).max() if len(closes) >= 90 else closes.max()
        drawdown_rolling_90 = (latest / float(rolling_max_90) - 1.0) if rolling_max_90 else 0.0

        dollar_volume = (history.get("Close", 0) * history.get("Volume", 0)).dropna()
        avg_volume_30 = float(volumes.tail(30).mean()) if not volumes.empty else 0.0
        median_dv_30 = float(dollar_volume.tail(30).median()) if not dollar_volume.empty else 0.0

        amihud = None
        tail = pd.DataFrame({"ret": returns.tail(30), "dv": dollar_volume.tail(30)})
        tail = tail.replace([np.inf, -np.inf], np.nan).dropna()
        tail = tail[tail["dv"] > 0]
        if not tail.empty:
            amihud = float((tail["ret"].abs() / tail["dv"]).mean())

        spread_proxy = None
        if "High" in history.columns and "Low" in history.columns:
            hl = history[["High", "Low", "Close"]].tail(30).replace([np.inf, -np.inf], np.nan).dropna()
            if not hl.empty:
                spread_proxy = float(((hl["High"] - hl["Low"]) / hl["Close"].replace(0, np.nan)).mean())

        return {
            "close": round(latest, 6),
            "momentum_7d": momentum(7),
            "momentum_30d": momentum(30),
            "momentum_90d": momentum(90),
            "volatility_30d": round(float(returns.tail(30).std(ddof=0)) if len(returns) >= 2 else 0.0, 6),
            "drawdown_rolling_90d": round(float(drawdown_rolling_90), 6),
            "avg_volume_30d": round(avg_volume_30, 2),
            "median_dollar_volume_30d": round(median_dv_30, 2),
            "amihud_30d": round(amihud, 12) if amihud is not None else None,
            "spread_proxy_30d": round(spread_proxy, 8) if spread_proxy is not None else None,
        }

    @staticmethod
    def _sector_dummy(sector: str) -> str:
        return str(sector or "Unknown_Sector").strip().replace(" ", "_").upper()

    @staticmethod
    def _macro_features(macro_doc: dict) -> dict:
        series = macro_doc.get("series", {})

        def v(name: str):
            row = series.get(name, {})
            return row.get("value") if row.get("status") == "ok" else None

        return {
            "macro_market_bias": macro_doc.get("market_bias"),
            "macro_regime": "RISK_ON" if (macro_doc.get("market_bias") or 0) > 0.15 else (
                "RISK_OFF" if (macro_doc.get("market_bias") or 0) < -0.15 else "NEUTRAL"
            ),
            "macro_fed_funds": v("fed_funds"),
            "macro_cpi_yoy": v("cpi_yoy"),
            "macro_vix": v("vix"),
            "macro_dxy": v("dxy"),
            "macro_qe_qt_proxy": v("fed_balance_sheet_walcl"),
            "macro_tga_proxy": v("tga_balance_wtregen"),
            "macro_real_rate_10y": v("real_rate_10y"),
            "macro_yield_spread_10y_2y": v("yield_spread_10y_2y"),
            "macro_forward_guidance_proxy": v("forward_guidance_proxy"),
        }

    def _write_rows(self, rows: list[dict]) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._output_format == "parquet":
            try:
                pd.DataFrame(rows).to_parquet(self._output_path.with_suffix(".parquet"), index=False)
                return
            except Exception as exc:
                self.logger.warning("Parquet write failed (%s), falling back to JSONL", exc)

        with open(self._output_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_history(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
