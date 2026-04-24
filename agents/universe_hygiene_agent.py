from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent, DATA_DIR


class UniverseHygieneAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("universe_hygiene_agent")
        cfg = self.config.get("universe_hygiene", {})
        self._enabled = bool(cfg.get("enabled", True))
        self._min_bars = int(cfg.get("min_bars", 60))
        self._min_dollar_volume = float(cfg.get("min_dollar_volume", 1_000_000.0))
        self._max_stale_days = int(cfg.get("max_stale_days", 7))
        self._lookback_period = str(cfg.get("lookback_period", "1y"))
        self._include_symbols = set(cfg.get("include_symbols", []))
        self._exclude_symbols = set(cfg.get("exclude_symbols", []))
        self._apply_curated_snapshot = bool(cfg.get("apply_curated_snapshot", False))
        self._discovery_enabled = bool(self.config.get("universe_discovery", {}).get("enabled", False))

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self._enabled:
                self.logger.info("Universe hygiene disabled by config")
                self.mark_done()
                return True

            bt_ctx = self.read_json(DATA_DIR / "backtest_context.json") or {}
            if bt_ctx.get("enabled"):
                self.logger.info("Universe hygiene skipped in backtest mode")
                self.mark_done()
                return True

            symbols, candidate_origin = self._collect_symbols()
            symbol_rows: list[dict] = []
            valid_symbols: list[str] = []

            for symbol in symbols:
                row = self._evaluate_symbol(symbol)
                row["candidate_origin"] = candidate_origin.get(symbol)
                symbol_rows.append(row)
                if row["status"] == "valid":
                    valid_symbols.append(symbol)

            generated_at = datetime.now(timezone.utc).isoformat()
            report = {
                "generated_at": generated_at,
                "config": {
                    "min_bars": self._min_bars,
                    "min_dollar_volume": self._min_dollar_volume,
                    "max_stale_days": self._max_stale_days,
                    "lookback_period": self._lookback_period,
                    "apply_curated_snapshot": self._apply_curated_snapshot,
                },
                "summary": {
                    "total_symbols": len(symbols),
                    "valid_symbols": len(valid_symbols),
                    "excluded_symbols": len(symbols) - len(valid_symbols),
                    "reason_counts": self._reason_counts(symbol_rows),
                },
                "symbols": symbol_rows,
            }

            curated = {
                "generated_at": generated_at,
                "symbols": valid_symbols,
                "metadata": {
                    "source_symbol_count": len(symbols),
                    "curated_symbol_count": len(valid_symbols),
                    "excluded_symbol_count": len(symbols) - len(valid_symbols),
                    "applied_to_runtime": self._apply_curated_snapshot,
                },
            }

            self.write_json(DATA_DIR / "universe_health_report.json", report)
            self.write_json(DATA_DIR / "universe_snapshot_curated.json", curated)

            if self._apply_curated_snapshot:
                self.write_json(DATA_DIR / "active_universe_snapshot.json", curated)

            self.update_shared_state("data_freshness.universe_health", generated_at)
            self.update_shared_state("data_freshness.universe_snapshot_curated", generated_at)
            self.logger.info(
                "Universe hygiene completed | total=%s valid=%s excluded=%s",
                len(symbols),
                len(valid_symbols),
                len(symbols) - len(valid_symbols),
            )
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    def _collect_symbols(self) -> tuple[list[str], dict[str, str | None]]:
        symbols: list[str] = []
        candidate_origin: dict[str, str | None] = {}
        master_universe = self.config.get("master_universe", {})
        for category, rows in master_universe.items():
            if str(category).startswith("_"):
                continue
            for symbol in rows:
                if symbol not in symbols:
                    symbols.append(symbol)
                    candidate_origin[symbol] = None

        for symbol in self.config.get("backtesting", {}).get("universe_snapshot", []):
            if symbol not in symbols:
                symbols.append(symbol)
                candidate_origin[symbol] = None

        if self._discovery_enabled:
            discovery = self.read_json(DATA_DIR / "universe_discovery_candidates.json")
            for row in discovery.get("candidates", []):
                symbol = row.get("symbol")
                if not symbol:
                    continue
                if symbol not in symbols:
                    symbols.append(symbol)
                candidate_origin[symbol] = row.get("candidate_origin", "discovery_provider")

        for symbol in self._include_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
                candidate_origin[symbol] = "manual_include"

        filtered = [symbol for symbol in symbols if symbol not in self._exclude_symbols]
        return filtered, candidate_origin

    def _evaluate_symbol(self, symbol: str) -> dict:
        history = self._fetch_history(symbol)
        if history is None or history.empty or "Close" not in history.columns:
            return self._excluded_row(symbol, ["NO_DATA"])

        closes = history["Close"].dropna()
        volumes = history.get("Volume", pd.Series(dtype=float)).fillna(0)
        if closes.empty:
            return self._excluded_row(symbol, ["NO_DATA"])

        bars = int(len(closes))
        reasons: list[str] = []

        if bars < self._min_bars:
            reasons.append("INSUFFICIENT_BARS")

        latest_dt = closes.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        stale_days = max(0, (datetime.now(timezone.utc) - latest_dt).days)
        if stale_days > self._max_stale_days:
            reasons.append("STALE_DATA")

        joined = history.copy()
        joined["dollar_volume"] = joined.get("Close", 0) * joined.get("Volume", 0)
        median_dollar_volume = float(joined["dollar_volume"].dropna().median()) if "dollar_volume" in joined else 0.0
        if median_dollar_volume < self._min_dollar_volume:
            reasons.append("ILLIQUID")

        status = "valid" if not reasons else "excluded"
        return {
            "symbol": symbol,
            "status": status,
            "reason_codes": reasons,
            "bars": bars,
            "latest_date": latest_dt.date().isoformat(),
            "stale_days": stale_days,
            "median_dollar_volume": round(median_dollar_volume, 2),
            "avg_volume": round(float(volumes.mean()) if not volumes.empty else 0.0, 2),
        }

    def _fetch_history(self, symbol: str) -> pd.DataFrame | None:
        try:
            return yf.Ticker(symbol).history(period=self._lookback_period, interval="1d", auto_adjust=True)
        except Exception as exc:
            self.logger.debug("History fetch failed for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _excluded_row(symbol: str, reasons: list[str]) -> dict:
        return {
            "symbol": symbol,
            "status": "excluded",
            "reason_codes": reasons,
            "bars": 0,
            "latest_date": None,
            "stale_days": None,
            "median_dollar_volume": 0.0,
            "avg_volume": 0.0,
        }

    @staticmethod
    def _reason_counts(rows: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for row in rows:
            reasons = row.get("reason_codes", [])
            if not reasons and row.get("status") == "valid":
                counts["VALID"] = counts.get("VALID", 0) + 1
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return counts
