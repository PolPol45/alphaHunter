from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf

from agents.base_agent import BaseAgent, DATA_DIR


class UniverseDiscoveryAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("universe_discovery_agent")
        cfg = self.config.get("universe_discovery", {})
        self._enabled = bool(cfg.get("enabled", True))
        self._watchlist = list(cfg.get("watchlist", []))
        self._max_candidates = int(cfg.get("max_candidates", 200))
        self._provider = str(cfg.get("provider", "yfinance"))

    def run(self) -> bool:
        self.mark_running()
        try:
            if not self._enabled:
                self.logger.info("Universe discovery disabled by config")
                self.mark_done()
                return True

            seeds = self._seed_symbols()
            candidates = []
            for symbol in seeds:
                row = self._describe_symbol(symbol)
                if row is not None:
                    candidates.append(row)
                if len(candidates) >= self._max_candidates:
                    break

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "provider": self._provider,
                "summary": {
                    "seed_symbols": len(seeds),
                    "candidates": len(candidates),
                },
                "candidates": candidates,
            }
            self.write_json(DATA_DIR / "universe_discovery_candidates.json", payload)
            self.update_shared_state("data_freshness.universe_discovery_candidates", payload["generated_at"])
            self.logger.info("Universe discovery generated | candidates=%s", len(candidates))
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    def _seed_symbols(self) -> list[str]:
        out: list[str] = []
        mu = self.config.get("master_universe", {})
        for key, rows in mu.items():
            if key.startswith("_"):
                continue
            if not (key.startswith("equities") or key.startswith("etf")):
                continue
            for symbol in rows:
                if symbol not in out:
                    out.append(symbol)

        for sector in self.config.get("sector_map", {}).values():
            for symbol in sector.get("members", []):
                if symbol not in out:
                    out.append(symbol)

        for symbol in self._watchlist:
            if symbol not in out:
                out.append(symbol)
        return out

    @staticmethod
    def _describe_symbol(symbol: str) -> dict | None:
        try:
            info = yf.Ticker(symbol).info or {}
            quote_type = str(info.get("quoteType", "")).upper()
            if quote_type and quote_type not in {"EQUITY", "ETF"}:
                return None
            return {
                "symbol": symbol,
                "long_name": info.get("longName") or info.get("shortName") or symbol,
                "quote_type": quote_type or None,
                "exchange": info.get("exchange"),
                "sector": info.get("sector"),
                "candidate_origin": "discovery_provider",
            }
        except Exception:
            return None
