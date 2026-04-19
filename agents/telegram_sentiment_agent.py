"""
TelegramSentimentAgent
======================
Legge messaggi da canali Telegram pubblici via Telethon (MTProto),
analizza il sentiment con Ollama (qwen2.5:3b), e scrive
data/telegram_sentiment.json ingestito da RiskAgent come news boost.

Canali monitorati:
  - unfolded_defi      (DeFi/on-chain)
  - TheCryptoGateway   (macro crypto)
  - unfolded           (general crypto news)
  - ayyyeandy          (crypto trader calls)
  - web3Mi             (Italian Web3 community)

Output: data/telegram_sentiment.json
  {
    "generated_at": "...",
    "overall_bias": 0.12,          # [-1, +1] media pesata
    "articles": [                  # formato compatibile RiskAgent _build_news_boost_map
      {"symbol": "BTC", "sentiment_score": 0.6, "ticker": "BTC", ...},
      ...
    ],
    "macro_bias": 0.15,
    "alerts": ["BTC breakout sopra 90k", ...],
    "channel_summary": { "unfolded_defi": {"bias": 0.3, "messages": 12}, ... }
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Allow running directly as script from trading_bot/
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from agents.base_agent import BaseAgent, DATA_DIR

# ── Canali da monitorare ────────────────────────────────────────────────────
CHANNELS = [
    "unfolded_defi",
    "TheCryptoGateway",
    "unfolded",
    "ayyyeandy",
    "web3Mi",
]

# ── Credenziali Telegram (da config o env) ──────────────────────────────────
TG_API_ID   = int(os.environ.get("TG_API_ID",   "32855229"))
TG_API_HASH = os.environ.get("TG_API_HASH", "3b6c403e1ffc82caea4322411a82ebb7")
TG_SESSION  = str(Path(__file__).parent.parent / "data" / "tg_alphahunter")

# ── Crypto symbol keywords per entity extraction ────────────────────────────
CRYPTO_KEYWORDS = {
    "BTC": ["btc", "bitcoin", "#btc", "#bitcoin"],
    "ETH": ["eth", "ethereum", "#eth", "#ethereum"],
    "SOL": ["sol", "solana", "#sol", "#solana"],
    "BNB": ["bnb", "binance", "#bnb"],
    "XRP": ["xrp", "ripple", "#xrp"],
    "AVAX": ["avax", "avalanche", "#avax"],
    "LINK": ["link", "chainlink", "#link"],
    "ARB": ["arb", "arbitrum", "#arb"],
    "OP": ["optimism", "#op"],
    "MATIC": ["matic", "polygon", "#matic"],
    "DOGE": ["doge", "dogecoin", "#doge"],
    "PEPE": ["pepe", "#pepe"],
    "SUI": ["sui", "#sui"],
    "APT": ["aptos", "#apt"],
    "TON": ["ton", "toncoin", "#ton"],
}

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5:3b"
MESSAGES_LIMIT  = 200  # ultimi N messaggi per canale
MAX_CHARS       = 3000  # tronca testo prima di mandarlo a Ollama


class TelegramSentimentAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__("telegram_sentiment_agent")
        self._ollama_endpoint = self.config.get("llm", {}).get(
            "endpoint", OLLAMA_ENDPOINT
        )
        self._ollama_model = self.config.get("llm", {}).get(
            "model", OLLAMA_MODEL
        )
        self._messages_limit = int(
            self.config.get("telegram_sentiment", {}).get("messages_limit", MESSAGES_LIMIT)
        )

    # ── Public interface ────────────────────────────────────────────────────

    def run(self) -> bool:
        self.mark_running()
        try:
            raw = asyncio.run(self._fetch_all_channels())
            if not raw:
                self.logger.warning("Nessun messaggio recuperato da Telegram")
                self.mark_done()
                return True

            result = self._analyze(raw)
            self.write_json(DATA_DIR / "telegram_sentiment.json", result)
            self.logger.info(
                f"Telegram sentiment scritto: overall_bias={result['overall_bias']:+.3f}, "
                f"assets={len(result['articles'])}, alerts={len(result['alerts'])}"
            )
            self.mark_done()
            return True
        except Exception as exc:
            self.mark_error(exc)
            return False

    # ── Telegram fetch ──────────────────────────────────────────────────────

    async def _fetch_all_channels(self) -> dict[str, list[str]]:
        try:
            from telethon import TelegramClient
            from telethon.errors import FloodWaitError
        except ImportError:
            self.logger.error("Telethon non installato. Esegui: pip install telethon")
            return {}

        result: dict[str, list[str]] = {}
        client = TelegramClient(TG_SESSION, TG_API_ID, TG_API_HASH)

        try:
            await client.connect()
            if not await client.is_user_authorized():
                self.logger.error(
                    "Telethon non autorizzato. Prima run interattiva richiesta: "
                    "esegui python -c \"from agents.telegram_sentiment_agent import *; "
                    "asyncio.run(tg_first_login())\""
                )
                return {}

            cutoff = datetime.now(timezone.utc) - timedelta(days=7)

            for channel in CHANNELS:
                try:
                    msgs: list[str] = []
                    async for msg in client.iter_messages(channel, limit=self._messages_limit):
                        if msg.date < cutoff:
                            break
                        if msg.text:
                            msgs.append(msg.text.strip())
                    result[channel] = msgs
                    self.logger.info(f"  {channel}: {len(msgs)} messaggi")
                except FloodWaitError as e:
                    self.logger.warning(f"FloodWait {channel}: aspetto {e.seconds}s")
                    await asyncio.sleep(e.seconds)
                except Exception as e:
                    self.logger.warning(f"Errore fetch {channel}: {e}")
                    result[channel] = []

        finally:
            await client.disconnect()

        return result

    # ── LLM analysis ────────────────────────────────────────────────────────

    def _analyze(self, raw: dict[str, list[str]]) -> dict:
        articles: list[dict] = []
        channel_summary: dict = {}
        all_biases: list[float] = []
        all_alerts: list[str] = []

        for channel, msgs in raw.items():
            if not msgs:
                channel_summary[channel] = {"bias": 0.0, "messages": 0}
                continue

            combined = "\n".join(msgs)[:MAX_CHARS]
            analysis = self._llm_analyze(channel, combined)

            bias = float(analysis.get("bias", 0.0))
            alerts = list(analysis.get("alerts", []))
            symbol_scores: dict[str, float] = analysis.get("symbol_scores", {})

            channel_summary[channel] = {
                "bias": round(bias, 3),
                "messages": len(msgs),
                "top_themes": analysis.get("themes", []),
            }
            all_biases.append(bias)
            all_alerts.extend(alerts[:3])

            # Converti in formato articles per _build_news_boost_map di RiskAgent
            for sym, score in symbol_scores.items():
                articles.append({
                    "symbol": sym,
                    "ticker": sym,
                    "sentiment_score": round(score, 4),
                    "relevance_score": 0.7,
                    "source": f"telegram:{channel}",
                    "headline": f"[{channel}] {analysis.get('summary', '')[:80]}",
                })

            # Se nessun simbolo specifico ma bias forte → applica a BTC/ETH come proxy macro
            if not symbol_scores and abs(bias) >= 0.2:
                for sym in ("BTC", "ETH"):
                    articles.append({
                        "symbol": sym,
                        "ticker": sym,
                        "sentiment_score": round(bias * 0.6, 4),
                        "relevance_score": 0.4,
                        "source": f"telegram:{channel}",
                        "headline": f"[{channel}] macro bias {bias:+.2f}",
                    })

        overall_bias = round(sum(all_biases) / len(all_biases), 4) if all_biases else 0.0

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_bias": overall_bias,
            "macro_bias": overall_bias,
            "articles": articles,
            "alerts": list(dict.fromkeys(all_alerts))[:10],  # dedup
            "channel_summary": channel_summary,
        }

    def _llm_analyze(self, channel: str, text: str) -> dict:
        crypto_list = ", ".join(CRYPTO_KEYWORDS.keys())
        prompt = f"""Sei un analista crypto. Analizza questi messaggi Telegram dal canale @{channel}.

MESSAGGI:
{text}

Rispondi SOLO con JSON valido (nessun testo extra) con questa struttura:
{{
  "bias": <float -1.0 a +1.0, sentiment generale del mercato crypto>,
  "summary": "<stringa, riassunto 1 frase>",
  "themes": ["<tema1>", "<tema2>"],
  "alerts": ["<alert importante se presente, es: BTC sopra 90k>"],
  "symbol_scores": {{
    "<SIMBOLO>": <float -1.0 a +1.0>
  }}
}}

Note:
- bias +1.0 = molto bullish, -1.0 = molto bearish, 0 = neutro
- symbol_scores solo per simboli menzionati esplicitamente: {crypto_list}
- alerts solo se c'è qualcosa di rilevante (price target, breakout, liquidazioni massive)
- se non ci sono messaggi significativi, bias=0 e arrays vuoti
"""
        try:
            resp = requests.post(
                self._ollama_endpoint,
                json={
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 400},
                },
                timeout=45,
            )
            if resp.status_code == 200:
                raw_json = resp.json().get("response", "{}")
                parsed = json.loads(raw_json)
                # Clamp bias in [-1, +1]
                parsed["bias"] = max(-1.0, min(1.0, float(parsed.get("bias", 0.0))))
                # Clamp symbol scores
                for sym in list(parsed.get("symbol_scores", {}).keys()):
                    parsed["symbol_scores"][sym] = max(
                        -1.0, min(1.0, float(parsed["symbol_scores"][sym]))
                    )
                return parsed
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM JSON parse error ({channel}): {e}")
        except Exception as e:
            self.logger.warning(f"LLM error ({channel}): {e}")

        # Fallback: keyword-based scoring
        return self._keyword_fallback(text)

    @staticmethod
    def _keyword_fallback(text: str) -> dict:
        text_lower = text.lower()
        bullish = sum(text_lower.count(w) for w in [
            "bullish", "pump", "ath", "breakout", "accumulate", "buy",
            "long", "moon", "rip", "surge", "rally", "uptrend",
        ])
        bearish = sum(text_lower.count(w) for w in [
            "bearish", "dump", "crash", "selloff", "short", "liquidation",
            "rekt", "bear", "falling", "breakdown", "risk off",
        ])
        total = bullish + bearish
        bias = round((bullish - bearish) / max(total, 1), 2) if total > 0 else 0.0

        symbol_scores: dict[str, float] = {}
        for sym, keywords in CRYPTO_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                symbol_scores[sym] = round(bias * 0.7, 3)

        return {"bias": bias, "symbol_scores": symbol_scores, "alerts": [], "themes": [], "summary": ""}


# ── First-login helper (da eseguire una volta manualmente) ──────────────────

async def tg_first_login():
    """Esegui una volta per autenticare la sessione Telethon."""
    try:
        from telethon import TelegramClient
    except ImportError:
        print("Installa telethon: pip install telethon")
        return
    client = TelegramClient(TG_SESSION, TG_API_ID, TG_API_HASH)
    await client.start()
    print("✅ Login Telegram completato. Sessione salvata in:", TG_SESSION)
    await client.disconnect()


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    import asyncio
    asyncio.run(tg_first_login())
