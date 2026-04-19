"""
KrakenExecutionAdapter — xStocks & Crypto live execution via Kraken REST API.

Reads api_key / api_secret from config.json["kraken"].
If config["kraken"]["paper_trading"] is true, logs orders without sending them.

Supported asset classes:
  - tokenized_asset  (xStocks: AAPLx, NVDAx, SPYx …)
  - spot             (BTC, ETH, SOL …)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Optional

import requests

XSTOCKS_PAIRS: Dict[str, str] = {
    "AAPL": "AAPLx/USD",  "MSFT": "MSFTx/USD",  "NVDA": "NVDAx/USD",
    "TSLA": "TSLAx/USD",  "META": "METAx/USD",   "AMZN": "AMZNx/USD",
    "GOOGL": "GOOGLx/USD","AMD":  "AMDx/USD",    "COIN": "COINx/USD",
    "MSTR": "MSTRx/USD",  "SPY":  "SPYx/USD",    "QQQ":  "QQQx/USD",
    "IWM":  "IWMx/USD",   "GS":   "GSx/USD",     "JPM":  "JPMx/USD",
    "INTC": "INTCx/USD",  "NFLX": "NFLXx/USD",   "BAC":  "BACx/USD",
    "XOM":  "XOMx/USD",   "CVX":  "CVXx/USD",
}

CRYPTO_PAIRS: Dict[str, str] = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "SOL": "SOLUSD",
    "BNB": "BNBUSD",
}

_BASE = "https://api.kraken.com"


class KrakenExecutionAdapter:
    """
    Thin wrapper around Kraken REST API for order placement and account queries.
    Respects paper_trading flag — no real orders sent in paper mode.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("kraken_execution_adapter")
        cfg_path = Path(__file__).parent.parent / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        kraken_cfg = cfg.get("kraken", {})

        self._api_key    = kraken_cfg.get("api_key", "")
        self._api_secret = kraken_cfg.get("api_secret", "")
        self._paper      = bool(kraken_cfg.get("paper_trading", True))
        self._timeout    = int(kraken_cfg.get("timeout_seconds", 15))

        if not self._api_key or not self._api_secret:
            raise ValueError("Kraken API key/secret missing in config.json['kraken']")

    def symbol_to_pair(self, symbol: str) -> Optional[str]:
        return XSTOCKS_PAIRS.get(symbol) or CRYPTO_PAIRS.get(symbol)

    def is_xstock(self, symbol: str) -> bool:
        return symbol in XSTOCKS_PAIRS

    def place_order(
        self,
        pair: str,
        side: str,
        volume: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Dict:
        """
        Place order on Kraken. Returns dict with txid, descr, paper, error.
        In paper mode: logs only, no HTTP call.
        """
        params: Dict = {
            "ordertype": order_type,
            "type": side,
            "volume": str(round(volume, 8)),
            "pair": pair,
        }
        if order_type == "limit" and limit_price:
            params["price"] = str(round(limit_price, 6))
        if pair.endswith("x/USD"):
            params["asset_class"] = "tokenized_asset"

        if self._paper:
            self.logger.info(f"[PAPER] Kraken {side.upper()} {volume} {pair} @ {order_type}")
            return {"txid": [], "descr": params, "paper": True, "error": None}

        try:
            resp = self._private_post("/0/private/AddOrder", params)
            if resp.get("error"):
                self.logger.error(f"Kraken order error: {resp['error']}")
                return {"txid": [], "descr": params, "paper": False, "error": str(resp["error"])}
            result = resp.get("result", {})
            self.logger.info(f"Kraken order placed | {side.upper()} {volume} {pair} | txid={result.get('txid')}")
            return {"txid": result.get("txid", []), "descr": result.get("descr", {}), "paper": False, "error": None}
        except Exception as e:
            self.logger.error(f"Kraken place_order: {e}")
            return {"txid": [], "descr": params, "paper": False, "error": str(e)}

    def cancel_order(self, txid: str) -> Dict:
        if self._paper:
            return {"cancelled": True, "paper": True}
        return self._private_post("/0/private/CancelOrder", {"txid": txid}).get("result", {})

    def get_open_orders(self) -> Dict:
        if self._paper:
            return {}
        return self._private_post("/0/private/OpenOrders", {}).get("result", {}).get("open", {})

    def get_balances(self) -> Dict[str, float]:
        if self._paper:
            return {}
        raw = self._private_post("/0/private/Balance", {}).get("result", {})
        return {k: float(v) for k, v in raw.items() if float(v) > 0}

    def get_ticker(self, pair: str) -> Optional[float]:
        try:
            resp = requests.get(f"{_BASE}/0/public/Ticker", params={"pair": pair}, timeout=self._timeout)
            data = resp.json()
            if data.get("error"):
                return None
            return float(list(data["result"].values())[0]["c"][0])
        except Exception as e:
            self.logger.debug(f"get_ticker {pair}: {e}")
            return None

    def _private_post(self, uri_path: str, data: Dict) -> Dict:
        nonce = str(int(time.time() * 1000))
        data["nonce"] = nonce
        post_data = urllib.parse.urlencode(data)
        encoded   = (nonce + post_data).encode()
        message   = uri_path.encode() + hashlib.sha256(encoded).digest()
        secret    = base64.b64decode(self._api_secret)
        signature = base64.b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode()
        headers = {
            "API-Key":  self._api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(f"{_BASE}{uri_path}", data=post_data, headers=headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()
