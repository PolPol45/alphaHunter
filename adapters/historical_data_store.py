import json
import random
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# --- Fase 6: Refactoring Polars e Parquet ---
import polars as pl
import requests

from agents.base_agent import BASE_DIR, DATA_DIR

logger = logging.getLogger("historical_store")


def fetch_kraken_ohlcv(symbol: str, interval: int = 1440, since: int = None) -> pl.DataFrame:
    """
    Scarica dati storici direttamente da Kraken API (Public), immune al survivorship bias.
    interval: 1440 = 1D, 240 = 4H
    """
    try:
        url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval}"
        if since:
            url += f"&since={since}"
            
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("error"):
            logger.warning(f"Kraken history error for {symbol}: {data['error']}")
            return pl.DataFrame()
            
        res = data.get("result", {})
        pair_key = [k for k in res.keys() if k != "last"][0]
        candles = res[pair_key]
        
        # [time, open, high, low, close, vwap, volume, count]
        pl_data = {
            "t": [int(c[0]) for c in candles],
            "o": [float(c[1]) for c in candles],
            "h": [float(c[2]) for c in candles],
            "l": [float(c[3]) for c in candles],
            "c": [float(c[4]) for c in candles],
            "v": [float(c[6]) for c in candles],
        }
        return pl.DataFrame(pl_data)
    except Exception as e:
        logger.error(f"Errore Kraken Fetch {symbol}: {e}")
        return pl.DataFrame()


def fetch_real_snapshots(symbols: list[str], start: date, end: date, bt_dir: Path) -> None:
    """
    Fase 6: Download storico massivo in Parquet con Polars.
    CRYPTO  -> Kraken API (pubblica, no survivorship bias)
    EQUITY  -> yfinance BATCH (tutte le azioni in 1 sola chiamata API, ~20x piu veloce)
    """
    bt_dir.mkdir(parents=True, exist_ok=True)
    fetch_start = start - timedelta(days=400)
    start_ts = int(datetime.combine(fetch_start, datetime.min.time(), tzinfo=timezone.utc).timestamp())

    written = 0

    # Separa simboli gia in cache, crypto e equity
    crypto_todo = []
    equity_todo = []
    for s in symbols:
        parquet_file = bt_dir / f"sym_{s}.parquet"
        if parquet_file.exists():
            continue  # gia in cache
        is_crypto = (
            s.endswith(("USDT", "XBT", "ETH", "XRP", "BNB", "SOL", "ADA", "DOT"))
            or (s.endswith("USD") and len(s) <= 8)
        )
        if is_crypto:
            crypto_todo.append(s)
        else:
            equity_todo.append(s)

    # -- 1. CRYPTO via Kraken --------------------------------------------------
    for s in crypto_todo:
        parquet_file = bt_dir / f"sym_{s}.parquet"
        print(f"[HistoricalStore/Kraken] Crypto {s}...")
        df_1d = fetch_kraken_ohlcv(s, interval=1440, since=start_ts)
        if len(df_1d) > 0:
            df_1d.write_parquet(parquet_file)
            written += 1

    # -- 2. EQUITY via yfinance BATCH ------------------------------------------
    if equity_todo:
        print(f"[HistoricalStore/yfinance] BATCH download di {len(equity_todo)} simboli equity in 1 chiamata...")
        try:
            import yfinance as yf
            import pandas as pd

            raw = yf.download(
                equity_todo,
                start=fetch_start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )

            if raw is None or raw.empty:
                print("[HistoricalStore/yfinance] Nessun dato ricevuto dal batch.")
            else:
                for sym in equity_todo:
                    parquet_file = bt_dir / f"sym_{sym}.parquet"
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            lvl0 = raw.columns.get_level_values(0).unique().tolist()
                            if sym not in lvl0:
                                continue
                            df_sym = raw[sym].copy()
                        else:
                            df_sym = raw.copy()

                        df_sym = df_sym.dropna(how="all").reset_index()
                        # Flatten eventuali tuple nei nomi colonne
                        df_sym.columns = [
                            c[0] if isinstance(c, tuple) else c for c in df_sym.columns
                        ]
                        time_col = df_sym.columns[0]

                        needed = {"Open", "High", "Low", "Close", "Volume"}
                        if not needed.issubset(set(df_sym.columns)):
                            continue

                        pl_data = {
                            "t": [int(pd.Timestamp(x).timestamp()) for x in df_sym[time_col]],
                            "o": df_sym["Open"].astype(float).values.flatten().tolist(),
                            "h": df_sym["High"].astype(float).values.flatten().tolist(),
                            "l": df_sym["Low"].astype(float).values.flatten().tolist(),
                            "c": df_sym["Close"].astype(float).values.flatten().tolist(),
                            "v": df_sym["Volume"].astype(float).values.flatten().tolist(),
                        }
                        df_pl = pl.DataFrame(pl_data)
                        if len(df_pl) > 0:
                            df_pl.write_parquet(parquet_file)
                            written += 1
                    except Exception as e:
                        logger.warning(f"Skip equity {sym}: {e}")
                        continue

        except Exception as e:
            logger.error(f"yfinance batch fallito: {e}")

    print(f"[HistoricalStore] Completato. {written} file Parquet generati.")


class HistoricalDataStore:
    """
    Fase 6: Memory-efficient storage. Legge file .parquet usando Polars lazy/eager loading
    garantendo velocita mostruose sul caricamento OOS Walk-Forward.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.base_dir = BASE_DIR
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.bt_dir = self.base_dir / "backtest_data"
        self.bt_dir.mkdir(parents=True, exist_ok=True)
        # Cache per tenere memorizzata la versione polars in RAM
        self._cache_pl: dict[str, pl.DataFrame] = {}

    def freeze_universe(self, universe: list[str]) -> list[str]:
        return list(dict.fromkeys(universe))

    def get_snapshot_at(self, day: date, universe: list[str]) -> dict:
        """Restituisce la foto del mercato alla data X. Estrazione ultra rapida O(1) con Polars."""
        cutoff = int(datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc).timestamp())
        assets: dict = {}

        for sym in universe:
            df = self._load_sym_parquet(sym)
            if df is None or len(df) == 0:
                continue
                
            # Filtro Polars vettorializzato: mantieni solo t <= cutoff
            filtered = df.filter(pl.col("t") <= cutoff)
            if len(filtered) == 0:
                continue
                
            # Conversione in dizionario nativo per la compatibilita downstream
            c1 = filtered.to_dicts()
            
            last_price = c1[-1]["c"]
            assets[sym] = {
                "last_price": last_price,
                "ohlcv_1d": c1,
                "ohlcv_4h": c1,  # placeholder per 4h
                "orderbook": {"bids": [], "asks": []},
                "volume_24h": c1[-1]["v"],
            }

        return {
            "timestamp": datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "data_source": "polars_parquet_kraken",
            "assets": assets,
            "world_events": [],
        }

    def _load_sym_parquet(self, sym: str) -> pl.DataFrame | None:
        if sym in self._cache_pl:
            return self._cache_pl[sym]
        fp = self.bt_dir / f"sym_{sym}.parquet"
        if not fp.exists():
            return None
        try:
            df = pl.read_parquet(fp)
            self._cache_pl[sym] = df
            return df
        except Exception as e:
            logger.error(f"Parquet Read Error per {sym}: {e}")
            return None
