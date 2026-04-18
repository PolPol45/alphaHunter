"""
fetch_backtest_data.py
======================
Scarica dati storici reali da yfinance e li salva nel formato atteso da
HistoricalDataStore: backtest_data/market_YYYY-MM-DD.json

Uso:
    python fetch_backtest_data.py                        # date da config.json
    python fetch_backtest_data.py --start 2024-01-01 --end 2024-12-31

Il file per ogni giorno viene saltato se esiste già e --force non è passato.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agents.base_agent import BASE_DIR, CONFIG_PATH

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Installa yfinance e pandas: pip install yfinance pandas")
    sys.exit(1)


BT_DIR = BASE_DIR / "backtest_data"
BT_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def universe_from_config(cfg: dict) -> list[str]:
    mu = cfg.get("master_universe", {})
    symbols: list[str] = []
    # Solo large cap + ETF (esclude small cap distressed che sono spesso delisted)
    for key in ("equities_large_cap", "etf_long", "etf_hedge", "crypto_core", "crypto_defi_bridge"):
        symbols.extend(mu.get(key, []))
    symbols.extend(cfg.get("backtesting", {}).get("universe_snapshot", []))
    # Deduplica mantenendo ordine
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def fetch_all(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Scarica tutti i simboli in un batch (più veloce di uno per uno)."""
    print(f"Download dati per {len(symbols)} simboli: {start} → {end} ...")
    # yfinance vuole end + 1 giorno per includere end
    end_yf = (datetime.combine(end, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=end_yf,
        interval="1d",
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            if len(symbols) == 1:
                df = raw
            else:
                df = raw[sym] if sym in raw.columns.get_level_values(0) else pd.DataFrame()
            if df is not None and not df.empty:
                df = df.dropna(how="all")
                result[sym] = df
        except Exception:
            pass
    print(f"  Ricevuti dati per {len(result)}/{len(symbols)} simboli")
    return result


def build_ohlcv_from_df(df: pd.DataFrame, as_of: date, bars: int = 60) -> list[dict]:
    """Converte un DataFrame yfinance in lista OHLCV compatibile con HistoricalDataStore."""
    df = df[df.index.date <= as_of].tail(bars)
    out: list[dict] = []
    for ts, row in df.iterrows():
        t = int(datetime.combine(ts.date(), datetime.min.time(), tzinfo=timezone.utc).timestamp())
        try:
            out.append({
                "t": t,
                "o": round(float(row["Open"]), 6),
                "h": round(float(row["High"]), 6),
                "l": round(float(row["Low"]), 6),
                "c": round(float(row["Close"]), 6),
                "v": round(float(row.get("Volume", 0)), 2),
            })
        except Exception:
            continue
    return out


def build_snapshot(day: date, symbols: list[str], data: dict[str, pd.DataFrame]) -> dict:
    assets: dict = {}
    for sym in symbols:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        # Prezzo del giorno (o ultimo disponibile <= day)
        df_to = df[df.index.date <= day]
        if df_to.empty:
            continue
        last_row = df_to.iloc[-1]
        last_price = float(last_row["Close"])
        volume_24h = float(last_row.get("Volume", 0))

        ohlcv_1d = build_ohlcv_from_df(df, day, bars=60)
        # 4h: usiamo i daily come proxy (il backtest accetta anche questo)
        ohlcv_4h = ohlcv_1d[-30:]

        # Orderbook sintetico attorno al prezzo reale
        spread = last_price * 0.0005
        orderbook = {
            "bids": [[round(last_price - spread * i, 6), 1.0] for i in range(1, 6)],
            "asks": [[round(last_price + spread * i, 6), 1.0] for i in range(1, 6)],
        }

        assets[sym] = {
            "last_price": round(last_price, 6),
            "ohlcv_1d": ohlcv_1d,
            "ohlcv_4h": ohlcv_4h,
            "orderbook": orderbook,
            "volume_24h": round(volume_24h, 2),
        }

    return {
        "timestamp": datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
        "data_source": "yfinance_real",
        "assets": assets,
        "world_events": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica dati reali per il backtest")
    parser.add_argument("--start", help="Data inizio YYYY-MM-DD (default: da config.json)")
    parser.add_argument("--end", help="Data fine YYYY-MM-DD (default: da config.json)")
    parser.add_argument("--force", action="store_true", help="Sovrascrive file già esistenti")
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = cfg.get("backtesting", {})

    start_s = args.start or bt_cfg.get("start_date", "2025-01-01")
    end_s = args.end or bt_cfg.get("end_date", "2025-06-30")
    start = date.fromisoformat(start_s)
    end = date.fromisoformat(end_s)

    symbols = universe_from_config(cfg)
    if not symbols:
        print("Nessun simbolo trovato in config.json")
        sys.exit(1)

    print(f"Universo: {len(symbols)} simboli | Finestra: {start} → {end}")

    # Scarica tutto in un batch
    data = fetch_all(symbols, start - timedelta(days=365), end)  # +1 anno di storico per gli indicatori

    # Genera un file per ogni giorno
    day = start
    written = 0
    skipped = 0
    while day <= end:
        fp = BT_DIR / f"market_{day.isoformat()}.json"
        if fp.exists() and not args.force:
            skipped += 1
            day += timedelta(days=1)
            continue

        snapshot = build_snapshot(day, symbols, data)
        fp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        written += 1

        if written % 20 == 0:
            print(f"  Scritti {written} file... (ultimo: {day})")

        day += timedelta(days=1)

    print(f"\nFatto! Scritti: {written} | Saltati (già esistenti): {skipped}")
    print(f"File in: {BT_DIR}")
    print("\nOra puoi lanciare il backtest con:")
    print("  python orchestrator.py   (con mode: backtest in config.json)")


if __name__ == "__main__":
    main()
