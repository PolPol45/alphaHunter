"""
test_feeds.py — verifica connessioni live a OpenBB/yfinance e Binance
Esegui: python test_feeds.py
"""

import sys
import time
from datetime import datetime, timezone

# ─── colori ANSI ───────────────────────────────────────────────────────────────
OK  = "\033[92m[OK] \033[0m"
ERR = "\033[91m[ERR]\033[0m"
INF = "\033[94m[..] \033[0m"
HDR = "\033[1;33m"
RST = "\033[0m"

CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
YF_SYMBOLS     = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]

failures: list[str] = []

def sep(title: str) -> None:
    print(f"\n{HDR}{'─'*60}{RST}")
    print(f"{HDR}  {title}{RST}")
    print(f"{HDR}{'─'*60}{RST}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. yfinance — prezzi live
# ══════════════════════════════════════════════════════════════════════════════
sep("1 · yfinance (OpenBB primary source)")

try:
    import yfinance as yf
    print(f"{OK} yfinance importato")

    print(f"{INF} download prezzi (period=1d interval=1d) …")
    t0 = time.time()
    data = yf.download(YF_SYMBOLS, period="2d", interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    elapsed = time.time() - t0

    if data.empty:
        print(f"{ERR} yfinance: DataFrame vuoto — nessun dato ricevuto")
        failures.append("yfinance: DataFrame vuoto")
    else:
        try:
            close = data["Close"]
        except KeyError:
            close = data.xs("Close", axis=1, level=0)

        print(f"{OK} Download completato in {elapsed:.2f}s | candles={len(data)}")
        print(f"\n  {'Simbolo':<12} {'Ultimo prezzo':>14}  {'Timestamp'}")
        print(f"  {'─'*12} {'─'*14}  {'─'*26}")
        for sym in YF_SYMBOLS:
            try:
                col = close[sym] if sym in close.columns else close
                price = float(col.dropna().iloc[-1])
                ts    = col.dropna().index[-1]
                print(f"  {sym:<12} {price:>14,.2f}  {ts}")
            except Exception as e:
                print(f"  {sym:<12} {'N/A':>14}  ({e})")
                failures.append(f"yfinance {sym}: {e}")

except ImportError:
    print(f"{ERR} yfinance non installato — esegui: pip install yfinance")
    failures.append("yfinance: ImportError")
except Exception as e:
    print(f"{ERR} yfinance errore: {e}")
    failures.append(f"yfinance: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. OpenBB SDK — connessione
# ══════════════════════════════════════════════════════════════════════════════
sep("2 · OpenBB SDK")

try:
    from openbb import obb
    print(f"{OK} OpenBB SDK importato")

    sym = "BTC-USD"
    print(f"{INF} obb.equity.price.historical({sym!r}) …")
    t0  = time.time()
    res = obb.equity.price.historical(sym, provider="yfinance")
    elapsed = time.time() - t0

    df = res.to_df()
    if df.empty:
        print(f"{ERR} OpenBB: DataFrame vuoto")
        failures.append("OpenBB: DataFrame vuoto")
    else:
        price = float(df["close"].dropna().iloc[-1])
        ts    = df.index[-1]
        print(f"{OK} OpenBB risponde in {elapsed:.2f}s | {sym} close={price:,.2f} @ {ts}")

except ImportError:
    print(f"{ERR} OpenBB non installato — esegui: pip install openbb")
    failures.append("OpenBB: ImportError")
except Exception as e:
    print(f"{ERR} OpenBB errore: {e}")
    failures.append(f"OpenBB: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Binance REST — ping + ticker + klines
# ══════════════════════════════════════════════════════════════════════════════
sep("3 · Binance public REST")

try:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from adapters.binance_client import BinanceClient

    client = BinanceClient({})

    # 3a. ping
    ok = client.connect()
    if ok:
        print(f"{OK} Binance /api/v3/ping OK")
    else:
        print(f"{ERR} Binance ping fallito: {client.last_error}")
        failures.append(f"Binance ping: {client.last_error}")

    # 3b. 24h ticker
    print(f"\n  {'Simbolo':<12} {'Last price':>12}  {'24h Vol':>14}  {'24h Chg%':>9}")
    print(f"  {'─'*12} {'─'*12}  {'─'*14}  {'─'*9}")
    for sym in CRYPTO_SYMBOLS:
        try:
            t0 = time.time()
            tick = client.get_ticker_24h(sym)
            ms   = (time.time() - t0) * 1000
            price  = float(tick.get("last_price", 0))
            vol    = float(tick.get("volume_24h", 0))
            chg    = float(tick.get("change_pct", 0))
            print(f"  {sym:<12} {price:>12,.2f}  {vol:>14,.0f}  {chg:>+8.2f}%  ({ms:.0f}ms)")
        except Exception as e:
            print(f"  {sym:<12} ERRORE: {e}")
            failures.append(f"Binance ticker {sym}: {e}")

    # 3c. klines (1d, ultime 3 candele)
    print(f"\n{INF} klines 1d per BTCUSDT (ultime 3) …")
    try:
        candles = client.get_ohlcv("BTCUSDT", limit=3)
        if candles:
            print(f"{OK} {len(candles)} candele ricevute")
            for c in candles:
                ts = datetime.fromtimestamp(c["t"], tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"    {ts}  o={c['o']:.2f}  h={c['h']:.2f}  l={c['l']:.2f}  c={c['c']:.2f}  v={c['v']:.2f}")
        else:
            print(f"{ERR} klines vuoto")
            failures.append("Binance klines BTCUSDT: vuoto")
    except Exception as e:
        print(f"{ERR} klines errore: {e}")
        failures.append(f"Binance klines: {e}")

except Exception as e:
    print(f"{ERR} BinanceClient errore: {e}")
    failures.append(f"BinanceClient: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Simulazione ciclo bot — MarketDataAgent.run_once()
# ══════════════════════════════════════════════════════════════════════════════
sep("4 · Ciclo MarketDataAgent (1 run)")

try:
    import json, os
    sys.path.insert(0, os.path.dirname(__file__))
    from agents.market_data_agent import MarketDataAgent

    agent = MarketDataAgent()
    print(f"{INF} run_once() in corso …")
    t0 = time.time()
    agent.run_once()
    elapsed = time.time() - t0

    data_path = os.path.join(os.path.dirname(__file__), "data", "market_data.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            md = json.load(f)
        src    = md.get("data_source", "?")
        assets = md.get("assets", {})
        ts     = md.get("timestamp", "?")
        print(f"{OK} run_once completato in {elapsed:.2f}s | source={src} | asset={len(assets)} | ts={ts}")

        print(f"\n  {'Asset':<12} {'Last price':>12}  {'Candles 1d':>10}  {'Candles 4h':>10}")
        print(f"  {'─'*12} {'─'*12}  {'─'*10}  {'─'*10}")
        for sym, v in assets.items():
            price = v.get("last_price", 0)
            n1d   = len(v.get("ohlcv_1d", []))
            n4h   = len(v.get("ohlcv_4h", []))
            print(f"  {sym:<12} {price:>12,.2f}  {n1d:>10}  {n4h:>10}")
    else:
        print(f"{ERR} market_data.json non trovato dopo run_once()")
        failures.append("MarketDataAgent: market_data.json assente")

except Exception as e:
    import traceback
    print(f"{ERR} MarketDataAgent errore: {e}")
    traceback.print_exc()
    failures.append(f"MarketDataAgent: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Riepilogo finale
# ══════════════════════════════════════════════════════════════════════════════
sep("RIEPILOGO")
if not failures:
    print(f"{OK} Tutte le connessioni funzionanti. Bot riceve feed live correttamente.")
else:
    print(f"{ERR} {len(failures)} problema/i rilevato/i:\n")
    for i, f in enumerate(failures, 1):
        print(f"  {i}. {f}")
    print(f"\n  Controlla rete / dipendenze / config.json")

print()
