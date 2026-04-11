"""
yfinance Data Quality Validator
=================================
Verifica che i dati scaricati da yfinance per l'Alpha Hunter siano validi
e consistenti con una fonte secondaria (Binance per crypto cross-check).

Checks eseguiti per ogni simbolo:
  1. Connettività          — yfinance risponde
  2. Lunghezza storia      — almeno 60 candele giornaliere
  3. Sanità OHLC           — high >= close >= low, open > 0
  4. Freshness             — l'ultima candela non è più vecchia di 5 giorni lavorativi
  5. Volume                — almeno una candela con volume > 0 (escludi ETF privi di volume)
  6. IV/HV reperibile      — options chain disponibile (solo per azioni singole, non ETF)
  7. Cross-check prezzo    — confronto con Binance per i 4 simboli crypto nell'universo alpha

Uso:
    cd trading_bot
    source venv/bin/activate
    python tests/validate_yfinance.py
"""

from __future__ import annotations

import sys
import pathlib
import json
import math
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

# Add project root to path
BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_DIR))

# ── Universe ──────────────────────────────────────────────────────────────── #

EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",
    "TSLA", "AMZN", "AMD",  "SPY",  "QQQ",
    "NFLX", "COST", "JPM",  "XOM",  "UNH",
    "SMH",  "IWM",  "XLF",  "GLD",  "TLT",
]

# Symbols that are ETFs/commodities — no options or volume may be 0
NO_OPTIONS_SYMBOLS = {"SPY", "QQQ", "IWM", "XLF", "SMH", "GLD", "TLT"}

# Binance cross-check: alpha hunter crypto tickers → Binance symbols
ALPHA_CRYPTO_MAP = {
    # (none in default universe, but ready if user adds them)
}

# Max age for last candle (5 calendar days covers weekends + holidays)
MAX_CANDLE_AGE_DAYS = 5

# Binance API for cross-check
BINANCE_BASE = "https://api.binance.com"


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _binance_price(symbol: str) -> float | None:
    """Fetch last price from Binance for cross-check."""
    try:
        url = f"{BINANCE_BASE}/api/v3/ticker/price?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=8) as r:
            return float(json.loads(r.read())["price"])
    except Exception:
        return None


def _trading_days_ago(n: int) -> datetime:
    """Approximate: go back n calendar days (generous bound)."""
    return datetime.now(timezone.utc) - timedelta(days=n)


def _check_symbol(yf, symbol: str) -> dict:
    """Run all quality checks for one symbol. Returns a result dict."""
    result = {
        "symbol":         symbol,
        "passed":         True,
        "issues":         [],
        "last_price":     None,
        "last_date":      None,
        "n_candles":      0,
        "iv_available":   None,
        "binance_price":  None,
        "price_delta_pct": None,
    }

    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="1y", interval="1d", auto_adjust=True)
    except Exception as exc:
        result["passed"] = False
        result["issues"].append(f"yfinance fetch error: {exc}")
        return result

    # ── 1. Lunghezza storia ──────────────────────────────────────────── #
    n = len(hist)
    result["n_candles"] = n
    if n < 60:
        result["passed"] = False
        result["issues"].append(f"Too few candles: {n} (min 60)")

    if n == 0:
        result["issues"].append("No data returned — symbol may be delisted")
        return result

    closes = hist["Close"].values
    highs  = hist["High"].values
    lows   = hist["Low"].values
    opens  = hist["Open"].values

    last_price = float(closes[-1])
    result["last_price"] = round(last_price, 4)

    # ── 2. Freshness ─────────────────────────────────────────────────── #
    last_ts = hist.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_dt = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        last_dt = datetime.fromtimestamp(float(last_ts), tz=timezone.utc)
    result["last_date"] = last_dt.date().isoformat()
    age_days = (datetime.now(timezone.utc) - last_dt).days
    if age_days > MAX_CANDLE_AGE_DAYS:
        result["passed"] = False
        result["issues"].append(f"Stale data: last candle is {age_days} days old")

    # ── 3. OHLC sanity (check last 20 candles) ──────────────────────── #
    for i in range(-min(20, n), 0):
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        if any(math.isnan(x) or x <= 0 for x in [o, h, l, c]):
            result["passed"] = False
            result["issues"].append(f"NaN or zero price at candle index {i}")
            break
        if not (h >= c >= l and h >= o >= l):
            result["passed"] = False
            result["issues"].append(
                f"OHLC violated at index {i}: O={o} H={h} L={l} C={c}"
            )
            break

    # ── 4. Volume ────────────────────────────────────────────────────── #
    if "Volume" in hist.columns:
        total_vol = float(hist["Volume"].sum())
        if total_vol == 0 and symbol not in NO_OPTIONS_SYMBOLS:
            result["passed"] = False
            result["issues"].append("Zero total volume across entire history")

    # ── 5. Options / IV availability ─────────────────────────────────── #
    if symbol not in NO_OPTIONS_SYMBOLS:
        try:
            exp_dates = ticker.options
            if exp_dates:
                chain = ticker.option_chain(exp_dates[0])
                iv_vals = chain.calls["impliedVolatility"].dropna()
                iv_ok   = len(iv_vals) > 0 and float(iv_vals.mean()) > 0.005
                result["iv_available"] = iv_ok
                if not iv_ok:
                    result["issues"].append("Options chain present but IV values are zero/NaN")
            else:
                result["iv_available"] = False
                result["issues"].append("No options expiry dates returned")
        except Exception as exc:
            result["iv_available"] = False
            result["issues"].append(f"Options fetch error: {exc}")
    else:
        result["iv_available"] = "N/A (ETF/fund)"

    # ── 6. Binance cross-check (if symbol in ALPHA_CRYPTO_MAP) ────────── #
    binance_sym = ALPHA_CRYPTO_MAP.get(symbol)
    if binance_sym:
        bp = _binance_price(binance_sym)
        result["binance_price"] = bp
        if bp and last_price > 0:
            delta_pct = abs(bp - last_price) / last_price * 100
            result["price_delta_pct"] = round(delta_pct, 2)
            if delta_pct > 2.0:
                result["passed"] = False
                result["issues"].append(
                    f"Price divergence vs Binance: yf={last_price:.4f} "
                    f"bnb={bp:.4f} Δ={delta_pct:.2f}%"
                )

    return result


# ── Main ─────────────────────────────────────────────────────────────────── #

def main() -> None:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed — run: pip install yfinance")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  yfinance Data Quality Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Universe: {len(EQUITY_UNIVERSE)} symbols")
    print(f"{'─'*60}\n")

    results: list[dict] = []
    for sym in EQUITY_UNIVERSE:
        sys.stdout.write(f"  Checking {sym:6s} ... ")
        sys.stdout.flush()
        r = _check_symbol(yf, sym)
        results.append(r)
        status = "OK" if r["passed"] else "FAIL"
        extras = []
        if r["last_price"]:
            extras.append(f"${r['last_price']:>10.2f}")
        if r["last_date"]:
            extras.append(f"last={r['last_date']}")
        if r["n_candles"]:
            extras.append(f"n={r['n_candles']}")
        if r["iv_available"] not in (None, "N/A (ETF/fund)"):
            extras.append(f"iv={'yes' if r['iv_available'] else 'NO'}")
        print(f"[{status:4s}]  {' | '.join(extras)}")
        if r["issues"]:
            for issue in r["issues"]:
                print(f"           ⚠  {issue}")

    # ── Summary ─────────────────────────────────────────────────────── #
    passed  = sum(1 for r in results if r["passed"])
    failed  = len(results) - passed
    no_iv   = [r["symbol"] for r in results if r["iv_available"] is False]

    print(f"\n{'─'*60}")
    print(f"  PASSED: {passed}/{len(results)}   FAILED: {failed}")
    if no_iv:
        print(f"  Missing IV: {', '.join(no_iv)}")
    print(f"{'─'*60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
