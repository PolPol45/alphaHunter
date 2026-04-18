"""
seed_ml_history.py
==================
Pre-popola cross_sectional_features_history.jsonl con dati storici reali
scaricati da yfinance, così il MLStrategyAgent può fare training subito
invece di aspettare settimane di cicli live.

Genera una riga per simbolo per ogni venerdì degli ultimi N mesi,
replicando esattamente il formato del FeatureStoreAgent.

Uso:
    python seed_ml_history.py              # ultimi 18 mesi, simboli da config
    python seed_ml_history.py --months 12  # ultimi 12 mesi
    python seed_ml_history.py --force      # sovrascrive file esistente
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("Installa: pip install yfinance pandas numpy")
    sys.exit(1)

BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
HIST_PATH = DATA_DIR / "cross_sectional_features_history.jsonl"
CFG_PATH  = BASE_DIR / "config.json"


def load_config() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def universe_symbols(cfg: dict, max_sym: int = 80) -> list[str]:
    mu = cfg.get("master_universe", {})
    syms: list[str] = []
    for key in ("equities_large_cap", "equities_small_cap", "etf_long", "etf_hedge", "equities_distressed"):
        syms.extend(mu.get(key, []))
    # Crypto
    for key in ("crypto_core", "crypto_defi_bridge", "crypto_altcoin"):
        syms.extend(mu.get(key, []))
    # Deduplica
    seen: set[str] = set()
    out: list[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:max_sym]


def fridays_in_range(start: date, end: date) -> list[date]:
    """Tutti i venerdì nell'intervallo [start, end]."""
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() == 4:  # venerdì
            out.append(d)
        d += timedelta(days=1)
    return out


def compute_features_at(df: pd.DataFrame, as_of: date) -> dict | None:
    """
    Calcola le stesse feature del FeatureStoreAgent per un simbolo
    alla data as_of (usa solo dati <= as_of per evitare lookahead).
    """
    sub = df[df.index.date <= as_of]
    if sub.empty:
        return None

    closes = sub["Close"].dropna()
    if len(closes) < 40:
        return None

    volumes = sub.get("Volume", pd.Series(dtype=float)).fillna(0)
    returns = closes.pct_change().dropna()
    latest  = float(closes.iloc[-1])

    def momentum(days: int) -> float | None:
        if len(closes) <= days:
            return None
        prev = float(closes.iloc[-1 - days])
        return round((latest / prev) - 1.0, 6) if prev != 0 else None

    rolling_max_90 = float(closes.tail(90).max()) if len(closes) >= 90 else float(closes.max())
    drawdown_90    = round((latest / rolling_max_90) - 1.0, 6) if rolling_max_90 else 0.0

    dollar_vol = (closes * volumes.reindex(closes.index, fill_value=0)).dropna()
    avg_vol_30 = float(volumes.tail(30).mean()) if not volumes.empty else 0.0
    med_dv_30  = float(dollar_vol.tail(30).median()) if not dollar_vol.empty else 0.0

    # Amihud illiquidity
    amihud = None
    tail = pd.DataFrame({"ret": returns.tail(30), "dv": dollar_vol.tail(30)})
    tail = tail.replace([np.inf, -np.inf], np.nan).dropna()
    tail = tail[tail["dv"] > 0]
    if not tail.empty:
        amihud = round(float((tail["ret"].abs() / tail["dv"]).mean()), 12)

    # Spread proxy
    spread = None
    if "High" in sub.columns and "Low" in sub.columns:
        hl = sub[["High", "Low", "Close"]].tail(30).replace([np.inf, -np.inf], np.nan).dropna()
        if not hl.empty:
            spread = round(float(((hl["High"] - hl["Low"]) / hl["Close"].replace(0, np.nan)).mean()), 8)

    # Target: excess return della settimana successiva (se disponibile)
    # Necessario per il modello ML supervised
    next_close_idx = df.index[df.index.date > as_of]
    if len(next_close_idx) >= 5:
        next_close = float(df.loc[next_close_idx[4], "Close"]) if "Close" in df.columns else latest
        symbol_ret_t1 = round((next_close / latest) - 1.0, 8)
    elif len(next_close_idx) >= 1:
        next_close = float(df.loc[next_close_idx[0], "Close"]) if "Close" in df.columns else latest
        symbol_ret_t1 = round((next_close / latest) - 1.0, 8)
    else:
        symbol_ret_t1 = None  # ultima settimana disponibile — verrà droppata dal builder

    return {
        "close":                    round(latest, 6),
        "momentum_7d":              momentum(7),
        "momentum_30d":             momentum(30),
        "momentum_90d":             momentum(90),
        "volatility_30d":           round(float(returns.tail(30).std(ddof=0)) if len(returns) >= 2 else 0.0, 6),
        "drawdown_rolling_90d":     drawdown_90,
        "avg_volume_30d":           round(avg_vol_30, 2),
        "median_dollar_volume_30d": round(med_dv_30, 2),
        "amihud_30d":               amihud,
        "spread_proxy_30d":         spread,
        # Colonne richieste dal WeeklyMLDatasetBuilder
        "symbol_return_t_plus_1":      symbol_ret_t1,
        "benchmark_return_t_plus_1":   None,  # verrà riempito dal builder con SPY
        "target_excess_return_t_plus_1": symbol_ret_t1,  # proxy, senza benchmark
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",   type=int, default=18, help="Mesi di storico (default: 18)")
    parser.add_argument("--max-sym",  type=int, default=60, help="Max simboli (default: 60)")
    parser.add_argument("--force",    action="store_true",  help="Sovrascrive file esistente")
    args = parser.parse_args()

    cfg     = load_config()
    symbols = universe_symbols(cfg, max_sym=args.max_sym)
    end     = date.today()
    start = date.fromordinal(end.toordinal() - args.months * 30)

    print(f"Seed ML history | {len(symbols)} simboli | {start} → {end} | {args.months} mesi")

    # Già esiste?
    if HIST_PATH.exists() and not args.force:
        existing = sum(1 for l in HIST_PATH.read_text().splitlines() if l.strip())
        print(f"File già esistente con {existing} righe. Usa --force per sovrascrivere.")
        # Controlla date esistenti
        dates_existing: set[str] = set()
        for line in HIST_PATH.read_text().splitlines():
            if line.strip():
                try:
                    dates_existing.add(json.loads(line).get("date", ""))
                except Exception:
                    pass
        if len(dates_existing) >= 10:
            print(f"Date distinte presenti: {len(dates_existing)} — il ML dovrebbe già funzionare.")
            print("Se ancora skippa, prova: python seed_ml_history.py --force")
            return
        print(f"Solo {len(dates_existing)} date distinte — continuo il seed.")

    fridays = fridays_in_range(start, end - timedelta(days=7))
    print(f"Venerdì da generare: {len(fridays)}")

    # Download storico in batch
    print(f"Download dati storici per {len(symbols)} simboli...")
    period_str = f"{args.months + 3}mo"
    try:
        raw = yf.download(
            tickers=symbols,
            period=period_str,
            interval="1d",
            auto_adjust=True,
            progress=True,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        print(f"Errore download batch: {e}")
        sys.exit(1)

    def get_df(sym: str) -> pd.DataFrame | None:
        try:
            df = raw[sym] if len(symbols) > 1 and sym in raw.columns.get_level_values(0) else raw
            if df is None or df.empty:
                return None
            return df.dropna(how="all")
        except Exception:
            return None

    # Genera righe
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.force and HIST_PATH.exists():
        HIST_PATH.unlink()

    written = 0
    skipped_sym = 0

    with open(HIST_PATH, "a", encoding="utf-8") as fh:
        for friday in fridays:
            rows_this_week: list[dict] = []
            for sym in symbols:
                df = get_df(sym)
                if df is None:
                    skipped_sym += 1
                    continue
                feats = compute_features_at(df, friday)
                if feats is None:
                    continue
                row = {
                    "date":   friday.isoformat(),
                    "date_t": friday.isoformat(),  # alias usato dal WeeklyMLDatasetBuilder
                    "symbol": sym,
                    "benchmark": "SPY",
                    "sector": "Unknown_Sector",
                    "sector_dummy": "UNKNOWN_SECTOR",
                    "macro_market_bias": 0.0,
                    "macro_regime": "NEUTRAL",
                    "macro_fed_funds": None,
                    "macro_cpi_yoy": None,
                    "macro_vix": None,
                    "macro_dxy": None,
                    "macro_qe_qt_proxy": None,
                    "macro_tga_proxy": None,
                    "macro_real_rate_10y": None,
                    "macro_yield_spread_10y_2y": None,
                    "macro_forward_guidance_proxy": None,
                    **feats,
                }
                rows_this_week.append(row)

            for row in rows_this_week:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if friday.day <= 7:  # primo venerdì del mese
                print(f"  {friday} — {len(rows_this_week)} simboli scritti (totale: {written})")

    print(f"\nFatto! Righe scritte: {written} | Simboli saltati: {skipped_sym}")
    print(f"Date distinte: {len(fridays)}")
    print(f"\nOra il MLStrategyAgent farà training al prossimo ciclo.")
    print("Puoi forzare il training subito azzerando ml_strategy_state.json:")
    print("  rm -f data/ml_strategy_state.json")


if __name__ == "__main__":
    main()
