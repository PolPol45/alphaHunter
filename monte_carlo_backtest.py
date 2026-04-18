"""
monte_carlo_backtest.py
=======================
1. Scarica uno snapshot reale di oggi da yfinance (prezzi correnti)
2. Genera N scenari GBM (Geometric Brownian Motion) con parametri configurabili
3. Fa girare il BacktestingAgent su ogni scenario (isolato, senza side-effects)
4. Produce un report con distribuzione delle metriche

Uso:
    python monte_carlo_backtest.py                  # defaults da config.json
    python monte_carlo_backtest.py --scenarios 200 --days 90
    python monte_carlo_backtest.py --scenarios 50 --days 180 --vol-multiplier 1.5 --drift -0.001
    python monte_carlo_backtest.py --scenarios 100 --days 60 --scenario-type crash
    python monte_carlo_backtest.py --no-download    # usa snapshot già in data/

Scenario types:
    base    — parametri da config / yfinance reali
    bull    — drift positivo (+0.001/giorno), vol normale
    bear    — drift negativo (-0.001/giorno), vol +50%
    crash   — shock iniziale -20%, poi alta volatilità
    lateral — drift ~0, vol bassa

Output:
    reports/monte_carlo_report_<timestamp>.json
    reports/monte_carlo_summary_<timestamp>.txt  (stampa anche a schermo)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import random
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agents.base_agent import BASE_DIR, DATA_DIR, REPORTS_DIR
from backtesting.backtest_metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    turnover,
)

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Configurazione default
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SCENARIOS = 100
DEFAULT_DAYS = 90
DEFAULT_BARS_1D = 300   # storico passato simulato per ogni candle
DEFAULT_BARS_4H = 200

SCENARIO_PRESETS: dict[str, dict] = {
    "base":    {"drift": 0.0002,  "vol_mult": 1.0,  "initial_shock": 0.0},
    "bull":    {"drift": 0.001,   "vol_mult": 0.85, "initial_shock": 0.0},
    "bear":    {"drift": -0.001,  "vol_mult": 1.5,  "initial_shock": 0.0},
    "crash":   {"drift": -0.0005, "vol_mult": 2.0,  "initial_shock": -0.20},
    "lateral": {"drift": 0.0,     "vol_mult": 0.6,  "initial_shock": 0.0},
}


# ──────────────────────────────────────────────────────────────────────────────
# Download snapshot reale
# ──────────────────────────────────────────────────────────────────────────────

def download_real_snapshot(symbols: list[str]) -> dict[str, dict]:
    """
    Scarica prezzi e volatilità storiche reali da yfinance.
    Ritorna { symbol: { "price": float, "vol_daily": float, "hist_returns": [float] } }
    """
    if yf is None:
        raise ImportError("yfinance non installato: pip install yfinance")

    print(f"  Download snapshot reale per {len(symbols)} simboli...")
    result: dict[str, dict] = {}

    # Batch download — 1 anno di storico per stimare volatilità
    try:
        raw = yf.download(
            tickers=symbols,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        print(f"  Warning: yfinance batch failed ({e}), provo simbolo per simbolo")
        raw = None

    for sym in symbols:
        try:
            if raw is not None and not raw.empty:
                df = raw[sym] if len(symbols) > 1 and sym in raw.columns.get_level_values(0) else raw
            else:
                df = yf.Ticker(sym).history(period="1y", interval="1d", auto_adjust=True)

            if df is None or df.empty or "Close" not in df.columns:
                continue

            closes = df["Close"].dropna()
            if len(closes) < 10:
                continue

            returns = closes.pct_change().dropna().tolist()
            price = float(closes.iloc[-1])
            vol_daily = float(closes.pct_change().dropna().std()) if len(returns) > 1 else 0.01

            result[sym] = {
                "price": price,
                "vol_daily": vol_daily,
                "hist_returns": returns[-252:],  # ultimo anno
            }
        except Exception:
            continue

    print(f"  Snapshot scaricato: {len(result)}/{len(symbols)} simboli con dati")
    return result


def load_snapshot_from_market_data() -> dict[str, dict]:
    """Fallback: legge market_data.json esistente e stima vol dai candle."""
    fp = DATA_DIR / "market_data.json"
    if not fp.exists():
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        doc = json.load(f)
    result: dict[str, dict] = {}
    for sym, info in doc.get("assets", {}).items():
        price = float(info.get("last_price", 0.0))
        if price <= 0:
            continue
        candles = info.get("ohlcv_1d", [])
        closes = [float(c["c"]) for c in candles if c.get("c")]
        returns: list[float] = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] / closes[i - 1]) - 1.0)
        vol = (sum(r ** 2 for r in returns[-60:]) / max(1, len(returns[-60:]) - 1)) ** 0.5 if returns else 0.01
        result[sym] = {"price": price, "vol_daily": vol, "hist_returns": returns[-252:]}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Generatore GBM
# ──────────────────────────────────────────────────────────────────────────────

def gbm_path(
    price0: float,
    vol_daily: float,
    days: int,
    drift: float,
    vol_mult: float,
    initial_shock: float,
    rng: random.Random,
) -> list[float]:
    """
    Genera un path di prezzi GBM con possibile shock iniziale.
    Drift e vol sono giornalieri.
    """
    sigma = vol_daily * vol_mult
    mu = drift - 0.5 * sigma ** 2  # drift corretto per log-normal
    price = price0 * (1.0 + initial_shock)
    path = [max(price, 1e-6)]
    for _ in range(days):
        z = rng.gauss(0.0, 1.0)
        price = price * math.exp(mu + sigma * z)
        path.append(max(price, 1e-6))
    return path


def build_ohlcv_from_path(
    daily_prices: list[float],
    base_date: date,
    bars_history: int,
    intraday_vol: float,
    rng: random.Random,
) -> list[dict]:
    """
    Costruisce una lista OHLCV daily dal path GBM.
    Genera bars_history candle passate + len(daily_prices) candle simulate.
    """
    out: list[dict] = []
    step = 86400  # 1 giorno in secondi

    # Candle storiche fittizie (stesso prezzo iniziale con rumore ridotto)
    p0 = daily_prices[0]
    hist_start_ts = int(
        datetime.combine(base_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    ) - bars_history * step

    p = p0
    for i in range(bars_history):
        z = rng.gauss(0.0, intraday_vol * 0.5)
        c = max(p * (1 + z), 1e-6)
        h = max(p, c) * (1 + abs(rng.gauss(0, intraday_vol * 0.3)))
        l = min(p, c) * (1 - abs(rng.gauss(0, intraday_vol * 0.3)))
        out.append({
            "t": hist_start_ts + i * step,
            "o": round(p, 6),
            "h": round(h, 6),
            "l": round(l, 6),
            "c": round(c, 6),
            "v": round(abs(rng.gauss(1000, 500)), 2),
        })
        p = c

    # Candle simulate dal path GBM
    for i, price in enumerate(daily_prices):
        prev = daily_prices[i - 1] if i > 0 else price
        h = max(prev, price) * (1 + abs(rng.gauss(0, intraday_vol * 0.2)))
        l = min(prev, price) * (1 - abs(rng.gauss(0, intraday_vol * 0.2)))
        out.append({
            "t": int(datetime.combine(base_date + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc).timestamp()),
            "o": round(prev, 6),
            "h": round(h, 6),
            "l": round(l, 6),
            "c": round(price, 6),
            "v": round(abs(rng.gauss(1000, 500)), 2),
        })

    return out


def generate_scenario_snapshots(
    snapshot: dict[str, dict],
    symbols: list[str],
    start: date,
    days: int,
    preset: dict,
    drift_override: float | None,
    vol_mult_override: float | None,
    rng: random.Random,
) -> dict[date, dict]:
    """
    Genera N giorni di snapshot per un singolo scenario MC.
    Ritorna { day: market_snapshot_dict }
    """
    drift = drift_override if drift_override is not None else preset["drift"]
    vol_mult = vol_mult_override if vol_mult_override is not None else preset["vol_mult"]
    initial_shock = preset["initial_shock"]

    # Genera path GBM per ogni simbolo
    paths: dict[str, list[float]] = {}
    for sym in symbols:
        info = snapshot.get(sym)
        if info is None:
            # Simbolo senza dati: prezzo fittizio stabile
            p0 = 100.0
            vol = 0.015
        else:
            p0 = info["price"]
            vol = info["vol_daily"]

        path = gbm_path(p0, vol, days, drift, vol_mult, initial_shock, rng)
        paths[sym] = path

    # Costruisce snapshot giornalieri
    daily_snapshots: dict[date, dict] = {}
    for day_offset in range(days):
        day = start + timedelta(days=day_offset)
        assets: dict = {}
        for sym in symbols:
            path = paths.get(sym, [100.0] * (days + 1))
            price_today = path[day_offset + 1]  # path[0] = prezzo base
            info = snapshot.get(sym, {})
            vol = info.get("vol_daily", 0.015)

            ohlcv_1d = build_ohlcv_from_path(
                path[: day_offset + 2], day, DEFAULT_BARS_1D, vol, rng
            )
            ohlcv_4h = ohlcv_1d[-DEFAULT_BARS_4H:]

            spread = price_today * 0.0005
            assets[sym] = {
                "last_price": round(price_today, 6),
                "ohlcv_1d": ohlcv_1d,
                "ohlcv_4h": ohlcv_4h,
                "orderbook": {
                    "bids": [[round(price_today - spread * i, 6), 1.0] for i in range(1, 6)],
                    "asks": [[round(price_today + spread * i, 6), 1.0] for i in range(1, 6)],
                },
                "volume_24h": round(abs(rng.gauss(1e6, 2e5)), 2),
            }

        daily_snapshots[day] = {
            "timestamp": datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
            "data_source": "monte_carlo",
            "assets": assets,
            "world_events": [],
        }

    return daily_snapshots


# ──────────────────────────────────────────────────────────────────────────────
# Runner scenario singolo
# ──────────────────────────────────────────────────────────────────────────────

def run_single_scenario(
    scenario_id: int,
    daily_snapshots: dict[date, dict],
    start: date,
    end: date,
    universe: list[str],
    bt_dir: pathlib.Path,
    data_dir: pathlib.Path,
) -> dict:
    """
    Scrive i file di snapshot, esegue RiskAgent + ExecutionAgent giorno per giorno,
    raccoglie equity curve e trade, ritorna le metriche del singolo scenario.
    Usa backtest_context.json per isolamento (stesso meccanismo di BacktestingAgent).
    """
    from agents.risk_agent import RiskAgent
    from agents.execution_agent import ExecutionAgent
    from agents.backtesting_agent import BacktestingAgent

    # Istanzia agent freschi per ogni scenario
    bt_agent = BacktestingAgent()
    risk = RiskAgent()
    execution = ExecutionAgent()

    # Backtest: disable turnover gate (cycles run instantly, timestamps are midnight)
    execution._turnover_cfg = {"enabled": False}

    # Reset portfolio state pulito
    for fp_name in [
        "portfolio_retail.json",
        "portfolio_institutional.json",
        "portfolio_alpha.json",
        "stop_orders_retail.json",
        "stop_orders_institutional.json",
        "validated_signals.json",
    ]:
        fp = data_dir / fp_name
        if fp.exists():
            fp.unlink()

    equity_curve: list[float] = []
    returns: list[float] = []
    prev_equity: float | None = None

    for day_offset, (day, snapshot) in enumerate(sorted(daily_snapshots.items())):
        # Advance simulated clock to noon of the simulated day
        sim_now = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=12)
        execution._now = lambda _d=sim_now: _d
        execution._now_iso = lambda _d=sim_now: _d.isoformat()

        # Scrivi snapshot nel formato atteso da HistoricalDataStore
        fp = bt_dir / f"market_{day.isoformat()}.json"
        fp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

        # Contesto backtest
        ctx = {
            "enabled": True,
            "simulated_now": sim_now.isoformat(),
            "lookahead_cutoff": sim_now.isoformat(),
            "frozen_universe": universe,
            "fees": {"commission_pct": 0.001},
            "slippage_model": {"default_bps": 10},
        }
        _write_json_atomic(data_dir / "backtest_context.json", ctx)

        # Scrivi market_data.json dal snapshot corrente
        _write_json_atomic(data_dir / "market_data.json", snapshot)

        # Genera segnali di backtest (usa i metodi del BacktestingAgent)
        market = snapshot
        macro = bt_agent._build_macro_snapshot(day, market)
        _write_json_atomic(data_dir / "macro_snapshot.json", macro)
        _write_json_atomic(data_dir / "market_regime.json", bt_agent._build_market_regime(day, macro))

        sector = bt_agent._build_sector_scorecard(day)
        _write_json_atomic(data_dir / "sector_scorecard.json", sector)

        stock_scores = bt_agent._build_stock_scores(day, universe)
        _write_json_atomic(data_dir / "stock_scores.json", stock_scores)

        bull = bt_agent._build_bull_signals(stock_scores, macro)
        bear = bt_agent._build_bear_signals(stock_scores, macro)
        crypto = bt_agent._build_crypto_signals(day, market)
        _write_json_atomic(data_dir / "bull_signals.json", bull)
        _write_json_atomic(data_dir / "bear_signals.json", bear)
        _write_json_atomic(data_dir / "crypto_signals.json", crypto)

        # Feed placeholder
        _write_json_atomic(data_dir / "news_feed.json", {
            "generated_at": day.isoformat(), "items": [], "top_alerts": []
        })
        _write_json_atomic(data_dir / "signals.json", {
            "timestamp": day.isoformat(),
            "retail": {}, "institutional": {},
            "scanner": {"retail_top_candidates": [], "institutional_top_candidates": []},
            "context": {"macro_bias": macro.get("market_bias", 0.0)},
        })
        _write_json_atomic(data_dir / "alpha_signals.json", {
            "timestamp": day.isoformat(), "signals": {},
            "scanner": {"top_candidates": []}
        })

        # Esegui pipeline
        try:
            risk.run()
            execution.run()
        except Exception:
            pass

        # Raccogli equity
        retail = _read_json(data_dir / "portfolio_retail.json")
        inst = _read_json(data_dir / "portfolio_institutional.json")
        eq = float(retail.get("total_equity", 0.0)) + float(inst.get("total_equity", 0.0))
        equity_curve.append(eq)
        if prev_equity and prev_equity > 0:
            returns.append((eq - prev_equity) / prev_equity)
        prev_equity = eq

    # Raccolta trade
    retail = _read_json(data_dir / "portfolio_retail.json")
    inst = _read_json(data_dir / "portfolio_institutional.json")
    all_trades = retail.get("trades", []) + inst.get("trades", [])
    avg_eq = sum(equity_curve) / len(equity_curve) if equity_curve else 1.0

    # Cleanup context
    ctx_fp = data_dir / "backtest_context.json"
    if ctx_fp.exists():
        ctx_fp.unlink()

    return {
        "scenario_id": scenario_id,
        "sharpe": round(sharpe_ratio(returns), 6),
        "sortino": round(sortino_ratio(returns), 6),
        "max_drawdown": round(max_drawdown(equity_curve), 6),
        "calmar": round(calmar_ratio(equity_curve), 6),
        "win_rate": round(win_rate(all_trades), 6),
        "turnover": round(turnover(all_trades, avg_eq), 6),
        "final_equity": round(equity_curve[-1], 4) if equity_curve else 0.0,
        "start_equity": round(equity_curve[0], 4) if equity_curve else 0.0,
        "total_return_pct": round(
            (equity_curve[-1] / equity_curve[0] - 1.0) * 100 if equity_curve and equity_curve[0] > 0 else 0.0, 4
        ),
        "trades": len(all_trades),
        "equity_curve": [round(e, 2) for e in equity_curve],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Analisi distribuzione
# ──────────────────────────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def analyse_results(results: list[dict], drawdown_threshold: float = 0.15) -> dict:
    def extract(key: str) -> list[float]:
        return [r[key] for r in results if r.get(key) is not None]

    def dist(vals: list[float]) -> dict:
        if not vals:
            return {}
        return {
            "p5":    round(percentile(vals, 5), 6),
            "p25":   round(percentile(vals, 25), 6),
            "p50":   round(percentile(vals, 50), 6),
            "p75":   round(percentile(vals, 75), 6),
            "p95":   round(percentile(vals, 95), 6),
            "mean":  round(sum(vals) / len(vals), 6),
            "min":   round(min(vals), 6),
            "max":   round(max(vals), 6),
        }

    sharpes = extract("sharpe")
    mdd = extract("max_drawdown")
    returns = extract("total_return_pct")
    win_rates = extract("win_rate")
    calmars = extract("calmar")
    sortinos = extract("sortino")

    n = len(results)
    prob_dd_over_threshold = round(sum(1 for d in mdd if d > drawdown_threshold) / max(n, 1), 4)
    prob_positive_return = round(sum(1 for r in returns if r > 0) / max(n, 1), 4)
    prob_sharpe_positive = round(sum(1 for s in sharpes if s > 0) / max(n, 1), 4)
    prob_ruin = round(sum(1 for d in mdd if d > 0.50) / max(n, 1), 4)

    return {
        "scenarios_run": n,
        "sharpe":         dist(sharpes),
        "sortino":        dist(sortinos),
        "max_drawdown":   dist(mdd),
        "calmar":         dist(calmars),
        "total_return_pct": dist(returns),
        "win_rate":       dist(win_rates),
        "probabilities": {
            f"drawdown_gt_{int(drawdown_threshold * 100)}pct": prob_dd_over_threshold,
            "positive_return":   prob_positive_return,
            "sharpe_positive":   prob_sharpe_positive,
            "ruin_gt_50pct_dd":  prob_ruin,
        },
    }


def format_summary(analysis: dict, args: argparse.Namespace, scenario_type: str) -> str:
    s = analysis["sharpe"]
    mdd = analysis["max_drawdown"]
    ret = analysis["total_return_pct"]
    prob = analysis["probabilities"]

    lines = [
        "=" * 60,
        "  MONTE CARLO BACKTEST — RISULTATI",
        "=" * 60,
        f"  Scenari:       {analysis['scenarios_run']}",
        f"  Orizzonte:     {args.days} giorni",
        f"  Scenario type: {scenario_type}",
        "",
        "  SHARPE RATIO",
        f"    P5 / P50 / P95:  {s['p5']:.3f} / {s['p50']:.3f} / {s['p95']:.3f}",
        f"    Media:           {s['mean']:.3f}",
        "",
        "  MAX DRAWDOWN",
        f"    P5 / P50 / P95:  {mdd['p5']:.1%} / {mdd['p50']:.1%} / {mdd['p95']:.1%}",
        f"    Peggiore:        {mdd['max']:.1%}",
        "",
        "  TOTAL RETURN %",
        f"    P5 / P50 / P95:  {ret['p5']:.2f}% / {ret['p50']:.2f}% / {ret['p95']:.2f}%",
        f"    Media:           {ret['mean']:.2f}%",
        "",
        "  PROBABILITÀ",
        f"    Drawdown > 15%:  {prob['drawdown_gt_15pct']:.1%}",
        f"    Ritorni positivi:{prob['positive_return']:.1%}",
        f"    Sharpe > 0:      {prob['sharpe_positive']:.1%}",
        f"    Rovina (DD>50%): {prob['ruin_gt_50pct_dd']:.1%}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers I/O
# ──────────────────────────────────────────────────────────────────────────────

def _write_json_atomic(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config() -> dict:
    from agents.base_agent import CONFIG_PATH
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def universe_from_config(cfg: dict) -> list[str]:
    mu = cfg.get("master_universe", {})
    symbols: list[str] = []
    for key in ("equities_large_cap", "equities_small_cap", "equities_distressed", "etf_long", "etf_hedge"):
        symbols.extend(mu.get(key, []))
    symbols.extend(cfg.get("backtesting", {}).get("universe_snapshot", []))
    # Cripto comuni
    symbols.extend(["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo Backtest")
    parser.add_argument("--scenarios",      type=int,   default=DEFAULT_SCENARIOS,
                        help=f"Numero di scenari MC (default: {DEFAULT_SCENARIOS})")
    parser.add_argument("--days",           type=int,   default=DEFAULT_DAYS,
                        help=f"Giorni di simulazione per scenario (default: {DEFAULT_DAYS})")
    parser.add_argument("--scenario-type",  type=str,   default="base",
                        choices=list(SCENARIO_PRESETS.keys()),
                        help="Preset scenario: base | bull | bear | crash | lateral")
    parser.add_argument("--drift",          type=float, default=None,
                        help="Override drift giornaliero (es. 0.001 per +0.1 perc/giorno)")
    parser.add_argument("--vol-multiplier", type=float, default=None,
                        help="Moltiplicatore volatilita rispetto al dato reale (es. 1.5 = +50 perc vol)")
    parser.add_argument("--dd-threshold",   type=float, default=0.15,
                        help="Soglia drawdown per calcolo probabilita (default: 0.15 = 15 perc)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Seed per riproducibilità (default: 42)")
    parser.add_argument("--no-download",    action="store_true",
                        help="Non scarica da yfinance, usa market_data.json esistente")
    parser.add_argument("--max-symbols",    type=int,   default=50,
                        help="Limita i simboli dell'universo (velocizza il test, default: 50)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MONTE CARLO BACKTEST")
    print("=" * 60)
    print(f"  Scenari: {args.scenarios} | Giorni: {args.days} | Type: {args.scenario_type}")
    print(f"  Seed: {args.seed} | Max simboli: {args.max_symbols}")

    cfg = load_config()
    all_symbols = universe_from_config(cfg)
    # Limita per performance
    symbols = all_symbols[:args.max_symbols]
    print(f"  Universo: {len(symbols)} simboli (totale config: {len(all_symbols)})")

    # 1. Snapshot reale
    print("\n[1/3] Snapshot prezzi reali...")
    if args.no_download:
        snapshot = load_snapshot_from_market_data()
        if not snapshot:
            print("  Nessun dato in market_data.json. Rimuovi --no-download o avvia prima il bot.")
            sys.exit(1)
        print(f"  Caricati {len(snapshot)} simboli da market_data.json")
    else:
        snapshot = download_real_snapshot(symbols)
        if not snapshot:
            print("  Download fallito. Provo a usare market_data.json...")
            snapshot = load_snapshot_from_market_data()
        if not snapshot:
            print("  Nessun dato disponibile. Assicurati di avere connessione internet.")
            sys.exit(1)

    # Simboli con dati reali
    valid_symbols = [s for s in symbols if s in snapshot]
    print(f"  Simboli con prezzi reali: {len(valid_symbols)}/{len(symbols)}")

    # 2. Genera ed esegui scenari
    preset = SCENARIO_PRESETS[args.scenario_type]
    start = date.today()
    end = start + timedelta(days=args.days - 1)

    bt_dir = BASE_DIR / "backtest_data"
    bt_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Esecuzione {args.scenarios} scenari ({start} → {end})...")
    print("  Ogni punto = 1 scenario completato\n  ", end="", flush=True)

    master_rng = random.Random(args.seed)
    results: list[dict] = []
    t0 = time.time()

    # Snapshot dei file live che il backtest sovrascrive — ripristinati dopo ogni scenario
    _LIVE_FILES = [
        "portfolio_retail.json", "portfolio_institutional.json", "portfolio_alpha.json",
        "stop_orders_retail.json", "stop_orders_institutional.json", "validated_signals.json",
        "market_data.json", "macro_snapshot.json", "market_regime.json",
        "sector_scorecard.json", "stock_scores.json", "bull_signals.json",
        "bear_signals.json", "crypto_signals.json", "news_feed.json",
        "signals.json", "alpha_signals.json", "backtest_context.json",
    ]
    _live_snapshots: dict[str, bytes | None] = {}
    for _fname in _LIVE_FILES:
        _p = DATA_DIR / _fname
        _live_snapshots[_fname] = _p.read_bytes() if _p.exists() else None

    def _restore_live_files() -> None:
        for _fname, _content in _live_snapshots.items():
            _p = DATA_DIR / _fname
            if _content is not None:
                _p.write_bytes(_content)
            elif _p.exists():
                _p.unlink(missing_ok=True)

    for i in range(args.scenarios):
        scenario_seed = master_rng.randint(0, 2**31)
        rng = random.Random(scenario_seed)

        daily_snapshots = generate_scenario_snapshots(
            snapshot=snapshot,
            symbols=valid_symbols,
            start=start,
            days=args.days,
            preset=preset,
            drift_override=args.drift,
            vol_mult_override=args.vol_multiplier,
            rng=rng,
        )

        try:
            result = run_single_scenario(
                scenario_id=i + 1,
                daily_snapshots=daily_snapshots,
                start=start,
                end=end,
                universe=valid_symbols,
                bt_dir=bt_dir,
                data_dir=DATA_DIR,
            )
        finally:
            _restore_live_files()
        results.append(result)

        # Progress
        print("." if (i + 1) % 50 != 0 else f". [{i+1}/{args.scenarios}]", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n\n  Completati in {elapsed:.1f}s ({elapsed / args.scenarios:.1f}s/scenario)")

    # 3. Analisi e report
    print("\n[3/3] Analisi risultati...")
    analysis = analyse_results(results, drawdown_threshold=args.dd_threshold)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "report_type": "monte_carlo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "scenarios": args.scenarios,
            "days": args.days,
            "scenario_type": args.scenario_type,
            "drift_override": args.drift,
            "vol_multiplier_override": args.vol_multiplier,
            "dd_threshold": args.dd_threshold,
            "seed": args.seed,
            "symbols_used": len(valid_symbols),
            "preset": preset,
        },
        "analysis": analysis,
        "scenario_results": results,
    }

    report_path = REPORTS_DIR / f"monte_carlo_report_{timestamp}.json"
    summary_path = REPORTS_DIR / f"monte_carlo_summary_{timestamp}.txt"

    _write_json_atomic(report_path, report)

    summary_text = format_summary(analysis, args, args.scenario_type)
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)
    print(f"\n  Report JSON: {report_path.name}")
    print(f"  Summary TXT: {summary_path.name}")


if __name__ == "__main__":
    main()
