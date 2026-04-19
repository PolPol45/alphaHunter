"""
backtest_monitor.py — Live terminal dashboard per walk-forward backtest
=======================================================================
Legge i file JSON prodotti dal backtest e mostra in tempo reale:
  - Progresso (giorno corrente / totale giorni / %)
  - Equity curve retail + institutional
  - P&L totale, drawdown corrente, trades eseguiti
  - ML status (IC, ultimo training)
  - Ultimo agente completato + eventuali errori

Uso:
    python backtest_monitor.py          # refresh ogni 2s
    python backtest_monitor.py --fast   # refresh ogni 0.5s
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

DATA_DIR = pathlib.Path(__file__).parent / "data"
REPORTS_DIR = pathlib.Path(__file__).parent / "reports"

START_DATE = "2025-04-01"
END_DATE   = "2026-04-18"

# Total trading days in period (approx)
TOTAL_DAYS = 264


def _read(path: pathlib.Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val*100:.2f}%"


def _money(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.0f}"


def _bar(value: float, total: float, width: int = 40, fill: str = "█", empty: str = "░") -> str:
    if total <= 0:
        return empty * width
    filled = int(round(value / total * width))
    filled = max(0, min(width, filled))
    return fill * filled + empty * (width - filled)


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str: return _color(t, "92")
def _red(t: str) -> str:   return _color(t, "91")
def _yellow(t: str) -> str: return _color(t, "93")
def _cyan(t: str) -> str:   return _color(t, "96")
def _bold(t: str) -> str:   return _color(t, "1")
def _dim(t: str) -> str:    return _color(t, "2")


def _signed_color(val: float, text: str) -> str:
    return _green(text) if val >= 0 else _red(text)


def render(interval: float) -> None:
    os.system("clear")

    now = datetime.now().strftime("%H:%M:%S")

    # ── backtest context ───────────────────────────────────────────────────── #
    ctx = _read(DATA_DIR / "backtest_context.json")
    sim_date_raw = ctx.get("simulated_now", "")
    sim_date = sim_date_raw[:10] if sim_date_raw else "?"

    # Compute progress
    try:
        from datetime import date
        d0 = date.fromisoformat(START_DATE)
        d1 = date.fromisoformat(END_DATE)
        dc = date.fromisoformat(sim_date) if sim_date != "?" else d0
        elapsed = (dc - d0).days
        total   = (d1 - d0).days
        pct_done = elapsed / total if total > 0 else 0
    except Exception:
        elapsed, total, pct_done = 0, TOTAL_DAYS, 0

    # ── portfolios ────────────────────────────────────────────────────────── #
    port_retail = _read(DATA_DIR / "portfolio_retail.json")
    port_inst   = _read(DATA_DIR / "portfolio_institutional.json")

    def _equity_pnl(p: dict) -> tuple[float, float, float, float]:
        equity   = float(p.get("equity",   p.get("total_value", 0)))
        cash     = float(p.get("cash",     0))
        cost     = float(p.get("total_cost_basis", p.get("initial_capital", equity)))
        pnl      = equity - cost
        dd       = float(p.get("drawdown_pct", 0))
        return equity, cash, pnl, dd

    r_eq, r_cash, r_pnl, r_dd = _equity_pnl(port_retail)
    i_eq, i_cash, i_pnl, i_dd = _equity_pnl(port_inst)

    r_init = float(port_retail.get("initial_capital", 20_000))
    i_init = float(port_inst.get("initial_capital", 980_000))

    # ── trades ────────────────────────────────────────────────────────────── #
    validated = _read(DATA_DIR / "validated_signals.json")
    n_retail_sigs  = len(validated.get("retail",  {}).get("approved", []))
    n_inst_sigs    = len(validated.get("institutional", {}).get("approved", []))

    # ── ML signals ────────────────────────────────────────────────────────── #
    ml = _read(DATA_DIR / "ml_signals.json")
    ml_ic   = float(ml.get("avg_ic_across_folds", 0) or 0)
    ml_ts   = ml.get("generated_at", "?")[:16]
    ml_conf = float(ml.get("avg_confidence", 0) or 0)
    n_ml_long  = len(ml.get("long_picks",  []))
    n_ml_short = len(ml.get("short_picks", []))

    # ── agent health ──────────────────────────────────────────────────────── #
    shared = _read(DATA_DIR / "shared_state.json")
    agents = shared.get("agents", {})

    def _agent_status(name: str) -> str:
        a = agents.get(name, {})
        status = a.get("status", "unknown")
        if status == "running": return _yellow("⟳ running")
        if status == "done":    return _green("✓ done")
        if status == "error":   return _red("✗ error")
        return _dim(status)

    # ── DD killswitch ─────────────────────────────────────────────────────── #
    dd_ks = _read(DATA_DIR / "dd_killswitch.json")
    dd_ks_active = bool(dd_ks.get("active", False))

    # ── latest report ─────────────────────────────────────────────────────── #
    latest_report = None
    try:
        reports = sorted(REPORTS_DIR.glob("backtest_report_*.json"), key=lambda p: p.stat().st_mtime)
        if reports:
            latest_report = _read(reports[-1])
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════════════ #
    # RENDER
    # ═══════════════════════════════════════════════════════════════════════ #
    W = 72

    print(_bold(_cyan("═" * W)))
    print(_bold(_cyan(f"  BACKTEST MONITOR   {now}   refresh {interval}s")))
    print(_bold(_cyan("═" * W)))

    # Progress
    bar = _bar(elapsed, total, width=45)
    print(f"\n  {_bold('PROGRESSO')}  {_cyan(sim_date)}  →  {END_DATE}")
    print(f"  [{_green(bar[:int(pct_done*45)])+_dim(bar[int(pct_done*45):])}]  {_bold(f'{pct_done*100:.1f}%')}  ({elapsed}/{total} giorni)")

    # Equity / P&L
    print(f"\n  {'─'*68}")
    print(f"  {_bold('PORTFOLIO'):<30}  {'EQUITY':>12}  {'P&L':>10}  {'DD':>8}")
    print(f"  {'─'*68}")

    r_pnl_pct = r_pnl / r_init if r_init else 0
    i_pnl_pct = i_pnl / i_init if i_init else 0

    print(f"  {'Retail  (personal $20K)':<30}  "
          f"{_bold(f'${r_eq:>10,.0f}')}  "
          f"{_signed_color(r_pnl, f'{_money(r_pnl):>10}'):>10}  "
          f"{_signed_color(-r_dd, f'{r_dd*100:>6.1f}%'):>8}")
    print(f"  {'Institutional ($980K)':<30}  "
          f"{_bold(f'${i_eq:>10,.0f}')}  "
          f"{_signed_color(i_pnl, f'{_money(i_pnl):>10}'):>10}  "
          f"{_signed_color(-i_dd, f'{i_dd*100:>6.1f}%'):>8}")

    total_pnl = r_pnl + i_pnl
    total_eq  = r_eq  + i_eq
    print(f"  {'─'*68}")
    print(f"  {_bold('TOTALE'):<30}  "
          f"{_bold(f'${total_eq:>10,.0f}')}  "
          f"{_signed_color(total_pnl, _bold(f'{_money(total_pnl):>10}')):>10}")

    # DD killswitch alert
    if dd_ks_active:
        dd_at = float(dd_ks.get("dd_at_trigger", 0))
        print(f"\n  {_red(_bold(f'🚨 HARD DD KILLSWITCH ATTIVO  (trigger a {dd_at*100:.1f}%  — BUY bloccati)'))} ")

    # Signals
    print(f"\n  {_bold('SEGNALI APPROVATI')}  Retail: {_cyan(str(n_retail_sigs))}   Inst: {_cyan(str(n_inst_sigs))}")

    # ML
    ic_str = _signed_color(ml_ic, f"{ml_ic:+.3f}")
    ic_ok  = _green("✓ OK") if ml_ic >= 0.05 else _red("✗ sotto soglia 0.05")
    print(f"\n  {_bold('ML')}  IC={ic_str} {ic_ok}   conf={ml_conf:.2f}   long={n_ml_long}  short={n_ml_short}   @{_dim(ml_ts)}")

    # Agent status
    print(f"\n  {_bold('AGENTI')}")
    agent_names = [
        ("market_data_agent",       "MarketData  "),
        ("technical_analysis_agent","TA          "),
        ("risk_agent",              "Risk        "),
        ("execution_agent",         "Execution   "),
        ("crypto_strategy_agent",   "Crypto      "),
        ("ml_strategy_agent",       "ML Strategy "),
    ]
    cols = []
    for key, label in agent_names:
        cols.append(f"  {label}: {_agent_status(key)}")
    # print 2 per row
    for i in range(0, len(cols), 2):
        row = cols[i]
        if i+1 < len(cols):
            row += "    " + cols[i+1].strip()
        print(row)

    # Latest report summary
    if latest_report:
        rp = latest_report
        rp_date = str(rp.get("generated_at", "?"))[:16]
        rp_retail = rp.get("retail", {})
        rp_ret_pnl = float(rp_retail.get("total_pnl", 0))
        rp_trades  = int(rp_retail.get("total_trades", 0))
        rp_wr      = float(rp_retail.get("win_rate", 0))
        print(f"\n  {_bold('ULTIMO REPORT')}  {_dim(rp_date)}")
        print(f"  Retail P&L={_signed_color(rp_ret_pnl, _money(rp_ret_pnl))}  "
              f"trades={rp_trades}  win_rate={rp_wr*100:.0f}%")

    print(f"\n  {_dim('Ctrl+C per uscire')}")
    print(_bold(_cyan("═" * W)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    interval = 0.5 if args.fast else args.interval

    try:
        while True:
            render(interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nUscito.")


if __name__ == "__main__":
    main()
