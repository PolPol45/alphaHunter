"""
dashboard.py — Terminal UI per il Trading Bot
==============================================
Mostra in tempo reale:
  - Stato sistema e ciclo corrente
  - P&L e equity per portfolio (retail + institutional)
  - Posizioni aperte per strategia (bull / bear / crypto)
  - Segnali approvati dal RiskAgent
  - Stato agenti (running / done / error)
  - Stato ML: ultimo training, prossimo refresh
  - Stato auto-backtest: ultimo run, prossimo trigger
  - Pesi strategia appresi (learned_strategy_weights)

Uso:
    python dashboard.py               # refresh ogni 2s
    python dashboard.py --interval 5  # refresh ogni 5s

Si può avviare in parallelo al bot (legge solo file JSON, nessun side-effect).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"


def _read(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_dt(iso: str | None, relative: bool = False) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if relative:
            diff = datetime.now(timezone.utc) - dt
            s = int(diff.total_seconds())
            if s < 60:    return f"{s}s fa"
            if s < 3600:  return f"{s//60}m fa"
            if s < 86400: return f"{s//3600}h fa"
            return f"{s//86400}g fa"
        return dt.strftime("%d/%m %H:%M:%S")
    except Exception:
        return iso[:16] if iso else "—"


def _pct(val: float, decimals: int = 2) -> str:
    return f"{val * 100:.{decimals}f}%"


def _money(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.2f}"


def _colored_pnl(val: float, fmt: str = "money") -> Text:
    s = _money(val) if fmt == "money" else _pct(val)
    if val > 0:
        return Text(f"▲ {s}", style="bold green")
    if val < 0:
        return Text(f"▼ {s}", style="bold red")
    return Text(f"  {s}", style="dim")


def _agent_color(status: str) -> str:
    return {
        "running": "bold yellow",
        "done":    "green",
        "error":   "bold red",
        "idle":    "dim white",
    }.get(status, "white")


def _sparkline(values: list[float], width: int = 28) -> Text:
    """Curva ASCII con braille blocks a 8 livelli."""
    CHARS = "▁▂▃▄▅▆▇█"
    if not values or len(values) < 2:
        return Text("─" * width, style="dim")
    # Campiona 'width' punti
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values + [values[-1]] * (width - len(values))
    lo, hi = min(sampled), max(sampled)
    span = hi - lo if hi != lo else 1.0
    chars = [CHARS[min(7, int((v - lo) / span * 7.999))] for v in sampled]
    # Colore in base all'andamento: verde se sale, rosso se scende
    trend = sampled[-1] - sampled[0]
    color = "bright_green" if trend > 0 else ("bright_red" if trend < 0 else "dim")
    return Text("".join(chars), style=color)


def _read_log_last_run(agent_name: str) -> str | None:
    """Legge l'ultima riga Done dal log dell'agente."""
    log_path = LOGS_DIR / f"{agent_name}.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines[-200:]):
            line_lower = line.lower()
            if "] [info]" in line_lower and ("done" in line_lower or "completed" in line_lower or "start" in line_lower):
                # Estrae timestamp ISO dalla riga di log: [2026-04-13 18:15:12]
                ts_part = line[1:20] if line.startswith("[") else None
                if ts_part:
                    try:
                        dt = datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S")
                        return dt.replace(tzinfo=timezone.utc).isoformat()
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def _read_log_last_error(agent_name: str) -> str | None:
    """Legge l'ultimo errore dal log."""
    log_path = LOGS_DIR / f"{agent_name}.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines[-100:]):
            if "] [ERROR]" in line or "] [WARNING]" in line and "failed" in line.lower():
                return line[22:80] if len(line) > 22 else line
    except Exception:
        pass
    return None


def _next_due(last_iso: str | None, interval_days: int) -> str:
    if not last_iso:
        return "subito"
    try:
        from datetime import timedelta
        last = datetime.fromisoformat(last_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        nxt = last + timedelta(days=interval_days)
        diff = nxt - datetime.now(timezone.utc)
        s = int(diff.total_seconds())
        if s <= 0:
            return "ora"
        if s < 3600:  return f"tra {s//60}m"
        if s < 86400: return f"tra {s//3600}h"
        return f"tra {s//86400}g"
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────
# Sezione: Header sistema
# ─────────────────────────────────────────────────────────────────

def build_header(state: dict, cfg: dict) -> Panel:
    sys_s = state.get("system", {})
    status    = sys_s.get("status", "unknown")
    cycle     = sys_s.get("cycle_count", 0)
    started   = _fmt_dt(sys_s.get("started_at"))
    last_c    = _fmt_dt(sys_s.get("last_cycle"), relative=True)
    paper     = sys_s.get("paper_trading", True)
    mode      = cfg.get("orchestrator", {}).get("mode", "?")
    interval  = cfg.get("orchestrator", {}).get("cycle_interval_seconds", 60)

    status_color = "green" if status == "running" else ("yellow" if status == "stopped" else "red")
    paper_tag = "[bold cyan]PAPER[/bold cyan]" if paper else "[bold red]LIVE[/bold red]"

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC")
    text = (
        f"  {paper_tag}  │  "
        f"Status: [{status_color}]{status.upper()}[/{status_color}]  │  "
        f"Mode: [cyan]{mode}[/cyan]  │  "
        f"Ciclo: [bold]{cycle}[/bold]  │  "
        f"Intervallo: {interval}s  │  "
        f"Avvio: {started}  │  "
        f"Ultimo ciclo: {last_c}  │  "
        f"[dim]{now}[/dim]"
    )
    return Panel(text, title="[bold white]🤖 CRYPTO TRADING BOT[/bold white]",
                 border_style="bright_blue", padding=(0, 1))


# ─────────────────────────────────────────────────────────────────
# Sezione: Portfolio P&L
# ─────────────────────────────────────────────────────────────────

def build_portfolio_table(retail: dict, inst: dict) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
              expand=True, padding=(0, 1))
    t.add_column("Portfolio",     style="bold white", width=18)
    t.add_column("Equity",        justify="right", width=12)
    t.add_column("Cash",          justify="right", width=12)
    t.add_column("P&L Tot.",      justify="right", width=14)
    t.add_column("P&L %",         justify="right", width=10)
    t.add_column("Realized",      justify="right", width=14)
    t.add_column("Drawdown",      justify="right", width=10)
    t.add_column("Posizioni",     justify="right", width=10)

    for label, port in [("Retail (Personal)", retail), ("Institutional", inst)]:
        if not port:
            t.add_row(label, "—", "—", "—", "—", "—", "—", "—")
            continue

        eq    = float(port.get("total_equity", 0))
        cash  = float(port.get("cash", 0))
        pnl   = float(port.get("total_pnl", 0))
        pnl_p = float(port.get("total_pnl_pct", 0))
        real  = float(port.get("realized_pnl", 0))
        dd    = float(port.get("drawdown_pct", 0))
        pos   = port.get("positions", {})
        n_pos = sum(1 for v in pos.values() if float(v.get("quantity", 0)) > 0)

        dd_color = "red" if dd > 0.10 else ("yellow" if dd > 0.05 else "green")
        t.add_row(
            label,
            _money(eq),
            _money(cash),
            _colored_pnl(pnl),
            _colored_pnl(pnl_p, fmt="pct"),
            _colored_pnl(real),
            Text(_pct(dd), style=f"bold {dd_color}"),
            str(n_pos),
        )

    combined_eq  = float(retail.get("total_equity", 0)) + float(inst.get("total_equity", 0))
    combined_pnl = float(retail.get("total_pnl", 0)) + float(inst.get("total_pnl", 0))
    init_cap     = float(retail.get("initial_capital", 0)) + float(inst.get("initial_capital", 0))
    combined_pct = combined_pnl / init_cap if init_cap > 0 else 0.0

    t.add_section()
    t.add_row(
        "[bold]TOTALE[/bold]",
        f"[bold]{_money(combined_eq)}[/bold]",
        "—", _colored_pnl(combined_pnl), _colored_pnl(combined_pct, fmt="pct"),
        "—", "—", "—",
    )

    return Panel(t, title="[bold cyan]💰 PORTFOLIO P&L[/bold cyan]",
                 border_style="cyan", padding=(0, 0))


# ─────────────────────────────────────────────────────────────────
# Sezione: Posizioni aperte per strategia
# ─────────────────────────────────────────────────────────────────

def build_positions_table(retail: dict, inst: dict) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta",
              expand=True, padding=(0, 1))
    t.add_column("Simbolo",   style="bold white", width=10)
    t.add_column("Portfolio", width=12)
    t.add_column("Strategia", width=10)
    t.add_column("Side",      width=6)
    t.add_column("Qty",       justify="right", width=12)
    t.add_column("Entry",     justify="right", width=10)
    t.add_column("Prezzo att.", justify="right", width=12)
    t.add_column("Unreal P&L", justify="right", width=14)
    t.add_column("Stop Loss", justify="right", width=10)
    t.add_column("Aperta",    width=12)

    rows: list[tuple] = []
    for label, port in [("Retail", retail), ("Inst.", inst)]:
        for sym, pos in port.get("positions", {}).items():
            qty = float(pos.get("quantity", 0))
            if qty <= 0:
                continue
            rows.append((sym, label, pos))

    if not rows:
        t.add_row("—", "—", "—", "—", "—", "—", "—", "—", "—", "—")
    else:
        for sym, label, pos in sorted(rows, key=lambda x: x[0]):
            strat  = pos.get("strategy_bucket", pos.get("agent_source", "?"))
            side   = pos.get("side", "long")
            qty    = float(pos.get("quantity", 0))
            entry  = float(pos.get("avg_entry_price", 0))
            cur    = float(pos.get("current_price", 0))
            upnl   = float(pos.get("unrealized_pnl", 0))
            sl     = float(pos.get("stop_loss_price", 0))
            opened = _fmt_dt(pos.get("opened_at"), relative=True)

            side_color = "green" if side == "long" else "red"
            strat_color = {
                "bull": "bright_green", "bear": "red",
                "crypto": "bright_cyan", "alpha": "magenta",
            }.get(strat, "white")

            t.add_row(
                sym,
                label,
                Text(strat.upper(), style=strat_color),
                Text(side.upper(), style=side_color),
                f"{qty:.4f}",
                f"${entry:.4f}",
                f"${cur:.4f}",
                _colored_pnl(upnl),
                f"${sl:.4f}" if sl > 0 else "—",
                opened,
            )

    return Panel(t, title="[bold magenta]📊 POSIZIONI APERTE[/bold magenta]",
                 border_style="magenta", padding=(0, 0))


# ─────────────────────────────────────────────────────────────────
# Sezione: Segnali RiskAgent
# ─────────────────────────────────────────────────────────────────

def build_signals_table(validated: dict) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold yellow",
              expand=True, padding=(0, 1))
    t.add_column("Simbolo",   style="bold white", width=10)
    t.add_column("Portfolio", width=10)
    t.add_column("Segnale",   width=8)
    t.add_column("Strategia", width=10)
    t.add_column("Score",     justify="right", width=8)
    t.add_column("Entry",     justify="right", width=10)
    t.add_column("Stop",      justify="right", width=10)
    t.add_column("Target",    justify="right", width=10)
    t.add_column("Size",      justify="right", width=10)
    t.add_column("ML Boost",  justify="right", width=10)

    rows: list[tuple] = []
    ts = validated.get("timestamp", "")
    for mode in ("retail", "institutional"):
        for sym, sig in validated.get(mode, {}).items():
            if sig.get("approved"):
                rows.append((sym, mode[:4].title(), sig))

    if not rows:
        t.add_row("Nessun segnale attivo", *["—"] * 9)
    else:
        for sym, mode, sig in sorted(rows, key=lambda x: -float(x[2].get("score", 0))):
            stype  = sig.get("signal_type", "?")
            strat  = sig.get("strategy_bucket", sig.get("agent_source", "?"))
            score  = float(sig.get("score", 0))
            entry  = float(sig.get("entry_price", 0))
            sl     = float(sig.get("stop_loss_price", 0))
            tp     = float(sig.get("take_profit_price", 0))
            size   = float(sig.get("position_size_usdt", 0))
            boost  = sig.get("ml_boost")

            stype_color = "green" if stype == "BUY" else "red"
            strat_color = {
                "bull": "bright_green", "bear": "red",
                "crypto": "bright_cyan",
            }.get(strat, "white")
            score_color = "green" if score >= 0.7 else ("yellow" if score >= 0.6 else "white")

            boost_txt = Text(f"+{boost:.3f}", style="bright_green") if boost and boost > 0 else (
                Text(f"{boost:.3f}", style="red") if boost and boost < 0 else Text("—", style="dim"))

            t.add_row(
                sym, mode,
                Text(stype, style=f"bold {stype_color}"),
                Text(strat.upper(), style=strat_color),
                Text(f"{score:.3f}", style=score_color),
                f"${entry:.3f}", f"${sl:.3f}", f"${tp:.3f}",
                _money(size), boost_txt,
            )

    age = _fmt_dt(ts, relative=True) if ts else "—"
    return Panel(t, title=f"[bold yellow]⚡ SEGNALI RISK AGENT[/bold yellow]  [dim](aggiornato {age})[/dim]",
                 border_style="yellow", padding=(0, 0))


# ─────────────────────────────────────────────────────────────────
# Sezione: Stato agenti
# ─────────────────────────────────────────────────────────────────

def build_agents_table(state: dict) -> Panel:
    agents = state.get("agents", {})

    AGENT_LABELS = {
        "market_data_agent":        "Market Data",
        "macro_analyzer_agent":     "Macro",
        "sector_analyzer_agent":    "Sector",
        "stock_analyzer_agent":     "Stock",
        "news_data_agent":          "News",
        "technical_analysis_agent": "TA",
        "risk_agent":               "Risk",
        "execution_agent":          "Execution",
        "report_agent":             "Report",
        "bull_strategy_agent":      "Bull Strategy",
        "bear_strategy_agent":      "Bear Strategy",
        "crypto_strategy_agent":    "Crypto Strategy",
        "alpha_hunter_agent":       "Alpha Hunter",
        "portfolio_manager":        "Portfolio Mgr",
        "feature_store_agent":      "Feature Store",
        "ml_strategy_agent":        "ML Strategy",
    }

    half = len(AGENT_LABELS) // 2
    items = list(AGENT_LABELS.items())
    left_items  = items[:half]
    right_items = items[half:]

    def make_col(items_list: list) -> Table:
        t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        t.add_column("Agent",  style="white", width=15)
        t.add_column("Status", width=8)
        t.add_column("Ultima", width=10)
        t.add_column("Note",   width=18, overflow="fold")
        for key, label in items_list:
            info   = agents.get(key, {})
            status = info.get("status", "idle")
            # Fallback ai log se shared_state è idle ma il bot ha girato
            last_run = info.get("last_run")
            if not last_run and status == "idle":
                last_run = _read_log_last_run(key)
                if last_run:
                    status = "done"  # era già stato eseguito

            last = _fmt_dt(last_run, relative=True)
            err  = info.get("last_error") or (_read_log_last_error(key) if status == "error" else None)

            status_icon = {
                "running": "⟳ RUN",
                "done":    "✓ OK",
                "error":   "✗ ERR",
                "idle":    "· idle",
            }.get(status, status)

            color = _agent_color(status)
            note = Text(str(err)[:18], style="dim red") if err else Text("", style="dim")
            t.add_row(label, Text(status_icon, style=color), last, note)
        return t

    cols = Columns([make_col(left_items), make_col(right_items)], expand=True)
    return Panel(cols, title="[bold white]🔧 STATO AGENTI[/bold white]",
                 border_style="white", padding=(0, 1))


# ─────────────────────────────────────────────────────────────────
# Sezione: ML + AutoBacktest + Pesi
# ─────────────────────────────────────────────────────────────────

def _equity_curve_from_portfolios(retail: dict, inst: dict) -> list[float]:
    """Ricostruisce curva equity dai trade in ordine cronologico."""
    trades = retail.get("trades", []) + inst.get("trades", [])
    if not trades:
        return []
    trades_sorted = sorted(
        [t for t in trades if t.get("timestamp") and t.get("realized_pnl") is not None],
        key=lambda x: x["timestamp"]
    )
    init = float(retail.get("initial_capital", 20000)) + float(inst.get("initial_capital", 880000))
    eq = init
    curve = [init]
    for t in trades_sorted:
        eq += float(t.get("realized_pnl", 0))
        curve.append(round(eq, 2))
    return curve


def build_equity_panel(retail: dict, inst: dict) -> Panel:
    """Pannello con sparkline curva equity combinata."""
    curve = _equity_curve_from_portfolios(retail, inst)
    if not curve:
        return Panel("[dim]Nessun trade ancora[/dim]",
                     title="[bold green]📈 CURVA EQUITY[/bold green]", border_style="green")

    spark = _sparkline(curve, width=60)
    init  = curve[0]
    last  = curve[-1]
    pnl   = last - init
    pnl_p = pnl / init if init > 0 else 0.0

    # Mini stats
    peak    = max(curve)
    trough  = min(curve)
    dd      = (peak - trough) / peak if peak > 0 else 0.0
    n_up    = sum(1 for i in range(1, len(curve)) if curve[i] > curve[i-1])
    n_dn    = len(curve) - 1 - n_up

    pnl_txt = _colored_pnl(pnl)
    pct_txt = _colored_pnl(pnl_p, fmt="pct")

    t = Table(box=None, show_header=False, padding=(0, 2), expand=True)
    t.add_column("", width=60)
    t.add_column("", width=30)
    t.add_row(spark, "")
    t.add_row(
        Text(f"  Inizio: {_money(init)}  →  Attuale: {_money(last)}", style="dim"),
        Text(f"P&L: ", style="white").append_text(pnl_txt).append("  ").append_text(pct_txt),
    )
    t.add_row(
        Text(f"  Peak: {_money(peak)}  |  Trough: {_money(trough)}  |  Max DD: {_pct(dd)}", style="dim"),
        Text(f"Trade ▲{n_up}  ▼{n_dn}  tot:{len(curve)-1}", style="dim"),
    )

    return Panel(t, title="[bold green]📈 CURVA EQUITY (trade cumulativi)[/bold green]",
                 border_style="green", padding=(0, 0))


def build_ml_panel(cfg: dict) -> Panel:
    ml_state    = _read(DATA_DIR / "ml_strategy_state.json")
    bt_state    = _read(DATA_DIR / "auto_backtest_state.json")
    weights_doc = _read(DATA_DIR / "learned_strategy_weights.json")

    ml_cfg  = cfg.get("ml_strategy", {})
    ab_cfg  = cfg.get("orchestrator", {}).get("auto_backtest", {})

    refresh_days = int(ml_cfg.get("refresh_days", 30))
    ml_last      = ml_state.get("last_run_at")
    ml_status    = ml_state.get("status", "—")
    ml_next      = _next_due(ml_last, refresh_days)
    ml_last_fmt  = _fmt_dt(ml_last, relative=True)

    bt_interval  = int(ab_cfg.get("interval_days", 7))
    bt_last      = bt_state.get("last_run_at")
    bt_ok        = bt_state.get("last_run_success", None)
    bt_next      = _next_due(bt_last, bt_interval)
    bt_last_fmt  = _fmt_dt(bt_last, relative=True)

    weights      = weights_doc.get("strategy_weights", {})
    w_gen        = _fmt_dt(weights_doc.get("generated_at"), relative=True)
    n_reports    = weights_doc.get("reports_count", 0)

    # ML row
    ml_color = "green" if ml_status == "ok" else ("yellow" if not ml_last else "red")

    # Backtest row
    bt_color = "green" if bt_ok else ("dim" if bt_ok is None else "red")
    bt_ok_str = "✓ ok" if bt_ok else ("—" if bt_ok is None else "✗ fail")

    t = Table(box=None, show_header=False, padding=(0, 2), expand=True)
    t.add_column("Label",  style="bold white", width=22)
    t.add_column("Valore", width=40)

    t.add_row("🧠 ML Strategy",    "")
    t.add_row("  Ultimo training",  Text(ml_last_fmt, style=ml_color))
    t.add_row("  Stato",            Text(ml_status, style=ml_color))
    t.add_row("  Prossimo refresh", Text(ml_next, style="cyan"))
    t.add_row("  Intervallo",       f"ogni {refresh_days} giorni")
    t.add_section()
    t.add_row("📈 Auto-Backtest",   "")
    t.add_row("  Ultimo run",       Text(bt_last_fmt, style=bt_color))
    t.add_row("  Esito",            Text(bt_ok_str, style=bt_color))
    t.add_row("  Prossimo run",     Text(bt_next, style="cyan"))
    t.add_row("  Intervallo",       f"ogni {bt_interval} giorni")
    t.add_section()
    t.add_row("⚖️  Pesi Strategia", f"[dim](da {n_reports} report, {w_gen})[/dim]")
    for strat, w in weights.items():
        bar_len = int(float(w) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = "bright_green" if float(w) > 1.0 else ("red" if float(w) < 0.9 else "yellow")
        t.add_row(f"  {strat.upper()}", Text(f"{bar}  {float(w):.3f}x", style=color))

    # Curva di apprendimento: Sharpe per fold dal summary ML
    ml_summary_files = sorted(REPORTS_DIR.glob("ml_cross_sectional_summary_*.json"))
    if ml_summary_files:
        latest_summary = _read(ml_summary_files[-1])
        folds = latest_summary.get("folds", [])
        if folds:
            sharpes = [f.get("val_sharpe", 0.0) for f in folds]
            t.add_section()
            t.add_row("📊 Curva apprendimento", f"[dim]({len(folds)} fold, ultimo summary)[/dim]")
            spark = _sparkline(sharpes, width=30)
            best_fold = max(folds, key=lambda f: f.get("val_sharpe", 0))
            t.add_row("  Val Sharpe/fold", spark)
            t.add_row("  Miglior modello", Text(best_fold.get("model", "?"), style="bright_green"))
            t.add_row("  Best val Sharpe", Text(f"{best_fold.get('val_sharpe', 0):.3f}", style="bright_green"))
            oos = latest_summary.get("oos", {})
            if oos:
                t.add_row("  OOS Sharpe", Text(f"{oos.get('sharpe', 0):.3f}", style="cyan"))
                t.add_row("  OOS Max DD",  Text(f"{_pct(oos.get('max_drawdown', 0))}", style="yellow"))

    return Panel(t, title="[bold green]🤖 ML & APPRENDIMENTO[/bold green]",
                 border_style="green", padding=(0, 1))


# ─────────────────────────────────────────────────────────────────
# Sezione: Statistiche trade per strategia
# ─────────────────────────────────────────────────────────────────

def build_strategy_stats(retail: dict, inst: dict) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold bright_cyan",
              expand=True, padding=(0, 1))
    t.add_column("Strategia",   style="bold white", width=12)
    t.add_column("Portfolio",   width=10)
    t.add_column("Trade tot.",  justify="right", width=10)
    t.add_column("Chiusi",      justify="right", width=8)
    t.add_column("Win Rate",    justify="right", width=10)
    t.add_column("Realized P&L",justify="right", width=14)
    t.add_column("Avg P&L/trade",justify="right", width=14)
    t.add_column("Stop Loss %", justify="right", width=12)

    for label, port in [("Retail", retail), ("Inst.", inst)]:
        trades = port.get("trades", [])
        if not trades:
            continue

        buckets: dict[str, list] = {}
        for tr in trades:
            b = str(tr.get("strategy_bucket", tr.get("agent_source", "unknown"))).lower()
            buckets.setdefault(b, []).append(tr)

        for strat in sorted(buckets.keys()):
            tlist  = buckets[strat]
            closed = [x for x in tlist if x.get("realized_pnl") is not None]
            wins   = [x for x in closed if float(x.get("realized_pnl", 0)) > 0]
            stops  = [x for x in closed if x.get("reason") == "STOP_LOSS_TRIGGERED"]
            real   = sum(float(x.get("realized_pnl", 0)) for x in closed)
            wr     = len(wins) / len(closed) if closed else 0.0
            avg_pnl = real / len(closed) if closed else 0.0
            stop_pct = len(stops) / len(closed) if closed else 0.0

            strat_color = {
                "bull": "bright_green", "bear": "red",
                "crypto": "bright_cyan", "alpha": "magenta",
            }.get(strat, "white")
            wr_color = "green" if wr >= 0.55 else ("yellow" if wr >= 0.45 else "red")

            t.add_row(
                Text(strat.upper(), style=strat_color),
                label,
                str(len(tlist)),
                str(len(closed)),
                Text(_pct(wr), style=wr_color),
                _colored_pnl(real),
                _colored_pnl(avg_pnl),
                Text(_pct(stop_pct), style="dim"),
            )

    return Panel(t, title="[bold bright_cyan]📉 STATISTICHE STRATEGIE[/bold bright_cyan]",
                 border_style="bright_cyan", padding=(0, 0))


# ─────────────────────────────────────────────────────────────────
# Layout principale
# ─────────────────────────────────────────────────────────────────

def build_layout(cfg: dict) -> Any:
    retail    = _read(DATA_DIR / "portfolio_retail.json")
    inst      = _read(DATA_DIR / "portfolio_institutional.json")
    state     = _read(DATA_DIR / "shared_state.json")
    validated = _read(DATA_DIR / "validated_signals.json")

    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="pnl",      size=9),
        Layout(name="equity",   size=7),
        Layout(name="middle",   size=20),
        Layout(name="signals",  size=15),
        Layout(name="bottom",   size=20),
        Layout(name="footer",   size=1),
    )

    layout["header"].update(build_header(state, cfg))
    layout["pnl"].update(build_portfolio_table(retail, inst))
    layout["equity"].update(build_equity_panel(retail, inst))

    layout["middle"].split_row(
        Layout(build_agents_table(state), name="agents", ratio=3),
        Layout(build_ml_panel(cfg),       name="ml",     ratio=2),
    )

    layout["signals"].update(build_signals_table(validated))
    layout["bottom"].split_row(
        Layout(build_positions_table(retail, inst),  name="positions",  ratio=3),
        Layout(build_strategy_stats(retail, inst),   name="stats",      ratio=2),
    )

    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    layout["footer"].update(
        Text(f"  [q] esci  │  Refresh ogni {cfg.get('_dash_interval', 2)}s  │  {now} UTC", style="dim")
    )

    return layout


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Bot Terminal Dashboard")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Secondi tra ogni refresh (default: 2)")
    args = parser.parse_args()

    cfg = {}
    cfg_path = BASE_DIR / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    console = Console()

    cfg["_dash_interval"] = args.interval

    try:
        with Live(
            build_layout(cfg),
            console=console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while True:
                time.sleep(args.interval)
                try:
                    live.update(build_layout(cfg))
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass

    console.print("\n[dim]Dashboard chiusa.[/dim]")


if __name__ == "__main__":
    main()
