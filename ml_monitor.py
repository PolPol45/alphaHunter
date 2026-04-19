#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║       ML WALK-FORWARD TRAINING MONITOR              ║
║       Live terminal dashboard — CryptoBro            ║
╚══════════════════════════════════════════════════════╝
Usage:  python ml_monitor.py
"""
import json
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

BASE = Path(__file__).parent
LOGS = BASE / "logs"
DATA = BASE / "data"

# ── Config ────────────────────────────────────────────────────────────────────
LOG_FILES = {
    "walk_forward": LOGS / "ml_training.log",
    "ml_agent":     LOGS / "ml_strategy_agent.log",
    "backtest":     LOGS / "backtesting_agent.log",
    "risk":         LOGS / "risk_agent.log",
}

REFRESH = 2  # seconds

# ── Helpers ───────────────────────────────────────────────────────────────────

def tail_lines(path: Path, n: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        with open(path, "r", errors="replace") as f:
            return deque(f, n)
    except Exception:
        return []


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def parse_walk_forward_state(lines: list[str]) -> dict:
    state = {
        "sim_day": 0,
        "ml_cycles": 0,
        "last_fold": 0,
        "models_tested": [],
        "last_ic": None,
        "last_status": "Waiting...",
        "errors": [],
        "last_event_ts": None,
    }
    for line in lines:
        # Simulated day
        m = re.search(r"Giorno di simulazione[:\s]+(\d+)", line)
        if m:
            state["sim_day"] = int(m.group(1))
            state["ml_cycles"] = state["sim_day"] // 30
            state["last_event_ts"] = _extract_ts(line)

        # Fold detection
        m = re.search(r"fold[=\s:]+(\d+)", line, re.IGNORECASE)
        if m:
            state["last_fold"] = int(m.group(1))

        # IC
        m = re.search(r"ic[=\s:>]+([0-9.\-]+)", line, re.IGNORECASE)
        if m:
            try:
                state["last_ic"] = float(m.group(1))
            except Exception:
                pass

        # Models
        for model in ["elastic_net", "random_forest", "mlp", "stacked_ensemble", "xgboost", "lightgbm"]:
            if model in line.lower() and "model" in line.lower():
                if model not in state["models_tested"]:
                    state["models_tested"].append(model)

        # Status lines
        if "MLStrategyAgent completed" in line:
            state["last_status"] = "✅ ML Cycle Done"
            state["last_event_ts"] = _extract_ts(line)
        elif "Walk-Forward Optimization" in line:
            state["last_status"] = "🔄 ML Cycle Running"
            state["last_event_ts"] = _extract_ts(line)
        elif "Training K-Fold" in line:
            state["last_status"] = "🧠 Training Models..."
            state["last_event_ts"] = _extract_ts(line)
        elif "Ottimizzazione Pesi" in line:
            state["last_status"] = "⚖️  Optimizing Weights..."
        elif "Elaborazione Storica" in line:
            state["last_status"] = "📊 Building Feature Store..."

        if "ERROR" in line or "Errore" in line:
            state["errors"].append(line.strip()[-120:])

    return state


def _extract_ts(line: str) -> str:
    m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
    return m.group(1) if m else "—"


def parse_ml_signals() -> dict:
    doc = read_json(DATA / "ml_signals.json")
    ranking = doc.get("latest_ranking", {})
    longs = ranking.get("top_decile_long", [])
    shorts = ranking.get("bottom_decile_short", [])
    return {
        "generated_at": doc.get("generated_at", "—"),
        "latest_date": doc.get("latest_date", "—"),
        "avg_ic": doc.get("avg_ic_across_folds", None),
        "target_type": doc.get("target_type", "—"),
        "risk_parity": doc.get("risk_parity", False),
        "stacking": doc.get("ensemble_stacking", False),
        "longs": longs[:8],
        "shorts": shorts[:8],
    }


def parse_key_findings() -> dict:
    """Read the latest ML summary report for key metrics."""
    import glob
    reports = sorted(glob.glob(str(BASE / "reports" / "ml_cross_sectional_summary_*.json")))
    findings = {
        "best_model": "—",
        "oos_sharpe": None,
        "oos_sortino": None,
        "oos_drawdown": None,
        "avg_ic": None,
        "fold_count": 0,
        "models_tested": [],
        "top_long": "—",
        "top_short": "—",
        "signal_date": "—",
        "report_ts": "—",
    }
    if not reports:
        return findings
    try:
        doc = json.loads(Path(reports[-1]).read_text())
        oos = doc.get("oos", {})
        findings["best_model"] = doc.get("best_model", "—")
        findings["oos_sharpe"] = oos.get("sharpe")
        findings["oos_sortino"] = oos.get("sortino")
        findings["oos_drawdown"] = oos.get("max_drawdown")
        findings["avg_ic"] = doc.get("avg_val_ic")
        findings["fold_count"] = doc.get("fold_count", 0)
        findings["models_tested"] = doc.get("models_tested", [])
        findings["report_ts"] = doc.get("generated_at", "—")[:19]

        # Pull latest ranking from signals
        sig_doc = read_json(DATA / "ml_signals.json")
        ranking = sig_doc.get("latest_ranking", {})
        longs = ranking.get("top_decile_long", [])
        shorts = ranking.get("bottom_decile_short", [])
        findings["top_long"] = ", ".join([s["symbol"] for s in longs[:5]]) if longs else "—"
        findings["top_short"] = ", ".join([s["symbol"] for s in shorts[:5]]) if shorts else "—"
        findings["signal_date"] = sig_doc.get("latest_date", "—")
    except Exception:
        pass
    return findings


def parse_backtest_log(lines: list[str]) -> dict:
    info = {"last_date": "—", "fills": 0, "equity": None, "pnl": None}
    for line in lines:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", line)
        if m and "start" not in line.lower():
            info["last_date"] = m.group(1)
        if "FILLED" in line:
            info["fills"] += 1
        m = re.search(r"equity[=\s]+([0-9,.]+)", line, re.IGNORECASE)
        if m:
            try:
                info["equity"] = float(m.group(1).replace(",", ""))
            except Exception:
                pass
    return info


# ── UI Builders ───────────────────────────────────────────────────────────────

def build_header() -> Panel:
    now = datetime.now().strftime("%H:%M:%S")
    txt = Text(justify="center")
    txt.append("🧠 ML WALK-FORWARD TRAINING MONITOR", style="bold magenta")
    txt.append(f"   ⏰ {now}", style="dim white")
    return Panel(txt, style="bright_black", padding=(0, 2))


def build_status_panel(state: dict) -> Panel:
    t = Table.grid(padding=1)
    t.add_column(style="dim cyan", width=22)
    t.add_column(style="bold white")

    sim_day = state["sim_day"]
    total_days = 365  # approximate
    pct = min(100, int((sim_day / total_days) * 100))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)

    t.add_row("Status:", Text(state["last_status"], style="bold yellow"))
    t.add_row("Sim. Day:", f"{sim_day}/~{total_days}  [{bar}] {pct}%")
    t.add_row("ML Cycles:", f"[bold green]{state['ml_cycles']}[/bold green] eseguiti")
    t.add_row("Last Fold:", f"[bold]{state['last_fold']}[/bold]")
    t.add_row("Last IC:", (
        f"[bold {'green' if (state['last_ic'] or 0) > 0 else 'red'}]{state['last_ic']:.4f}[/]"
        if state["last_ic"] is not None else "[dim]—[/]"
    ))
    t.add_row("Last Event:", f"[dim]{state['last_event_ts']}[/]")
    t.add_row("Models Tested:", ", ".join(state["models_tested"]) or "[dim]—[/]")

    return Panel(t, title="[bold cyan]🔄 TRAINING PROGRESS[/bold cyan]", border_style="cyan")


def build_signals_panel(sig: dict) -> Panel:
    t = Table.grid(padding=1)
    t.add_column(style="dim cyan", width=20)
    t.add_column(style="white")

    avg_ic = sig["avg_ic"]
    t.add_row("Generated:", f"[dim]{sig['generated_at'][:19]}[/]")
    t.add_row("Market Date:", f"[bold]{sig['latest_date']}[/]")
    t.add_row("Target Method:", f"[yellow]{sig['target_type']}[/]")
    t.add_row("Risk Parity:", "✅" if sig["risk_parity"] else "❌")
    t.add_row("Stacking:", "✅" if sig["stacking"] else "❌")
    t.add_row("Avg IC:", (
        f"[bold {'green' if (avg_ic or 0) > 0 else 'red'}]{avg_ic:.4f}[/]"
        if avg_ic is not None else "[dim]Waiting...[/]"
    ))

    return Panel(t, title="[bold green]📊 LAST ML SIGNALS[/bold green]", border_style="green")


def build_rankings_panel(sig: dict) -> Panel:
    longs = sig["longs"]
    shorts = sig["shorts"]

    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    t.add_column("LONG 📈", style="green", width=8)
    t.add_column("W%", style="bold green", width=7)
    t.add_column("SHORT 📉", style="red", width=8)
    t.add_column("W%", style="bold red", width=7)

    max_rows = max(len(longs), len(shorts))
    for i in range(max_rows):
        lcell = longs[i]["symbol"] if i < len(longs) else ""
        lw = f"{abs(longs[i]['weight']*100):.1f}%" if i < len(longs) else ""
        scell = shorts[i]["symbol"] if i < len(shorts) else ""
        sw = f"{abs(shorts[i]['weight']*100):.1f}%" if i < len(shorts) else ""
        t.add_row(lcell, lw, scell, sw)

    return Panel(t, title="[bold]⚖️  LATEST PORTFOLIO RANKINGS[/bold]", border_style="bright_black")


def build_key_findings_panel(f: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim cyan", width=20)
    t.add_column(style="bold white", width=32)
    t.add_column(style="dim cyan", width=20)
    t.add_column(style="bold white")

    def fmt_metric(val, good_positive=True) -> Text:
        if val is None:
            return Text("Waiting...", style="dim")
        color = "green" if (val > 0) == good_positive else "red"
        return Text(f"{val:+.4f}", style=f"bold {color}")

    sharpe = f["oos_sharpe"]
    sortino = f["oos_sortino"]
    drawdown = f["oos_drawdown"]
    ic = f["avg_ic"]

    # Interpretazioni testuali
    sharpe_note = ""
    if sharpe is not None:
        if sharpe > 1.5:   sharpe_note = " 🔥 Excellent"
        elif sharpe > 0.8: sharpe_note = " ✅ Good"
        elif sharpe > 0:   sharpe_note = " ⚠️  Weak"
        else:              sharpe_note = " ❌ Negative"

    ic_note = ""
    if ic is not None:
        if ic > 0.05:  ic_note = " 🔥 Strong signal"
        elif ic > 0.02: ic_note = " ✅ Moderate"
        elif ic > 0:   ic_note = " ⚠️  Weak"
        else:          ic_note = " ❌ No signal"

    t.add_row(
        "Best Model:", Text(f["best_model"], style="bold yellow"),
        "Folds run:", Text(str(f["fold_count"]), style="bold"),
    )
    t.add_row(
        "OOS Sharpe:", Text((f"{sharpe:+.3f}{sharpe_note}" if sharpe else "Waiting..."), style=("bold green" if (sharpe or 0) > 0 else "bold red")),
        "OOS Sortino:", Text((f"{sortino:+.3f}" if sortino else "—"), style=("bold green" if (sortino or 0) > 0 else "bold red")),
    )
    t.add_row(
        "Max Drawdown:", Text((f"{drawdown:.1%}" if drawdown else "—"), style="bold red" if drawdown else "dim"),
        "Avg IC:", Text((f"{ic:+.4f}{ic_note}" if ic else "Waiting..."), style=("bold green" if (ic or 0) > 0 else "dim")),
    )
    t.add_row(
        "Top LONG picks:", Text(f["top_long"], style="bold green"),
        "Signal Date:", Text(f["signal_date"], style="dim"),
    )
    t.add_row(
        "Top SHORT picks:", Text(f["top_short"], style="bold red"),
        "Report at:", Text(f["report_ts"], style="dim"),
    )
    if f["models_tested"]:
        t.add_row("Models tested:", Text(", ".join(f["models_tested"]), style="dim yellow"), "", "")

    return Panel(t, title="[bold yellow]🏆 KEY FINDINGS — Latest ML Report[/bold yellow]", border_style="yellow")


def build_errors_panel(state: dict) -> Panel:
    errors = state["errors"][-6:]
    if not errors:
        txt = Text("No errors ✅", style="green")
    else:
        txt = Text("\n".join(errors), style="red", overflow="fold")
    return Panel(txt, title="[bold red]⚠️  ERRORS[/bold red]", border_style="red")


def build_recent_log(lines: list[str]) -> Panel:
    # Filter relevant lines
    keywords = ["ML", "Walk", "fold", "IC", "Training", "Ottimizza", "feat", "ERROR",
                 "FILLED", "equity", "Giorno"]
    filtered = [l.rstrip() for l in lines if any(k in l for k in keywords)][-14:]
    if not filtered:
        filtered = list(lines)[-10:]

    txt = Text(overflow="fold")
    for line in filtered:
        if "ERROR" in line or "Errore" in line:
            txt.append(line[-110:] + "\n", style="bold red")
        elif "completed" in line.lower() or "DONE" in line:
            txt.append(line[-110:] + "\n", style="bold green")
        elif "Walk-Forward" in line or "Training" in line:
            txt.append(line[-110:] + "\n", style="bold yellow")
        else:
            txt.append(line[-110:] + "\n", style="dim white")

    return Panel(txt, title="[bold]📋 LIVE LOG (ML events)[/bold]", border_style="bright_black")


# ── Main loop ─────────────────────────────────────────────────────────────────

def make_layout(state, sig, bt, findings, log_lines) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="key_findings", size=9),
        Layout(name="log", size=10),
    )
    layout["body"].split_row(
        Layout(name="status"),
        Layout(name="signals"),
        Layout(name="rankings"),
    )
    layout["header"].update(build_header())
    layout["status"].update(build_status_panel(state))
    layout["signals"].update(build_signals_panel(sig))
    layout["rankings"].update(build_rankings_panel(sig))
    layout["key_findings"].update(build_key_findings_panel(findings))
    layout["log"].update(build_recent_log(log_lines))
    return layout


def main():
    console = Console()
    console.print("[bold magenta]Avvio ML Monitor...[/] Ctrl+C per uscire\n")
    time.sleep(1)

    with Live(console=console, refresh_per_second=0.5, screen=True) as live:
        while True:
            try:
                # Read logs
                wf_lines = tail_lines(LOG_FILES["walk_forward"])
                ml_lines = tail_lines(LOG_FILES["ml_agent"])
                bt_lines = tail_lines(LOG_FILES["backtest"])

                all_lines = list(wf_lines) + list(ml_lines)

                state = parse_walk_forward_state(all_lines)
                sig = parse_ml_signals()
                bt = parse_backtest_log(bt_lines)
                findings = parse_key_findings()

                layout = make_layout(state, sig, bt, findings, all_lines)
                live.update(layout)

                time.sleep(REFRESH)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Monitor error: {e}[/]")
                time.sleep(REFRESH)

    console.print("\n[bold]Monitor chiuso.[/]")


if __name__ == "__main__":
    main()
