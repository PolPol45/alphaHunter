"""
monte_carlo_dashboard.py
========================
Dashboard terminale per monitorare e visualizzare i risultati Monte Carlo.

Funzionalità:
- Legge l'ultimo report MC da reports/
- Mostra distribuzione equity curve (fan chart ASCII)
- Distribuzione Sharpe / Drawdown / Return in percentili
- Probabilità chiave
- Istogramma ASCII dei ritorni
- Lancia un nuovo run MC in background e si aggiorna in tempo reale

Uso:
    python monte_carlo_dashboard.py                      # guarda ultimo report
    python monte_carlo_dashboard.py --run                # lancia nuovo run e monitora
    python monte_carlo_dashboard.py --run --scenarios 50 --days 60
    python monte_carlo_dashboard.py --run --scenario-type crash
    python monte_carlo_dashboard.py --interval 3         # refresh ogni 3s
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

try:
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("Installa rich: pip install rich")
    sys.exit(1)

BASE_DIR = pathlib.Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BARS = "▁▂▃▄▅▆▇█"

def _sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return "─" * width
    mn, mx = min(values), max(values)
    rng = mx - mn or 1.0
    idxs = [int((v - mn) / rng * (len(BARS) - 1)) for v in values]
    chars = [BARS[i] for i in idxs]
    # resample to width
    if len(chars) > width:
        step = len(chars) / width
        chars = [chars[int(i * step)] for i in range(width)]
    elif len(chars) < width:
        chars = chars + ["─"] * (width - len(chars))
    return "".join(chars)


def _bar(value: float, mn: float, mx: float, width: int = 20, char: str = "█") -> str:
    rng = mx - mn or 1.0
    filled = int((value - mn) / rng * width)
    filled = max(0, min(width, filled))
    return char * filled + "░" * (width - filled)


def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def _load_latest_report() -> dict | None:
    files = sorted(REPORTS_DIR.glob("monte_carlo_report_*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_all_reports() -> list[dict]:
    files = sorted(REPORTS_DIR.glob("monte_carlo_report_*.json"), key=lambda f: f.stat().st_mtime)
    out = []
    for f in files[-10:]:  # ultimi 10
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pannelli UI
# ─────────────────────────────────────────────────────────────────────────────

def build_header(report: dict | None, running: bool, run_progress: str) -> Panel:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if report:
        meta = report.get("parameters", report.get("meta", {}))
        n_scen = report.get("analysis", {}).get("scenarios_run", "?")
        days = meta.get("days", "?")
        stype = meta.get("scenario_type", "?")
        run_ts = report.get("generated_at", meta.get("timestamp", "?"))[:19]
        subtitle = f"[dim]{n_scen} scenari · {days}g · tipo=[bold]{stype}[/bold] · generato {run_ts}[/dim]"
    else:
        subtitle = "[dim]Nessun report trovato — usa --run per generarne uno[/dim]"

    status = f"[yellow]⟳ {run_progress}[/yellow]" if running else "[green]● Idle[/green]"
    title = Text()
    title.append("⬡  MONTE CARLO DASHBOARD  ", style="bold cyan")
    title.append(f"  {ts}  ", style="dim")
    title.append(status)
    return Panel(Align.center(title), subtitle=subtitle, style="cyan", height=3)


def build_fan_chart(report: dict) -> Panel:
    """Fan chart ASCII delle equity curve per percentile."""
    scenarios = report.get("scenario_results", report.get("scenarios", []))
    if not scenarios:
        return Panel("[dim]Nessuna equity curve disponibile[/dim]", title="📈 Fan Chart Equity", border_style="blue")

    curves = [s["equity_curve"] for s in scenarios if s.get("equity_curve")]
    if not curves:
        return Panel("[dim]Equity curve mancanti[/dim]", title="📈 Fan Chart Equity", border_style="blue")

    min_len = min(len(c) for c in curves)
    if min_len < 2:
        return Panel("[dim]Curve troppo corte[/dim]", title="📈 Fan Chart Equity", border_style="blue")

    # Tronca tutte alla lunghezza minima
    curves = [c[:min_len] for c in curves]

    WIDTH = 70
    HEIGHT = 16

    # Per ogni step campionato, calcola percentili
    steps = [int(i * (min_len - 1) / (WIDTH - 1)) for i in range(WIDTH)]

    def pct_at(step: int, p: float) -> float:
        vals = sorted(c[step] for c in curves)
        idx = (len(vals) - 1) * p / 100
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])

    p5  = [pct_at(s, 5)  for s in steps]
    p25 = [pct_at(s, 25) for s in steps]
    p50 = [pct_at(s, 50) for s in steps]
    p75 = [pct_at(s, 75) for s in steps]
    p95 = [pct_at(s, 95) for s in steps]

    all_vals = p5 + p95
    mn = min(all_vals)
    mx = max(all_vals)
    rng = mx - mn or 1.0

    def to_row(v: float) -> int:
        return int((v - mn) / rng * (HEIGHT - 1))

    # Costruisce griglia HEIGHT x WIDTH
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]

    def set_char(col: int, row_val: float, ch: str, force: bool = False) -> None:
        row = HEIGHT - 1 - to_row(row_val)
        row = max(0, min(HEIGHT - 1, row))
        if force or grid[row][col] == " ":
            grid[row][col] = ch

    # Banda p25-p75 (verde scuro)
    for col in range(WIDTH):
        lo_row = HEIGHT - 1 - to_row(p25[col])
        hi_row = HEIGHT - 1 - to_row(p75[col])
        lo_row = max(0, min(HEIGHT - 1, lo_row))
        hi_row = max(0, min(HEIGHT - 1, hi_row))
        for r in range(min(lo_row, hi_row), max(lo_row, hi_row) + 1):
            if grid[r][col] == " ":
                grid[r][col] = "▒"

    # Banda p5-p95 (più tenue)
    for col in range(WIDTH):
        lo_row = HEIGHT - 1 - to_row(p5[col])
        hi_row = HEIGHT - 1 - to_row(p95[col])
        lo_row = max(0, min(HEIGHT - 1, lo_row))
        hi_row = max(0, min(HEIGHT - 1, hi_row))
        for r in range(min(lo_row, hi_row), max(lo_row, hi_row) + 1):
            if grid[r][col] == " ":
                grid[r][col] = "░"

    # Mediana (bold)
    for col in range(WIDTH):
        set_char(col, p50[col], "─", force=True)

    # Scala Y (etichette equity)
    y_labels = []
    for row in range(HEIGHT):
        val = mn + (HEIGHT - 1 - row) / (HEIGHT - 1) * rng
        if row % 4 == 0:
            y_labels.append(f"{val:>10,.0f}")
        else:
            y_labels.append(" " * 10)

    lines = []
    for row in range(HEIGHT):
        label = y_labels[row]
        line_chars = grid[row]
        colored = Text()
        colored.append(f"{label} │", style="dim")
        for ch in line_chars:
            if ch == "─":
                colored.append(ch, style="bold yellow")
            elif ch == "▒":
                colored.append(ch, style="green")
            elif ch == "░":
                colored.append(ch, style="dim green")
            else:
                colored.append(ch)
        lines.append(colored)

    # Asse X
    x_axis = Text()
    x_axis.append("           └" + "─" * WIDTH, style="dim")
    lines.append(x_axis)

    # Legenda
    legend = Text()
    legend.append("           ")
    legend.append("░ P5-P95  ", style="dim green")
    legend.append("▒ P25-P75  ", style="green")
    legend.append("─ P50 (mediana)", style="bold yellow")

    start_eq = curves[0][0] if curves else 0
    final_p50 = p50[-1]
    ret_p50 = (final_p50 / start_eq - 1) * 100 if start_eq > 0 else 0

    content = Text()
    for line in lines:
        content.append_text(line)
        content.append("\n")
    content.append_text(legend)
    content.append(f"\n           Start: {start_eq:,.0f}  |  P50 finale: {final_p50:,.0f}  ({ret_p50:+.2f}%)", style="dim")

    return Panel(content, title="[bold blue]📈 Fan Chart Equity Curve[/bold blue]", border_style="blue")


def build_distribution_table(report: dict) -> Panel:
    analysis = report.get("analysis", {})
    if not analysis:
        return Panel("[dim]Nessuna analisi[/dim]", title="Distribuzione", border_style="magenta")

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", expand=True)
    t.add_column("Metrica", style="cyan", width=22)
    t.add_column("P5", justify="right", width=10)
    t.add_column("P25", justify="right", width=10)
    t.add_column("P50", justify="right", width=10)
    t.add_column("P75", justify="right", width=10)
    t.add_column("P95", justify="right", width=10)
    t.add_column("Media", justify="right", width=10)

    def add_row(label: str, key: str, fmt_fn=_fmt, style: str = ""):
        d = analysis.get(key, {})
        if not d:
            return
        t.add_row(
            label,
            fmt_fn(d.get("p5")),
            fmt_fn(d.get("p25")),
            fmt_fn(d.get("p50")),
            fmt_fn(d.get("p75")),
            fmt_fn(d.get("p95")),
            fmt_fn(d.get("mean")),
            style=style,
        )

    def fmt_pct_val(v):
        return f"{v:.2f}%" if v is not None else "n/a"

    add_row("Sharpe Ratio",      "sharpe",            _fmt)
    add_row("Sortino Ratio",     "sortino",           _fmt)
    add_row("Max Drawdown",      "max_drawdown",      lambda v: _pct(v) if v is not None else "n/a")
    add_row("Calmar Ratio",      "calmar",            _fmt)
    add_row("Total Return %",    "total_return_pct",  fmt_pct_val, style="bold")
    add_row("Win Rate",          "win_rate",          lambda v: f"{v:.1%}" if v is not None else "n/a")

    return Panel(t, title="[bold magenta]📊 Distribuzione Metriche[/bold magenta]", border_style="magenta")


def build_probability_panel(report: dict) -> Panel:
    analysis = report.get("analysis", {})
    prob = analysis.get("probabilities", {})
    meta = report.get("parameters", report.get("meta", {}))

    if not prob:
        return Panel("[dim]—[/dim]", title="Probabilità", border_style="yellow")

    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("Evento", style="white", width=30)
    t.add_column("Prob", justify="right", width=8)
    t.add_column("Barra", width=24)

    def add_prob(label: str, key: str, bad: bool = False):
        v = prob.get(key)
        if v is None:
            return
        bar = _bar(v, 0, 1, width=20)
        style = ("red" if v > 0.5 else "yellow") if bad else ("green" if v > 0.5 else "yellow")
        t.add_row(label, f"[{style}]{v:.1%}[/{style}]", f"[{style}]{bar}[/{style}]")

    add_prob("Drawdown > 15%",         "drawdown_gt_15pct",   bad=True)
    add_prob("Rovina (DD > 50%)",      "ruin_gt_50pct_dd",    bad=True)
    add_prob("Ritorno positivo",        "positive_return",     bad=False)
    add_prob("Sharpe > 0",             "sharpe_positive",     bad=False)

    # Parametri scenario
    t.add_row("", "", "")
    t.add_row("[dim]Scenario Type[/dim]",  f"[cyan]{meta.get('scenario_type','?')}[/cyan]", "")
    t.add_row("[dim]Scenari[/dim]",        f"{analysis.get('scenarios_run','?')}", "")
    t.add_row("[dim]Orizzonte[/dim]",      f"{meta.get('days','?')}g", "")

    return Panel(t, title="[bold yellow]⚡ Probabilità & Scenario[/bold yellow]", border_style="yellow")


def build_histogram(report: dict) -> Panel:
    """Istogramma ASCII dei total return %."""
    scenarios = report.get("scenario_results", report.get("scenarios", []))
    returns = [s.get("total_return_pct", 0) for s in scenarios if s.get("total_return_pct") is not None]
    if not returns:
        return Panel("[dim]Nessun dato[/dim]", title="Istogramma Ritorni", border_style="green")

    BINS = 20
    mn, mx = min(returns), max(returns)
    rng = mx - mn or 1.0
    bin_size = rng / BINS
    counts = [0] * BINS
    for r in returns:
        idx = min(int((r - mn) / rng * BINS), BINS - 1)
        counts[idx] += 1

    max_count = max(counts) or 1
    BAR_W = 30

    content = Text()
    zero_bin = int((0 - mn) / rng * BINS) if mn < 0 < mx else -1

    for i, cnt in enumerate(counts):
        bin_lo = mn + i * bin_size
        bin_hi = bin_lo + bin_size
        bar_len = int(cnt / max_count * BAR_W)
        bar = "█" * bar_len

        if i == zero_bin or (bin_lo < 0 <= bin_hi):
            style = "yellow"
            marker = "◀0"
        elif bin_lo >= 0:
            style = "green"
            marker = ""
        else:
            style = "red"
            marker = ""

        label = f"{bin_lo:+6.1f}%"
        content.append(f"  {label} │", style="dim")
        content.append(f"{bar:<{BAR_W}}", style=style)
        content.append(f" {cnt:3d} {marker}\n", style="dim")

    mean_ret = sum(returns) / len(returns)
    pos = sum(1 for r in returns if r > 0)
    content.append(f"\n  Media: {mean_ret:+.2f}%  |  Positivi: {pos}/{len(returns)} ({pos/len(returns):.0%})", style="dim")

    return Panel(content, title="[bold green]📉 Distribuzione Ritorni[/bold green]", border_style="green")


def build_history_panel(reports: list[dict]) -> Panel:
    """Pannello storico degli ultimi run MC."""
    if not reports:
        return Panel("[dim]Nessun report storico[/dim]", title="Storico Run", border_style="dim")

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold", expand=True)
    t.add_column("Data", style="dim", width=18)
    t.add_column("Tipo", width=8)
    t.add_column("Scen", justify="right", width=5)
    t.add_column("P50 Ret%", justify="right", width=9)
    t.add_column("P50 Sharpe", justify="right", width=10)
    t.add_column("P50 MDD", justify="right", width=9)
    t.add_column("P(pos)", justify="right", width=7)

    for rep in reversed(reports[-8:]):
        meta = rep.get("parameters", rep.get("meta", {}))
        an = rep.get("analysis", {})
        prob = an.get("probabilities", {})
        ret = an.get("total_return_pct", {})
        sh  = an.get("sharpe", {})
        mdd = an.get("max_drawdown", {})
        ts  = rep.get("generated_at", meta.get("timestamp", ""))[:16].replace("T", " ")
        stype = meta.get("scenario_type", "?")
        n_scen = an.get("scenarios_run", "?")
        p50_ret = ret.get("p50")
        p50_sh  = sh.get("p50")
        p50_mdd = mdd.get("p50")
        ppos    = prob.get("positive_return")

        ret_style = "green" if (p50_ret or 0) >= 0 else "red"
        t.add_row(
            ts,
            f"[cyan]{stype}[/cyan]",
            str(n_scen),
            f"[{ret_style}]{p50_ret:+.2f}%[/{ret_style}]" if p50_ret is not None else "n/a",
            f"{p50_sh:.3f}" if p50_sh is not None else "n/a",
            f"{p50_mdd:.1%}" if p50_mdd is not None else "n/a",
            f"{ppos:.0%}" if ppos is not None else "n/a",
        )

    return Panel(t, title="[bold]📋 Storico Run Monte Carlo[/bold]", border_style="dim")


def build_running_panel(run_progress: str, run_log: list[str]) -> Panel:
    content = Text()
    content.append(f"⟳ {run_progress}\n\n", style="bold yellow")
    for line in run_log[-12:]:
        content.append(line + "\n", style="dim")
    return Panel(content, title="[yellow]Monte Carlo in esecuzione...[/yellow]", border_style="yellow")


# ─────────────────────────────────────────────────────────────────────────────
# Background runner
# ─────────────────────────────────────────────────────────────────────────────

class MCRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = False
        self.progress = ""
        self.log: list[str] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cmd = [sys.executable, str(BASE_DIR / "monte_carlo_backtest.py")]
        cmd += ["--scenarios", str(self.args.scenarios)]
        cmd += ["--days", str(self.args.days)]
        cmd += ["--scenario-type", self.args.scenario_type]
        if self.args.vol_multiplier:
            cmd += ["--vol-multiplier", str(self.args.vol_multiplier)]
        if self.args.drift is not None:
            cmd += ["--drift", str(self.args.drift)]

        self.progress = f"Avvio: {' '.join(cmd[2:])}"
        self.log.append(f"$ {' '.join(cmd[1:])}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR),
            )
            for line in proc.stdout:  # type: ignore
                line = line.rstrip()
                self.log.append(line)
                self.progress = line[:80] if line else self.progress
            proc.wait()
            self.progress = f"Completato (exit {proc.returncode})"
        except Exception as e:
            self.progress = f"Errore: {e}"
            self.log.append(str(e))
        finally:
            self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo Dashboard")
    parser.add_argument("--run",            action="store_true",    help="Lancia un nuovo run MC")
    parser.add_argument("--scenarios",      type=int, default=50,   help="Numero scenari (default: 50)")
    parser.add_argument("--days",           type=int, default=90,   help="Orizzonte giorni (default: 90)")
    parser.add_argument("--scenario-type",  default="base",         choices=["base","bull","bear","crash","lateral"])
    parser.add_argument("--vol-multiplier", type=float, default=None)
    parser.add_argument("--drift",          type=float, default=None)
    parser.add_argument("--interval",       type=float, default=2.0, help="Refresh interval secondi")
    args = parser.parse_args()

    runner: MCRunner | None = None
    if args.run:
        runner = MCRunner(args)
        runner.start()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with Live(refresh_per_second=1, screen=True) as live:
        while True:
            report = _load_latest_report()
            all_reports = _load_all_reports()
            running = runner.running if runner else False
            progress = runner.progress if runner else ""
            run_log  = runner.log if runner else []

            # Layout
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3),
            )
            layout["main"].split_column(
                Layout(name="top"),
                Layout(name="mid"),
                Layout(name="bottom"),
            )

            layout["header"].update(build_header(report, running, progress))

            if running and not report:
                layout["top"].update(build_running_panel(progress, run_log))
                layout["mid"].update(Panel("[dim]In attesa del primo report...[/dim]"))
                layout["bottom"].update(Panel(""))
            elif report:
                layout["top"].update(build_fan_chart(report))
                layout["mid"].split_row(
                    Layout(build_distribution_table(report), name="dist"),
                    Layout(build_probability_panel(report),  name="prob"),
                )
                layout["bottom"].split_row(
                    Layout(build_histogram(report),         name="hist"),
                    Layout(build_history_panel(all_reports), name="hist2"),
                )
            else:
                msg = Panel(
                    Align.center(Text(
                        "\nNessun report Monte Carlo trovato.\n\n"
                        "Lancia un run con:\n"
                        "  python monte_carlo_dashboard.py --run\n\n"
                        "Oppure in un altro terminale:\n"
                        "  python monte_carlo_backtest.py --scenarios 50 --days 90\n",
                        style="dim"
                    )),
                    title="[dim]Monte Carlo Dashboard[/dim]",
                )
                layout["top"].update(msg)
                layout["mid"].update(Panel(""))
                layout["bottom"].update(Panel(""))

            # Footer
            shortcuts = Text()
            shortcuts.append("  q[/q]uit    ", style="dim")
            shortcuts.append("  Tipi scenario: ", style="dim")
            for stype in ["base", "bull", "bear", "crash", "lateral"]:
                shortcuts.append(f"[{stype}] ", style="cyan dim")
            if running:
                shortcuts.append(f"  ⟳ Running: {progress[:60]}", style="yellow")
            layout["footer"].update(Panel(shortcuts, style="dim"))

            live.update(layout)

            # Stop se run finito e non stiamo più girando
            if runner and not runner.running and hasattr(runner, "_done_shown"):
                pass
            if runner and not runner.running:
                runner._done_shown = True  # type: ignore

            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Uscito.[/dim]")
