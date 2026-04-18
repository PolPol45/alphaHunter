"""
backtest_dashboard.py
=====================
Dashboard interattiva Plotly per risultati Monte Carlo.

Uso:
    python backtest_dashboard.py                          # ultimo report
    python backtest_dashboard.py --report reports/monte_carlo_report_XYZ.json
    python backtest_dashboard.py --port 8051              # porta custom
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import webbrowser
from datetime import datetime

BASE_DIR = pathlib.Path(__file__).parent

REPORTS_DIR = BASE_DIR / "reports"


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────

def load_report(path: pathlib.Path | None) -> dict:
    if path is None:
        candidates = sorted(REPORTS_DIR.glob("monte_carlo_report_*.json"))
        if not candidates:
            sys.exit("Nessun report Monte Carlo trovato in reports/. Lancia prima monte_carlo_backtest.py")
        path = candidates[-1]
    print(f"  Report: {path.name}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Build dashboard
# ──────────────────────────────────────────────────────────────────────────────

def build_app(report: dict):
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        sys.exit("Installa dipendenze: pip install dash plotly")

    scenarios = report["scenario_results"]
    params    = report["parameters"]
    analysis  = report["analysis"]
    n         = len(scenarios)

    # ── helpers ──
    def vals(key):
        return [s[key] for s in scenarios if s.get(key) is not None]

    sharpes      = vals("sharpe")
    drawdowns    = [abs(x) for x in vals("max_drawdown")]
    returns      = vals("total_return_pct")
    win_rates    = vals("win_rate")
    calmar       = vals("calmar")
    trades_count = vals("trades")

    # Equity curves (sample max 30 for readability)
    step = max(1, n // 30)
    sampled = scenarios[::step]

    # ── COLORS ──
    BG      = "#0d1117"
    CARD    = "#161b22"
    BORDER  = "#30363d"
    GREEN   = "#3fb950"
    RED     = "#f85149"
    YELLOW  = "#d29922"
    BLUE    = "#58a6ff"
    PURPLE  = "#bc8cff"
    TEXT    = "#e6edf3"
    SUBTEXT = "#8b949e"

    # ── FIGURES ──

    # 1. Equity curves
    fig_equity = go.Figure()
    days = params.get("days", 90)
    x_axis = list(range(days + 1))

    for s in sampled:
        curve = s.get("equity_curve", [])
        if not curve:
            continue
        ret = s["total_return_pct"]
        color = f"rgba(63,185,80,0.15)" if ret >= 0 else f"rgba(248,81,73,0.15)"
        fig_equity.add_trace(go.Scatter(
            x=x_axis[:len(curve)], y=curve,
            mode="lines", line=dict(width=1, color=color),
            showlegend=False, hovertemplate="Giorno %{x}<br>Equity: %{y:,.0f}<extra></extra>",
        ))

    # Median curve
    if scenarios:
        max_len = max(len(s.get("equity_curve", [])) for s in scenarios)
        medians = []
        for i in range(max_len):
            vals_i = [s["equity_curve"][i] for s in scenarios if i < len(s.get("equity_curve", []))]
            medians.append(sorted(vals_i)[len(vals_i) // 2])
        fig_equity.add_trace(go.Scatter(
            x=x_axis[:len(medians)], y=medians,
            mode="lines", line=dict(width=2.5, color=BLUE),
            name="Mediana", hovertemplate="Giorno %{x}<br>Mediana: %{y:,.0f}<extra></extra>",
        ))

    fig_equity.update_layout(
        title="Equity Curve — tutti gli scenari", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font_color=TEXT, xaxis_title="Giorno", yaxis_title="Equity ($)",
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # 2. Return distribution
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=returns, nbinsx=40,
        marker_color=[GREEN if r >= 0 else RED for r in returns],
        marker_line_width=0,
        hovertemplate="Return: %{x:.1f}%<br>Scenari: %{y}<extra></extra>",
    ))
    fig_ret.add_vline(x=0, line_dash="dash", line_color=YELLOW, line_width=1.5)
    fig_ret.add_vline(
        x=analysis["total_return_pct"].get("p50", 0),
        line_dash="dot", line_color=BLUE, line_width=2,
        annotation_text=f"P50: {analysis['total_return_pct'].get('p50', 0):.1f}%",
        annotation_font_color=BLUE,
    )
    fig_ret.update_layout(
        title="Distribuzione Return %", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font_color=TEXT, xaxis_title="Return (%)", yaxis_title="Scenari",
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # 3. Sharpe vs Drawdown scatter
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=drawdowns, y=sharpes,
        mode="markers",
        marker=dict(
            size=7,
            color=returns,
            colorscale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
            colorbar=dict(title=dict(text="Return %", font=dict(color=TEXT)), tickfont=dict(color=TEXT)),
            showscale=True,
            opacity=0.8,
        ),
        hovertemplate="Drawdown: %{x:.1%}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig_scatter.add_hline(y=1.0, line_dash="dash", line_color=GREEN, line_width=1,
                          annotation_text="Sharpe=1", annotation_font_color=GREEN)
    fig_scatter.add_hline(y=0.0, line_dash="dash", line_color=YELLOW, line_width=1)
    fig_scatter.update_layout(
        title="Sharpe vs Max Drawdown", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font_color=TEXT, xaxis_title="Max Drawdown", yaxis_title="Sharpe Ratio",
        xaxis=dict(gridcolor=BORDER, tickformat=".0%"), yaxis=dict(gridcolor=BORDER),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # 4. Metrics box plots
    fig_box = make_subplots(rows=1, cols=3, subplot_titles=["Sharpe", "Max Drawdown", "Win Rate"])
    for col, (data, name, color) in enumerate([
        (sharpes,    "Sharpe",      BLUE),
        (drawdowns,  "Drawdown",    RED),
        (win_rates,  "Win Rate",    GREEN),
    ], 1):
        fig_box.add_trace(go.Box(
            y=data, name=name,
            marker_color=color, line_color=color,
            boxmean=True,
        ), row=1, col=col)
    fig_box.update_layout(
        title="Distribuzione Metriche", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font_color=TEXT, showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    for i in range(1, 4):
        fig_box.update_xaxes(gridcolor=BORDER, row=1, col=i)
        fig_box.update_yaxes(gridcolor=BORDER, row=1, col=i)

    # ── STAT CARDS ──
    prob = analysis.get("probabilities", {})
    p_ret = analysis["total_return_pct"]
    p_sh  = analysis["sharpe"]
    p_dd  = analysis["max_drawdown"]

    def stat_card(label, value, sub=None, color=TEXT):
        return html.Div([
            html.Div(label, style={"color": SUBTEXT, "fontSize": "12px", "marginBottom": "4px"}),
            html.Div(value, style={"color": color, "fontSize": "22px", "fontWeight": "700"}),
            html.Div(sub or "", style={"color": SUBTEXT, "fontSize": "11px", "marginTop": "2px"}),
        ], style={
            "background": CARD, "border": f"1px solid {BORDER}",
            "borderRadius": "8px", "padding": "16px 20px",
            "minWidth": "150px", "flex": "1",
        })

    med_ret = p_ret.get("p50", 0)
    med_sh  = p_sh.get("p50", 0)
    med_dd  = abs(p_dd.get("p50", 0))
    p_pos   = prob.get("positive_return_pct", 0)
    p_sharp = prob.get("sharpe_above_1_pct", 0)
    p_risk  = prob.get("drawdown_above_threshold_pct", 0)

    cards = html.Div([
        stat_card("Return mediano",  f"{med_ret:+.1f}%",
                  f"P5: {p_ret.get('p5',0):+.1f}% | P95: {p_ret.get('p95',0):+.1f}%",
                  GREEN if med_ret >= 0 else RED),
        stat_card("Sharpe mediano",  f"{med_sh:.2f}",
                  f"P5: {p_sh.get('p5',0):.2f} | P95: {p_sh.get('p95',0):.2f}",
                  GREEN if med_sh >= 1 else (YELLOW if med_sh >= 0 else RED)),
        stat_card("Max Drawdown med", f"{med_dd:.1%}",
                  f"Peggiore: {abs(p_dd.get('min',0)):.1%}",
                  RED if med_dd > 0.15 else YELLOW),
        stat_card("P(return > 0)",   f"{p_pos:.0f}%",  f"{n} scenari", GREEN if p_pos >= 60 else YELLOW),
        stat_card("P(Sharpe > 1)",   f"{p_sharp:.0f}%", None, GREEN if p_sharp >= 40 else YELLOW),
        stat_card("P(DD > soglia)",  f"{p_risk:.0f}%",  None, RED if p_risk >= 30 else YELLOW),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "16px 0"})

    # ── LAYOUT ──
    scenario_type = params.get("scenario_type", "?")
    generated     = report.get("generated_at", "")[:19].replace("T", " ")

    app = dash.Dash(__name__, title="Monte Carlo Backtest")
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Monte Carlo Backtest", style={"margin": 0, "fontSize": "24px", "color": TEXT}),
            html.Div([
                html.Span(f"Scenari: {n}", style={"marginRight": "16px"}),
                html.Span(f"Giorni: {params.get('days','?')}", style={"marginRight": "16px"}),
                html.Span(f"Type: {scenario_type.upper()}", style={"marginRight": "16px"}),
                html.Span(f"Generato: {generated}", style={"color": SUBTEXT}),
            ], style={"color": SUBTEXT, "fontSize": "13px", "marginTop": "4px"}),
        ], style={
            "background": CARD, "borderBottom": f"1px solid {BORDER}",
            "padding": "20px 32px",
        }),

        # Content
        html.Div([
            cards,
            html.Div([
                dcc.Graph(figure=fig_equity, style={"height": "380px"}),
            ], style={"marginBottom": "16px"}),
            html.Div([
                html.Div(dcc.Graph(figure=fig_ret,     style={"height": "320px"}), style={"flex": "1"}),
                html.Div(dcc.Graph(figure=fig_scatter, style={"height": "320px"}), style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
            html.Div([
                dcc.Graph(figure=fig_box, style={"height": "320px"}),
            ]),
        ], style={"padding": "16px 32px"}),
    ], style={"background": BG, "minHeight": "100vh", "fontFamily": "monospace"})

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dashboard interattiva Monte Carlo Backtest")
    parser.add_argument("--report", type=pathlib.Path, default=None, help="Path al report JSON (default: ultimo)")
    parser.add_argument("--port",   type=int, default=8051, help="Porta HTTP (default: 8051)")
    parser.add_argument("--no-browser", action="store_true", help="Non aprire browser automaticamente")
    args = parser.parse_args()

    print("\n Monte Carlo Backtest Dashboard")
    print("=" * 40)

    report = load_report(args.report)
    app    = build_app(report)

    url = f"http://localhost:{args.port}"
    print(f"  URL: {url}")
    print(f"  Ctrl+C per fermare\n")

    if not args.no_browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
