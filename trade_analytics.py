"""
trade_analytics.py
==================
Dashboard interattiva Plotly — analisi trade live del bot.

Mostra:
  - Portfolio summary (equity, cash, P&L, drawdown)
  - Posizioni aperte con P&L, stop-loss, take-profit
  - Trade history con decisione, sizing, strategy bucket
  - Esposizione per asset / strategia
  - Variance / risk breakdown

Uso:
    python trade_analytics.py
    python trade_analytics.py --port 8052
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import webbrowser
import threading
from datetime import datetime, timezone

BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"

# ── palette ──
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
BLUE    = "#58a6ff"
PURPLE  = "#bc8cff"
ORANGE  = "#e3b341"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_portfolio(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_signals() -> dict:
    for fname in ["validated_signals.json", "signals.json"]:
        p = DATA_DIR / fname
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def load_all() -> dict:
    retail = load_portfolio("portfolio_retail.json")
    inst   = load_portfolio("portfolio_institutional.json")
    alpha  = load_portfolio("portfolio_alpha.json")
    sigs   = load_signals()

    all_trades    = retail.get("trades", []) + inst.get("trades", []) + alpha.get("trades", [])
    all_positions = {}
    for port, label in [(retail, "retail"), (inst, "institutional"), (alpha, "alpha")]:
        for sym, pos in port.get("positions", {}).items():
            key = f"{sym} [{label}]"
            all_positions[key] = {**pos, "symbol": sym, "portfolio": label}

    total_equity    = sum(p.get("total_equity", 0) for p in [retail, inst, alpha] if p)
    total_cash      = sum(p.get("cash", 0) for p in [retail, inst, alpha] if p)
    total_realized  = sum(p.get("realized_pnl", 0) for p in [retail, inst, alpha] if p)
    total_unrealized= sum(p.get("unrealized_pnl", 0) for p in [retail, inst, alpha] if p)
    total_initial   = sum(p.get("initial_capital", p.get("equity_basis", 0)) for p in [retail, inst, alpha] if p)
    max_dd          = max(abs(p.get("drawdown_pct", 0)) for p in [retail, inst, alpha] if p) if any([retail, inst, alpha]) else 0

    return {
        "retail": retail, "institutional": inst, "alpha": alpha,
        "trades": all_trades, "positions": all_positions, "signals": sigs,
        "summary": {
            "total_equity": total_equity,
            "total_cash": total_cash,
            "total_realized": total_realized,
            "total_unrealized": total_unrealized,
            "total_initial": total_initial,
            "max_dd": max_dd,
            "n_positions": len(all_positions),
            "n_trades": len(all_trades),
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build app
# ──────────────────────────────────────────────────────────────────────────────

def build_app(data: dict):
    try:
        import dash
        from dash import dcc, html, dash_table
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        sys.exit("Installa: pip install dash plotly")

    trades    = data["trades"]
    positions = data["positions"]
    summary   = data["summary"]
    retail    = data["retail"]
    inst      = data["institutional"]

    # ── helpers ──────────────────────────────────────────────────────────────

    def card(label, value, sub=None, color=TEXT, border_color=BORDER):
        return html.Div([
            html.Div(label, style={"color": SUBTEXT, "fontSize": "11px", "marginBottom": "3px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(value, style={"color": color, "fontSize": "20px", "fontWeight": "700"}),
            html.Div(sub or "", style={"color": SUBTEXT, "fontSize": "11px", "marginTop": "3px"}),
        ], style={
            "background": CARD, "border": f"1px solid {border_color}",
            "borderRadius": "8px", "padding": "14px 18px", "flex": "1", "minWidth": "140px",
        })

    def section(title, content):
        return html.Div([
            html.Div(title, style={"color": SUBTEXT, "fontSize": "12px", "textTransform": "uppercase",
                                   "letterSpacing": "1px", "marginBottom": "10px", "paddingBottom": "6px",
                                   "borderBottom": f"1px solid {BORDER}"}),
            content,
        ], style={"marginBottom": "24px"})

    # ── SUMMARY CARDS ────────────────────────────────────────────────────────
    s = summary
    total_pnl = s["total_realized"] + s["total_unrealized"]
    pnl_pct   = (total_pnl / s["total_initial"] * 100) if s["total_initial"] > 0 else 0
    pnl_color = GREEN if total_pnl >= 0 else RED
    dd_color  = RED if s["max_dd"] > 0.10 else (YELLOW if s["max_dd"] > 0.05 else GREEN)

    summary_cards = html.Div([
        card("Total Equity",    f"${s['total_equity']:,.0f}",
             f"Iniziale: ${s['total_initial']:,.0f}"),
        card("Cash disponibile",f"${s['total_cash']:,.0f}",
             f"{s['total_cash']/s['total_equity']*100:.1f}% dell'equity" if s['total_equity'] > 0 else ""),
        card("P&L Totale",      f"${total_pnl:+,.0f}",
             f"Realized: ${s['total_realized']:+,.0f} | Unrealized: ${s['total_unrealized']:+,.0f}",
             pnl_color),
        card("Return",          f"{pnl_pct:+.2f}%", None, pnl_color),
        card("Max Drawdown",    f"{s['max_dd']:.1%}", None, dd_color),
        card("Posizioni aperte",str(s["n_positions"]), f"{s['n_trades']} trade totali"),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "20px"})

    # ── POSIZIONI APERTE ─────────────────────────────────────────────────────
    pos_rows = []
    for key, pos in positions.items():
        sym   = pos["symbol"]
        pnl   = pos.get("unrealized_pnl", 0)
        pct   = pos.get("unrealized_pnl_pct", 0)
        entry = pos.get("avg_entry_price", 0)
        curr  = pos.get("current_price", entry)
        sl    = pos.get("stop_loss_price", 0)
        tp    = pos.get("take_profit_price", 0)
        val   = pos.get("position_value_usdt", 0)
        atr   = pos.get("entry_atr", None)
        sl_dist = ((curr - sl) / curr * 100) if curr > 0 and sl > 0 else None
        tp_dist = ((tp - curr) / curr * 100) if curr > 0 and tp > 0 else None
        pos_rows.append({
            "Portfolio": pos.get("portfolio", "?").capitalize(),
            "Symbol":    sym,
            "Strategy":  pos.get("strategy_bucket", pos.get("agent_source", "?")),
            "Qty":       f"{pos.get('quantity', 0):.4f}",
            "Entry":     f"${entry:,.4f}",
            "Current":   f"${curr:,.4f}",
            "Value":     f"${val:,.0f}",
            "P&L $":     f"${pnl:+,.2f}",
            "P&L %":     f"{pct*100:+.2f}%",
            "Stop-Loss": f"${sl:,.4f}" if sl else "—",
            "SL dist":   f"{sl_dist:.1f}%" if sl_dist is not None else "—",
            "Take-Profit": f"${tp:,.4f}" if tp else "—",
            "TP dist":   f"{tp_dist:.1f}%" if tp_dist is not None else "—",
            "ATR entry": f"{atr:.4f}" if atr else "—",
        })

    pos_table = dash_table.DataTable(
        data=pos_rows,
        columns=[{"name": c, "id": c} for c in (pos_rows[0].keys() if pos_rows else [])],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": BG, "color": SUBTEXT, "fontSize": "11px",
                       "fontWeight": "600", "border": f"1px solid {BORDER}", "textTransform": "uppercase"},
        style_cell={"backgroundColor": CARD, "color": TEXT, "fontSize": "12px",
                     "border": f"1px solid {BORDER}", "padding": "8px 12px", "fontFamily": "monospace"},
        style_data_conditional=[
            {"if": {"filter_query": '{P&L $} contains "+"'}, "color": GREEN},
            {"if": {"filter_query": '{P&L $} contains "-"'}, "color": RED},
        ],
    ) if pos_rows else html.Div("Nessuna posizione aperta", style={"color": SUBTEXT, "padding": "12px"})

    # ── TRADE HISTORY TABLE ──────────────────────────────────────────────────
    trade_rows = []
    for t in sorted(trades, key=lambda x: x.get("timestamp",""), reverse=True):
        ts = t.get("timestamp","")[:19].replace("T"," ")
        notional = t.get("notional", 0)
        comm     = t.get("commission", 0)
        sl       = t.get("stop_loss_price")
        trade_rows.append({
            "Time":      ts,
            "Portfolio": t.get("mode","?").capitalize(),
            "Symbol":    t.get("symbol","?"),
            "Side":      t.get("side","?"),
            "Qty":       f"{t.get('quantity',0):.4f}",
            "Fill $":    f"${t.get('fill_price',0):,.4f}",
            "Notional":  f"${notional:,.0f}",
            "Commission":f"${comm:.2f}",
            "Stop-Loss": f"${sl:,.4f}" if sl else "—",
            "Strategy":  t.get("strategy_bucket", t.get("agent_source","?")),
            "Reason":    t.get("reason","?"),
            "Source":    t.get("source","?"),
        })

    trade_table = dash_table.DataTable(
        data=trade_rows,
        columns=[{"name": c, "id": c} for c in (trade_rows[0].keys() if trade_rows else [])],
        page_size=20,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": BG, "color": SUBTEXT, "fontSize": "11px",
                       "fontWeight": "600", "border": f"1px solid {BORDER}", "textTransform": "uppercase"},
        style_cell={"backgroundColor": CARD, "color": TEXT, "fontSize": "12px",
                     "border": f"1px solid {BORDER}", "padding": "8px 12px", "fontFamily": "monospace"},
        style_data_conditional=[
            {"if": {"filter_query": '{Side} = "BUY"'},  "color": GREEN},
            {"if": {"filter_query": '{Side} = "SELL"'}, "color": RED},
        ],
    ) if trade_rows else html.Div("Nessun trade", style={"color": SUBTEXT, "padding": "12px"})

    # ── CHARTS ───────────────────────────────────────────────────────────────

    # 1. Esposizione per strategia (pie)
    bucket_val: dict[str, float] = {}
    for pos in positions.values():
        b   = pos.get("strategy_bucket", pos.get("agent_source", "unknown"))
        val = pos.get("position_value_usdt", 0)
        bucket_val[b] = bucket_val.get(b, 0) + val

    fig_pie = go.Figure(go.Pie(
        labels=list(bucket_val.keys()),
        values=list(bucket_val.values()),
        hole=0.45,
        marker=dict(colors=[BLUE, GREEN, PURPLE, ORANGE, RED, YELLOW]),
        textfont=dict(color=TEXT),
        hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig_pie.update_layout(
        title="Esposizione per strategia", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD, font_color=TEXT,
        legend=dict(font=dict(color=TEXT)),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # 2. P&L per posizione (bar)
    pos_syms = [p["symbol"] + f"\n[{p['portfolio'][:4]}]" for p in positions.values()]
    pos_pnls = [p.get("unrealized_pnl", 0) for p in positions.values()]
    fig_pnl = go.Figure(go.Bar(
        x=pos_syms, y=pos_pnls,
        marker_color=[GREEN if v >= 0 else RED for v in pos_pnls],
        hovertemplate="%{x}<br>P&L: $%{y:+,.2f}<extra></extra>",
    ))
    fig_pnl.update_layout(
        title="Unrealized P&L per posizione", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD, font_color=TEXT,
        xaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=BORDER, title="P&L ($)"),
        margin=dict(l=50, r=20, t=50, b=60),
    )

    # 3. Sizing analysis — notional per trade
    if trades:
        trade_syms      = [t.get("symbol","?") for t in trades]
        trade_notionals = [t.get("notional", 0) for t in trades]
        trade_modes     = [t.get("mode","?") for t in trades]
        colors_sizing   = [BLUE if m == "retail" else (PURPLE if m == "institutional" else ORANGE) for m in trade_modes]

        fig_sizing = go.Figure(go.Bar(
            x=trade_syms, y=trade_notionals,
            marker_color=colors_sizing,
            hovertemplate="%{x}<br>Notional: $%{y:,.0f}<extra></extra>",
            text=[m[:4] for m in trade_modes],
            textposition="outside",
            textfont=dict(size=9, color=SUBTEXT),
        ))
        fig_sizing.update_layout(
            title="Trade sizing — notional per operazione", title_font_color=TEXT,
            paper_bgcolor=CARD, plot_bgcolor=CARD, font_color=TEXT,
            xaxis=dict(gridcolor=BORDER, tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=BORDER, title="Notional ($)"),
            margin=dict(l=50, r=20, t=50, b=80),
        )
    else:
        fig_sizing = go.Figure()
        fig_sizing.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD)

    # 4. Risk heatmap — SL distance vs position size
    sl_dists, pos_vals, syms_h = [], [], []
    for pos in positions.values():
        curr = pos.get("current_price", 0)
        sl   = pos.get("stop_loss_price", 0)
        val  = pos.get("position_value_usdt", 0)
        if curr > 0 and sl > 0 and val > 0:
            sl_dists.append((curr - sl) / curr * 100)
            pos_vals.append(val)
            syms_h.append(pos["symbol"])

    fig_risk = go.Figure(go.Scatter(
        x=sl_dists, y=pos_vals,
        mode="markers+text",
        text=syms_h, textposition="top center",
        textfont=dict(size=10, color=TEXT),
        marker=dict(
            size=[max(8, v / 500) for v in pos_vals],
            color=sl_dists,
            colorscale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
            showscale=True,
            colorbar=dict(title=dict(text="SL dist %", font=dict(color=TEXT)), tickfont=dict(color=TEXT)),
        ),
        hovertemplate="%{text}<br>SL dist: %{x:.1f}%<br>Value: $%{y:,.0f}<extra></extra>",
    )) if sl_dists else go.Figure()
    fig_risk.add_vline(x=10, line_dash="dash", line_color=YELLOW, line_width=1,
                       annotation_text="SL 10%", annotation_font_color=YELLOW) if sl_dists else None
    fig_risk.update_layout(
        title="Risk map — distanza stop-loss vs size posizione", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD, font_color=TEXT,
        xaxis=dict(gridcolor=BORDER, title="Distanza dal Stop-Loss (%)"),
        yaxis=dict(gridcolor=BORDER, title="Valore posizione ($)"),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # 5. Portfolio breakdown per account
    port_names = ["Retail", "Institutional", "Alpha"]
    port_equities = [
        data["retail"].get("total_equity", 0),
        data["institutional"].get("total_equity", 0),
        data["alpha"].get("total_equity", 0),
    ]
    port_cash = [
        data["retail"].get("cash", 0),
        data["institutional"].get("cash", 0),
        data["alpha"].get("cash", 0),
    ]
    port_invested = [e - c for e, c in zip(port_equities, port_cash)]

    fig_port = go.Figure()
    fig_port.add_trace(go.Bar(name="Investito", x=port_names, y=port_invested,
                               marker_color=BLUE, hovertemplate="%{x}<br>Investito: $%{y:,.0f}<extra></extra>"))
    fig_port.add_trace(go.Bar(name="Cash",      x=port_names, y=port_cash,
                               marker_color=GREEN, hovertemplate="%{x}<br>Cash: $%{y:,.0f}<extra></extra>"))
    fig_port.update_layout(
        barmode="stack", title="Portfolio breakdown per account", title_font_color=TEXT,
        paper_bgcolor=CARD, plot_bgcolor=CARD, font_color=TEXT,
        legend=dict(font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, title="$"),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # ── PER-PORTFOLIO STATS ───────────────────────────────────────────────────
    def port_cards(port: dict, label: str):
        eq   = port.get("total_equity", 0)
        cash = port.get("cash", 0)
        rpnl = port.get("realized_pnl", 0)
        upnl = port.get("unrealized_pnl", 0)
        dd   = abs(port.get("drawdown_pct", 0))
        init = port.get("initial_capital", port.get("equity_basis", eq))
        ret  = (eq - init) / init * 100 if init > 0 else 0
        n_pos = len([p for p in port.get("positions", {}).values() if p.get("quantity", 0) > 0])
        return html.Div([
            html.Div(label, style={"color": TEXT, "fontWeight": "700", "marginBottom": "10px", "fontSize": "14px"}),
            html.Div([
                card("Equity",     f"${eq:,.0f}",   f"Iniziale: ${init:,.0f}"),
                card("Cash",       f"${cash:,.0f}",  f"{cash/eq*100:.0f}%" if eq > 0 else ""),
                card("Return",     f"{ret:+.2f}%",   None, GREEN if ret >= 0 else RED),
                card("Realized",   f"${rpnl:+,.0f}", None, GREEN if rpnl >= 0 else RED),
                card("Unrealized", f"${upnl:+,.0f}", None, GREEN if upnl >= 0 else RED),
                card("Drawdown",   f"{dd:.1%}",      f"{n_pos} posizioni", RED if dd > 0.10 else YELLOW if dd > 0.05 else GREEN),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        ], style={"background": BG, "border": f"1px solid {BORDER}", "borderRadius": "8px",
                  "padding": "16px", "marginBottom": "12px"})

    # ── LAYOUT ───────────────────────────────────────────────────────────────
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    app = dash.Dash(__name__, title="Trade Analytics")
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Trade Analytics", style={"margin": 0, "fontSize": "22px", "color": TEXT}),
            html.Div(f"Live data — {now_str}", style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "4px"}),
        ], style={"background": CARD, "borderBottom": f"1px solid {BORDER}", "padding": "18px 32px"}),

        html.Div([
            # Summary
            section("Portfolio Summary", summary_cards),

            # Per-account breakdown
            section("Per Account", html.Div([
                port_cards(data["retail"], "Retail"),
                port_cards(data["institutional"], "Institutional"),
                port_cards(data["alpha"], "Alpha Hunter"),
            ])),

            # Charts row 1
            section("Allocazione & P&L", html.Div([
                html.Div(dcc.Graph(figure=fig_pie,  style={"height": "320px"}), style={"flex": "1"}),
                html.Div(dcc.Graph(figure=fig_pnl,  style={"height": "320px"}), style={"flex": "2"}),
            ], style={"display": "flex", "gap": "16px"})),

            # Charts row 2
            section("Sizing & Risk", html.Div([
                html.Div(dcc.Graph(figure=fig_sizing, style={"height": "320px"}), style={"flex": "1"}),
                html.Div(dcc.Graph(figure=fig_risk,   style={"height": "320px"}), style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px"})),

            # Portfolio allocation
            section("Cash vs Investito", dcc.Graph(figure=fig_port, style={"height": "280px"})),

            # Posizioni aperte
            section("Posizioni Aperte", pos_table),

            # Trade history
            section("Trade History", trade_table),

        ], style={"padding": "20px 32px"}),
    ], style={"background": BG, "minHeight": "100vh", "fontFamily": "monospace"})

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int,  default=8052)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("\n Trade Analytics Dashboard")
    print("=" * 40)

    data = load_all()
    s    = data["summary"]
    print(f"  Posizioni: {s['n_positions']} | Trade: {s['n_trades']} | Equity: ${s['total_equity']:,.0f}")

    app = build_app(data)
    url = f"http://localhost:{args.port}"
    print(f"  URL: {url}")
    print(f"  Ctrl+C per fermare\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
