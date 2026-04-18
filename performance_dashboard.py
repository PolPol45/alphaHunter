"""
Performance Dashboard — Bloomberg-style analytics for the trading bot.
Run: python performance_dashboard.py [--port PORT]
"""

import argparse
import glob
import json
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

# ── Colour palette ─────────────────────────────────────────────────────────────
BG = "#0d1117"
CARD = "#161b22"
CARD_LIGHT = "#1c2230"
BORDER = "#30363d"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
TEXT = "#e6edf3"
SUBTEXT = "#8b949e"

def base_layout(**kwargs):
    """Return common dark layout kwargs, merging any caller overrides."""
    defaults = dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    defaults.update(kwargs)
    return defaults

# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_all_reports():
    """Return list of report dicts sorted by generated_at ascending."""
    pattern = str(REPORTS_DIR / "report_*.json")
    files = sorted(glob.glob(pattern))
    reports = []
    for fp in files:
        d = safe_load(fp)
        if d and "generated_at" in d:
            reports.append(d)
    reports.sort(key=lambda r: r["generated_at"])
    return reports


def load_ml_cross_sectional():
    pattern = str(REPORTS_DIR / "ml_cross_sectional_summary_*.json")
    files = sorted(glob.glob(pattern))
    results = []
    for fp in files:
        d = safe_load(fp)
        if d:
            # Attach filename timestamp as fallback
            d["_file"] = fp
            results.append(d)
    return results


def placeholder_fig(msg="Dati non ancora disponibili"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=SUBTEXT),
    )
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
    )
    return fig


def explanation_box(text: str) -> html.Div:
    return html.Div(
        html.P(text, style={"margin": 0, "color": SUBTEXT, "fontStyle": "italic", "fontSize": "13px", "lineHeight": "1.6"}),
        style={
            "borderLeft": f"3px solid {BLUE}",
            "background": CARD_LIGHT,
            "padding": "12px 16px",
            "marginTop": "12px",
            "borderRadius": "4px",
        },
    )


def section_title(title: str) -> html.H2:
    return html.H2(title, style={"color": TEXT, "fontSize": "16px", "fontWeight": "600",
                                  "marginBottom": "8px", "marginTop": "0",
                                  "borderBottom": f"1px solid {BORDER}", "paddingBottom": "8px"})


def card(children, style=None):
    base = {"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "8px",
            "padding": "20px", "marginBottom": "20px"}
    if style:
        base.update(style)
    return html.Div(children, style=base)


def metric_box(label, value, color=TEXT):
    return html.Div([
        html.Div(label, style={"color": SUBTEXT, "fontSize": "11px", "textTransform": "uppercase",
                                "letterSpacing": "0.05em", "marginBottom": "4px"}),
        html.Div(value, style={"color": color, "fontSize": "20px", "fontWeight": "700", "fontFamily": "monospace"}),
    ], style={"background": CARD_LIGHT, "border": f"1px solid {BORDER}", "borderRadius": "6px",
              "padding": "12px 16px", "minWidth": "130px", "flex": "1"})


# ── Section builders ───────────────────────────────────────────────────────────

def build_equity_section(reports):
    if not reports:
        fig = placeholder_fig()
        return card([section_title("1 — Equity Curve & Returns"), dcc.Graph(figure=fig),
                     explanation_box("La curva mostra l'andamento del valore del portfolio nel tempo. Una curva crescente indica che la strategia sta generando valore. Drawdown temporanei sono normali — quello che conta è il trend di lungo periodo.")])

    times, retail, inst, combined = [], [], [], []
    for r in reports:
        times.append(r["generated_at"])
        retail.append(r.get("combined", {}).get("retail_equity") or r.get("retail", {}).get("portfolio_summary", {}).get("total_equity", 0))
        inst.append(r.get("combined", {}).get("inst_equity") or r.get("institutional", {}).get("portfolio_summary", {}).get("total_equity", 0))
        combined.append(r.get("combined", {}).get("total_equity", 0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=combined, name="Combined", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=times, y=inst, name="Institutional", line=dict(color=PURPLE, width=1.5)))
    fig.add_trace(go.Scatter(x=times, y=retail, name="Retail", line=dict(color=GREEN, width=1.5)))
    fig.update_layout(
        **base_layout(),
        height=320,
        title=dict(text="Equity nel Tempo", font=dict(color=TEXT, size=13)),
        yaxis=dict(tickprefix="$", gridcolor=BORDER, zerolinecolor=BORDER),
    )

    return card([
        section_title("1 — Equity Curve & Returns"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        explanation_box(
            "La curva mostra l'andamento del valore del portfolio nel tempo. "
            "Una curva crescente indica che la strategia sta generando valore. "
            "Drawdown temporanei sono normali — quello che conta è il trend di lungo periodo."
        ),
    ])


def build_trade_section(reports):
    """Read live trades from portfolio JSON files — NOT from reports (which only have new_trades=[])."""
    # Load from live portfolio files — the authoritative source
    seen_ids: set = set()
    all_trades: list = []
    open_positions: dict = {}

    for fname, label in [("portfolio_retail.json", "retail"), ("portfolio_institutional.json", "institutional")]:
        port = safe_load(DATA_DIR / fname) or {}
        for t in port.get("trades", []):
            tid = t.get("id") or f"{t.get('symbol','?')}|{t.get('timestamp','?')}|{t.get('side','?')}"
            if tid not in seen_ids:
                seen_ids.add(tid)
                t["_portfolio"] = label
                all_trades.append(t)
        for sym, pos in (port.get("positions") or {}).items():
            if float(pos.get("quantity", 0)) > 0:
                open_positions[f"{label}:{sym}"] = {**pos, "symbol": sym}

    # Separate BUY / SELL
    buy_trades  = [t for t in all_trades if t.get("side","").upper() == "BUY"]
    sell_trades = [t for t in all_trades if t.get("side","").upper() in ("SELL","SELL_SHORT","CLOSE")]

    # Match SELL to BUY by symbol to compute realized P&L
    buy_map: dict = {}   # symbol → list of (fill_price, qty)
    for t in sorted(all_trades, key=lambda x: x.get("timestamp","")):
        sym = t.get("symbol","?")
        side = t.get("side","").upper()
        qty  = float(t.get("quantity", 0))
        fp   = float(t.get("fill_price", 0))
        if side == "BUY":
            buy_map.setdefault(sym, []).append((fp, qty))
        elif side in ("SELL","SELL_SHORT","CLOSE") and sym in buy_map and buy_map[sym]:
            entry_fp, _ = buy_map[sym].pop(0)
            t["_realized_pnl"] = (fp - entry_fp) * qty

    closed_pnls = [t.get("_realized_pnl", 0.0) for t in sell_trades if "_realized_pnl" in t]
    open_pnls   = [pos.get("unrealized_pnl", 0.0) for pos in open_positions.values()]

    # P&L chart — prefer realized closed trades, fallback to open unrealized
    if closed_pnls:
        labels = [f"{t.get('symbol','?')}\n{t.get('timestamp','?')[:10]}" for t in sell_trades if "_realized_pnl" in t]
        pnls   = closed_pnls
        title_str = "P&L Realizzato per Trade Chiuso"
        pnl_note  = "realized"
    else:
        labels = [pos["symbol"] + f"\n[{k.split(':')[0][:4]}]" for k, pos in open_positions.items()]
        pnls   = open_pnls
        title_str = "P&L Non Realizzato — Posizioni Aperte"
        pnl_note  = "unrealized"

    # Metrics
    wins   = [p for p in (closed_pnls if closed_pnls else open_pnls) if p > 0]
    losses = [p for p in (closed_pnls if closed_pnls else open_pnls) if p < 0]
    n      = len(closed_pnls if closed_pnls else open_pnls)
    win_rate      = len(wins) / max(n, 1) * 100
    avg_win        = sum(wins)   / max(len(wins),   1)
    avg_loss       = sum(losses) / max(len(losses), 1)
    profit_factor  = sum(wins) / abs(sum(losses)) if losses else float("inf")
    total_notional = sum(float(t.get("notional", 0)) for t in all_trades)
    total_comm     = sum(float(t.get("commission", 0)) for t in all_trades)

    pf_str   = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
    pf_color = GREEN if profit_factor >= 1.5 else (YELLOW if profit_factor >= 1.0 else RED)
    wr_color = GREEN if win_rate >= 50 else (YELLOW if win_rate >= 40 else RED)

    metrics_row = html.Div([
        metric_box("Win Rate",      f"{win_rate:.1f}%",       wr_color),
        metric_box("Avg Win $",     f"${avg_win:,.2f}",       GREEN),
        metric_box("Avg Loss $",    f"${avg_loss:,.2f}",      RED),
        metric_box("Profit Factor", pf_str,                   pf_color),
        metric_box("Total Trades",  str(len(all_trades)),     BLUE),
        metric_box("Commissioni",   f"${total_comm:,.2f}",    YELLOW),
        metric_box("Notional tot.", f"${total_notional:,.0f}", TEXT),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"})

    if pnls:
        colors = [GREEN if p >= 0 else RED for p in pnls]
        fig = go.Figure(go.Bar(
            x=labels, y=pnls,
            marker_color=colors,
            text=[f"${p:+,.0f}" for p in pnls],
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="%{x}<br>P&L: $%{y:+,.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=BORDER, line_width=1)
        fig.update_layout(
            **base_layout(),
            height=320,
            title=dict(text=title_str, font=dict(color=TEXT, size=13)),
            xaxis=dict(tickangle=-45, gridcolor=BORDER, tickfont=dict(size=9)),
            yaxis=dict(tickprefix="$", gridcolor=BORDER, zerolinecolor=BORDER),
        )
        chart = dcc.Graph(figure=fig, config={"displayModeBar": False})
    else:
        chart = dcc.Graph(figure=placeholder_fig("Nessun trade ancora registrato"),
                          config={"displayModeBar": False})

    source_note = f"Fonte: portfolio_retail + portfolio_institutional | {len(buy_trades)} BUY · {len(sell_trades)} SELL · {len(open_positions)} posizioni aperte"

    return card([
        section_title("2 — Trade Performance"),
        metrics_row,
        html.Div(source_note, style={"color": SUBTEXT, "fontSize": "11px", "marginBottom": "8px"}),
        chart,
        explanation_box(
            "Il Profit Factor è il rapporto tra guadagni totali e perdite totali. "
            "> 1.5 è considerato buono. Il Win Rate da solo non basta: una strategia con 40% win rate "
            "ma profit factor 2.0 è migliore di una con 60% win rate e profit factor 0.8. "
            "Le commissioni vengono sottratte ad ogni trade — tenerle basse è essenziale per la profittabilità."
        ),
    ])


def build_ml_learning_section():
    ml_reports = load_ml_cross_sectional()
    learned = safe_load(DATA_DIR / "learned_strategy_weights.json") or {}

    # Learning curve chart
    timestamps, val_sharpes, test_sharpes, oos_sharpes = [], [], [], []
    for r in ml_reports:
        ts = r.get("generated_at", "")
        folds = r.get("folds", [])
        if folds:
            fold = folds[0]
            val_sharpes.append(fold.get("val_sharpe", None))
            test_sharpes.append(fold.get("test_sharpe", None))
        else:
            val_sharpes.append(None)
            test_sharpes.append(None)
        oos_sharpes.append(r.get("oos", {}).get("sharpe", None))
        timestamps.append(ts or r.get("_file", "?")[-20:])

    if timestamps:
        fig_lc = go.Figure()
        if any(v is not None for v in val_sharpes):
            fig_lc.add_trace(go.Scatter(x=timestamps, y=val_sharpes, name="Val Sharpe", line=dict(color=GREEN, width=2)))
        if any(v is not None for v in test_sharpes):
            fig_lc.add_trace(go.Scatter(x=timestamps, y=test_sharpes, name="Test Sharpe", line=dict(color=YELLOW, width=2)))
        if any(v is not None for v in oos_sharpes):
            fig_lc.add_trace(go.Scatter(x=timestamps, y=oos_sharpes, name="OOS Sharpe", line=dict(color=PURPLE, width=2)))
        fig_lc.update_layout(
            **base_layout(),
            height=280,
            title=dict(text="Learning Curve del Modello ML", font=dict(color=TEXT, size=13)),
            yaxis=dict(title="Sharpe Ratio", gridcolor=BORDER),
        )
    else:
        fig_lc = placeholder_fig()

    # Walk-forward bar chart
    wf = learned.get("walk_forward", {})
    model_scores = wf.get("model_scores", {})
    if model_scores:
        # Shorten param combo labels
        labels, scores = [], []
        for i, (k, v) in enumerate(model_scores.items()):
            labels.append(f"Combo {i+1}")
            scores.append(v)
        colors_wf = [GREEN if s == max(scores) else BLUE for s in scores]
        fig_wf = go.Figure(go.Bar(x=labels, y=scores, marker_color=colors_wf,
                                   text=[f"{s:.3f}" for s in scores], textposition="outside"))
        fig_wf.update_layout(
            **base_layout(),
            height=240,
            title=dict(text="Walk-Forward: Score per Combinazione di Parametri", font=dict(color=TEXT, size=13)),
            yaxis=dict(gridcolor=BORDER),
        )
    else:
        fig_wf = placeholder_fig("Walk-forward scores non disponibili")

    return card([
        section_title("3 — ML Model Learning Curve"),
        dcc.Graph(figure=fig_lc, config={"displayModeBar": False}),
        dcc.Graph(figure=fig_wf, config={"displayModeBar": False}),
        explanation_box(
            "La learning curve mostra come il modello migliora nel tempo con più dati. "
            "Se val_score e train_score convergono, il modello sta generalizzando bene. "
            "Una grande divergenza indica overfitting."
        ),
    ])


def build_ml_signal_section():
    ml = safe_load(DATA_DIR / "ml_signals.json") or {}

    ranking = ml.get("latest_ranking", {})
    longs = ranking.get("top_decile_long", [])
    shorts = ranking.get("bottom_decile_short", [])

    if longs or shorts:
        symbols = [e["symbol"] for e in longs] + [e["symbol"] for e in shorts]
        returns_ = [e["predicted_excess_return"] for e in longs] + [e["predicted_excess_return"] for e in shorts]
        colors = [GREEN] * len(longs) + [RED] * len(shorts)
        labels = ["LONG"] * len(longs) + ["SHORT"] * len(shorts)

        fig_rank = go.Figure(go.Bar(
            y=symbols,
            x=returns_,
            orientation="h",
            marker_color=colors,
            text=[f"{l} {r:.5f}" for l, r in zip(labels, returns_)],
            textposition="outside",
        ))
        fig_rank.update_layout(
            **base_layout(),
            height=max(300, len(symbols) * 28),
            title=dict(text="Segnali ML — Top & Bottom Decile", font=dict(color=TEXT, size=13)),
            xaxis=dict(title="Predicted Excess Return", gridcolor=BORDER),
            yaxis=dict(autorange="reversed", gridcolor=BORDER),
        )
    else:
        fig_rank = placeholder_fig()

    # History: frequency of appearance in top decile
    history = ml.get("history", {})
    symbol_freq = {}
    for date_entry in history.values():
        for item in date_entry.get("top_decile_long", []):
            sym = item["symbol"]
            symbol_freq[sym] = symbol_freq.get(sym, 0) + 1

    if symbol_freq:
        sorted_sf = sorted(symbol_freq.items(), key=lambda x: -x[1])[:20]
        fig_freq = go.Figure(go.Bar(
            x=[s[0] for s in sorted_sf],
            y=[s[1] for s in sorted_sf],
            marker_color=PURPLE,
            text=[str(s[1]) for s in sorted_sf],
            textposition="outside",
        ))
        fig_freq.update_layout(
            **base_layout(),
            height=260,
            title=dict(text="Frequenza Apparizione nel Top Decile (storia)", font=dict(color=TEXT, size=13)),
            yaxis=dict(title="Numero di periodi", gridcolor=BORDER),
        )
        freq_chart = dcc.Graph(figure=fig_freq, config={"displayModeBar": False})
    else:
        freq_chart = html.Div()

    return card([
        section_title("4 — ML Signal Quality"),
        dcc.Graph(figure=fig_rank, config={"displayModeBar": False}),
        freq_chart,
        explanation_box(
            "I segnali ML rappresentano le previsioni del modello sul rendimento atteso nelle prossime settimane. "
            "I simboli nel top decile sono quelli con le previsioni migliori. "
            "Nota: tutti i segnali con predicted_excess_return identico indicano che il modello non ha ancora "
            "abbastanza dati per discriminare — è normale nelle prime settimane."
        ),
    ])


def build_adaptive_section():
    adaptive = safe_load(DATA_DIR / "adaptive_params.json") or {}
    learned = safe_load(DATA_DIR / "learned_strategy_weights.json") or {}

    # min_score_by_bucket bar chart
    bucket_scores = adaptive.get("min_score_by_bucket", {})
    strat_weights_adaptive = adaptive.get("strategy_weights", {})
    selected_model = learned.get("selected_model", {})
    strat_weights_learned = learned.get("strategy_weights", {})

    charts = []

    if bucket_scores:
        sorted_bs = sorted(bucket_scores.items(), key=lambda x: -x[1])
        fig_bs = go.Figure(go.Bar(
            x=[b[0] for b in sorted_bs],
            y=[b[1] for b in sorted_bs],
            marker_color=YELLOW,
            text=[f"{b[1]:.2f}" for b in sorted_bs],
            textposition="outside",
        ))
        fig_bs.update_layout(
            **base_layout(),
            height=260,
            title=dict(text="Soglia Minima di Score per Bucket", font=dict(color=TEXT, size=13)),
            yaxis=dict(range=[0, 1.1], gridcolor=BORDER),
        )
        charts.append(dcc.Graph(figure=fig_bs, config={"displayModeBar": False}))

    # Side by side pies
    pie_row_children = []

    if strat_weights_adaptive:
        fig_pie_a = go.Figure(go.Pie(
            labels=list(strat_weights_adaptive.keys()),
            values=list(strat_weights_adaptive.values()),
            hole=0.4,
            marker=dict(colors=[GREEN, YELLOW, BLUE, PURPLE, RED]),
        ))
        fig_pie_a.update_layout(
            **base_layout(),
            height=260,
            title=dict(text="Strategy Weights (Adaptive)", font=dict(color=TEXT, size=13)),
        )
        pie_row_children.append(html.Div(dcc.Graph(figure=fig_pie_a, config={"displayModeBar": False}),
                                          style={"flex": "1"}))

    if selected_model:
        fig_pie_m = go.Figure(go.Pie(
            labels=list(selected_model.keys()),
            values=list(selected_model.values()),
            hole=0.4,
            marker=dict(colors=[GREEN, YELLOW, BLUE, PURPLE, RED]),
        ))
        fig_pie_m.update_layout(
            **base_layout(),
            height=260,
            title=dict(text="Selected Model Weights (Walk-Forward)", font=dict(color=TEXT, size=13)),
        )
        pie_row_children.append(html.Div(dcc.Graph(figure=fig_pie_m, config={"displayModeBar": False}),
                                          style={"flex": "1"}))

    if strat_weights_learned:
        fig_pie_l = go.Figure(go.Pie(
            labels=list(strat_weights_learned.keys()),
            values=list(strat_weights_learned.values()),
            hole=0.4,
            marker=dict(colors=[GREEN, YELLOW, BLUE]),
        ))
        fig_pie_l.update_layout(
            **base_layout(),
            height=260,
            title=dict(text="Strategy Weights (Learned)", font=dict(color=TEXT, size=13)),
        )
        pie_row_children.append(html.Div(dcc.Graph(figure=fig_pie_l, config={"displayModeBar": False}),
                                          style={"flex": "1"}))

    # Metadata row
    cycle = adaptive.get("cycle_count", "?")
    last_upd = adaptive.get("last_updated", "?")
    sharpe = adaptive.get("metrics", {}).get("sharpe_recent", "?")
    max_dd = adaptive.get("metrics", {}).get("max_drawdown", "?")
    frozen = adaptive.get("metrics", {}).get("frozen", False)

    meta_row = html.Div([
        metric_box("Cicli completati", str(cycle), BLUE),
        metric_box("Sharpe recente", f"{sharpe:.2f}" if isinstance(sharpe, float) else str(sharpe),
                   GREEN if isinstance(sharpe, float) and sharpe > 0 else RED),
        metric_box("Max Drawdown", f"{float(max_dd)*100:.2f}%" if isinstance(max_dd, (int, float)) else str(max_dd), RED),
        metric_box("Frozen", "SI" if frozen else "NO", RED if frozen else GREEN),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"})

    return card([
        section_title("5 — Adaptive Parameters Evolution"),
        meta_row,
        *charts,
        html.Div(pie_row_children, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        explanation_box(
            "I parametri adattativi vengono aggiornati automaticamente in base alle performance recenti. "
            "Un peso alto su 'bull' significa che la strategia bull sta performando meglio delle altre. "
            "La soglia minima di score per bucket filtra i segnali di bassa qualità."
        ),
    ])


def build_risk_section(reports):
    if not reports:
        return card([
            section_title("6 — Risk Overview"),
            dcc.Graph(figure=placeholder_fig()),
            explanation_box(
                "Ogni posizione ha uno stop-loss automatico che limita la perdita massima. "
                "La zona di rischio è la distanza tra il prezzo attuale e lo stop-loss. "
                "Una posizione con stop-loss vicino (< 3%) è più a rischio di essere chiusa da un movimento temporaneo."
            ),
        ])

    latest = reports[-1]
    rows = []
    for pk in ("retail", "institutional"):
        ps = latest.get(pk, {}).get("portfolio_summary", {})
        total_equity = ps.get("total_equity", 1) or 1
        for sym, pos in (latest.get(pk, {}).get("open_positions", {}) or {}).items():
            cur = pos.get("current_price", 0) or 0
            sl = pos.get("stop_loss_price", 0) or 0
            tp = pos.get("take_profit_price", 0) or 0
            pv = pos.get("position_value_usdt", 0) or 0

            dist_sl = abs((cur - sl) / cur * 100) if cur and sl else 0
            dist_tp = abs((tp - cur) / cur * 100) if cur and tp and tp != 0 else 0
            pct_portfolio = pv / total_equity * 100 if total_equity else 0

            rows.append({
                "label": f"{pk[:4].upper()}:{sym}",
                "dist_sl": dist_sl,
                "dist_tp": dist_tp,
                "pct_portfolio": pct_portfolio,
                "cur": cur,
                "sl": sl,
                "tp": tp,
            })

    if not rows:
        return card([
            section_title("6 — Risk Overview"),
            dcc.Graph(figure=placeholder_fig("Nessuna posizione aperta")),
            explanation_box(
                "Ogni posizione ha uno stop-loss automatico che limita la perdita massima. "
                "La zona di rischio è la distanza tra il prezzo attuale e lo stop-loss. "
                "Una posizione con stop-loss vicino (< 3%) è più a rischio di essere chiusa da un movimento temporaneo."
            ),
        ])

    rows = rows[:25]  # cap for readability
    labels = [r["label"] for r in rows]

    # Stacked horizontal bar: safe zone, risk zone, upside
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Zona Rischio (% a Stop-Loss)",
        y=labels,
        x=[r["dist_sl"] for r in rows],
        orientation="h",
        marker_color=RED,
        text=[f"{r['dist_sl']:.1f}%" for r in rows],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Zona Upside (% a Take-Profit)",
        y=labels,
        x=[r["dist_tp"] for r in rows],
        orientation="h",
        marker_color=GREEN,
        text=[f"{r['dist_tp']:.1f}%" for r in rows],
        textposition="inside",
    ))
    fig.update_layout(
        **base_layout(),
        barmode="stack",
        height=max(300, len(rows) * 28),
        title=dict(text="Distanza Stop-Loss / Take-Profit per Posizione", font=dict(color=TEXT, size=13)),
        xaxis=dict(title="% dal prezzo corrente", gridcolor=BORDER),
    )

    # Position size chart
    fig2 = go.Figure(go.Bar(
        x=labels,
        y=[r["pct_portfolio"] for r in rows],
        marker_color=BLUE,
        text=[f"{r['pct_portfolio']:.1f}%" for r in rows],
        textposition="outside",
    ))
    fig2.update_layout(
        **base_layout(),
        height=280,
        title=dict(text="Peso Posizione come % del Portafoglio", font=dict(color=TEXT, size=13)),
        xaxis=dict(tickangle=-45, gridcolor=BORDER),
        yaxis=dict(title="% del portafoglio", gridcolor=BORDER),
    )

    return card([
        section_title("6 — Risk Overview"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dcc.Graph(figure=fig2, config={"displayModeBar": False}),
        explanation_box(
            "Ogni posizione ha uno stop-loss automatico che limita la perdita massima. "
            "La zona di rischio è la distanza tra il prezzo attuale e lo stop-loss. "
            "Una posizione con stop-loss vicino (< 3%) è più a rischio di essere chiusa da un movimento temporaneo."
        ),
    ])


# ── App assembly ───────────────────────────────────────────────────────────────

def build_layout():
    reports = load_all_reports()

    header = html.Div([
        html.Div([
            html.H1("Trading Bot — Performance Dashboard",
                    style={"color": TEXT, "fontSize": "20px", "fontWeight": "700", "margin": 0}),
            html.Span(
                f"Aggiornato: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
                f"{len(reports)} report caricati",
                style={"color": SUBTEXT, "fontSize": "12px"},
            ),
        ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
    ], style={
        "background": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px 20px",
        "marginBottom": "20px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
    })

    return html.Div([
        header,
        build_equity_section(reports),
        build_trade_section(reports),
        build_ml_learning_section(),
        build_ml_signal_section(),
        build_adaptive_section(),
        build_risk_section(reports),
    ], style={
        "background": BG,
        "minHeight": "100vh",
        "padding": "20px",
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
        "maxWidth": "1400px",
        "margin": "0 auto",
    })


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Performance Dashboard")
    parser.add_argument("--port", type=int, default=8053, help="Port to run dashboard on (default: 8053)")
    args = parser.parse_args()

    app = dash.Dash(
        __name__,
        title="Trading Bot Dashboard",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; background: """ + BG + """; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: """ + BG + """; }
        ::-webkit-scrollbar-thumb { background: """ + BORDER + """; border-radius: 3px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""

    app.layout = build_layout()

    url = f"http://localhost:{args.port}"
    print(f"Dashboard running at {url}")
    webbrowser.open(url)

    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
