"""
backtest_performance.py
=======================
Dashboard Plotly/Dash dedicata ai risultati di backtest storico.
Legge tutti i reports/backtest_report_*.json esistenti.

Uso:
    python backtest_performance.py
    python backtest_performance.py --port 8054
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import threading
import webbrowser
from datetime import datetime, timezone

BASE_DIR    = pathlib.Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
CARD_L  = "#1c2230"
BORDER  = "#30363d"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
BLUE    = "#58a6ff"
PURPLE  = "#bc8cff"
ORANGE  = "#e3b341"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

BUCKET_COLORS = {"bull": GREEN, "bear": RED, "crypto": BLUE, "unknown": SUBTEXT}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_reports() -> list[dict]:
    files = sorted(glob.glob(str(REPORTS_DIR / "backtest_report_*.json")))
    out = []
    for fp in files:
        try:
            d = json.loads(pathlib.Path(fp).read_text(encoding="utf-8"))
            if d.get("report_type") == "backtest":
                d["_file"] = pathlib.Path(fp).name
                out.append(d)
        except Exception:
            pass
    return out


# ── UI helpers ────────────────────────────────────────────────────────────────

def _layout(**kw):
    base = dict(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
        margin=dict(l=55, r=20, t=45, b=45),
    )
    base.update(kw)
    return base


def _explain(text: str):
    from dash import html
    return html.Div(
        html.P(text, style={"margin": 0, "color": SUBTEXT, "fontStyle": "italic",
                             "fontSize": "13px", "lineHeight": "1.7"}),
        style={
            "borderLeft": f"3px solid {BLUE}", "background": CARD_L,
            "padding": "12px 16px", "marginTop": "14px", "borderRadius": "4px",
        },
    )


def _section(title: str, children, style=None):
    from dash import html
    base = {"background": CARD, "border": f"1px solid {BORDER}",
            "borderRadius": "8px", "padding": "20px", "marginBottom": "20px"}
    if style:
        base.update(style)
    return html.Div([
        html.H2(title, style={"color": TEXT, "fontSize": "15px", "fontWeight": "700",
                               "marginTop": 0, "marginBottom": "14px",
                               "borderBottom": f"1px solid {BORDER}", "paddingBottom": "8px"}),
        *([children] if not isinstance(children, list) else children),
    ], style=base)


def _metric(label, value, sub=None, color=TEXT):
    from dash import html
    return html.Div([
        html.Div(label, style={"color": SUBTEXT, "fontSize": "10px", "textTransform": "uppercase",
                                "letterSpacing": "0.6px", "marginBottom": "4px"}),
        html.Div(value, style={"color": color, "fontSize": "22px", "fontWeight": "700"}),
        html.Div(sub or "", style={"color": SUBTEXT, "fontSize": "11px", "marginTop": "3px"}),
    ], style={"background": CARD_L, "border": f"1px solid {BORDER}", "borderRadius": "6px",
              "padding": "14px 16px", "flex": "1", "minWidth": "130px"})


def _placeholder(msg="Dati non disponibili", h=280):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=14, color=SUBTEXT))
    fig.update_layout(**_layout(), xaxis_visible=False, yaxis_visible=False,
                      height=h, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_header(reports):
    from dash import html
    if not reports:
        return html.Div("Nessun report backtest trovato in reports/",
                         style={"color": RED, "padding": "20px"})
    items = []
    for r in reports:
        w  = r.get("window", {})
        s  = r.get("summary", {})
        m  = r.get("metrics", {})
        ret = (s.get("final_equity", 0) / s.get("start_equity", 1) - 1) * 100
        items.append(html.Div([
            html.Div(r["_file"].replace("backtest_report_","").replace(".json",""),
                     style={"color": BLUE, "fontWeight": "700", "fontSize": "13px"}),
            html.Div(f"{w.get('start_date','')} → {w.get('end_date','')}",
                     style={"color": SUBTEXT, "fontSize": "11px"}),
            html.Div([
                html.Span(f"Return: ", style={"color": SUBTEXT}),
                html.Span(f"{ret:+.1f}%", style={"color": GREEN if ret >= 0 else RED, "fontWeight": "700"}),
                html.Span(f"  |  Sharpe: ", style={"color": SUBTEXT}),
                html.Span(f"{m.get('sharpe',0):.2f}", style={"color": BLUE}),
                html.Span(f"  |  Trades: ", style={"color": SUBTEXT}),
                html.Span(str(s.get("trades",0)), style={"color": TEXT}),
            ], style={"fontSize": "12px", "marginTop": "4px"}),
        ], style={"background": CARD_L, "border": f"1px solid {BORDER}", "borderRadius": "6px",
                  "padding": "12px 16px", "flex": "1", "minWidth": "220px"}))

    return html.Div([
        html.Div("Backtest Reports disponibili",
                 style={"color": SUBTEXT, "fontSize": "11px", "textTransform": "uppercase",
                         "letterSpacing": "0.6px", "marginBottom": "10px"}),
        html.Div(items, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ], style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "8px",
              "padding": "16px 20px", "marginBottom": "20px"})


def sec_equity(reports):
    import plotly.graph_objects as go
    from dash import dcc

    if not reports:
        return _section("1 — Equity Curve", _explain("Nessun dato disponibile."))

    fig = go.Figure()
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    for i, r in enumerate(reports):
        ec  = r.get("equity_curve", [])
        if not ec:
            continue
        w   = r.get("window", {})
        lbl = f"{w.get('start_date','')} → {w.get('end_date','')}"
        fig.add_trace(go.Scatter(
            y=ec, x=list(range(len(ec))),
            name=lbl, line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{lbl}<br>Giorno %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>",
        ))

    # Drawdown overlay for first report
    r0 = reports[0]
    ec0 = r0.get("equity_curve", [])
    if ec0:
        peak = ec0[0]
        dd_series = []
        for v in ec0:
            peak = max(peak, v)
            dd_series.append((v - peak) / peak * 100)
        fig.add_trace(go.Scatter(
            y=dd_series, x=list(range(len(dd_series))),
            name="Drawdown %", yaxis="y2",
            line=dict(color=RED, width=1, dash="dot"),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.07)",
            hovertemplate="Giorno %{x}<br>DD: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_layout(height=360),
        yaxis=dict(title="Equity ($)", tickprefix="$", gridcolor=BORDER),
        yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right",
                    ticksuffix="%", gridcolor=BORDER, showgrid=False),
        xaxis=dict(title="Giorni simulati", gridcolor=BORDER),
    )

    return _section("1 — Equity Curve & Drawdown", [
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        _explain(
            "La curva mostra come il capitale è cresciuto (o diminuito) giorno per giorno durante il backtest. "
            "La linea rossa tratteggiata mostra il drawdown percentuale rispetto al massimo precedente. "
            "Un drawdown profondo indica un periodo di perdite — è normale, ma deve essere limitato. "
            "Il Calmar Ratio misura il return annuo diviso il max drawdown: > 1.0 è eccellente."
        ),
    ])


def sec_metrics(reports):
    from dash import html, dcc
    import plotly.graph_objects as go

    if not reports:
        return _section("2 — Metriche di Performance", _explain("Nessun dato."))

    blocks = []
    for r in reports:
        m   = r.get("metrics", {})
        s   = r.get("summary", {})
        w   = r.get("window", {})
        ret = (s.get("final_equity", 0) / max(s.get("start_equity", 1), 1) - 1) * 100
        dd  = abs(m.get("max_drawdown", 0))
        sh  = m.get("sharpe", 0)
        so  = m.get("sortino", 0)
        cal = m.get("calmar", 0)
        wr  = m.get("win_rate", 0) * 100
        to  = m.get("turnover", 0)
        trades = s.get("trades", 0)
        final  = s.get("final_equity", 0)
        start  = s.get("start_equity", 0)

        label = f"{w.get('start_date','')} → {w.get('end_date','')}"
        blocks.append(html.Div([
            html.Div(label, style={"color": BLUE, "fontSize": "12px",
                                    "fontWeight": "700", "marginBottom": "10px"}),
            html.Div([
                _metric("Return",      f"{ret:+.2f}%",     None, GREEN if ret >= 0 else RED),
                _metric("Sharpe",      f"{sh:.3f}",        "obiettivo > 1.0",
                         GREEN if sh >= 1 else (YELLOW if sh >= 0.5 else RED)),
                _metric("Sortino",     f"{so:.3f}",        "obiettivo > 1.5",
                         GREEN if so >= 1.5 else (YELLOW if so >= 0.8 else RED)),
                _metric("Calmar",      f"{cal:.3f}",       "obiettivo > 1.0",
                         GREEN if cal >= 1 else (YELLOW if cal >= 0.5 else RED)),
                _metric("Max DD",      f"{dd:.1%}",        "obiettivo < 15%",
                         GREEN if dd < 0.10 else (YELLOW if dd < 0.15 else RED)),
                _metric("Win Rate",    f"{wr:.1f}%",       "obiettivo > 50%",
                         GREEN if wr >= 50 else (YELLOW if wr >= 40 else RED)),
                _metric("Turnover",    f"{to:.1f}x",       "rotazione capitale"),
                _metric("Trades",      str(trades),        f"${final-start:+,.0f} P&L"),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        ], style={"background": BG, "border": f"1px solid {BORDER}", "borderRadius": "6px",
                  "padding": "14px", "marginBottom": "12px"}))

    # Radar chart per confronto se > 1 report
    if len(reports) >= 2:
        cats   = ["Sharpe", "Sortino", "Win Rate", "Calmar", "DD score"]
        traces = []
        for i, r in enumerate(reports):
            m  = r.get("metrics", {})
            w  = r.get("window", {})
            dd = abs(m.get("max_drawdown", 0))
            vals = [
                min(m.get("sharpe", 0) / 2, 1),
                min(m.get("sortino", 0) / 3, 1),
                m.get("win_rate", 0),
                min(m.get("calmar", 0) / 5, 1),
                max(0, 1 - dd / 0.3),
            ]
            lbl = f"{w.get('start_date','')} → {w.get('end_date','')}"
            traces.append(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=lbl,
                line=dict(color=[BLUE, GREEN, PURPLE][i % 3]),
                fillcolor=f"rgba({[88,63,188][i%3]},{[166,185,140][i%3]},{[255,80,255][i%3]},0.12)",
            ))
        fig_radar = go.Figure(traces)
        fig_radar.update_layout(
            **_layout(height=320),
            polar=dict(
                bgcolor=CARD_L,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER, tickfont=dict(color=SUBTEXT)),
                angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT)),
            ),
            title=dict(text="Confronto backtest (normalizzato)", font=dict(color=TEXT, size=13)),
        )
        blocks.append(dcc.Graph(figure=fig_radar, config={"displayModeBar": False}))

    blocks.append(_explain(
        "Sharpe Ratio: return medio diviso volatilità (annualizzato). > 1.0 = buono, > 2.0 = eccellente. "
        "Sortino: come Sharpe ma penalizza solo le perdite, non la volatilità al rialzo — più realistico. "
        "Calmar: return annuo / max drawdown — misura l'efficienza del rischio. "
        "Turnover: quante volte il capitale è stato reinvestito — alto turnover = più commissioni."
    ))

    return _section("2 — Metriche di Performance", blocks)


def sec_strategy(reports):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dash import dcc

    if not reports:
        return _section("3 — Strategia per Bucket", _explain("Nessun dato."))

    figs = []
    for r in reports:
        sb  = r.get("strategy_breakdown", {})
        w   = r.get("window", {})
        if not sb:
            continue

        buckets = [b for b, v in sb.items() if v.get("trades", 0) > 0]
        if not buckets:
            continue

        label = f"{w.get('start_date','')} → {w.get('end_date','')}"

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["P&L Realizzato ($)", "Win Rate (%)", "N° Trade"],
        )
        colors = [BUCKET_COLORS.get(b, SUBTEXT) for b in buckets]

        fig.add_trace(go.Bar(
            x=buckets, y=[sb[b].get("realized_pnl", 0) for b in buckets],
            marker_color=[GREEN if sb[b].get("realized_pnl",0) >= 0 else RED for b in buckets],
            name="P&L", showlegend=False,
            hovertemplate="%{x}<br>P&L: $%{y:+,.0f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=buckets, y=[sb[b].get("win_rate", 0) * 100 for b in buckets],
            marker_color=colors, name="Win Rate", showlegend=False,
            hovertemplate="%{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=buckets, y=[sb[b].get("trades", 0) for b in buckets],
            marker_color=colors, name="Trades", showlegend=False,
            hovertemplate="%{x}<br>Trades: %{y}<extra></extra>",
        ), row=1, col=3)

        fig.add_hline(y=50, row=1, col=2, line_dash="dash", line_color=YELLOW, line_width=1)

        fig.update_layout(
            **_layout(height=300),
            title=dict(text=label, font=dict(color=TEXT, size=12)),
        )
        for col in range(1, 4):
            fig.update_xaxes(gridcolor=BORDER, row=1, col=col)
            fig.update_yaxes(gridcolor=BORDER, row=1, col=col)

        figs.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

    figs.append(_explain(
        "Il breakdown per bucket mostra quale strategia ha contribuito di più al risultato. "
        "'bull' = posizioni long su azioni; 'bear' = hedge/short; 'crypto' = asset digitali. "
        "Un P&L positivo con win rate > 50% indica una strategia robusta. "
        "Se un bucket ha molti trade ma P&L basso, le commissioni stanno erodendo i guadagni."
    ))

    return _section("3 — Performance per Strategia (Bucket)", figs)


def sec_decisions(reports):
    from dash import html, dcc

    if not reports:
        return _section("4 — Logica Decisionale", _explain("Nessun dato."))

    def agent_box(title, freq, color, inputs, outputs, desc):
        return html.Div([
            html.Div([
                html.Span(title, style={"color": color, "fontWeight": "700", "fontSize": "13px"}),
                html.Span(f"  [{freq}]", style={"color": SUBTEXT, "fontSize": "11px"}),
            ], style={"marginBottom": "6px"}),
            html.Div([
                html.Div([
                    html.Div("INPUT", style={"color": SUBTEXT, "fontSize": "10px", "marginBottom": "3px"}),
                    html.Div(inputs, style={"color": TEXT, "fontSize": "11px", "lineHeight": "1.6"}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Div("OUTPUT", style={"color": SUBTEXT, "fontSize": "10px", "marginBottom": "3px"}),
                    html.Div(outputs, style={"color": color, "fontSize": "11px", "lineHeight": "1.6"}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Div("COSA FA", style={"color": SUBTEXT, "fontSize": "10px", "marginBottom": "3px"}),
                    html.Div(desc, style={"color": SUBTEXT, "fontSize": "11px", "lineHeight": "1.6", "fontStyle": "italic"}),
                ], style={"flex": "2"}),
            ], style={"display": "flex", "gap": "16px"}),
        ], style={
            "background": CARD_L, "border": f"2px solid {color}",
            "borderRadius": "6px", "padding": "12px 16px", "marginBottom": "8px",
        })

    def arrow():
        return html.Div("↓", style={"textAlign": "center", "color": BORDER,
                                     "fontSize": "20px", "margin": "-2px 0"})

    def bg_row(agents):
        """Row of background agents running in parallel threads."""
        return html.Div([
            html.Div("BACKGROUND THREADS (asincroni, ogni N cicli)", style={
                "color": SUBTEXT, "fontSize": "10px", "textTransform": "uppercase",
                "letterSpacing": "0.6px", "marginBottom": "6px",
            }),
            html.Div(agents, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        ], style={
            "border": f"1px dashed {BORDER}", "borderRadius": "6px",
            "padding": "12px", "marginBottom": "8px",
        })

    def bg_agent(title, color, cadence, desc):
        return html.Div([
            html.Div(title, style={"color": color, "fontWeight": "700", "fontSize": "12px"}),
            html.Div(cadence, style={"color": SUBTEXT, "fontSize": "10px"}),
            html.Div(desc, style={"color": SUBTEXT, "fontSize": "11px", "marginTop": "4px", "fontStyle": "italic"}),
        ], style={
            "background": CARD, "border": f"1px solid {color}",
            "borderRadius": "4px", "padding": "10px 12px", "flex": "1", "minWidth": "160px",
        })

    pipeline = html.Div([
        html.H3("Pipeline completa del bot", style={"color": TEXT, "fontSize": "14px",
                                                     "marginBottom": "14px", "marginTop": 0}),

        # Step 1 — Market Data
        agent_box(
            "MarketDataAgent", "ogni ciclo ~60s", BLUE,
            "config.json (simboli BTC/ETH/SOL/BNB)", "market_data.json",
            "Simula OHLCV con Geometric Brownian Motion (GBM) + noise. "
            "Genera orderbook sintetico. In backtest usa dati storici reali da yfinance.",
        ),
        arrow(),

        # Background agents — run in parallel
        bg_row([
            bg_agent("MacroAnalyzerAgent", BLUE, "ogni ciclo",
                      "Fed funds, CPI, DXY, VIX → macro_snapshot.json + market_regime.json. "
                      "Calcola market_bias [-1,+1] da BTC momentum proxy."),
            bg_agent("SectorAnalyzerAgent", PURPLE, "ogni 10 cicli (~10 min)",
                      "Legge stock_scores.json → calcola score per settore (Tech, Health, Finance…) "
                      "→ sector_scorecard.json. Rank settori per momentum relativo."),
            bg_agent("StockAnalyzerAgent", ORANGE, "ogni 5 cicli (~5 min)",
                      "Fetcha P/E, EPS, insider flow via yfinance per ~350 simboli "
                      "→ stock_scores.json (composite_score tecnico + fondamentale)."),
            bg_agent("FeatureStoreAgent", YELLOW, "ogni 3 cicli (~3 min)",
                      "Calcola feature cross-sezionali per il modello ML "
                      "(momentum, volatilità, correlazioni) → cross_sectional_features.jsonl."),
            bg_agent("MLStrategyAgent", GREEN, "ogni 30 cicli (~30 min)",
                      "Riaddestra modelli ElasticNet/RandomForest/MLP su feature store "
                      "→ ml_signals.json con predicted_excess_return per simbolo."),
        ]),
        arrow(),

        # Step 2 — Strategy agents
        agent_box(
            "BullStrategyAgent", "ogni ciclo", GREEN,
            "stock_scores.json + macro_snapshot.json + sector_scorecard.json",
            "bull_signals.json",
            "Seleziona top ETF (50%), large cap (30%), small cap (20%) con composite_score > soglia. "
            "Applica macro_multiplier dal bias di mercato.",
        ),
        agent_box(
            "BearStrategyAgent", "ogni ciclo", RED,
            "stock_scores.json + macro_snapshot.json",
            "bear_signals.json",
            "Seleziona ETF hedge (SQQQ, YANG, SOXS…) e short candidates con score < 0.4. "
            "Modalità hedge_only: max 2% del capitale per hedging.",
        ),
        agent_box(
            "CryptoStrategyAgent", "ogni ciclo", BLUE,
            "market_data.json (OHLCV crypto) + signals.json (TA)",
            "crypto_signals.json",
            "Combina segnali TA (RSI/MACD/BB/EMA) con momentum 7d. "
            "Divide in core (BTC/ETH), DeFi bridge, altcoin.",
        ),
        agent_box(
            "AlphaHunterAgent", "ogni ciclo", PURPLE,
            "yfinance live (284 simboli: equity + ETF)",
            "alpha_signals.json",
            "3 fattori: IV/HV mispricing (40%) + momentum RSI/EMA (35%) + "
            "value/mean-reversion 52w z-score (25%). Portfolio isolato $100K.",
        ),
        arrow(),

        # Step 3 — Technical Analysis
        agent_box(
            "TechnicalAnalysisAgent", "ogni ciclo", PURPLE,
            "market_data.json (OHLCV BTC/ETH/SOL/BNB + altri crypto)",
            "signals.json (TA signals con last_price embedded)",
            "RSI-14, MACD(12/26/9), Bollinger(20,2), EMA(20/50/200). "
            "Genera BUY/SELL/HOLD per ogni simbolo con stop-loss/take-profit/ATR.",
        ),
        arrow(),

        # Step 4 — Risk
        agent_box(
            "RiskAgent", "ogni ciclo", YELLOW,
            "bull/bear/crypto_signals + signals.json + ml_signals.json + adaptive_params.json",
            "validated_signals.json",
            "Calcola composite_score combinando signal + ML boost (se fresco < 3h). "
            "Applica 6 filtri: score >= soglia, drawdown < 15%, sub-exposure < 95%, "
            "multi_tf_confirm, turnover_gate (prezzo mosso > 2%), max 12 posizioni.",
        ),
        arrow(),

        # Step 5 — Execution
        agent_box(
            "ExecutionAgent", "ogni ciclo", GREEN,
            "validated_signals.json + market_data.json + portfolio_*.json",
            "portfolio_retail.json + portfolio_institutional.json + portfolio_alpha.json",
            "Controlla exits (stop-loss 10%, take-profit 8%, trailing stop) PRIMA delle entries. "
            "Fill simulato con slippage (0.02%) + commissioni (0.05%). "
            "Gestisce 3 portfolio separati: retail ($20K), institutional ($880K), alpha ($100K).",
        ),
        arrow(),

        # Step 6 — Report
        agent_box(
            "ReportAgent + AdaptiveLearner", "orario", ORANGE,
            "portfolio_*.json + validated_signals.json",
            "reports/report_*.json + adaptive_params.json + learned_strategy_weights.json",
            "Salva report orario con P&L, posizioni, metriche. "
            "AdaptiveLearner aggiorna soglie per bucket e pesi strategia in base alle performance recenti.",
        ),
    ])

    # Sizing + filters tables
    tables = html.Div([
        html.H3("Dettaglio sizing & filtri", style={"color": TEXT, "fontSize": "13px",
                                                     "marginBottom": "12px", "marginTop": "20px"}),
        html.Div([
            html.Div([
                html.Div("Formula sizing", style={"color": YELLOW, "fontWeight": "700", "marginBottom": "6px"}),
                html.Pre(
                    "qty = (equity × stake_pct) / price\n\n"
                    "Retail:        stake=30%, reserve=20%\n"
                    "Institutional: stake=20%, reserve=15%\n"
                    "Alpha Hunter:  stake=20%, reserve=10%\n\n"
                    "max_exposure = 20% per singolo asset\n"
                    "stop_loss    = entry × 0.90  (−10%)\n"
                    "take_profit  = entry × 1.08  (+8%)",
                    style={"color": GREEN, "fontSize": "12px", "background": BG,
                           "padding": "10px", "borderRadius": "4px", "margin": 0}
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("Filtri RiskAgent (tutti devono passare)", style={"color": YELLOW, "fontWeight": "700", "marginBottom": "6px"}),
                html.Pre(
                    "1. composite_score >= soglia_bucket\n"
                    "   crypto:    0.65\n"
                    "   bull:      0.72\n"
                    "   bear_hedge:0.56 / bear_short:0.68\n"
                    "   ta_crypto: 0.65\n\n"
                    "2. portfolio_drawdown < 15%\n"
                    "3. sub_exposure < 95% budget bucket\n"
                    "4. multi_tf_confirm (trend su più TF)\n"
                    "5. turnover_gate: Δprice > 2% da entry\n"
                    "6. max_open_positions <= 12",
                    style={"color": BLUE, "fontSize": "12px", "background": BG,
                           "padding": "10px", "borderRadius": "4px", "margin": 0}
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("Exit conditions (ordine di priorità)", style={"color": YELLOW, "fontWeight": "700", "marginBottom": "6px"}),
                html.Pre(
                    "1. STOP LOSS\n"
                    "   price <= entry × 0.90\n\n"
                    "2. TRAILING STOP\n"
                    "   attiva dopo +1.5% dal entry\n"
                    "   chiude se high × (1 − 0.02)\n\n"
                    "3. TAKE PROFIT\n"
                    "   price >= entry × 1.08\n\n"
                    "4. SEGNALE SELL\n"
                    "   RiskAgent emette SELL signal",
                    style={"color": RED, "fontSize": "12px", "background": BG,
                           "padding": "10px", "borderRadius": "4px", "margin": 0}
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("Chi calcola settori & mercati", style={"color": YELLOW, "fontWeight": "700", "marginBottom": "6px"}),
                html.Pre(
                    "SETTORI:\n"
                    "  SectorAnalyzerAgent\n"
                    "  → sector_scorecard.json\n"
                    "  ogni 10 cicli (~10 min)\n\n"
                    "MACRO / REGIME:\n"
                    "  MacroAnalyzerAgent\n"
                    "  → macro_snapshot.json\n"
                    "  → market_regime.json\n"
                    "  ogni ciclo\n\n"
                    "STOCK SCORES:\n"
                    "  StockAnalyzerAgent\n"
                    "  → stock_scores.json\n"
                    "  ogni 5 cicli (~5 min)",
                    style={"color": PURPLE, "fontSize": "12px", "background": BG,
                           "padding": "10px", "borderRadius": "4px", "margin": 0}
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ])

    return _section("4 — Pipeline Completa, Logica Decisionale & Sizing", [
        pipeline,
        tables,
        _explain(
            "Il bot non prende mai decisioni discrezionali — ogni trade è il risultato deterministico "
            "della pipeline. I settori vengono calcolati da SectorAnalyzerAgent ogni ~10 minuti in un thread separato. "
            "Il regime di mercato (RISK_ON / NEUTRAL / RISK_OFF) viene aggiornato ogni ciclo da MacroAnalyzerAgent "
            "usando il momentum BTC a 30 giorni come proxy. StockAnalyzerAgent calcola composite_score tecnico + "
            "fondamentale per ~350 simboli. Il ML boost viene applicato dal RiskAgent solo se i segnali ML "
            "sono stati aggiornati nelle ultime 3 ore — altrimenti viene ignorato per evitare segnali stantii."
        ),
    ])


def sec_variance(reports):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dash import dcc, html

    if not reports:
        return _section("5 — Variance & Risk", _explain("Nessun dato."))

    figs = []
    for r in reports:
        ec  = r.get("equity_curve", [])
        m   = r.get("metrics", {})
        w   = r.get("window", {})
        if len(ec) < 2:
            continue

        # Daily returns
        rets = [(ec[i] - ec[i-1]) / ec[i-1] for i in range(1, len(ec)) if ec[i-1] > 0]
        if not rets:
            continue

        mean_r = sum(rets) / len(rets)
        var_r  = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        std_r  = math.sqrt(var_r)
        skew   = sum((r - mean_r) ** 3 for r in rets) / (len(rets) * std_r ** 3) if std_r > 0 else 0
        kurt   = sum((r - mean_r) ** 4 for r in rets) / (len(rets) * std_r ** 4) - 3 if std_r > 0 else 0
        neg_r  = [r for r in rets if r < 0]
        down_std = math.sqrt(sum(r**2 for r in neg_r) / max(len(neg_r), 1))
        var_95  = sorted(rets)[max(0, int(len(rets) * 0.05))]
        var_99  = sorted(rets)[max(0, int(len(rets) * 0.01))]

        label = f"{w.get('start_date','')} → {w.get('end_date','')}"

        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=["Distribuzione Ritorni Giornalieri", "Rolling Volatilità (7gg)"])

        # Histogram of returns
        fig.add_trace(go.Histogram(
            x=[r * 100 for r in rets], nbinsx=30,
            marker_color=BLUE, opacity=0.8,
            hovertemplate="Return: %{x:.2f}%<br>Giorni: %{y}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig.add_vline(x=0,            line_dash="dash", line_color=YELLOW,  line_width=1, row=1, col=1)
        fig.add_vline(x=var_95 * 100, line_dash="dot",  line_color=RED,    line_width=1.5, row=1, col=1,
                       annotation_text="VaR 95%", annotation_font_color=RED, annotation_font_size=10)

        # Rolling volatility
        window = 7
        roll_vol = []
        for i in range(window, len(rets)):
            chunk = rets[i-window:i]
            m2    = sum(chunk) / window
            rv    = math.sqrt(sum((x - m2)**2 for x in chunk) / window) * 100
            roll_vol.append(rv)

        fig.add_trace(go.Scatter(
            y=roll_vol, x=list(range(window, len(rets))),
            line=dict(color=ORANGE, width=1.5), showlegend=False,
            fill="tozeroy", fillcolor="rgba(227,179,65,0.1)",
            hovertemplate="Giorno %{x}<br>Vol: %{y:.2f}%<extra></extra>",
        ), row=1, col=2)

        fig.update_layout(**_layout(height=300), title=dict(text=label, font=dict(color=TEXT, size=12)))
        for col in [1, 2]:
            fig.update_xaxes(gridcolor=BORDER, row=1, col=col)
            fig.update_yaxes(gridcolor=BORDER, row=1, col=col)

        figs.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

        # Stats row
        figs.append(html.Div([
            _metric("Volatilità daily",  f"{std_r*100:.2f}%",   "std dev ritorni",  YELLOW),
            _metric("Downside vol",      f"{down_std*100:.2f}%", "solo ritorni neg", RED),
            _metric("VaR 95%",           f"{var_95*100:.2f}%",  "perdita max 1/20", RED),
            _metric("VaR 99%",           f"{var_99*100:.2f}%",  "perdita max 1/100", RED),
            _metric("Skewness",          f"{skew:.2f}",          "> 0 = coda destra", BLUE),
            _metric("Excess Kurtosis",   f"{kurt:.2f}",          "> 0 = code pesanti", PURPLE),
            _metric("Max Drawdown",      f"{abs(m.get('max_drawdown',0)):.1%}", None,
                     RED if abs(m.get("max_drawdown",0)) > 0.15 else YELLOW),
        ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "12px",
                   "marginBottom": "8px"}))

    figs.append(_explain(
        "La distribuzione dei ritorni ideale è simmetrica con code sottili (kurtosis bassa). "
        "Skewness positiva è buona — significa più giorni con guadagni grandi che perdite grandi. "
        "Il VaR 95% dice: 'nel 5% dei giorni peggiori, la perdita è almeno questa'. "
        "La volatilità rolling mostra se il rischio è stabile o aumenta in certi periodi — "
        "picchi di volatilità coincidono spesso con fasi di mercato difficili."
    ))

    return _section("5 — Variance & Distribuzione dei Ritorni", figs)


# ── App assembly ──────────────────────────────────────────────────────────────

def build_app(port: int):
    import dash
    from dash import html

    reports = load_reports()
    print(f"  Report backtest trovati: {len(reports)}")
    for r in reports:
        w = r.get("window", {})
        s = r.get("summary", {})
        print(f"    {r['_file']} | {w.get('start_date','')} → {w.get('end_date','')} | trades={s.get('trades',0)}")

    app = dash.Dash(__name__, title="Backtest Performance")
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Backtest Performance Dashboard",
                     style={"margin": 0, "fontSize": "20px", "color": TEXT}),
            html.Div("Analisi storica · Dati da reports/backtest_report_*.json",
                      style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "4px"}),
        ], style={"background": CARD, "borderBottom": f"1px solid {BORDER}", "padding": "18px 32px"}),

        html.Div([
            sec_header(reports),
            sec_equity(reports),
            sec_metrics(reports),
            sec_strategy(reports),
            sec_decisions(reports),
            sec_variance(reports),
        ], style={"padding": "20px 32px"}),
    ], style={"background": BG, "minHeight": "100vh", "fontFamily": "monospace"})

    return app


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Performance Dashboard")
    parser.add_argument("--port", type=int, default=8054)
    args = parser.parse_args()

    print("\n Backtest Performance Dashboard")
    print("=" * 40)

    app = build_app(args.port)
    url = f"http://localhost:{args.port}"
    print(f"  URL: {url}")
    print(f"  Ctrl+C per fermare\n")

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
