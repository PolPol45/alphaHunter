"""
backtest_live_dashboard.py — Dashboard HTML live per walk-forward backtest
==========================================================================
Serve una pagina HTML con Plotly che si auto-aggiorna ogni 5s.
Legge i JSON prodotti dal backtest in tempo reale.

Uso:
    python backtest_live_dashboard.py          # porta 8055
    python backtest_live_dashboard.py --port 8056
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE_DIR  = pathlib.Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

START_DATE = "2025-04-01"
END_DATE   = "2026-04-18"


def _read(path: pathlib.Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _latest_report() -> dict:
    try:
        reports = sorted(REPORTS_DIR.glob("backtest_report_*.json"), key=lambda p: p.stat().st_mtime)
        return _read(reports[-1]) if reports else {}
    except Exception:
        return {}


def build_data() -> dict:
    ctx       = _read(DATA_DIR / "backtest_context.json")
    port_r    = _read(DATA_DIR / "portfolio_retail.json")
    port_i    = _read(DATA_DIR / "portfolio_institutional.json")
    ml        = _read(DATA_DIR / "ml_signals.json")
    shared    = _read(DATA_DIR / "shared_state.json")
    dd_ks     = _read(DATA_DIR / "dd_killswitch.json")
    validated = _read(DATA_DIR / "validated_signals.json")
    report    = _latest_report()

    sim_date = ctx.get("simulated_now", "")[:10] or START_DATE
    try:
        d0 = date.fromisoformat(START_DATE)
        d1 = date.fromisoformat(END_DATE)
        dc = date.fromisoformat(sim_date)
        elapsed = (dc - d0).days
        total   = (d1 - d0).days
        pct     = round(elapsed / total * 100, 1) if total else 0
    except Exception:
        elapsed, total, pct = 0, 264, 0

    def _ep(p: dict, init: float) -> dict:
        eq  = float(p.get("equity", p.get("total_value", init)))
        pnl = eq - init
        dd  = float(p.get("drawdown_pct", 0))
        cash= float(p.get("cash", 0))
        return {"equity": eq, "pnl": pnl, "pnl_pct": pnl/init*100 if init else 0,
                "dd": dd*100, "cash": cash}

    r = _ep(port_r, float(port_r.get("initial_capital", 20_000)))
    i = _ep(port_i, float(port_i.get("initial_capital", 980_000)))

    # equity curve from latest report
    eq_curve  = report.get("equity_curve", [])
    eq_dates  = report.get("equity_dates", [])
    trades    = report.get("trades", [])
    metrics   = report.get("metrics", {})
    breakdown = report.get("strategy_breakdown", {})

    # Per-strategy P&L
    strat_pnl = {}
    for k, v in breakdown.items():
        if isinstance(v, dict):
            strat_pnl[k] = round(float(v.get("total_pnl", 0)), 2)

    # ML
    ml_ic   = float(ml.get("avg_ic_across_folds", 0) or 0)
    ml_conf = float(ml.get("avg_confidence", 0) or 0)
    n_long  = len(ml.get("long_picks", []))
    n_short = len(ml.get("short_picks", []))

    # Agent health
    agents_raw = shared.get("agents", {})
    agents = {k: v.get("status", "?") for k, v in agents_raw.items()}

    return {
        "sim_date": sim_date,
        "elapsed": elapsed,
        "total": total,
        "pct": pct,
        "retail": r,
        "inst": i,
        "total_pnl": r["pnl"] + i["pnl"],
        "total_equity": r["equity"] + i["equity"],
        "eq_curve": eq_curve,
        "eq_dates": eq_dates,
        "trades": trades,
        "metrics": metrics,
        "strat_pnl": strat_pnl,
        "ml": {"ic": ml_ic, "conf": ml_conf, "long": n_long, "short": n_short,
               "ts": ml.get("generated_at", "")[:16]},
        "agents": agents,
        "dd_ks": {"active": bool(dd_ks.get("active")), "at": float(dd_ks.get("dd_at_trigger", 0))},
        "n_approved_retail": len(validated.get("retail", {}).get("approved", [])),
        "n_approved_inst":   len(validated.get("institutional", {}).get("approved", [])),
        "n_trades": len(trades),
        "now": datetime.now().strftime("%H:%M:%S"),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'SF Mono',monospace;font-size:13px}
  h1{font-size:18px;font-weight:700;color:#58a6ff;padding:16px 20px 8px}
  .subtitle{color:#8b949e;padding:0 20px 16px;font-size:12px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;padding:0 20px 16px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
  .card-title{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
  .card-value{font-size:22px;font-weight:700}
  .card-sub{color:#8b949e;font-size:11px;margin-top:4px}
  .green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}.blue{color:#58a6ff}.dim{color:#8b949e}
  .progress-wrap{padding:0 20px 16px}
  .progress-label{display:flex;justify-content:space-between;margin-bottom:6px;color:#8b949e;font-size:12px}
  .progress-bar-bg{background:#21262d;border-radius:4px;height:14px;overflow:hidden}
  .progress-bar{background:linear-gradient(90deg,#1f6feb,#58a6ff);height:100%;transition:width .5s;border-radius:4px}
  .charts{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:0 20px 16px}
  .chart-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
  .chart-title{color:#8b949e;font-size:11px;text-transform:uppercase;margin-bottom:8px}
  .agents{padding:0 20px 16px;display:flex;flex-wrap:wrap;gap:8px}
  .agent-pill{padding:4px 10px;border-radius:20px;font-size:11px;border:1px solid}
  .agent-done{border-color:#238636;color:#3fb950;background:#0d1117}
  .agent-running{border-color:#9e6a03;color:#d29922;background:#0d1117}
  .agent-error{border-color:#da3633;color:#f85149;background:#0d1117}
  .agent-unknown{border-color:#30363d;color:#8b949e;background:#0d1117}
  .alert{margin:0 20px 16px;padding:10px 14px;background:#3d1f1f;border:1px solid #f85149;border-radius:6px;color:#f85149;font-size:12px}
  .metrics-row{display:flex;flex-wrap:wrap;gap:8px;padding:0 20px 16px}
  .metric-chip{background:#21262d;border-radius:6px;padding:6px 12px;font-size:12px}
  .metric-chip span{color:#8b949e;margin-right:4px}
  .strat-bars{padding:0 20px 16px}
  footer{padding:12px 20px;color:#30363d;font-size:11px;border-top:1px solid #21262d}
</style>
</head>
<body>
<h1>⚡ Backtest Live Monitor</h1>
<div class="subtitle" id="subtitle">Caricamento...</div>

<div class="progress-wrap">
  <div class="progress-label">
    <span id="prog-label">Progresso</span>
    <span id="prog-pct">0%</span>
  </div>
  <div class="progress-bar-bg"><div class="progress-bar" id="prog-bar" style="width:0%"></div></div>
</div>

<div id="dd-alert" class="alert" style="display:none"></div>

<div class="grid" id="kpi-grid"></div>

<div class="metrics-row" id="metrics-row"></div>

<div class="charts">
  <div class="chart-card">
    <div class="chart-title">Equity Curve (Institutional)</div>
    <div id="chart-equity" style="height:240px"></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">P&L per Strategia</div>
    <div id="chart-strat" style="height:240px"></div>
  </div>
</div>

<div class="agents" id="agents-row"></div>

<footer id="footer">Auto-refresh ogni 5s</footer>

<script>
const FMT_MONEY = v => (v>=0?'+':'')+v.toLocaleString('it-IT',{style:'currency',currency:'USD',maximumFractionDigits:0});
const FMT_PCT   = v => (v>=0?'+':'')+v.toFixed(2)+'%';
const COLOR     = v => v>=0?'#3fb950':'#f85149';

function card(title, value, sub, colorClass='') {
  return `<div class="card">
    <div class="card-title">${title}</div>
    <div class="card-value ${colorClass}">${value}</div>
    ${sub?`<div class="card-sub">${sub}</div>`:''}
  </div>`;
}

function render(d) {
  document.getElementById('subtitle').textContent =
    `Simulated: ${d.sim_date}  →  2026-04-18   |   Aggiornato: ${d.now}`;

  // progress
  document.getElementById('prog-label').textContent = `Giorno ${d.elapsed} / ${d.total}  (${d.sim_date})`;
  document.getElementById('prog-pct').textContent = d.pct + '%';
  document.getElementById('prog-bar').style.width = d.pct + '%';

  // DD killswitch
  const ddEl = document.getElementById('dd-alert');
  if (d.dd_ks.active) {
    ddEl.style.display = '';
    ddEl.textContent = `🚨 HARD DD KILLSWITCH ATTIVO — trigger a ${(d.dd_ks.at*100).toFixed(1)}%  — tutti i BUY bloccati`;
  } else { ddEl.style.display = 'none'; }

  // KPIs
  const r = d.retail, i = d.inst;
  const tp = d.total_pnl, te = d.total_equity;
  document.getElementById('kpi-grid').innerHTML = [
    card('Equity Totale', '$'+te.toLocaleString('it-IT',{maximumFractionDigits:0}), 'Retail + Institutional'),
    card('P&L Totale', FMT_MONEY(tp), '', tp>=0?'green':'red'),
    card('Retail P&L', FMT_MONEY(r.pnl), FMT_PCT(r.pnl_pct), r.pnl>=0?'green':'red'),
    card('Retail DD', r.dd.toFixed(1)+'%', `Cash: $${r.cash.toLocaleString('it-IT',{maximumFractionDigits:0})}`, r.dd>10?'red':r.dd>5?'yellow':'green'),
    card('Inst P&L', FMT_MONEY(i.pnl), FMT_PCT(i.pnl_pct), i.pnl>=0?'green':'red'),
    card('Inst DD', i.dd.toFixed(1)+'%', `Cash: $${i.cash.toLocaleString('it-IT',{maximumFractionDigits:0})}`, i.dd>10?'red':i.dd>5?'yellow':'green'),
    card('Trades Totali', d.n_trades, `Retail segnali: ${d.n_approved_retail}`),
    card('ML IC', (d.ml.ic>=0?'+':'')+d.ml.ic.toFixed(3), `conf: ${d.ml.conf.toFixed(2)}  L:${d.ml.long} S:${d.ml.short}`,
         d.ml.ic>=0.05?'green':'red'),
  ].join('');

  // Metrics chips
  const m = d.metrics;
  const chips = [
    ['Sharpe', m.sharpe!=null?(+m.sharpe).toFixed(2):'—'],
    ['Sortino', m.sortino!=null?(+m.sortino).toFixed(2):'—'],
    ['Max DD', m.max_drawdown!=null?((+m.max_drawdown)*100).toFixed(1)+'%':'—'],
    ['Calmar', m.calmar!=null?(+m.calmar).toFixed(2):'—'],
    ['Win Rate', m.win_rate!=null?((+m.win_rate)*100).toFixed(0)+'%':'—'],
    ['Turnover', m.turnover!=null?(+m.turnover).toFixed(1)+'x':'—'],
  ];
  document.getElementById('metrics-row').innerHTML = chips.map(
    ([k,v]) => `<div class="metric-chip"><span>${k}</span>${v}</div>`
  ).join('');

  // Equity curve
  if (d.eq_curve.length > 0) {
    const xs = d.eq_dates.length ? d.eq_dates : d.eq_curve.map((_,i)=>i);
    Plotly.react('chart-equity', [{
      x: xs, y: d.eq_curve, type:'scatter', mode:'lines',
      line:{color:'#58a6ff', width:2}, fill:'tozeroy',
      fillcolor:'rgba(88,166,255,0.08)', name:'Equity'
    }], {
      paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
      font:{color:'#8b949e', size:11},
      xaxis:{gridcolor:'#21262d', linecolor:'#30363d'},
      yaxis:{gridcolor:'#21262d', linecolor:'#30363d', tickprefix:'$'},
      margin:{l:55,r:10,t:5,b:30}, showlegend:false
    }, {responsive:true, displayModeBar:false});
  }

  // Strategy P&L bar chart
  const sp = d.strat_pnl;
  const spKeys = Object.keys(sp);
  const spVals = spKeys.map(k=>sp[k]);
  if (spKeys.length) {
    Plotly.react('chart-strat', [{
      x: spKeys, y: spVals, type:'bar',
      marker:{color: spVals.map(v=>v>=0?'#238636':'#da3633')},
      text: spVals.map(v=>FMT_MONEY(v)), textposition:'outside',
    }], {
      paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
      font:{color:'#8b949e', size:11},
      xaxis:{gridcolor:'#21262d'},
      yaxis:{gridcolor:'#21262d', tickprefix:'$'},
      margin:{l:60,r:10,t:5,b:30}, showlegend:false
    }, {responsive:true, displayModeBar:false});
  }

  // Agents
  const statusClass = s => s==='done'?'agent-done':s==='running'?'agent-running':s==='error'?'agent-error':'agent-unknown';
  const statusIcon  = s => s==='done'?'✓':s==='running'?'⟳':s==='error'?'✗':'?';
  document.getElementById('agents-row').innerHTML = Object.entries(d.agents).map(
    ([k,v]) => `<div class="agent-pill ${statusClass(v)}">${statusIcon(v)} ${k.replace('_agent','')}</div>`
  ).join('');

  document.getElementById('footer').textContent =
    `Auto-refresh 5s  |  Ultimo aggiornamento: ${d.now}  |  Report: ${d.n_trades} trades registrati`;
}

async function fetchAndRender() {
  try {
    const res = await fetch('/api/data');
    const d   = await res.json();
    render(d);
  } catch(e) {
    document.getElementById('subtitle').textContent = 'Errore caricamento dati: ' + e;
  }
}

fetchAndRender();
setInterval(fetchAndRender, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/api/data":
            data = build_data()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML_TEMPLATE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8055)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Dashboard → {url}  (Ctrl+C per fermare)\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFermato.")


if __name__ == "__main__":
    main()
