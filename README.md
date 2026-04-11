# Crypto Trading Bot — Multi-Agent System

> **PAPER TRADING ONLY** — the system can consume live market data and talk to a paper IBKR account, but it is designed to avoid real-money execution.

---

## Overview

This repository currently has two distinct runtimes:

1. `trading_bot/` — the Python multi-agent engine that generates signals, validates risk, executes paper trades, and writes JSON state.
2. `server/` + `crypto_dashboard_app.html` — the Node/Socket.IO dashboard backend and static frontend.

The dashboard does not run the strategy. It reads the JSON files produced by `trading_bot/` and also opens separate Bybit market/account streams for the UI.

### Current architecture in code

The Python pipeline manages three portfolios:

- `retail` crypto portfolio
- `institutional` crypto portfolio
- `alpha` US-equity sidecar portfolio driven by `AlphaHunterAgent`

Each agent communicates through files in `data/`, and `orchestrator.py` runs the pipeline sequentially every 60 seconds.

---

## Integrations

The code supports these adapters:

| Integration | Used by | Role | Fallback |
|-------------|---------|------|----------|
| **OpenBB / yfinance** | `MarketDataAgent` | Crypto OHLCV data | Local simulation |
| **World Monitor** | `MarketDataAgent` | Macro / geopolitical events | Stub demo events |
| **IBKR paper account** | `ExecutionAgent` | Crypto paper execution | Local simulation |
| **yfinance options/equities** | `AlphaHunterAgent` | US equity alpha scan | HOLD signals on failure |
| **Bybit REST / WS** | `server/` only | Dashboard market/account feed | None |

Important: the current `config.json` in this repo already has `openbb.enabled`, `alpha_hunter.enabled`, and `ibkr.enabled` set to `true`. Do not assume pure simulation unless you explicitly disable them.

---

## Prerequisites

- Python 3.10 or higher
- pip
- Node.js for the dashboard server

Optional only if you want live paper execution:

- IB Gateway or TWS running on a paper account
- API access enabled in IBKR

---

## Setup

### Python bot

```bash
cd trading_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dashboard server

```bash
cd ..
npm install
```

### Environment variables

The repo-root `.env` is used by the Node dashboard backend, not by the Python multi-agent bot.

Current variables:

- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_MODE`
- `PORT`

IBKR settings for the Python bot live in `trading_bot/config.json`, not in `.env`.

---

## Running

### Only the Python bot

```bash
cd trading_bot
source venv/bin/activate
python orchestrator.py
```

### Bot + dashboard

```bash
# Terminal 1
npm start

# Terminal 2
cd trading_bot
source venv/bin/activate
python orchestrator.py
```

If the dashboard is running without the Python orchestrator, the Bybit panels can still update, but the bot panels will show stale or empty data.

---

## Real Pipeline

```text
orchestrator.py
│
├─► MarketDataAgent
│     Reads live crypto OHLCV from OpenBB/yfinance when enabled.
│     Falls back to local simulation when data fetch fails.
│     Writes data/market_data.json
│
├─► TechnicalAnalysisAgent
│     Builds retail + institutional crypto signals from 4h and 1d candles.
│     Uses EMA, RSI, MACD, ADX, ATR, volume ratio, ROC.
│     Writes data/signals.json
│
├─► RiskAgent
│     Validates retail + institutional signals.
│     Retail: circuit breaker + stake sizing.
│     Institutional: Black-Litterman + Kelly + Greeks guardrails.
│     Writes data/validated_signals.json and data/bl_state.json
│
├─► AlphaHunterAgent
│     Separate US-equity scanner using yfinance + options data.
│     Writes data/alpha_signals.json
│
├─► ExecutionAgent
│     Executes retail + institutional crypto trades through IBKR paper if connected,
│     otherwise in local simulation.
│     Executes alpha portfolio only in simulation.
│     Writes portfolio_*.json and stop_orders_*.json
│
└─► ReportAgent
      Builds hourly JSON reports from the latest state.
      Writes reports/report_YYYY-MM-DD_HH-MM-SS.json
```

Failure safety remains the same: if an upstream agent fails, downstream crypto execution is skipped for that cycle.

---

## Portfolio Accounting Notes

Retail and institutional are strategy sleeves, not isolated IBKR sub-accounts.

When IBKR execution is live:

- both sleeves share one paper IBKR account
- total equity and cash are split proportionally using configured capital weights
- reports expose this as `equity_basis = proportional_split_of_shared_ibkr_account`

This means report-level equity for `retail` and `institutional` is synthetic bookkeeping, not two separate broker balances.

When IBKR is not connected, each sleeve is marked-to-market independently in local simulation.

---

## Output Files

| File | Writer | Meaning |
|------|--------|---------|
| `data/market_data.json` | `MarketDataAgent` | Crypto OHLCV, orderbook, last price, world events |
| `data/signals.json` | `TechnicalAnalysisAgent` | Retail + institutional crypto signals |
| `data/validated_signals.json` | `RiskAgent` | Approved / rejected crypto signals |
| `data/bl_state.json` | `RiskAgent` | Cached Black-Litterman rebalance state |
| `data/alpha_signals.json` | `AlphaHunterAgent` | US-equity alpha signals |
| `data/portfolio_retail.json` | `ExecutionAgent` | Retail portfolio state |
| `data/portfolio_institutional.json` | `ExecutionAgent` | Institutional portfolio state |
| `data/portfolio_alpha.json` | `ExecutionAgent` | Alpha portfolio state |
| `data/stop_orders_retail.json` | `ExecutionAgent` | Retail stop-order bookkeeping |
| `data/stop_orders_institutional.json` | `ExecutionAgent` | Institutional stop-order bookkeeping |
| `shared_state.json` | orchestrator + agents | Health, timestamps, cycle counters |
| `reports/report_*.json` | `ReportAgent` | Hourly portfolio/system report |

---

## Legacy Files

These root-level Python files are not the main entrypoint of the current multi-agent system:

- `crypto_momentum_bot.py`
- `crypto_momentum_bot_v2.py`
- `institutional_bot.py`

They are legacy standalone implementations and reference material for logic that was later ported into `trading_bot/agents/`.

---

## Monitoring

```bash
tail -f trading_bot/logs/orchestrator.log
cat trading_bot/shared_state.json | python3 -m json.tool
cat trading_bot/data/portfolio_retail.json | python3 -m json.tool
cat trading_bot/data/portfolio_institutional.json | python3 -m json.tool
ls -lh trading_bot/reports/
```

---

## Tests

The repo did not originally include an automated test suite.
There is now a small offline regression layer under `trading_bot/tests/` focused on reporting/accounting metadata.

Run it with:

```bash
python3 -m unittest discover -s trading_bot/tests
```

---

## Safety Rules

1. `trading.paper_trading` must remain `true`.
2. `ExecutionAgent` hard-fails if `paper_trading` is not `True`.
3. IBKR execution is intended only for paper accounts.
4. Every position is expected to carry a stop-loss.
5. Agents still communicate only through JSON files in `data/`; there are no direct inter-agent method calls.
