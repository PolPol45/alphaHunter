# Trading Bot — Multi-Agent Top-Down Engine

Paper-trading multi-agent system with top-down analysis:
`Market -> Macro -> Sector -> Stock -> Strategies -> Risk -> Execution -> Report`.

The bot supports:
- live cycle mode (`orchestrator.mode = live_cycle`)
- historical backtest mode (`orchestrator.mode = backtest`)
- AutoML retrain/deploy loop for strategy weights (from backtest reports)

## 1. Architecture Overview

The runtime is orchestrated by [orchestrator.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/orchestrator.py).

Main pipeline order in live mode:

1. `MarketDataAgent`
2. `MacroAnalyzerAgent`
3. `SectorAnalyzerAgent`
4. `StockAnalyzerAgent`
5. `NewsDataAgent`
6. `TechnicalAnalysisAgent`
7. `BullStrategyAgent`, `BearStrategyAgent`, `CryptoStrategyAgent`
8. `RiskAgent`
9. `ExecutionAgent`
10. `PortfolioManager`
11. `ReportAgent` (hourly cadence)

Backtest mode runs via `BacktestingAgent`, which replays historical dates and calls risk/execution on each simulated step.

## 2. Agent Team and Connections

All agents inherit from [base_agent.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/agents/base_agent.py) and communicate only through JSON files in `data/`.

Core agents:

- `MarketDataAgent`: market feed ingest (Binance/OpenBB fallback, world monitor events), writes `data/market_data.json`.
- `MacroAnalyzerAgent`: macro regime snapshot, writes `data/macro_snapshot.json` and `data/market_regime.json`.
- `SectorAnalyzerAgent`: sector scorecard/ranking, writes `data/sector_scorecard.json`.
- `StockAnalyzerAgent`: per-ticker scoring dataset, writes `data/stock_scores.json`.
- `NewsDataAgent`: normalized news feed and alerts, writes `data/news_feed.json`.
- `TechnicalAnalysisAgent`: technical signals layer from candles/events/news, writes `data/signals.json`.
- `BullStrategyAgent`: long bias allocation from stock/sector/macro context, writes `data/bull_signals.json`.
- `BearStrategyAgent`: hedge/short allocation with `hedge_only` support, writes `data/bear_signals.json`.
- `CryptoStrategyAgent`: crypto regime and bucket allocations, writes `data/crypto_signals.json`.
- `RiskAgent`: merges/filters all strategy signals, applies portfolio constraints, writes `data/validated_signals.json`.
- `ExecutionAgent`: paper execution (IBKR paper if reachable, otherwise simulation), writes `data/portfolio_*.json`.
- `PortfolioManager`: consolidates sub-portfolios for dashboard/reporting.
- `ReportAgent`: periodic operational reports into `reports/`.
- `BacktestingAgent`: end-to-end historical replay and metrics report.

## 3. Top-Down Analysis (Current Implementation)

### 3.1 Market Layer

Inputs:
- crypto OHLCV/orderbook
- macro/geopolitical events

Outputs:
- `market_data.json`
- market event context used downstream

### 3.2 Macro Layer

Inputs:
- market feed + macro sources

Outputs:
- `macro_snapshot.json`
- `market_regime.json` with regime and risk flags

Used by:
- strategy gating
- risk-off / risk-on filtering in `RiskAgent`

### 3.3 Sector Layer

Inputs:
- sector map and benchmarks from config

Outputs:
- sector scorecard with relative ranking

Used by:
- stock and strategy prioritization

### 3.4 Stock Layer

Inputs:
- ticker universe + technical/fundamental blocks

Outputs:
- `stock_scores.json` with composite score per symbol

In backtest implementation the core formula is explicit:
- `composite_score = 0.6 * technical_score + 0.4 * fundamental_score`

### 3.5 Strategy Layer

Bull/Bear/Crypto consume the top-down datasets and generate candidate allocations.

### 3.6 Risk Layer

`RiskAgent` is the execution gate. It does:
- score thresholding by strategy bucket
- market regime filter (risk-on/risk-off logic)
- multi-timeframe confirmation (`1d` and `4h` alignment)
- correlation clustering / covariance penalty
- volatility-targeted sizing
- max exposure caps
- turnover controls (via execution constraints)
- dynamic strategy weighting (rolling performance + learned weights)

### 3.7 Execution and Reporting

`ExecutionAgent` applies approved orders and updates simulated/paper portfolio state.
`ReportAgent` emits cycle reports.

## 4. Mathematical Core (Implemented Formulas)

### 4.1 Stock composite score (backtest path)

- `score = 0.6 * technical + 0.4 * fundamental`

### 4.2 Volatility-targeted risk sizing

In [risk_agent.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/agents/risk_agent.py):

- `score_mult = 0.7 + clamp(score,0,1) * 0.6`
- `vol_mult = clamp(target_vol_daily / max(vol_pct, min_vol_floor), 0.5, 1.5)`
- `corr_mult = 1 - min(0.9, corr_penalty * covariance_penalty)`
- `size_mult = clamp(score_mult * vol_mult * corr_mult, 0.35, 1.6)`

Final stake is then clipped by:
- asset cap: `max_asset_exposure_pct * mode_equity`
- sub-portfolio cap: `max_sub_exposure_pct * sub_capital`
- available cash

### 4.3 Dynamic strategy weighting

Rolling performance multiplier per bucket:

- `edge = realized_pnl_sum / gross_notional_sum`
- `rolling_mult = clamp(1 + 10 * edge, strategy_weight_min, strategy_weight_max)`

Combined with learned AutoML weight:

- `final_mult = clamp(rolling_mult * learned_mult, strategy_weight_min, strategy_weight_max)`

### 4.4 Correlation gate

Candidate rejected when:
- `abs(corr(candidate, any_selected)) >= correlation_threshold`

### 4.5 Backtest performance metrics

Computed in [backtest_metrics.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/backtesting/backtest_metrics.py):

- Sharpe
- Sortino
- Max Drawdown
- Calmar
- Win rate
- Turnover

Backtest report also includes:
- equity curve
- strategy breakdown (`bull`, `bear`, `crypto`, `unknown`)

## 5. AutoML Loop (Learn -> Retrain -> Deploy)

Implemented in:
- [auto_ml_pipeline.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/backtesting/auto_ml_pipeline.py)
- [train_and_deploy.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/backtesting/train_and_deploy.py)

Process:

1. Load historical `backtest_report_*.json` from:
   - `trading_bot/reports/`
   - zip artifacts in workspace root (if they contain report JSON)
2. Walk-forward model selection on a parameter grid.
3. Fit final strategy multipliers for `bull/bear/crypto`.
4. Deploy to `data/learned_strategy_weights.json`.
5. `RiskAgent` consumes deployed weights automatically.

AutoML can run:
- automatically after each backtest (if enabled in `config.backtesting.auto_ml`)
- manually via CLI:

```bash
cd trading_bot
../venv/bin/python -m backtesting.train_and_deploy --repo-dir . --workspace-dir ".."
```

## 6. GitHub Backtest Report Publishing

Workflow:
- [.github/workflows/backtest.yml](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/.github/workflows/backtest.yml)

Current behavior:

- runs backtest on GitHub Actions
- uploads artifact
- commits generated report JSON into branch `backtest-reports`

This avoids manual zip handling and gives direct Git access to historical reports.

## 7. Setup and Run

### 7.1 Install

```bash
cd trading_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 7.2 Live cycle

```bash
cd trading_bot
source venv/bin/activate
python orchestrator.py
```

### 7.3 Backtest cycle

Set in `config.json`:
- `orchestrator.mode = "backtest"`
- `backtesting.start_date`
- `backtesting.end_date`

Then run:

```bash
cd trading_bot
source venv/bin/activate
python orchestrator.py
```

## 8. Key Files

- Runtime entrypoint: [orchestrator.py](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/orchestrator.py)
- Configuration: [config.json](/Users/paolomaizza/Desktop/crypto challenge/trading_bot/config.json)
- Agents folder: `/Users/paolomaizza/Desktop/crypto challenge/trading_bot/agents`
- Backtesting folder: `/Users/paolomaizza/Desktop/crypto challenge/trading_bot/backtesting`
- Data contracts: `/Users/paolomaizza/Desktop/crypto challenge/trading_bot/data_contracts`
- Reports output: `/Users/paolomaizza/Desktop/crypto challenge/trading_bot/reports`

## 9. Safety and Scope

- Paper-trading only (`trading.paper_trading` must remain `true`).
- If IBKR paper connection is unavailable, execution falls back to simulation.
- Agent communication is file-based and auditable through JSON outputs.


LIVE BOT
cd "/Users/paolomaizza/Desktop/crypto challenge/trading_bot"
source "../venv/bin/activate"
python orchestrator.py


MONTE CARLO BACKTEST (nuovo terminale)
cd "/Users/paolomaizza/Desktop/crypto challenge/trading_bot"
source "../venv/bin/activate"
python monte_carlo_backtest.py --scenarios 50 --days 90 --scenario-type base


DASHBOARD LIVE — trade analytics (porta 8052)
cd "/Users/paolomaizza/Desktop/crypto challenge/trading_bot"
source "../venv/bin/activate"
python trade_analytics.py


DASHBOARD PERFORMANCE LIVE — equity curve + ML + adaptive params (porta 8053)
cd "/Users/paolomaizza/Desktop/crypto challenge/trading_bot"
source "../venv/bin/activate"
python performance_dashboard.py


DASHBOARD BACKTEST STORICO — metriche, pipeline, variance (porta 8054)
cd "/Users/paolomaizza/Desktop/crypto challenge/trading_bot"
source "../venv/bin/activate"
python backtest_performance.py


DASHBOARD MONTE CARLO — distribuzione scenari (porta 8051)
cd "/Users/paolomaizza/Desktop/crypto challenge/trading_bot"
source "../venv/bin/activate"
python backtest_dashboard.py


Porte summary:

Porta	Dashboard
8051	Monte Carlo scenari
8052	Trade live (posizioni, P&L, sizing)
8053	Performance live (equity curve, ML)
8054	Backtest storico (metriche, pipeline