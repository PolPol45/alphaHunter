# AutoML Backtest Loop (learn -> retrain -> deploy)

Questo modulo implementa un loop automatico sui report di backtest:

1. **Learn**: legge tutti i `backtest_report_*.json` disponibili
   - da `trading_bot/reports/`
   - da zip artifact nella root workspace (`*.zip`) se contengono report JSON
2. **Retrain**: esegue selezione modello con **walk-forward validation**
3. **Deploy**: salva i pesi strategia in `data/learned_strategy_weights.json`
4. **Use in runtime**: `RiskAgent` legge questi pesi e li combina con il rolling edge locale.

## File principali

- `backtesting/auto_ml_pipeline.py`
- `backtesting/train_and_deploy.py`
- `agents/backtesting_agent.py` (trigger automatico post-backtest)
- `agents/risk_agent.py` (consumo pesi appresi)

## Config

In `config.json`:

- `backtesting.auto_ml.enabled` (default `true`)
- `backtesting.auto_ml.min_train_reports` (default `2`)
- `risk_agent.learned_strategy_weights` fallback statico

## Esecuzione manuale

```bash
cd trading_bot
../venv/bin/python -m backtesting.train_and_deploy --repo-dir . --workspace-dir ".." --min-train-reports 2
```

## Esecuzione automatica

Ogni run in modalità backtest (`orchestrator.mode = backtest`) al termine:

- genera il report backtest
- lancia AutoML
- salva:
  - `data/learned_strategy_weights.json`
  - `reports/auto_ml_summary_<start>_<end>.json`

## Nota pratica

Con pochi report storici (es. 2-3) il sistema tende a pesi quasi neutri.
Il modello diventa informativo quando accumuli più backtest con finestre/regimi diversi.
