import json
from pathlib import Path

from agents.backtesting_agent import BacktestingAgent


def test_backtesting_pipeline_runs_short_window(tmp_path):
    cfg_path = Path('config.json')
    original = json.loads(cfg_path.read_text())

    try:
        cfg = json.loads(cfg_path.read_text())
        cfg.setdefault('orchestrator', {})['mode'] = 'backtest'
        cfg['backtesting'] = {
            'enabled': True,
            'start_date': '2026-01-01',
            'end_date': '2026-01-03',
            'fees': {'commission_pct': 0.001},
            'slippage_model': {'default_bps': 8},
            'universe_snapshot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'NVDA', 'SOXX', 'QQQ'],
        }
        cfg_path.write_text(json.dumps(cfg, indent=2))

        agent = BacktestingAgent()
        assert agent.run() is True

        reports = sorted(Path('reports').glob('backtest_report_*.json'))
        assert reports, 'Expected at least one backtest report file'
        latest = json.loads(reports[-1].read_text())
        assert latest['report_type'] == 'backtest'
        for k in ['sharpe', 'sortino', 'max_drawdown', 'calmar', 'win_rate', 'turnover']:
            assert k in latest['metrics']
    finally:
        cfg_path.write_text(json.dumps(original, indent=2))
        ctx = Path('data/backtest_context.json')
        if ctx.exists():
            ctx.unlink()
