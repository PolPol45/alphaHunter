import pytest
import math
from backtesting.backtest_metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, win_rate, turnover
)

class TestRiskMetrics:

    def test_sharpe_ratio_normal(self):
        # A simple winning sequence of daily returns
        returns = [0.01, 0.02, 0.015, -0.005, 0.01]
        sr = sharpe_ratio(returns, rf_annual=0.0, periods_per_year=252)
        assert sr > 0
        assert isinstance(sr, float)

    def test_sharpe_ratio_empty(self):
        # Edge case: no trades/returns
        assert sharpe_ratio([]) == 0.0

    def test_max_drawdown_clean(self):
        # Equity curve going up and down
        equity = [100.0, 110.0, 90.0, 95.0, 80.0, 105.0]
        # Max peak is 110.0. Min point after peak is 80.0.
        # Drawdown = (110 - 80) / 110 = 30 / 110 = 0.2727...
        mdd = max_drawdown(equity)
        assert math.isclose(mdd, 0.2727, rel_tol=1e-3)

    def test_max_drawdown_flat(self):
        # Equity never dropping
        equity = [100.0, 110.0, 110.0, 120.0]
        assert max_drawdown(equity) == 0.0

    def test_win_rate(self):
        # Normal trades
        trades = [
            {"realized_pnl": 10},
            {"realized_pnl": -5},
            {"realized_pnl": 0},
            {"realized_pnl": 20}
        ]
        # 4 closed trades, 2 winning
        assert win_rate(trades) == 0.5

    def test_win_rate_no_closed(self):
        trades = [{"status": "OPEN", "notional": 100}]
        assert win_rate(trades) == 0.0

    def test_calmar_ratio(self):
        # Equity doubling in 1 year (approx 252 days) with 50% max drawdown
        equity = [100.0] * 126 + [50.0] + [200.0] * 125
        # Start = 100, End = 200 => CAGR = 100%
        # MDD = (100 - 50)/100 = 0.5
        # Calmar = 1.0 / 0.5 = 2.0
        cr = calmar_ratio(equity, periods_per_year=252)
        assert math.isclose(cr, 2.0, rel_tol=1e-2)
