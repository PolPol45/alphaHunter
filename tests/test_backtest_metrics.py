from backtesting.backtest_metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio


def test_metrics_basic_shapes():
    rets = [0.01, -0.005, 0.007, 0.002, -0.003, 0.004]
    eq = [100, 101, 100.5, 101.2, 100.9, 101.3]

    s = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    mdd = max_drawdown(eq)
    cal = calmar_ratio(eq)

    assert isinstance(s, float)
    assert isinstance(so, float)
    assert 0 <= mdd <= 1
    assert isinstance(cal, float)


def test_max_drawdown_zero_for_monotonic_up():
    eq = [100, 101, 102, 103]
    assert max_drawdown(eq) == 0
