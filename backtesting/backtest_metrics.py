from __future__ import annotations

import math


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(var, 0.0))


def sharpe_ratio(returns: list[float], rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    rf = rf_annual / periods_per_year
    ex = [r - rf for r in returns]
    sd = _safe_std(ex)
    if sd == 0:
        return 0.0
    return (_safe_mean(ex) / sd) * math.sqrt(periods_per_year)


def sortino_ratio(returns: list[float], rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    rf = rf_annual / periods_per_year
    ex = [r - rf for r in returns]
    downside = [min(0.0, r) for r in ex]
    dd = _safe_std(downside)
    if dd == 0:
        return 0.0
    return (_safe_mean(ex) / dd) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: list[float]) -> float:
    peak = 0.0
    mdd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        if peak <= 0:
            continue
        dd = (peak - e) / peak
        mdd = max(mdd, dd)
    return mdd


def calmar_ratio(equity_curve: list[float], periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2:
        return 0.0
    start = equity_curve[0]
    end = equity_curve[-1]
    if start <= 0:
        return 0.0
    years = max(len(equity_curve) / periods_per_year, 1 / periods_per_year)
    cagr = (end / start) ** (1 / years) - 1
    mdd = max_drawdown(equity_curve)
    if mdd <= 0:
        return 0.0
    return cagr / mdd


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    closed = [t for t in trades if t.get("realized_pnl") is not None]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if float(t.get("realized_pnl", 0.0)) > 0)
    return wins / len(closed)


def turnover(trades: list[dict], avg_equity: float) -> float:
    if avg_equity <= 0:
        return 0.0
    gross = sum(abs(float(t.get("notional", 0.0))) for t in trades)
    return gross / avg_equity
