"""
PR3.4 quality gate.

Compares candidate report vs frozen baseline and enforces rollout KPI gates:
- max drawdown <= baseline
- turnover increase <= threshold
- sharpe >= baseline
- calmar >= baseline
- profit_factor >= 1
- bear trades > 0
- crypto trades > 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(doc: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float((doc.get("metrics", {}) or {}).get(key, default) or default)


def _strategy_trades(doc: dict[str, Any], key: str) -> int:
    br = doc.get("strategy_breakdown", {}) if isinstance(doc, dict) else {}
    return int((br.get(key, {}) or {}).get("trades", 0) or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Frozen baseline backtest report json")
    parser.add_argument("--candidate", required=True, help="Candidate backtest report json")
    parser.add_argument("--max-turnover-increase-pct", type=float, default=10.0, help="Allowed turnover increase vs baseline")
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    candidate = _load(Path(args.candidate))

    b_dd = _metric(baseline, "max_drawdown", 1.0)
    c_dd = _metric(candidate, "max_drawdown", 1.0)
    b_to = _metric(baseline, "turnover", 0.0)
    c_to = _metric(candidate, "turnover", 0.0)
    b_sh = _metric(baseline, "sharpe", -999.0)
    c_sh = _metric(candidate, "sharpe", -999.0)
    b_ca = _metric(baseline, "calmar", -999.0)
    c_ca = _metric(candidate, "calmar", -999.0)
    b_pf = _metric(baseline, "profit_factor", 0.0)
    c_pf = _metric(candidate, "profit_factor", 0.0)
    bear_trades = _strategy_trades(candidate, "bear")
    crypto_trades = _strategy_trades(candidate, "crypto")

    turnover_cap = b_to * (1.0 + max(0.0, args.max_turnover_increase_pct) / 100.0)
    checks = {
        "max_drawdown_le_baseline": c_dd <= b_dd,
        "turnover_increase_within_cap": c_to <= turnover_cap,
        "sharpe_ge_baseline": c_sh >= b_sh,
        "calmar_ge_baseline": c_ca >= b_ca,
        "profit_factor_ge_1": c_pf >= 1.0,
        "bear_trades_gt_zero": bear_trades > 0,
        "crypto_trades_gt_zero": crypto_trades > 0,
    }

    print("== PR3.4 Quality Gate ==")
    print(f"maxDD: baseline={b_dd:.4f} candidate={c_dd:.4f}")
    print(f"turnover: baseline={b_to:.4f} candidate={c_to:.4f} cap={turnover_cap:.4f}")
    print(f"sharpe: baseline={b_sh:.4f} candidate={c_sh:.4f}")
    print(f"calmar: baseline={b_ca:.4f} candidate={c_ca:.4f}")
    print(f"profit_factor: baseline={b_pf:.4f} candidate={c_pf:.4f}")
    print(f"bear trades: {bear_trades} | crypto trades: {crypto_trades}")
    print("")
    for key, ok in checks.items():
        print(f"- {key}: {'PASS' if ok else 'FAIL'}")

    passed = all(checks.values())
    print("")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
