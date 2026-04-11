"""
PR2 Quality Gate
================
Compare a candidate backtest report against PR1 baseline and enforce go/no-go targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TARGETS = {
    "max_drawdown": 0.18,   # lower is better
    "turnover": 110.0,      # lower is better
    "sharpe": 0.45,         # higher is better
    "min_return_pct": 0.04, # higher is better
}


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def _return_pct(report: dict) -> float:
    s = report.get("summary", {})
    start = float(s.get("start_equity", 0.0))
    end = float(s.get("final_equity", 0.0))
    if start <= 0:
        return 0.0
    return (end / start) - 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to PR1 baseline report JSON")
    parser.add_argument("--candidate", required=True, help="Path to PR2 candidate report JSON")
    args = parser.parse_args()

    baseline = _load_report(Path(args.baseline))
    candidate = _load_report(Path(args.candidate))

    bm = baseline.get("metrics", {})
    cm = candidate.get("metrics", {})

    b_ret = _return_pct(baseline)
    c_ret = _return_pct(candidate)

    checks = {
        "max_drawdown": float(cm.get("max_drawdown", 1.0)) < DEFAULT_TARGETS["max_drawdown"],
        "turnover": float(cm.get("turnover", 1e9)) < DEFAULT_TARGETS["turnover"],
        "sharpe": float(cm.get("sharpe", -1.0)) > DEFAULT_TARGETS["sharpe"],
        "return": c_ret >= DEFAULT_TARGETS["min_return_pct"],
    }

    print("== PR2 Quality Gate ==")
    print(f"Baseline return:  {b_ret*100:.2f}%")
    print(f"Candidate return: {c_ret*100:.2f}%")
    print("")
    print(f"Baseline metrics:  sharpe={float(bm.get('sharpe', 0.0)):.4f}  maxDD={float(bm.get('max_drawdown', 0.0))*100:.2f}%  turnover={float(bm.get('turnover', 0.0)):.2f}")
    print(f"Candidate metrics: sharpe={float(cm.get('sharpe', 0.0)):.4f}  maxDD={float(cm.get('max_drawdown', 0.0))*100:.2f}%  turnover={float(cm.get('turnover', 0.0)):.2f}")
    print("")
    print("Targets:")
    print(f"- max_drawdown < {DEFAULT_TARGETS['max_drawdown']*100:.2f}%  => {'PASS' if checks['max_drawdown'] else 'FAIL'}")
    print(f"- turnover < {DEFAULT_TARGETS['turnover']:.2f}               => {'PASS' if checks['turnover'] else 'FAIL'}")
    print(f"- sharpe > {DEFAULT_TARGETS['sharpe']:.2f}                  => {'PASS' if checks['sharpe'] else 'FAIL'}")
    print(f"- return >= {DEFAULT_TARGETS['min_return_pct']*100:.2f}%            => {'PASS' if checks['return'] else 'FAIL'}")

    passed = all(checks.values())
    print("")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
