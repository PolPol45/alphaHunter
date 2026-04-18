from __future__ import annotations

import json
import pathlib
import subprocess
import sys


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]


def test_pr34_gate_includes_profit_factor(tmp_path):
    baseline = {
        "metrics": {"max_drawdown": 0.25, "turnover": 100.0, "sharpe": 0.2, "calmar": 0.1, "profit_factor": 1.1},
        "strategy_breakdown": {"bear": {"trades": 1}, "crypto": {"trades": 1}},
    }
    candidate = {
        "metrics": {"max_drawdown": 0.2, "turnover": 105.0, "sharpe": 0.3, "calmar": 0.2, "profit_factor": 0.9},
        "strategy_breakdown": {"bear": {"trades": 2}, "crypto": {"trades": 2}},
    }
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(baseline), encoding="utf-8")
    c.write_text(json.dumps(candidate), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(BOT_DIR / "backtesting" / "pr34_quality_gate.py"), "--baseline", str(b), "--candidate", str(c)],
        capture_output=True,
        text=True,
        cwd=BOT_DIR,
    )
    assert proc.returncode != 0
    assert "profit_factor_ge_1: FAIL" in proc.stdout

