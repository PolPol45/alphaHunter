#!/usr/bin/env python3
"""Generate delta report between current and previous backtest reports.

Outputs:
- JSON delta payload for dashboards/automation
- Markdown report for GitHub Step Summary and human review
- Gate JSON with pass/fail and violation reasons
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _report_stats(report: dict[str, Any]) -> dict[str, float]:
    summary = report.get("summary", {}) or {}
    metrics = report.get("metrics", {}) or {}

    start_equity = _safe_num(summary.get("start_equity"), 0.0)
    final_equity = _safe_num(summary.get("final_equity"), 0.0)
    pnl = final_equity - start_equity
    ret_pct = (pnl / start_equity * 100.0) if start_equity else 0.0

    return {
        "start_equity": start_equity,
        "final_equity": final_equity,
        "pnl": pnl,
        "return_pct": ret_pct,
        "trades": _safe_num(summary.get("trades"), 0.0),
        "sharpe": _safe_num(metrics.get("sharpe"), 0.0),
        "sortino": _safe_num(metrics.get("sortino"), 0.0),
        "max_drawdown": _safe_num(metrics.get("max_drawdown"), 0.0),
        "calmar": _safe_num(metrics.get("calmar"), 0.0),
        "win_rate": _safe_num(metrics.get("win_rate"), 0.0),
        "turnover": _safe_num(metrics.get("turnover"), 0.0),
    }


def _build_delta(current: dict[str, float], previous: dict[str, float]) -> dict[str, float]:
    return {
        "final_equity": current["final_equity"] - previous["final_equity"],
        "pnl": current["pnl"] - previous["pnl"],
        "return_pct_points": current["return_pct"] - previous["return_pct"],
        "trades": current["trades"] - previous["trades"],
        "sharpe": current["sharpe"] - previous["sharpe"],
        "sortino": current["sortino"] - previous["sortino"],
        "max_drawdown": current["max_drawdown"] - previous["max_drawdown"],
        "calmar": current["calmar"] - previous["calmar"],
        "win_rate": current["win_rate"] - previous["win_rate"],
        "turnover": current["turnover"] - previous["turnover"],
    }


def _gate(
    delta: dict[str, float],
    max_return_drop_pp: float,
    max_sharpe_drop: float,
    max_dd_increase: float,
) -> tuple[bool, list[str]]:
    violations: list[str] = []

    if delta["return_pct_points"] < -max_return_drop_pp:
        violations.append(
            f"Return drop too high: {delta['return_pct_points']:.2f}pp < -{max_return_drop_pp:.2f}pp"
        )
    if delta["sharpe"] < -max_sharpe_drop:
        violations.append(
            f"Sharpe drop too high: {delta['sharpe']:.3f} < -{max_sharpe_drop:.3f}"
        )
    if delta["max_drawdown"] > max_dd_increase:
        violations.append(
            f"Drawdown increase too high: +{delta['max_drawdown']:.4f} > {max_dd_increase:.4f}"
        )

    return len(violations) == 0, violations


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_sign(value: float, decimals: int = 2, suffix: str = "") -> str:
    return f"{value:+.{decimals}f}{suffix}"


def _build_markdown(
    current_name: str,
    previous_name: str,
    current: dict[str, float],
    previous: dict[str, float],
    delta: dict[str, float],
    gate_passed: bool,
    violations: list[str],
) -> str:
    lines: list[str] = []
    lines.append("## Backtest Delta Report")
    lines.append("")
    lines.append(f"Current: `{current_name}`")
    lines.append(f"Previous: `{previous_name}`")
    lines.append("")
    lines.append("| Metric | Current | Previous | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Final Equity | {_fmt_money(current['final_equity'])} | {_fmt_money(previous['final_equity'])} | {_fmt_sign(delta['final_equity'])} |"
    )
    lines.append(
        f"| PnL | {_fmt_money(current['pnl'])} | {_fmt_money(previous['pnl'])} | {_fmt_sign(delta['pnl'])} |"
    )
    lines.append(
        f"| Return % | {current['return_pct']:.2f}% | {previous['return_pct']:.2f}% | {_fmt_sign(delta['return_pct_points'], 2, 'pp')} |"
    )
    lines.append(
        f"| Sharpe | {current['sharpe']:.3f} | {previous['sharpe']:.3f} | {_fmt_sign(delta['sharpe'], 3)} |"
    )
    lines.append(
        f"| Sortino | {current['sortino']:.3f} | {previous['sortino']:.3f} | {_fmt_sign(delta['sortino'], 3)} |"
    )
    lines.append(
        f"| Max Drawdown | {current['max_drawdown']*100:.2f}% | {previous['max_drawdown']*100:.2f}% | {_fmt_sign(delta['max_drawdown']*100, 2, 'pp')} |"
    )
    lines.append(
        f"| Calmar | {current['calmar']:.3f} | {previous['calmar']:.3f} | {_fmt_sign(delta['calmar'], 3)} |"
    )
    lines.append(
        f"| Win Rate | {current['win_rate']*100:.2f}% | {previous['win_rate']*100:.2f}% | {_fmt_sign(delta['win_rate']*100, 2, 'pp')} |"
    )
    lines.append(
        f"| Turnover | {current['turnover']:.3f}x | {previous['turnover']:.3f}x | {_fmt_sign(delta['turnover'], 3)}x |"
    )
    lines.append(
        f"| Trades | {int(current['trades'])} | {int(previous['trades'])} | {_fmt_sign(delta['trades'], 0)} |"
    )
    lines.append("")

    status = "PASS" if gate_passed else "FAIL"
    lines.append(f"Quality Gate: **{status}**")

    if violations:
        lines.append("")
        lines.append("Violations:")
        for v in violations:
            lines.append(f"- {v}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate delta report and quality gate")
    parser.add_argument("--current", required=True, help="Current backtest report JSON path")
    parser.add_argument("--previous", required=True, help="Previous backtest report JSON path")
    parser.add_argument("--json-out", required=True, help="Output delta JSON path")
    parser.add_argument("--md-out", required=True, help="Output Markdown path")
    parser.add_argument("--gate-out", required=True, help="Output gate JSON path")
    parser.add_argument("--max-return-drop-pp", type=float, default=2.0)
    parser.add_argument("--max-sharpe-drop", type=float, default=0.25)
    parser.add_argument("--max-dd-increase", type=float, default=0.03)
    args = parser.parse_args()

    current_path = Path(args.current)
    previous_path = Path(args.previous)
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    gate_out = Path(args.gate_out)

    current_report = _load_json(current_path)
    previous_report = _load_json(previous_path)

    current_stats = _report_stats(current_report)
    previous_stats = _report_stats(previous_report)
    delta_stats = _build_delta(current_stats, previous_stats)

    gate_passed, violations = _gate(
        delta=delta_stats,
        max_return_drop_pp=args.max_return_drop_pp,
        max_sharpe_drop=args.max_sharpe_drop,
        max_dd_increase=args.max_dd_increase,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_report": current_path.name,
        "previous_report": previous_path.name,
        "current": current_stats,
        "previous": previous_stats,
        "delta": delta_stats,
        "quality_gate": {
            "passed": gate_passed,
            "thresholds": {
                "max_return_drop_pp": args.max_return_drop_pp,
                "max_sharpe_drop": args.max_sharpe_drop,
                "max_dd_increase": args.max_dd_increase,
            },
            "violations": violations,
        },
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_out.write_text(
        _build_markdown(
            current_name=current_path.name,
            previous_name=previous_path.name,
            current=current_stats,
            previous=previous_stats,
            delta=delta_stats,
            gate_passed=gate_passed,
            violations=violations,
        ),
        encoding="utf-8",
    )
    gate_out.write_text(
        json.dumps(
            {
                "passed": gate_passed,
                "violations": violations,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Delta JSON: {json_out}")
    print(f"Delta Markdown: {md_out}")
    print(f"Gate: {'PASS' if gate_passed else 'FAIL'}")

    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
