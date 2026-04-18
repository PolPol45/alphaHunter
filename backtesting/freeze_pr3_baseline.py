"""
Freeze PR3 baseline artifacts for deterministic strategy comparisons.

Creates a timestamped snapshot containing:
- backtest report (golden baseline)
- effective config values (date range, universe, fees, slippage, seed)
- summary manifest for traceability
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _freeze_payload(config: dict[str, Any], report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    bt = config.get("backtesting", {}) if isinstance(config, dict) else {}
    fees = bt.get("fees", {}) if isinstance(bt, dict) else {}
    slip = bt.get("slippage_model", {}) if isinstance(bt, dict) else {}
    universe = bt.get("universe_snapshot", []) if isinstance(bt, dict) else []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}

    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "config_snapshot": {
            "start_date": bt.get("start_date"),
            "end_date": bt.get("end_date"),
            "universe_size": len(universe) if isinstance(universe, list) else None,
            "universe_snapshot": universe if isinstance(universe, list) else [],
            "fees": fees if isinstance(fees, dict) else {},
            "slippage_model": slip if isinstance(slip, dict) else {},
            "seed": bt.get("seed"),
        },
        "report_kpi": {
            "start_equity": summary.get("start_equity"),
            "final_equity": summary.get("final_equity"),
            "total_return_pct": (
                ((float(summary.get("final_equity", 0.0)) / float(summary.get("start_equity", 1.0))) - 1.0) * 100.0
                if float(summary.get("start_equity", 0.0) or 0.0) > 0
                else None
            ),
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "max_drawdown": metrics.get("max_drawdown"),
            "calmar": metrics.get("calmar"),
            "turnover": metrics.get("turnover"),
            "profit_factor": metrics.get("profit_factor"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze PR3 golden baseline artifacts")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--report", required=True, help="Path to backtest report json")
    parser.add_argument("--out-dir", default="reports/pr3_golden", help="Output folder for frozen artifacts")
    parser.add_argument("--label", default="", help="Optional custom label")
    args = parser.parse_args()

    config_path = Path(args.config)
    report_path = Path(args.report)
    out_dir = Path(args.out_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    config = _read_json(config_path)
    report = _read_json(report_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label.strip() or "golden"
    prefix = f"pr3_baseline_{label}_{stamp}"

    frozen_payload = _freeze_payload(config, report, report_path)
    baseline_meta = out_dir / f"{prefix}.meta.json"
    baseline_report = out_dir / f"{prefix}.report.json"
    baseline_config = out_dir / f"{prefix}.config.json"

    _write_json(baseline_meta, frozen_payload)
    _write_json(baseline_report, report)
    _write_json(baseline_config, config)

    print("== PR3 Baseline Frozen ==")
    print(f"meta:   {baseline_meta}")
    print(f"report: {baseline_report}")
    print(f"config: {baseline_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
