from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtesting.auto_ml_pipeline import AutoMLBacktestPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Train strategy weights from historical backtest reports and deploy.")
    parser.add_argument("--repo-dir", default=".", help="Path to trading_bot repo")
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Workspace path containing zip artifacts (defaults to parent of repo)",
    )
    parser.add_argument("--min-train-reports", type=int, default=2)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    workspace_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else repo_dir.parent

    pipeline = AutoMLBacktestPipeline(
        repo_dir=repo_dir,
        workspace_dir=workspace_dir,
        min_train_reports=args.min_train_reports,
    )
    out = pipeline.run()
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
