from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelParams:
    pnl_weight: float
    win_weight: float
    turnover_weight: float
    trades_weight: float
    floor_weight: float

    def as_dict(self) -> dict:
        return {
            "pnl_weight": self.pnl_weight,
            "win_weight": self.win_weight,
            "turnover_weight": self.turnover_weight,
            "trades_weight": self.trades_weight,
            "floor_weight": self.floor_weight,
        }


class AutoMLBacktestPipeline:
    """
    Learn -> retrain -> deploy loop on historical backtest reports.

    Inputs:
    - backtest_report_*.json in trading_bot/reports
    - backtest_report_*.json found inside zip archives in workspace root

    Outputs:
    - trading_bot/data/learned_strategy_weights.json
    - training summary dict returned by run()
    """

    STRATS = ("bull", "bear", "crypto", "ml")

    def __init__(
        self,
        repo_dir: Path,
        workspace_dir: Path,
        min_train_reports: int = 2,
    ) -> None:
        self.repo_dir = repo_dir
        self.workspace_dir = workspace_dir
        self.reports_dir = repo_dir / "reports"
        self.data_dir = repo_dir / "data"
        self.min_train_reports = max(1, int(min_train_reports))

    def run(self) -> dict:
        reports = self._load_reports()
        if not reports:
            return {
                "status": "skipped",
                "reason": "no_backtest_reports_found",
                "reports_count": 0,
            }

        ordered = sorted(reports, key=self._report_sort_key)
        grid = self._candidate_grid()

        wf = self._walk_forward_select(ordered, grid)
        best_params = ModelParams(**wf["best_params"])

        final_weights = self._fit_weights(ordered, best_params)
        deploy_doc = {
            "version": "v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reports_count": len(ordered),
            "selected_model": best_params.as_dict(),
            "walk_forward": wf,
            "strategy_weights": final_weights,
        }

        self.data_dir.mkdir(parents=True, exist_ok=True)
        target = self.data_dir / "learned_strategy_weights.json"
        target.write_text(json.dumps(deploy_doc, indent=2), encoding="utf-8")
        return {"status": "ok", **deploy_doc}

    def _load_reports(self) -> list[dict]:
        out: list[dict] = []

        for fp in sorted(self.reports_dir.glob("backtest_report_*.json")):
            try:
                out.append(json.loads(fp.read_text(encoding="utf-8")))
            except Exception:
                continue

        for zip_fp in sorted(self.workspace_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(zip_fp) as zf:
                    for name in zf.namelist():
                        if "backtest_report_" not in name or not name.endswith(".json"):
                            continue
                        try:
                            obj = json.loads(zf.read(name).decode("utf-8"))
                        except Exception:
                            continue
                        out.append(obj)
            except Exception:
                continue

        unique: dict[tuple[str, str], dict] = {}
        for r in out:
            w = r.get("window", {})
            key = (str(w.get("start_date", "")), str(w.get("end_date", "")))
            if key == ("", ""):
                continue
            if key not in unique:
                unique[key] = r
        return list(unique.values())

    @staticmethod
    def _report_sort_key(report: dict) -> tuple:
        w = report.get("window", {})
        return (str(w.get("start_date", "")), str(w.get("end_date", "")))

    def _candidate_grid(self) -> list[ModelParams]:
        grid: list[ModelParams] = []
        for pnl_w in (0.35, 0.45, 0.55):
            for win_w in (0.25, 0.35, 0.45):
                for turn_w in (0.10, 0.20, 0.30):
                    tr_w = round(1.0 - pnl_w - win_w - turn_w, 2)
                    if tr_w < 0.05:
                        continue
                    grid.append(
                        ModelParams(
                            pnl_weight=pnl_w,
                            win_weight=win_w,
                            turnover_weight=turn_w,
                            trades_weight=tr_w,
                            floor_weight=0.15,
                        )
                    )
        return grid

    def _walk_forward_select(self, reports: list[dict], grid: list[ModelParams]) -> dict:
        if len(reports) <= self.min_train_reports:
            # fallback on default params when there is not enough folds
            best = ModelParams(0.45, 0.35, 0.15, 0.05, 0.15)
            return {
                "folds": 0,
                "model_scores": {json.dumps(best.as_dict(), sort_keys=True): 0.0},
                "best_params": best.as_dict(),
            }

        scores: dict[str, float] = {}
        folds = 0
        for i in range(self.min_train_reports, len(reports)):
            train = reports[:i]
            valid = reports[i]
            folds += 1
            for params in grid:
                weights = self._fit_weights(train, params)
                s = self._evaluate_weights(weights, valid)
                key = json.dumps(params.as_dict(), sort_keys=True)
                scores[key] = scores.get(key, 0.0) + s

        best_key = max(scores, key=scores.get)
        best_params_dict = json.loads(best_key)
        best = ModelParams(**best_params_dict)
        return {"folds": folds, "model_scores": scores, "best_params": best.as_dict()}

    def _fit_weights(self, reports: list[dict], params: ModelParams) -> dict:
        accum = {s: 0.0 for s in self.STRATS}
        for report in reports:
            feat = self._strategy_features(report)
            for s in self.STRATS:
                pnl_s = feat[s]["pnl_signal"]
                win_s = feat[s]["win_signal"]
                turn_s = feat[s]["turnover_signal"]
                tr_s = feat[s]["trades_signal"]
                util = (
                    params.pnl_weight * pnl_s
                    + params.win_weight * win_s
                    + params.turnover_weight * turn_s
                    + params.trades_weight * tr_s
                )
                accum[s] += util

        # softmax with floor so no strategy gets to zero
        exps = {k: math.exp(v) for k, v in accum.items()}
        den = sum(exps.values()) or 1.0
        raw = {k: exps[k] / den for k in self.STRATS}
        floored = {k: max(params.floor_weight, raw[k]) for k in self.STRATS}
        total = sum(floored.values()) or 1.0
        normalized = {k: floored[k] / total for k in self.STRATS}

        # Convert normalized weights around multiplier 1.0
        # 1/3 => neutral, >1/3 overweight, <1/3 underweight
        multipliers = {}
        neutral = 1.0 / len(self.STRATS)
        for k, w in normalized.items():
            tilt = (w - neutral) / neutral  # approx [-1,+1]
            m = 1.0 + 0.35 * tilt
            multipliers[k] = round(max(0.7, min(1.3, m)), 4)
        return multipliers

    def _evaluate_weights(self, weights: dict, report: dict) -> float:
        feat = self._strategy_features(report)
        # target = per-strategy pnl efficiency proxy
        score = 0.0
        for s in self.STRATS:
            target = feat[s]["pnl_signal"] * 0.6 + feat[s]["win_signal"] * 0.4
            score += weights.get(s, 1.0) * target

        m = report.get("metrics", {})
        max_dd = float(m.get("max_drawdown", 0.0) or 0.0)
        turn = float(m.get("turnover", 0.0) or 0.0)
        penalty = (max_dd * 0.6) + (min(turn, 200.0) / 200.0) * 0.2
        return score - penalty

    def _strategy_features(self, report: dict) -> dict:
        start_eq = float(report.get("summary", {}).get("start_equity", 1.0) or 1.0)
        sb = report.get("strategy_breakdown", {}) or {}

        out = {}
        for s in self.STRATS:
            row = sb.get(s, {}) or {}
            pnl = float(row.get("realized_pnl", 0.0) or 0.0)
            win = float(row.get("win_rate", 0.5) or 0.5)
            turn = float(row.get("turnover", 0.0) or 0.0)
            trades = float(row.get("trades", 0.0) or 0.0)

            pnl_rate = pnl / max(start_eq, 1.0)
            pnl_signal = max(-1.0, min(1.0, pnl_rate * 20.0))
            win_signal = max(0.0, min(1.0, win))
            turnover_signal = max(0.0, min(1.0, 1.0 - (turn / 3.0)))
            trades_signal = max(0.0, min(1.0, trades / 250.0))

            out[s] = {
                "pnl_signal": pnl_signal,
                "win_signal": win_signal,
                "turnover_signal": turnover_signal,
                "trades_signal": trades_signal,
            }
        return out
