from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitWindow:
    train_dates: set[pd.Timestamp]
    val_dates: set[pd.Timestamp]
    test_dates: set[pd.Timestamp]


class CrossSectionalMLPipeline:
    def __init__(
        self,
        dataset_path: Path,
        signals_path: Path,
        summary_path: Path,
        config: dict,
    ) -> None:
        self.dataset_path = dataset_path
        self.signals_path = signals_path
        self.summary_path = summary_path
        self.cfg = config

    def run(self) -> dict:
        df = self._load_dataset()
        if df.empty:
            result = {"status": "skipped", "reason": "empty_dataset"}
            self._write_outputs(result, {"generated_at": self._now(), "dates": []})
            return result

        feature_cols = self._feature_columns(df)
        splits = self._rolling_splits(pd.Series(sorted(df["date_t"].unique())))
        if not splits:
            result = {"status": "skipped", "reason": "insufficient_dates"}
            self._write_outputs(result, {"generated_at": self._now(), "dates": []})
            return result

        fold_rows: list[dict] = []
        all_test_rows: list[pd.DataFrame] = []
        best_global = None

        for idx, split in enumerate(splits, start=1):
            fold_df = self._run_fold(df, feature_cols, split, idx)
            fold_rows.append(fold_df["summary"])
            all_test_rows.append(fold_df["test_predictions"])
            if best_global is None or fold_df["summary"]["val_sharpe"] > best_global["val_sharpe"]:
                best_global = fold_df["summary"]

        test_all = pd.concat(all_test_rows, ignore_index=True)
        signals = self._build_signals(test_all)
        summary = self._build_summary(fold_rows, test_all, feature_cols, best_global)

        self._write_outputs(summary, signals)
        return summary

    def _load_dataset(self) -> pd.DataFrame:
        path = self.dataset_path
        if not path.exists() and path.suffix == ".parquet":
            alt = path.with_suffix(".jsonl")
            if alt.exists():
                path = alt
        if not path.exists():
            return pd.DataFrame()

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            rows = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
            df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date_t"] = pd.to_datetime(df["date_t"], utc=True, errors="coerce")
        df = df.dropna(subset=["date_t", "symbol", "target_excess_return_t_plus_1"]).copy()
        return df

    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list[str]:
        excluded = {
            "date_t",
            "symbol",
            "benchmark",
            "bench_symbol",
            "symbol_next_close",
            "symbol_return_t_plus_1",
            "benchmark_return_t_plus_1",
            "target_excess_return_t_plus_1",
        }
        cols = [c for c in df.columns if c not in excluded]
        def _is_numeric(dtype) -> bool:
            try:
                return np.issubdtype(dtype, np.number)
            except TypeError:
                return False
        return [c for c in cols if _is_numeric(df[c].dtype)]

    def _rolling_splits(self, dates: pd.Series) -> list[SplitWindow]:
        dates = pd.to_datetime(dates, utc=True).sort_values().reset_index(drop=True)
        if dates.empty:
            return []

        split_cfg = self.cfg.get("split", {})
        train_years = int(split_cfg.get("train_years", 10))
        val_years = int(split_cfg.get("val_years", 1))
        test_years = int(split_cfg.get("test_years", 1))

        weeks_per_year = 52
        train_w = train_years * weeks_per_year
        val_w = val_years * weeks_per_year
        test_w = test_years * weeks_per_year

        total = len(dates)
        need = train_w + val_w + test_w
        if total < need:
            train_w = max(26, int(total * 0.6))
            val_w = max(8, int(total * 0.2))
            test_w = max(8, total - train_w - val_w)

        splits: list[SplitWindow] = []
        step = max(4, test_w // 4)
        start = 0
        while True:
            a = start
            b = a + train_w
            c = b + val_w
            d = c + test_w
            if d > total:
                break
            splits.append(
                SplitWindow(
                    train_dates=set(dates.iloc[a:b].tolist()),
                    val_dates=set(dates.iloc[b:c].tolist()),
                    test_dates=set(dates.iloc[c:d].tolist()),
                )
            )
            start += step

        if not splits and total >= 24:
            b = max(8, int(total * 0.6))
            c = max(b + 4, int(total * 0.8))
            splits.append(
                SplitWindow(
                    train_dates=set(dates.iloc[:b].tolist()),
                    val_dates=set(dates.iloc[b:c].tolist()),
                    test_dates=set(dates.iloc[c:].tolist()),
                )
            )
        return splits

    def _run_fold(self, df: pd.DataFrame, feature_cols: list[str], split: SplitWindow, fold_id: int) -> dict:
        train = df[df["date_t"].isin(split.train_dates)].copy()
        val = df[df["date_t"].isin(split.val_dates)].copy()
        test = df[df["date_t"].isin(split.test_dates)].copy()

        models = self._candidate_models()
        best = None
        for name, model, with_interactions in models:
            x_train, x_val, x_test, cols = self._prepare_features(train, val, test, feature_cols, with_interactions)
            y_train = train["target_excess_return_t_plus_1"].values
            y_val = val["target_excess_return_t_plus_1"].values
            y_test = test["target_excess_return_t_plus_1"].values

            fitted = model.fit(x_train, y_train)
            val_pred = fitted.predict(x_val)
            test_pred = fitted.predict(x_test)
            val_stats = self._ranking_metrics(val.assign(pred=val_pred))
            test_stats = self._ranking_metrics(test.assign(pred=test_pred))
            row = {
                "model": name,
                "rmse_val": float(math.sqrt(mean_squared_error(y_val, val_pred))),
                "val_sharpe": float(val_stats["sharpe"]),
                "val_hit_rate": float(val_stats["hit_rate_top_vs_bottom"]),
                "test_sharpe": float(test_stats["sharpe"]),
                "test_hit_rate": float(test_stats["hit_rate_top_vs_bottom"]),
                "test_turnover": float(test_stats["turnover"]),
                "test_returns": test_stats["returns"],
                "test_pred": test_pred,
                "feature_columns": cols,
            }
            if best is None or row["val_sharpe"] > best["val_sharpe"]:
                best = row

        assert best is not None
        test_predictions = test[["date_t", "symbol", "target_excess_return_t_plus_1"]].copy()
        test_predictions["predicted_excess_return"] = best["test_pred"]

        summary = {
            "fold": fold_id,
            "model": best["model"],
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "val_sharpe": round(best["val_sharpe"], 6),
            "test_sharpe": round(best["test_sharpe"], 6),
            "val_hit_rate": round(best["val_hit_rate"], 6),
            "test_hit_rate": round(best["test_hit_rate"], 6),
            "test_turnover": round(best["test_turnover"], 6),
            "test_sortino": round(self._sortino(best["test_returns"]), 6),
            "test_max_drawdown": round(self._max_drawdown(best["test_returns"]), 6),
        }
        return {"summary": summary, "test_predictions": test_predictions}

    def _candidate_models(self):
        model_cfg = self.cfg.get("models", {})
        models = []
        if bool(model_cfg.get("elasticnet", True)):
            models.append(("elastic_net", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.02, l1_ratio=0.5, random_state=42, max_iter=5000)),
            ]), True))
        if bool(model_cfg.get("random_forest", True)):
            models.append(("random_forest", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)),
            ]), False))
        if bool(model_cfg.get("mlp", True)):
            models.append(("mlp", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True, max_iter=350, random_state=42)),
            ]), False))
        if bool(model_cfg.get("xgboost", False)):
            try:
                from xgboost import XGBRegressor

                models.append(("xgboost", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=42,
                        objective="reg:squarederror",
                    )),
                ]), False))
            except Exception:
                pass
        if bool(model_cfg.get("lightgbm", False)):
            try:
                from lightgbm import LGBMRegressor

                models.append(("lightgbm", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LGBMRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=42,
                    )),
                ]), False))
            except Exception:
                pass
        if not models:
            models.append(("elastic_net", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.02, l1_ratio=0.5, random_state=42, max_iter=5000)),
            ]), True))
        return models

    @staticmethod
    def _prepare_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], add_interactions: bool):
        cols = list(feature_cols)
        if add_interactions:
            for left, right, name in [
                ("momentum_30d", "median_dollar_volume_30d", "interaction_mom30_liquidity"),
                ("volatility_30d", "beta", "interaction_vol_beta"),
            ]:
                if left in train.columns and right in train.columns:
                    for df in (train, val, test):
                        df[name] = df[left] * df[right]
                    cols.append(name)

        def to_float(df: pd.DataFrame) -> pd.DataFrame:
            """Cast tutte le feature a float64, rimuove colonne non castabili."""
            result = pd.DataFrame(index=df.index)
            for c in cols:
                if c not in df.columns:
                    result[c] = 0.0
                else:
                    try:
                        result[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").fillna(0.0)
                    except Exception:
                        result[c] = 0.0
            return result

        return to_float(train), to_float(val), to_float(test), cols

    def _build_signals(self, preds: pd.DataFrame) -> dict:
        preds = preds.copy()
        preds["date_t"] = pd.to_datetime(preds["date_t"], utc=True)
        dates = sorted(preds["date_t"].unique())
        by_date = {}
        for d in dates:
            block = preds[preds["date_t"] == d].sort_values("predicted_excess_return", ascending=False).reset_index(drop=True)
            if block.empty:
                continue
            n = len(block)
            dec = max(1, n // 10)
            top = block.head(dec)
            bottom = block.tail(dec)
            top_w = (top["predicted_excess_return"].clip(lower=0.0) + 1e-9)
            bot_w = (bottom["predicted_excess_return"].abs() + 1e-9)
            top = top.assign(weight=(top_w / top_w.sum()).round(8))
            bottom = bottom.assign(weight=(-(bot_w / bot_w.sum())).round(8))
            by_date[str(pd.Timestamp(d).date())] = {
                "top_decile_long": top[["symbol", "predicted_excess_return", "weight"]].to_dict("records"),
                "bottom_decile_short": bottom[["symbol", "predicted_excess_return", "weight"]].to_dict("records"),
            }

        latest_date = max(by_date.keys()) if by_date else None
        latest = by_date.get(latest_date, {"top_decile_long": [], "bottom_decile_short": []})
        return {
            "generated_at": self._now(),
            "runtime_mode": self.cfg.get("runtime_mode", "informative_only"),
            "latest_date": latest_date,
            "latest_ranking": latest,
            "history": by_date,
        }

    def _build_summary(self, folds: list[dict], preds: pd.DataFrame, feature_cols: list[str], best_global: dict | None) -> dict:
        returns = self._series_returns(preds)
        summary = {
            "status": "ok",
            "generated_at": self._now(),
            "folds": folds,
            "fold_count": len(folds),
            "models_tested": sorted({f["model"] for f in folds}),
            "dataset_rows": int(len(preds)),
            "feature_columns": feature_cols,
            "oos": {
                "sharpe": round(self._sharpe(returns), 6),
                "sortino": round(self._sortino(returns), 6),
                "max_drawdown": round(self._max_drawdown(returns), 6),
                "turnover": round(self._turnover(preds), 6),
                "returns_distribution": {
                    "count": len(returns),
                    "mean": round(float(np.mean(returns)) if returns else 0.0, 8),
                    "std": round(float(np.std(returns)) if returns else 0.0, 8),
                    "min": round(float(np.min(returns)) if returns else 0.0, 8),
                    "max": round(float(np.max(returns)) if returns else 0.0, 8),
                },
            },
            "best_model": best_global["model"] if best_global else None,
        }
        return summary

    def _write_outputs(self, summary: dict, signals: dict) -> None:
        self.signals_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.signals_path.write_text(json.dumps(signals, indent=2), encoding="utf-8")
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @staticmethod
    def _ranking_metrics(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"sharpe": 0.0, "hit_rate_top_vs_bottom": 0.0, "turnover": 0.0, "returns": []}
        n = len(df)
        dec = max(1, n // 10)
        block = df.sort_values("pred", ascending=False)
        top = block.head(dec)
        bottom = block.tail(dec)
        top_ret = float(top["target_excess_return_t_plus_1"].mean())
        bot_ret = float(bottom["target_excess_return_t_plus_1"].mean())
        spread = top_ret - bot_ret
        returns = [spread]
        hit = 1.0 if spread > 0 else 0.0
        return {
            "sharpe": CrossSectionalMLPipeline._sharpe(returns),
            "hit_rate_top_vs_bottom": hit,
            "turnover": float(2.0 * dec / max(1, n)),
            "returns": returns,
        }

    @staticmethod
    def _series_returns(preds: pd.DataFrame) -> list[float]:
        out = []
        for d, block in preds.groupby("date_t"):
            b = block.sort_values("predicted_excess_return", ascending=False)
            n = len(b)
            dec = max(1, n // 10)
            top = float(b.head(dec)["target_excess_return_t_plus_1"].mean())
            bot = float(b.tail(dec)["target_excess_return_t_plus_1"].mean())
            out.append(top - bot)
        return out

    @staticmethod
    def _turnover(preds: pd.DataFrame) -> float:
        prev = None
        changes = []
        for _, block in preds.groupby("date_t"):
            b = block.sort_values("predicted_excess_return", ascending=False)
            n = len(b)
            dec = max(1, n // 10)
            current = set(b.head(dec)["symbol"].tolist())
            if prev is not None:
                overlap = len(prev & current)
                changes.append(1.0 - overlap / max(1, len(current)))
            prev = current
        return float(np.mean(changes)) if changes else 0.0

    @staticmethod
    def _sharpe(returns: list[float]) -> float:
        if not returns:
            return 0.0
        arr = np.array(returns, dtype=float)
        sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
        if sd <= 0:
            return 0.0
        return float(arr.mean() / sd * math.sqrt(52))

    @staticmethod
    def _sortino(returns: list[float]) -> float:
        if not returns:
            return 0.0
        arr = np.array(returns, dtype=float)
        downside = arr[arr < 0]
        if len(downside) == 0:
            return float(arr.mean() * math.sqrt(52))
        dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
        if dd <= 0:
            return 0.0
        return float(arr.mean() / dd * math.sqrt(52))

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        if not returns:
            return 0.0
        eq = np.cumprod([1.0 + r for r in returns])
        peaks = np.maximum.accumulate(eq)
        dd = (eq - peaks) / peaks
        return float(abs(np.min(dd)))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
