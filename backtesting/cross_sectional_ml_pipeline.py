"""
CrossSectionalMLPipeline — Advanced v2
======================================
Improvements over v1:
  - Cross-Sectional Rank (CSR) as training target (more stable, hedge-fund standard)
  - ElasticNet with GridSearchCV (alpha auto-tuned on val)
  - Ensemble stacking: ElasticNet + RF + MLP predictions → Ridge meta-model
  - Optional XGBoost / LightGBM (enabled in config)
  - IC-weighted signal construction (conviction-proportional weights)
  - Risk-Parity position sizing (equal volatility contribution)
  - Sector-neutral constraints (max 40% per sector)
  - Turnover penalty in signal construction
  - Seasonality features injected at dataset level
  - More extended interaction features
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from backtesting.montecarlo_simulator import MontecarloSimulator



@dataclass(frozen=True)
class SplitWindow:
    train_dates: set
    val_dates: set
    test_dates: set


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
        self._use_rank_target = bool(config.get("use_rank_target", True))
        self._use_stacking = bool(config.get("use_stacking", True))
        self._use_risk_parity = bool(config.get("use_risk_parity", True))
        self._sector_cap = float(config.get("sector_cap", 0.40))
        self._turnover_penalty = float(config.get("turnover_penalty", 0.002))
        self._ic_decay = float(config.get("ic_decay", 0.85))  # IC weighting decay per fold
        import logging
        self.logger = logging.getLogger("cross_sectional_ml_pipeline")

    # ─── PUBLIC ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        df = self._load_dataset()
        if df.empty:
            result = {"status": "skipped", "reason": "empty_dataset"}
            self._write_outputs(result, {"generated_at": self._now(), "dates": []})
            return result

        # Inject seasonality features
        df = self._add_seasonality_features(df)

        # Use rank target (CSE standard) instead of raw excess return
        if self._use_rank_target:
            df = self._add_rank_target(df)
            target_col = "target_rank"
        else:
            target_col = "target_excess_return_t_plus_1"

        feature_cols = self._feature_columns(df, target_col)
        splits = self._rolling_splits(pd.Series(sorted(df["date_t"].unique())))
        if not splits:
            result = {"status": "skipped", "reason": "insufficient_dates"}
            self._write_outputs(result, {"generated_at": self._now(), "dates": []})
            return result

        fold_rows: list[dict] = []
        all_test_rows: list[pd.DataFrame] = []
        fold_ic_scores: list[float] = []  # IC per fold for weighting
        best_global = None

        # --- FASE 13: Reinforcement Learning (PPO Agent) ---
        from backtesting.rl_ppo_agent import RLEnsemblePipeline
        rl_pipeline = RLEnsemblePipeline()

        for idx, split in enumerate(splits, start=1):
            fold_df = self._run_fold(df, feature_cols, split, idx, target_col, rl_pipeline)
            fold_rows.append(fold_df["summary"])
            all_test_rows.append(fold_df["test_predictions"])
            fold_ic_scores.append(fold_df["summary"].get("val_ic", 0.0))
            if best_global is None or fold_df["summary"]["val_sharpe"] > best_global["val_sharpe"]:
                best_global = fold_df["summary"]

            # Guardrail RL
            if "rl_returns" in fold_df:
                # We need benchmark returns for this fold. (Using simple mean market return)
                fold_test_df = df[df["date_t"].isin(split.test_dates)]
                bench_returns = fold_test_df.groupby("date_t")["target_excess_return_t_plus_1"].mean().tolist()
                rl_pipeline.evaluate_guardrail(fold_df["rl_returns"], bench_returns)

        test_all = pd.concat(all_test_rows, ignore_index=True)
        # Build IC-weighted signals
        signals = self._build_signals_ic_weighted(test_all, fold_ic_scores, df)
        summary, mc_results = self._build_summary(fold_rows, test_all, feature_cols, best_global)

        self._write_outputs(summary, signals, mc_results)
        return summary

    # ─── DATA LOADING ─────────────────────────────────────────────────────────

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

    # ─── FEATURE ENGINEERING ─────────────────────────────────────────────────

    @staticmethod
    def _add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
        """Inject month/week dummies, options expiry flag, Q-end flag."""
        df = df.copy()
        dt = df["date_t"]
        df["season_month"] = dt.dt.month
        df["season_week"] = dt.dt.isocalendar().week.astype(int)
        df["season_day_of_week"] = dt.dt.dayofweek
        # January effect
        df["season_is_january"] = (dt.dt.month == 1).astype(int)
        # Options expiry: 3rd Friday of month, proxy = week 3 of month
        df["season_is_expiry_week"] = ((dt.dt.day >= 15) & (dt.dt.day <= 21) & (dt.dt.dayofweek == 4)).astype(int)
        # Quarter end
        df["season_is_quarter_end"] = dt.dt.month.isin([3, 6, 9, 12]).astype(int)
        # Sin/cos encoding for cyclical features
        df["season_month_sin"] = np.sin(2 * np.pi * df["season_month"] / 12)
        df["season_month_cos"] = np.cos(2 * np.pi * df["season_month"] / 12)
        df["season_week_sin"] = np.sin(2 * np.pi * df["season_week"] / 52)
        df["season_week_cos"] = np.cos(2 * np.pi * df["season_week"] / 52)
        return df

    @staticmethod
    def _add_rank_target(df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw excess return to cross-sectional percentile rank per date (0-1).
        Rank target is more robust: removes absolute return bias and focuses model
        on relative ordering — the actual signal we care about."""
        df = df.copy()
        df["target_rank"] = (
            df.groupby("date_t")["target_excess_return_t_plus_1"]
            .rank(pct=True, method="average")
        )
        return df

    # 15 features with highest empirical IC in equity/crypto cross-sectional models
    CORE_FEATURES = [
        "momentum_30d",
        "momentum_7d",
        "momentum_90d",
        "volatility_30d",
        "rsi_14d",
        "momentum_zscore_30d",
        "pct_from_52w_high",
        "macro_vix",
        "macro_market_bias",
        "macro_regime_numeric",
        "amihud_30d",
        "avg_volume_30d",
        "alt_fear_greed",
        "alt_btc_bid_ask_imbalance",
        "sentiment_market",
    ]

    def _feature_columns(self, df: pd.DataFrame, target_col: str) -> list[str]:
        excluded = {
            "date_t", "symbol", "benchmark", "bench_symbol",
            "symbol_next_close", "symbol_return_t_plus_1",
            "benchmark_return_t_plus_1", "target_excess_return_t_plus_1",
            "target_rank", "sector",
        }
        excluded.add(target_col)

        # Use config whitelist if provided, else CORE_FEATURES
        cfg_features = self.cfg.get("selected_features", [])
        whitelist = cfg_features if cfg_features else self.CORE_FEATURES

        def _is_numeric(dtype) -> bool:
            try:
                return np.issubdtype(dtype, np.number)
            except TypeError:
                return False

        available = {c for c in df.columns if c not in excluded and _is_numeric(df[c].dtype)}
        selected = [f for f in whitelist if f in available]

        # Fallback: if fewer than 5 whitelisted features exist, use all numeric
        if len(selected) < 5:
            selected = [c for c in df.columns if c not in excluded and _is_numeric(df[c].dtype)]

        return selected

    # ─── ROLLING SPLITS ───────────────────────────────────────────────────────

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
            a, b, c, d = start, start + train_w, start + train_w + val_w, start + train_w + val_w + test_w
            if d > total:
                break
            splits.append(SplitWindow(
                train_dates=set(dates.iloc[a:b].tolist()),
                val_dates=set(dates.iloc[b:c].tolist()),
                test_dates=set(dates.iloc[c:d].tolist()),
            ))
            start += step

        if not splits and total >= 24:
            b = max(8, int(total * 0.6))
            c = max(b + 4, int(total * 0.8))
            splits.append(SplitWindow(
                train_dates=set(dates.iloc[:b].tolist()),
                val_dates=set(dates.iloc[b:c].tolist()),
                test_dates=set(dates.iloc[c:].tolist()),
            ))
        return splits

    # ─── FOLD EXECUTION ───────────────────────────────────────────────────────

    def _run_fold(self, df: pd.DataFrame, feature_cols: list[str],
                  split: SplitWindow, fold_id: int, target_col: str, rl_pipeline) -> dict:
        train = df[df["date_t"].isin(split.train_dates)].copy()
        val = df[df["date_t"].isin(split.val_dates)].copy()
        test = df[df["date_t"].isin(split.test_dates)].copy()

        # FASE 13: Allena l'Agente RL sul train fold attuale!
        if rl_pipeline and rl_pipeline.is_active:
            # Training PPO Agent (Gymnasium Env)
            rl_pipeline.train_ppo(train)

        candidate_models = self._candidate_models()
        base_predictions_val: dict[str, np.ndarray] = {}
        base_predictions_test: dict[str, np.ndarray] = {}
        fold_results = []
        best = None

        for name, model, with_interactions in candidate_models:
            x_train, x_val, x_test, cols = self._prepare_features(
                train, val, test, feature_cols, with_interactions
            )
            y_train = train[target_col].values
            y_val = val[target_col].values
            y_test = test[target_col].values

            try:
                fitted = model.fit(x_train, y_train)
                val_pred = fitted.predict(x_val)
                test_pred = fitted.predict(x_test)
            except Exception:
                continue

            val_stats = self._ranking_metrics(val.assign(pred=val_pred))
            test_stats = self._ranking_metrics(test.assign(pred=test_pred))

            # IC (Information Coefficient) = Spearman rank correlation
            val_ic = self._information_coefficient(y_val, val_pred)

            row = {
                "model": name,
                "fitted_pipeline": fitted,
                "rmse_val": float(math.sqrt(mean_squared_error(y_val, val_pred))),
                "val_sharpe": float(val_stats["sharpe"]),
                "val_ic": float(val_ic),
                "val_hit_rate": float(val_stats["hit_rate_top_vs_bottom"]),
                "test_sharpe": float(test_stats["sharpe"]),
                "test_hit_rate": float(test_stats["hit_rate_top_vs_bottom"]),
                "test_turnover": float(test_stats["turnover"]),
                "test_returns": test_stats["returns"],
                "test_pred": test_pred,
                "feature_columns": cols,
            }
            base_predictions_val[name] = val_pred
            base_predictions_test[name] = test_pred
            fold_results.append(row)

            if best is None or row["val_sharpe"] > best["val_sharpe"]:
                best = row

        # ── Ensemble Stacking ──────────────────────────────────────────────
        if self._use_stacking and len(base_predictions_val) >= 2:
            val_stack_X = np.column_stack(list(base_predictions_val.values()))
            test_stack_X = np.column_stack(list(base_predictions_test.values()))
            y_val = val[target_col].values
            y_test = test[target_col].values

            meta = Ridge(alpha=1.0)
            try:
                meta.fit(val_stack_X, y_val)
                stack_val_pred = meta.predict(val_stack_X)
                stack_test_pred = meta.predict(test_stack_X)
                stack_val_stats = self._ranking_metrics(val.assign(pred=stack_val_pred))
                stack_test_stats = self._ranking_metrics(test.assign(pred=stack_test_pred))
                stack_ic = self._information_coefficient(y_val, stack_val_pred)
                stack_row = {
                    "model": "stacked_ensemble",
                    "rmse_val": float(math.sqrt(mean_squared_error(y_val, stack_val_pred))),
                    "val_sharpe": float(stack_val_stats["sharpe"]),
                    "val_ic": float(stack_ic),
                    "val_hit_rate": float(stack_val_stats["hit_rate_top_vs_bottom"]),
                    "test_sharpe": float(stack_test_stats["sharpe"]),
                    "test_hit_rate": float(stack_test_stats["hit_rate_top_vs_bottom"]),
                    "test_turnover": float(stack_test_stats["turnover"]),
                    "test_returns": stack_test_stats["returns"],
                    "test_pred": stack_test_pred,
                    "feature_columns": list(base_predictions_val.keys()),
                }
                fold_results.append(stack_row)
                if best is None or stack_row["val_sharpe"] > best["val_sharpe"]:
                    best = stack_row
            except Exception:
                pass

        if best is None:
            # Emergency fallback
            x_train, x_val, x_test, cols = self._prepare_features(
                train, val, test, feature_cols, False
            )
            y_train = train[target_col].values
            fb = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("m", ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=5000)),
            ])
            fb.fit(x_train, y_train)
            best = {
                "model": "fallback_elastic_net",
                "val_sharpe": 0.0,
                "val_ic": 0.0,
                "val_hit_rate": 0.0,
                "test_sharpe": 0.0,
                "test_hit_rate": 0.0,
                "test_turnover": 0.0,
                "test_returns": [],
                "test_pred": fb.predict(x_test),
                "feature_columns": cols,
            }

        test_predictions = test[["date_t", "symbol", "target_excess_return_t_plus_1"]].copy()
        
        # --- FASE 13: ENSEMBLE SUPERVISED + REINFORCEMENT LEARNING ---
        base_preds = best["test_pred"]
        if rl_pipeline and rl_pipeline.is_active:
            rl_preds = rl_pipeline.predict(test)
            # Normalizziamo le scale per l'Ensemble (entrambi dovrebbero essere centrati sullo 0)
            supervised_w = 1.0 - rl_pipeline.ensemble_weight
            rl_w = rl_pipeline.ensemble_weight
            
            ensemble_pred = (base_preds * supervised_w) + (rl_preds * rl_w)
            test_predictions["predicted_excess_return"] = ensemble_pred
            test_predictions["rl_raw_action"] = rl_preds
            
            # Simulated RL return to trigger guardrail
            rl_returns = (rl_preds * test["target_excess_return_t_plus_1"].values)
        else:
            test_predictions["predicted_excess_return"] = base_preds
            rl_returns = []

        # --- FASE 2: META-LABELING ---
        # 1. Definiamo quale era la "verità", ovvero se il trade primario sarebbe stato profittevole.
        # Poiché usiamo target_rank o excess_return continui, noi vogliamo scommettere
        # che predizione positiva = rendimento positivo reale.
        x_train, x_val, x_test, _ = self._prepare_features(train, val, test, feature_cols, False)
        
        # Recupera le previsioni in validazione dal miglior modello base
        if best["model"] == "stacked_ensemble":
            val_preds_for_meta = stack_val_pred
        elif best["model"] == "fallback_elastic_net":
            val_preds_for_meta = fb.predict(x_val)
        else:
            val_preds_for_meta = base_predictions_val.get(best["model"], None)

        if val_preds_for_meta is not None:
            # Crea feature addizionali per il meta-modello: le features originali + la primary prediction
            X_meta_val = np.column_stack((x_val, val_preds_for_meta))
            X_meta_test = np.column_stack((x_test, best["test_pred"]))
            
            # Target Meta: successo = (valore reale > mediana) == (predizione > mediana)
            y_val_real = val["target_excess_return_t_plus_1"].values
            val_real_median = np.median(y_val_real)
            val_pred_median = np.median(val_preds_for_meta)
            
            # 1 se la direzione predetta correla con la spinta reale
            meta_y_val = ((y_val_real > val_real_median) == (val_preds_for_meta > val_pred_median)).astype(int)
            
            try:
                meta_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                meta_clf.fit(X_meta_val, meta_y_val)
                # Probabilità di successo (classe 1)
                meta_probs = meta_clf.predict_proba(X_meta_test)[:, 1]
            except Exception:
                # Fallback: nessuna opinione (50%)
                meta_probs = np.full(len(test), 0.5)
        else:
            meta_probs = np.full(len(test), 0.5)

        test_predictions["meta_confidence"] = meta_probs

        # Also store original raw return for IC computation
        if "target_rank" in test.columns:
            test_predictions["target_rank"] = test["target_rank"].values
        if "sector" in test.columns:
            test_predictions["sector"] = test["sector"].values
        if "volatility_30d" in test.columns:
            test_predictions["volatility_30d"] = test["volatility_30d"].values

        # --- FASE 12: Explainability & Drill-Down (SHAP Values) ---
        shap_explanations = ["N/A"] * len(test_predictions)
        try:
            import shap
            import matplotlib.pyplot as plt
            import os
            import json

            if best["model"] != "stacked_ensemble" and "fitted_pipeline" in best:
                pipeline = best["fitted_pipeline"]
                final_model = pipeline.named_steps.get("model")
                
                # Applica SHAP se il modello supporta la feature importance (es. Random Forest)
                if final_model is not None and hasattr(final_model, "feature_importances_"):
                    # Trasforma x_test tramite la pipeline (escluso il modello predittivo)
                    X_processed = x_test
                    for name, step in pipeline.steps[:-1]:
                        X_processed = step.transform(X_processed)
                    
                    explainer = shap.TreeExplainer(final_model)
                    # shap_values ha dimensione (num_campioni, num_features)
                    shap_values = explainer.shap_values(X_processed)
                    
                    # 1. Global Feature Importance Diagram
                    reports_dir = DATA_DIR.parent / "reports"
                    os.makedirs(str(reports_dir), exist_ok=True)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_processed, feature_names=best["feature_columns"], show=False)
                    plt.savefig(str(reports_dir / f"shap_summary_fold_{fold_id}.png"), bbox_inches="tight")
                    plt.close()
                    
                    # 2. Local Detailed Explanation per trade
                    for i in range(len(test_predictions)):
                        val_vector = shap_values[i]
                        feat_contrib = list(zip(best["feature_columns"], val_vector))
                        feat_contrib.sort(key=lambda x: x[1], reverse=True)
                        
                        top_pos = [f"{f} (+{v:.2f})" for f, v in feat_contrib[:3] if v > 0]
                        top_neg = [f"{f} ({v:.2f})" for f, v in feat_contrib[-3:] if v < 0]
                        
                        reason = "Perché: " + ", ".join(top_pos) if top_pos else "Perché: Nessun segnale netto."
                        if top_neg:
                            reason += " | Freno: " + ", ".join(top_neg)
                        
                        shap_explanations[i] = reason
                        
                    # 3. Concept Drift Detection
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    feat_imp = dict(zip(best["feature_columns"], mean_abs_shap.tolist()))
                    
                    drift_path = DATA_DIR / "shap_feat_imp.json"
                    if drift_path.exists():
                        with open(drift_path, "r") as df:
                            prev_imp = json.load(df)
                        for f_name, cur_val in feat_imp.items():
                            prev_val = float(prev_imp.get(f_name, 0.0))
                            if prev_val > 0.01 and cur_val < (prev_val * 0.1):
                                self.logger.critical(f"⚠️ CONCEPT DRIFT ALERT: Feature '{f_name}' importanza crollata da {prev_val:.3f} a {cur_val:.3f}!")
                                
                    with open(drift_path, "w") as df:
                        json.dump(feat_imp, df)
                        
        except Exception as e:
            self.logger.warning(f"Failed to generate SHAP explanations: {e}")
            
        test_predictions["shap_explanation"] = shap_explanations

        summary = {
            "fold": fold_id,
            "model": best["model"],
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "val_sharpe": round(best["val_sharpe"], 6),
            "val_ic": round(best.get("val_ic", 0.0), 6),
            "test_sharpe": round(best["test_sharpe"], 6),
            "val_hit_rate": round(best["val_hit_rate"], 6),
            "test_hit_rate": round(best["test_hit_rate"], 6),
            "test_turnover": round(best["test_turnover"], 6),
            "test_sortino": round(self._sortino(best["test_returns"]), 6),
            "test_max_drawdown": round(self._max_drawdown(best["test_returns"]), 6),
            "models_compared": [r["model"] for r in fold_results],
        }
        return {"summary": summary, "test_predictions": test_predictions, "rl_returns": rl_returns}

    # ─── MODELS ───────────────────────────────────────────────────────────────

    def _candidate_models(self):
        model_cfg = self.cfg.get("models", {})
        models = []

        if bool(model_cfg.get("elasticnet", True)):
            # Use ElasticNetCV for auto-tuned alpha
            models.append(("elastic_net_cv", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    alphas=[0.001, 0.01, 0.05, 0.1, 0.5],
                    cv=TimeSeriesSplit(n_splits=3),
                    max_iter=5000,
                    random_state=42,
                )),
            ]), True))

        if bool(model_cfg.get("random_forest", True)):
            models.append(("random_forest", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=300, max_depth=8,
                    min_samples_leaf=5, random_state=42, n_jobs=-1,
                )),
            ]), False))

        if bool(model_cfg.get("mlp", True)):
            models.append(("mlp", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    alpha=1e-3,
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    max_iter=500,
                    random_state=42,
                )),
            ]), False))

        if bool(model_cfg.get("xgboost", False)):
            try:
                from xgboost import XGBRegressor
                models.append(("xgboost", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(
                        n_estimators=400, max_depth=5, learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.85,
                        reg_lambda=1.0, reg_alpha=0.1,
                        random_state=42, objective="reg:squarederror",
                        eval_metric="rmse", verbosity=0,
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
                        n_estimators=400, max_depth=6, learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.85,
                        reg_lambda=1.0, reg_alpha=0.1,
                        random_state=42, verbosity=-1,
                    )),
                ]), False))
            except Exception:
                pass

        if not models:
            models.append(("elastic_net_fallback", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.02, l1_ratio=0.5, random_state=42, max_iter=5000)),
            ]), True))

        return models

    # ─── FEATURE PREPARATION ─────────────────────────────────────────────────

    @staticmethod
    def _prepare_features(train, val, test, feature_cols, add_interactions):
        cols = list(feature_cols)

        if add_interactions:
            interaction_pairs = [
                # Original pairs (numeric only)
                ("momentum_30d", "median_dollar_volume_30d", "ix_mom30_liq"),
                ("volatility_30d", "beta", "ix_vol_beta"),
                # Regime × Momentum
                ("macro_market_bias", "momentum_30d", "ix_regime_mom30"),
                ("macro_vix", "volatility_30d", "ix_vix_vol"),
                ("macro_regime_numeric", "momentum_90d", "ix_regime_mom90"),
                # Short/Long momentum divergence
                ("momentum_7d", "momentum_90d", "ix_short_long_mom"),
                # Quality × Momentum
                ("roe", "momentum_30d", "ix_roe_mom"),
                # VIX fear × volatility
                ("macro_vix_regime", "volatility_30d", "ix_vix_regime_vol"),
            ]
            for left, right, name in interaction_pairs:
                if left in train.columns and right in train.columns:
                    try:
                        for df in (train, val, test):
                            l_num = pd.to_numeric(df[left], errors="coerce").fillna(0.0)
                            r_num = pd.to_numeric(df[right], errors="coerce").fillna(0.0)
                            df[name] = l_num * r_num
                        cols.append(name)
                    except Exception:
                        pass  # skip broken interaction pairs silently

        def to_float(df):
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

    # ─── IC-WEIGHTED SIGNAL CONSTRUCTION ─────────────────────────────────────

    def _build_signals_ic_weighted(
        self, preds: pd.DataFrame, fold_ics: list[float], raw_df: pd.DataFrame
    ) -> dict:
        """
        IC-Weighted signal construction:
        - Weights proportional to predicted rank (not raw pred value)  
        - Risk-parity size: inverse volatility weighting
        - Sector neutrality: cap max sector allocation
        - Turnover memory: penalize position changes
        """
        preds = preds.copy()
        preds["date_t"] = pd.to_datetime(preds["date_t"], utc=True)

        # Merge volatility if available
        if "volatility_30d" not in preds.columns and "volatility_30d" in raw_df.columns:
            vol_map = raw_df.set_index(["date_t", "symbol"])["volatility_30d"].to_dict()
            preds["volatility_30d"] = [
                vol_map.get((r.date_t, r.symbol), 0.02)
                for r in preds.itertuples()
            ]

        # Merge sector if available
        if "sector" not in preds.columns and "sector" in raw_df.columns:
            sec_map = raw_df.set_index(["date_t", "symbol"])["sector"].to_dict()
            preds["sector"] = [
                sec_map.get((r.date_t, r.symbol), "Unknown")
                for r in preds.itertuples()
            ]

        dates = sorted(preds["date_t"].unique())
        by_date: dict = {}
        prev_longs: set = set()
        prev_shorts: set = set()

        for d in dates:
            block = preds[preds["date_t"] == d].copy()
            if block.empty:
                continue

            n = len(block)
            dec = max(1, n // 10)

            # Cross-sectional rank within this date
            block["cs_rank"] = block["predicted_excess_return"].rank(pct=True)
            block_sorted = block.sort_values("cs_rank", ascending=False)

            top = block_sorted.head(dec).copy()
            bottom = block_sorted.tail(dec).copy()

            # ── Risk-Parity sizing ────────────────────────────────────────
            if self._use_risk_parity and "volatility_30d" in top.columns:
                top_vol = top["volatility_30d"].clip(lower=1e-4)
                top["rp_weight"] = (1.0 / top_vol) / (1.0 / top_vol).sum()
                bot_vol = bottom["volatility_30d"].clip(lower=1e-4)
                bottom["rp_weight"] = (1.0 / bot_vol) / (1.0 / bot_vol).sum()
            else:
                top["rp_weight"] = 1.0 / len(top)
                bottom["rp_weight"] = 1.0 / len(bottom)

            # ── Sector constraint ─────────────────────────────────────────
            if "sector" in top.columns:
                top = self._apply_sector_cap(top, "rp_weight")
                bottom = self._apply_sector_cap(bottom, "rp_weight")

            # ── Meta-Labeling overlay (Fase 2) ──
            if "meta_confidence" in top.columns:
                top["meta_confidence"] = top["meta_confidence"].fillna(0.5)
                bottom["meta_confidence"] = bottom["meta_confidence"].fillna(0.5)
            else:
                top["meta_confidence"] = 0.5
                bottom["meta_confidence"] = 0.5

            # ── IC-scaling: scale by prediction conviction ────────────────
            top_rank_w = top["cs_rank"].clip(lower=0.5)  # only top half meaningful
            # Moltiplichiamo il peso di rischio per il rank di alpha e la confidenza del Meta-Model
            top["ic_weight"] = top["rp_weight"] * top_rank_w * top["meta_confidence"]
            top["ic_weight"] /= top["ic_weight"].sum()

            bot_rank_w = (1.0 - bottom["cs_rank"]).clip(lower=0.0) + 1e-9
            bottom["ic_weight"] = bottom["rp_weight"] * bot_rank_w * bottom["meta_confidence"]
            bottom["ic_weight"] /= bottom["ic_weight"].sum()

            # ── Turnover penalty: discount if same stock was in last period ─
            if self._turnover_penalty > 0:
                top["ic_weight"] *= top["symbol"].apply(
                    lambda s: 1.0 if s not in prev_longs else (1.0 - self._turnover_penalty)
                )
                top_sum = top["ic_weight"].sum()
                if top_sum > 0:
                    top["ic_weight"] /= top_sum

            prev_longs = set(top["symbol"].tolist())
            prev_shorts = set(bottom["symbol"].tolist())

            top_records = top[["symbol", "predicted_excess_return"]].copy()
            top_records["weight"] = top["ic_weight"].round(8)
            top_records["cs_rank"] = top["cs_rank"].round(4)
            if "meta_confidence" in top.columns:
                top_records["meta_confidence"] = top["meta_confidence"].round(4)
            if "shap_explanation" in top.columns:
                top_records["shap_explanation"] = top["shap_explanation"]

            bot_records = bottom[["symbol", "predicted_excess_return"]].copy()
            bot_records["weight"] = (-bottom["ic_weight"]).round(8)
            bot_records["cs_rank"] = bottom["cs_rank"].round(4)
            if "meta_confidence" in bottom.columns:
                bot_records["meta_confidence"] = bottom["meta_confidence"].round(4)
            if "shap_explanation" in bottom.columns:
                bot_records["shap_explanation"] = bottom["shap_explanation"]

            by_date[str(pd.Timestamp(d).date())] = {
                "top_decile_long": top_records.to_dict("records"),
                "bottom_decile_short": bot_records.to_dict("records"),
            }

        latest_date = max(by_date.keys()) if by_date else None
        latest = by_date.get(latest_date, {"top_decile_long": [], "bottom_decile_short": []})

        avg_ic = float(np.mean(fold_ics)) if fold_ics else 0.0

        return {
            "generated_at": self._now(),
            "runtime_mode": self.cfg.get("runtime_mode", "informative_only"),
            "target_type": "cross_sectional_rank" if self._use_rank_target else "excess_return",
            "ensemble_stacking": self._use_stacking,
            "risk_parity": self._use_risk_parity,
            "avg_ic_across_folds": round(avg_ic, 6),
            "latest_date": latest_date,
            "latest_ranking": latest,
            "history": by_date,
        }

    def _apply_sector_cap(self, df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
        """Rescale weights so no sector exceeds sector_cap."""
        df = df.copy()
        if "sector" not in df.columns:
            return df
        df[weight_col] = df[weight_col] / df[weight_col].sum()
        for _iter in range(10):
            sector_sums = df.groupby("sector")[weight_col].sum()
            over_cap = sector_sums[sector_sums > self._sector_cap]
            if over_cap.empty:
                break
            for sec, total in over_cap.items():
                mask = df["sector"] == sec
                df.loc[mask, weight_col] *= self._sector_cap / total
            total_w = df[weight_col].sum()
            if total_w > 0:
                df[weight_col] /= total_w
        return df

    # ─── SUMMARY & OUTPUT ─────────────────────────────────────────────────────

    def _build_summary(self, folds, preds, feature_cols, best_global):
        returns = self._series_returns(preds)
        ic_scores = [f.get("val_ic", 0.0) for f in folds]
        
        # --- Fase 3: Montecarlo Stress Testing ---
        mc = MontecarloSimulator(trials=5000, confidence_level=0.99)
        mc_results = mc.run_simulation(returns)

        summary = {
            "status": "ok",
            "generated_at": self._now(),
            "folds": folds,
            "fold_count": len(folds),
            "models_tested": sorted({f["model"] for f in folds}),
            "dataset_rows": int(len(preds)),
            "feature_columns": feature_cols,
            "avg_val_ic": round(float(np.mean(ic_scores)) if ic_scores else 0.0, 6),
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
                "montecarlo_stress_test": mc_results,
            },
            "best_model": best_global["model"] if best_global else None,
        }
        return summary, mc_results

    def _write_outputs(self, summary, signals, mc_results=None):
        self.signals_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.signals_path.write_text(json.dumps(signals, indent=2), encoding="utf-8")
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if mc_results:
            mc_path = self.signals_path.parent / "montecarlo_metrics.json"
            mc_path.write_text(json.dumps(mc_results, indent=2), encoding="utf-8")

    # ─── METRICS ─────────────────────────────────────────────────────────────

    @staticmethod
    def _information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Spearman rank correlation — the hedge-fund standard IC metric."""
        from scipy.stats import spearmanr
        if len(y_true) < 4:
            return 0.0
        try:
            corr, _ = spearmanr(y_true, y_pred)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

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
