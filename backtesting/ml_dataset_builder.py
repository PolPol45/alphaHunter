from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd


@dataclass
class BuildResult:
    rows: int
    symbols: int
    dates: int
    output_path: Path


class WeeklyMLDatasetBuilder:
    def __init__(
        self,
        features_path: Path,
        output_dataset_path: Path,
        output_meta_path: Path,
        benchmark_default: str = "SPY",
    ) -> None:
        self.features_path = features_path
        self.output_dataset_path = output_dataset_path
        self.output_meta_path = output_meta_path
        self.benchmark_default = benchmark_default

    def run(self) -> BuildResult:
        df = self._load_features()
        if df.empty:
            output_path = self._write_empty()
            return BuildResult(0, 0, 0, output_path)

        df = self._normalize(df)
        weekly = self._to_weekly(df)
        dataset = self._add_excess_return_target(weekly)
        dataset = dataset.dropna(subset=["target_excess_return_t_plus_1"])

        output_path = self._write_dataset(dataset)

        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(dataset)),
            "symbols": int(dataset["symbol"].nunique()) if not dataset.empty else 0,
            "dates": int(dataset["date_t"].nunique()) if not dataset.empty else 0,
            "benchmark_default": self.benchmark_default,
            "features_source": str(self.features_path.name),
            "dataset_path": output_path.name,
        }
        self.output_meta_path.write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")
        return BuildResult(meta["rows"], meta["symbols"], meta["dates"], output_path)

    def _load_features(self) -> pd.DataFrame:
        if not self.features_path.exists():
            return pd.DataFrame()
        if self.features_path.suffix == ".parquet":
            return pd.read_parquet(self.features_path)

        rows = []
        with open(self.features_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Se esiste già date_t (alias del seed), usala come date e droppa date_t
        # per evitare colonne duplicate dopo il rename date→date_t
        if "date_t" in out.columns and "date" not in out.columns:
            out = out.rename(columns={"date_t": "date"})
        elif "date_t" in out.columns:
            out = out.drop(columns=["date_t"])
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.dropna(subset=["date", "symbol", "close"]).copy()
        out["symbol"] = out["symbol"].astype(str)
        out["benchmark"] = out["benchmark"].fillna("SPY") if "benchmark" in out.columns else "SPY"
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        return out

    @staticmethod
    def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        # Includi target pre-calcolato nel resample così è disponibile come fallback
        exclude = {"symbol_return_t_plus_1", "benchmark_return_t_plus_1"}
        keep_cols = [c for c in df.columns if c not in exclude]
        weekly = (
            df[keep_cols]
            .groupby("symbol")
            .resample("W-FRI", on="date")
            .last()
            .reset_index()
        )
        weekly = weekly.dropna(subset=["close"]).copy()
        weekly = weekly.rename(columns={"date": "date_t"})
        return weekly

    def _add_excess_return_target(self, weekly: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        out = weekly.copy().sort_values(["symbol", "date_t"]).reset_index(drop=True)

        # Salva il target già presente nel seed (se disponibile)
        seed_target = out.get("target_excess_return_t_plus_1", None)
        seed_sym_ret = out.get("symbol_return_t_plus_1", None)

        # Ricalcola symbol return da prezzi consecutivi
        out["symbol_next_close"] = out.groupby("symbol")["close"].shift(-1)
        out["symbol_return_t_plus_1"] = (out["symbol_next_close"] / out["close"]) - 1.0

        # Se il ricalcolo produce NaN (ultima settimana), usa il valore del seed
        if seed_sym_ret is not None:
            mask = out["symbol_return_t_plus_1"].isna()
            out.loc[mask, "symbol_return_t_plus_1"] = seed_sym_ret[mask]

        # Benchmark symbol — usa colonna se esiste, altrimenti default
        if "benchmark" in out.columns:
            out["bench_symbol"] = out["benchmark"].fillna(self.benchmark_default)
        else:
            out["bench_symbol"] = self.benchmark_default

        # Calcola benchmark return usando SPY come proxy (se presente nel dataset)
        spy_data = out[out["symbol"] == self.benchmark_default][["date_t", "close"]].copy()
        if not spy_data.empty:
            spy_data = spy_data.sort_values("date_t").copy()
            spy_data["spy_next_close"] = spy_data["close"].shift(-1)
            spy_data["benchmark_return_t_plus_1"] = (spy_data["spy_next_close"] / spy_data["close"]) - 1.0
            spy_data = spy_data[["date_t", "benchmark_return_t_plus_1"]].dropna()
            out = out.merge(spy_data, on="date_t", how="left")
        elif "benchmark_return_t_plus_1" in out.columns:
            # Usa il valore già nel seed (calcolato da seed_ml_history.py)
            pass
        else:
            out["benchmark_return_t_plus_1"] = 0.0

        # Target = excess return rispetto al benchmark
        # Usa il target pre-calcolato dal seed come prima fonte
        if seed_target is not None and seed_target.notna().any():
            out["target_excess_return_t_plus_1"] = seed_target
            # Riempi i pochi NaN con symbol_return
            mask = out["target_excess_return_t_plus_1"].isna()
            out.loc[mask, "target_excess_return_t_plus_1"] = out.loc[mask, "symbol_return_t_plus_1"]
        else:
            out["target_excess_return_t_plus_1"] = (
                out["symbol_return_t_plus_1"] - out.get("benchmark_return_t_plus_1", 0.0)
            )

        return out

    def _write_empty(self) -> Path:
        self.output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = self._write_dataset(pd.DataFrame())
        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": 0,
            "symbols": 0,
            "dates": 0,
            "benchmark_default": self.benchmark_default,
            "features_source": str(self.features_path.name),
            "dataset_path": output_path.name,
        }
        self.output_meta_path.write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")
        return output_path

    def _write_dataset(self, dataset: pd.DataFrame) -> Path:
        out = self.output_dataset_path.with_suffix(".jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for row in dataset.to_dict("records"):
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return out
