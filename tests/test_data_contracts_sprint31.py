from __future__ import annotations

import json
import pathlib
import unittest

BOT_DIR = pathlib.Path(__file__).resolve().parents[1]


class DataContractsSprint31Tests(unittest.TestCase):
    def test_universe_health_schema_exists_and_has_required_keys(self) -> None:
        path = BOT_DIR / "data_contracts" / "universe_health_report.schema.json"
        self.assertTrue(path.exists())
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("generated_at", schema["required"])
        self.assertIn("summary", schema["required"])
        self.assertIn("symbols", schema["required"])

    def test_cross_sectional_features_schema_exists_and_has_required_keys(self) -> None:
        path = BOT_DIR / "data_contracts" / "cross_sectional_features.schema.json"
        self.assertTrue(path.exists())
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("date", schema["required"])
        self.assertIn("symbol", schema["required"])
        self.assertIn("macro_market_bias", schema["required"])

    def test_ml_dataset_weekly_schema_exists(self) -> None:
        path = BOT_DIR / "data_contracts" / "ml_dataset_weekly.schema.json"
        self.assertTrue(path.exists())
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("date_t", schema["required"])
        self.assertIn("target_excess_return_t_plus_1", schema["required"])

    def test_ml_signals_schema_exists(self) -> None:
        path = BOT_DIR / "data_contracts" / "ml_signals.schema.json"
        self.assertTrue(path.exists())
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("runtime_mode", schema["required"])

    def test_universe_discovery_schema_exists(self) -> None:
        path = BOT_DIR / "data_contracts" / "universe_discovery_candidates.schema.json"
        self.assertTrue(path.exists())
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("candidates", schema["required"])


if __name__ == "__main__":
    unittest.main()
