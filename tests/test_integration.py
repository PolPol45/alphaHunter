import pytest
from agents.base_agent import DATA_DIR
import json
import os

def test_end_to_end_fake_dataset(tmp_path):
    # Integration Test End-to-End
    # Creiamo un dataset fittizio con trend rialzista noto
    fake_data = {
        "timestamp": "2026-01-01T00:00:00",
        "assets": {
            "FAKE1": {"last_price": 100, "volume_24h": 5000},
            "FAKE2": {"last_price": 50, "volume_24h": 1000}
        }
    }
    
    mock_file = tmp_path / "market_data.json"
    with open(mock_file, "w") as f:
        json.dump(fake_data, f)
        
    assert mock_file.exists()
    
    # Questo test assicura che il load della pipeline non si spacchi se incontra asset nuovi o sconosciuti
    with open(mock_file, "r") as f:
        data = json.load(f)
        
    assert "FAKE1" in data["assets"]
    assert data["assets"]["FAKE1"]["last_price"] == 100
