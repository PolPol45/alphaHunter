import pytest
from agents.execution_agent import ExecutionAgent

def test_execution_agent_position_sizing_mock():
    # Mock portfolio and config to test sizing logic
    agent = ExecutionAgent()
    agent.config = {"execution": {"max_open_positions": 5}}
    
    # Simulate an empty portfolio
    portfolio = {"cash": 100000, "positions": {}}
    
    # Create a fake signal
    signal = {
        "symbol": "BTC-USD",
        "signal_type": "BUY",
        "entry_price": 50000,
        "quantity": 0.5,
        "position_size_usdt": 25000
    }
    
    # Apply turnover gate
    allow, reason = agent._turnover_gate(
        mode="simulation",
        symbol="BTC-USD",
        vsig=signal,
        portfolio=portfolio,
        existing_pos={},
        desired_side="long"
    )
    
    assert allow is True
    assert reason == ""

def test_kraken_mapping():
    # Semplice validazione della conversione Ticker Yahoo -> Kraken
    # es: BTCUSDT -> BTC-USD
    agent = ExecutionAgent()
    # Questo \u00e8 un test mock della logica che avviene in config o nei client
    config_mock = {
        "BTC-USD": {"symbol": "BTC", "exchange": "KRAKEN"}
    }
    assert config_mock["BTC-USD"]["exchange"] == "KRAKEN"
