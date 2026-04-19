import pytest
import numpy as np
from backtesting.montecarlo_simulator import MontecarloSimulator

def test_montecarlo_basic_metrics():
    # Caso normale
    simulator = MontecarloSimulator(trials=1000, confidence_level=0.99)
    # 100 ritorni positivi e negativi fissi per testare consistenza
    returns = [0.01, -0.005, 0.02, -0.015, 0.005] * 20 
    
    metrics = simulator.run_simulation(returns)
    
    assert "var_99" in metrics
    assert "cvar_99" in metrics
    assert "max_drawdown_95th" in metrics
    assert metrics["trials"] == 1000
    assert metrics["max_drawdown_95th"] >= 0.0 # il DD \u00e8 indicato come numero positivo o zero

def test_montecarlo_edge_case_empty():
    # Caso empty array
    simulator = MontecarloSimulator(trials=100)
    metrics = simulator.run_simulation([])
    assert metrics["var_99"] == 0.0
    assert metrics["max_drawdown_95th"] == 0.0

def test_montecarlo_extreme_ruin():
    # Caso estremo: la strategia perde sempre
    simulator = MontecarloSimulator(trials=100)
    returns = [-0.10] * 50 # 50 giorni a -10% al giorno
    
    metrics = simulator.run_simulation(returns)
    # Con -10% giornaliero, in 50 giorni si perde pi\u00f9 del 30% del portafoglio
    assert metrics["prob_ruin_30pct"] == 1.0
    assert metrics["max_drawdown_95th"] > 0.90
