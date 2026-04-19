import numpy as np

class MontecarloSimulator:
    """
    Fase 3: Montecarlo Path Generator per Stress Testing.
    Prende un array cronologico di ritorni giornalieri o trade e genera N path alternativi
    ricampionando con rimpiazzo (Bootstrapping). Calcola Value at Risk (VaR), CVaR 
    e stima il Drawdown nel peggiore dei casi.
    """

    def __init__(self, trials: int = 5000, confidence_level: float = 0.99):
        self.trials = trials
        self.confidence_level = confidence_level

    def run_simulation(self, returns: list[float]) -> dict:
        """
        Esegue la simulazione Montecarlo sui rendimenti forniti.
        """
        if not returns or len(returns) < 5:
            return {
                "var_99": 0.0,
                "cvar_99": 0.0,
                "max_drawdown_95th": 0.0,
                "expected_return_annualized": 0.0,
                "prob_ruin_30pct": 0.0,
            }

        arr = np.array(returns)
        n_periods = len(arr)
        
        # Generiamo 'trials' percorsi casuali di lunghezza 'n_periods' pescando dai ritorni storici
        # con rimpiazzo (Metodo Bootstrap non parametrico)
        simulated_paths = np.random.choice(arr, size=(self.trials, n_periods), replace=True)
        
        # Calcolo Equity Curves per ogni path: (1 + r1)(1 + r2)...
        # cumulative product lungo l'asse 1 (tempo)
        equity_curves = np.cumprod(1.0 + simulated_paths, axis=1)

        # 1. VaR e CVaR a livello di rendimento del singolo periodo simulato
        # Calcoliamo la distribuzione dei ritorni finali (l'ultimo valore di ogni curve - 1)
        final_returns = equity_curves[:, -1] - 1.0
        
        # VaR: Il percentile della coda sinistra. Es. se confidence_level = 0.99, cerchiamo l'1% peggiore.
        alpha = 1.0 - self.confidence_level
        var_percentile = alpha * 100
        var = np.percentile(final_returns, var_percentile)
        
        # CVaR (Expected Shortfall): La media dei ritorni *peggiori* del VaR
        worst_cases = final_returns[final_returns <= var]
        cvar = np.mean(worst_cases) if len(worst_cases) > 0 else var

        # 2. Maximum Drawdown 95th Percentile
        # Per ogni path, calcoliamo il max drawdown
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (running_max - equity_curves) / running_max
        max_drawdowns_per_path = np.max(drawdowns, axis=1)
        
        # Troviamo il Drawdown estremo: es. il 95° percentile dei drawdown PEGGIORI
        # (cioè ordiniamo i max drawdown e prendiamo il 95% più alto)
        dd_95th = np.percentile(max_drawdowns_per_path, 95)

        # 3. Probabilità matematica di perdere il 30% del portafoglio (Ruin Probability)
        ruined_paths = np.sum(max_drawdowns_per_path >= 0.30)
        prob_ruin_30pct = ruined_paths / self.trials

        return {
            "var_99": round(float(var), 6),
            "cvar_99": round(float(cvar), 6),
            "max_drawdown_95th": round(float(dd_95th), 6),
            "expected_return_annualized": round(float(np.mean(final_returns)), 6),
            "prob_ruin_30pct": round(float(prob_ruin_30pct), 6),
            "trials": self.trials
        }
