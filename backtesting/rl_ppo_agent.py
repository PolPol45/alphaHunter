import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Dict, Any, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None
    DummyVecEnv = None


class TradingEnv(gym.Env if gym else object):
    """
    Fase 13: Simulated Trading Environment for RL agent.
    State: Market features (momentum, vol, RSI), macro (VIX), current portfolio weights.
    Action: Continuous [-1.0, 1.0] for each asset. Represents conviction (Long vs Short).
    Reward: Market excess return * action - transaction costs (Daily Sharpe proxy).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_balance=10000.0, fee=0.001, is_curriculum_step=False):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.initial_balance = initial_balance
        
        # Curriculum Learning: zero fee and limited slippage for first curriculum stage
        self.fee = 0.0 if is_curriculum_step else fee
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0 
        
        self.feature_cols = [c for c in data.columns if c not in ["date_t", "symbol", "target_excess_return_t_plus_1", "target_rank"]]
        obs_dim = len(self.feature_cols) + 2 

        if spaces:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.returns_history = []
        
    def reset(self, seed=None, options=None):
        if hasattr(super(), 'reset'):
            super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.returns_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        feats = row[self.feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats)
        state = np.append(feats, [self.balance / self.initial_balance, self.position])
        return state.astype(np.float32)

    def step(self, action):
        target_pos = float(action[0])
        trade_size = abs(target_pos - self.position)
        cost = trade_size * self.fee
        
        ret = float(self.data.iloc[self.current_step].get("target_excess_return_t_plus_1", 0.0))
        # Reward is the risk-adjusted step return 
        step_return = (target_pos * ret) - cost
        self.returns_history.append(step_return)
        
        self.position = target_pos
        self.current_step += 1
        
        done = self.current_step >= self.n_steps - 1
        truncated = False
        reward = step_return
            
        info = {"step_return": step_return}
        return self._get_obs(), reward, done, truncated, info


class RLEnsemblePipeline:
    def __init__(self):
        self.model = None
        self.is_active = True
        self.consecutive_underperformance = 0
        self.logger = logging.getLogger("RLEnsemble")
        self.ensemble_weight = 0.20 # Start conservative (20% RL, 80% Supervised)

    def train_ppo(self, train_df: pd.DataFrame, epochs: int = 10000):
        if not PPO or not gym:
            self.logger.warning("gymnasium / stable_baselines3 non installati. RL disattivato.")
            self.is_active = False
            return False
            
        if len(train_df) < 100:
            return False
            
        # Curriculum Learning Phase 1: Simple env
        env_simple = DummyVecEnv([lambda: TradingEnv(train_df, is_curriculum_step=True)])
        self.model = PPO("MlpPolicy", env_simple, verbose=0, learning_rate=5e-4, clip_range=0.2)
        
        try:
            # Curriculum Step 1 (Easy)
            self.model.learn(total_timesteps=int(epochs * 0.3))
            
            # Curriculum Step 2 (Hard - Real Fees)
            env_hard = DummyVecEnv([lambda: TradingEnv(train_df, is_curriculum_step=False)])
            self.model.set_env(env_hard)
            self.model.learn(total_timesteps=int(epochs * 0.7))
            
            return True
        except Exception as e:
            self.logger.error(f"PPO training failed: {e}")
            self.is_active = False
            return False

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.model is None or not self.is_active:
            return np.zeros(len(test_df))
            
        env = TradingEnv(test_df)
        preds = []
        obs, _ = env.reset()
        for i in range(len(test_df)):
            action, _ = self.model.predict(obs, deterministic=True)
            preds.append(action[0])
            if i < len(test_df) - 1:
                obs, _, _, _, _ = env.step(action)
                
        return np.array(preds)
        
    def evaluate_guardrail(self, rl_returns: List[float], benchmark_returns: List[float]):
        """Fase 13: Guardrail Anti-Overfitting RL."""
        if not self.is_active: return
        
        rl_sharpe = np.mean(rl_returns) / (np.std(rl_returns) + 1e-9)
        bench_sharpe = np.mean(benchmark_returns) / (np.std(benchmark_returns) + 1e-9)
        
        if rl_sharpe < bench_sharpe:
            self.consecutive_underperformance += 1
            # Riduci peso RL se perde
            self.ensemble_weight = max(0.0, self.ensemble_weight - 0.10)
        else:
            self.consecutive_underperformance = 0
            # Aumenta peso RL se batte il mercato!
            self.ensemble_weight = min(0.60, self.ensemble_weight + 0.10)
            
        if self.consecutive_underperformance >= 3:
            self.logger.critical("🚨 GUARDRAIL RL TRIPPED! L'agente PPO ha perso dal Benchmark (S&P500) per 3 fold consecutivi. Reinforcement Learning SILENZIATO. Torno al 100% Supervised Models (RandomForest).")
            self.is_active = False
            self.ensemble_weight = 0.0
