import gym
import numpy as np
from gym import spaces
from typing import Optional, Tuple, Dict

class BitcoinScalperEnv(gym.Env):
    """
    Environnement de trading RL pour Bitcoin Scalper, compatible OpenAI Gym.
    Simule le spread, les frais, la liquidité et permet l'apprentissage par renforcement (DQN, PPO, etc).
    Actions : 0 = Hold, 1 = Buy, 2 = Sell
    Observation : état du marché (features sélectionnées)
    Reward : profit net (après frais, spread, slippage)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: np.ndarray, fee: float = 0.0005, spread: float = 0.0002, window_size: int = 30, initial_balance: float = 10000.0):
        super().__init__()
        self.df = df
        self.fee = fee
        self.spread = spread
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0
        self.done = False
        self.equity_curve = [self.balance]
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        return self.df[self.current_step - self.window_size:self.current_step, :]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self._get_observation(), 0.0, True, {}
        price = self.df[self.current_step, 0]  # suppose la première colonne = prix
        reward = 0.0
        info = {}
        # Exécution de l'action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price * (1 + self.spread)
                self.balance -= self.entry_price * self.fee
            elif self.position == -1:
                # Fermer short, ouvrir long
                pnl = (self.entry_price - price * (1 + self.spread))
                reward = pnl - self.entry_price * self.fee - price * self.fee
                self.balance += pnl - self.entry_price * self.fee - price * self.fee
                self.position = 1
                self.entry_price = price * (1 + self.spread)
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price * (1 - self.spread)
                self.balance -= self.entry_price * self.fee
            elif self.position == 1:
                # Fermer long, ouvrir short
                pnl = (price * (1 - self.spread) - self.entry_price)
                reward = pnl - self.entry_price * self.fee - price * self.fee
                self.balance += pnl - self.entry_price * self.fee - price * self.fee
                self.position = -1
                self.entry_price = price * (1 - self.spread)
        else:  # Hold
            reward = 0.0
        # Gestion de la fin d'épisode
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True
        self.equity_curve.append(self.balance)
        return self._get_observation(), reward, self.done, info

    def render(self, mode: str = "human") -> None:
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}, Equity: {self.equity_curve[-1]}")

    def close(self):
        pass 