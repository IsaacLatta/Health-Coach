import random
import numpy as np
import gym
from gym import spaces
from typing import Any, List, Dict

from health_coach.rl_data_gen.drift import simulate_drift, discretize_probability
from health_coach.rl_data_gen.generate import predict_proba

ACTIONS = [
    "INCREASE_MODERATE_THRESHOLD",
    "DECREASE_MODERATE_THRESHOLD",
    "INCREASE_HIGH_THRESHOLD",
    "DECREASE_HIGH_THRESHOLD",
    "INCREASE_TOP_K",
    "DECREASE_TOP_K",
    "NOP",
]

class HeartDiseaseEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        anchors: List[Dict[str, Any]],
        model: Any,
        feature_names: List[str],
        min_steps: int = 20,
        max_steps: int = 50,
        gaussian_std: float = 0.02,
        delta_min: float = 0.01,
        delta_max: float = 0.05,
        seed: int = None,
    ):
        super().__init__()
        self.anchors = anchors
        self.model = model
        self.feature_names = feature_names
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.gaussian_std = gaussian_std
        self.delta_min = delta_min
        self.delta_max = delta_max
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._trajectory: List[Dict[str, Any]] = []
        self._step_idx: int = 0

    def reset(self, *, seed=None, options=None): 
        row0, row1 = random.choice(self.anchors)
        p0 = predict_proba(self.model, row0)
        p1 = predict_proba(self.model, row1)

        self._trajectory = simulate_drift(
            p0,
            p1,
            length=random.randint(self.min_steps, self.max_steps),
            gaussian_std=self.gaussian_std,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
        )
        self._step_idx = 0

        first = self._trajectory[0]
        return discretize_probability(first["prediction"]), {}

    def step(self, action: int):
        entry = self._trajectory[self._step_idx]
        a_str = ACTIONS[action]

        p_base = entry["prediction"]
        delta = random.uniform(self.delta_min, self.delta_max)
        if a_str.startswith("DECREASE_"):
            p_next = p_base - delta
        elif a_str.startswith("INCREASE_"):
            p_next = p_base + delta
        else:
            p_next = p_base
        p_next = max(0.0, min(1.0, p_next))
        next_state = discretize_probability(p_next)

        if p_next < p_base:
            reward = 1
        elif p_next > p_base:
            reward = -1
        else:
            reward = 0

        self._step_idx += 1
        done = self._step_idx >= len(self._trajectory)
        return next_state, reward, done, False, {}

    def render(self, mode="human"):
        print(f"Step {self._step_idx}/{len(self._trajectory)}")

    def close(self):
        pass
