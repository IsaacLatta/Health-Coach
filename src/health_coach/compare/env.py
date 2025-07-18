import random
import numpy as np
import gym
from gym import spaces
from typing import Any, List, Dict, Callable
from pydantic import BaseModel
from crewai.flow.flow import Flow

from health_coach.flows import RLFlow
from health_coach.rl_data_gen.drift import simulate_drift

ACTIONS = [
    "INCREASE_MODERATE_THRESHOLD",
    "DECREASE_MODERATE_THRESHOLD",
    "INCREASE_HIGH_THRESHOLD",
    "DECREASE_HIGH_THRESHOLD",
    "INCREASE_TOP_K",
    "DECREASE_TOP_K",
    "NOP",
]

def generate_base_trajectory(
    p_start: float,
    p_end:   float,
    *,
    min_steps:   int = 20,
    max_steps:   int = 50,
    gaussian_std: float = 0.02,
    seed:       int = None,
) -> List[float]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = random.randint(min_steps, max_steps)
    trajectory = []
    for t in range(T):
        alpha = t / (T - 1) if T > 1 else 0.0
        p = (1 - alpha) * p_start + alpha * p_end \
            + np.random.normal(0, gaussian_std)
        p = float(np.clip(p, 0.0, 1.0))
        trajectory.append(p)
    return trajectory

class AgentQLearningEnvironment(gym.Env):
    def __init__(
            self, 
            flow: RLFlow,
            q_table: np.ndarray,
            state_mapper: Callable[[float], int],
            action_to_noise_mapper: Callable[[int], float]
            ):
        self.rl_flow = flow
        self.q_table = q_table
        self.state_mapper = state_mapper
        self.action_to_noise_mapper = action_to_noise_mapper

    def run_episode(self, p_start: float, p_end: float) -> np.ndarray:
        trajectory = generate_base_trajectory(p_start, p_end)
        
        action = -1
        last_state = trajectory[0]
        for prob in trajectory:
            noise = 0 if action < 0 else self.action_noise_mapper(action)
            state = self.state_mapper(prob) + noise
            action = self.apply_policy(last_state, state)

        return self.q_table

    def reset_for_episode(
            self, 
            new_episode: List[List[Dict[str, Any]]]):
        self.episode = new_episode

    def apply_policy(self, previous_state: int, current_state: int) -> int:
        try:
            return self.rl_flow.kickoff(input={
                "previous_state": previous_state,
                "current_state": current_state
                })
        
        except Exception as e:
            return