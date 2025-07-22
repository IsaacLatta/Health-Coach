import random
import numpy as np
import gym
from gym import spaces
from typing import Any, List, Dict, Callable, Tuple
from pydantic import BaseModel
from crewai.flow.flow import Flow

from health_coach.flows import RLFlow
from health_coach.rl_data_gen.drift import simulate_drift

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
        p = (1 - alpha) * p_start + alpha * p_end + np.random.normal(0, gaussian_std)
        p = float(np.clip(p, 0.0, 1.0))
        trajectory.append(p)
    return trajectory

class AgentQLearningEnvironment:
    def __init__(
            self, 
            flow: RLFlow,
            state_mapper: Callable[[float], int],
            action_to_noise_mapper: Callable[[int], float]
            ):
        self.rl_flow = flow
        self.state_mapper = state_mapper
        self.action_to_noise_mapper = action_to_noise_mapper
        self.q_table = np.zeros((10, 7), dtype=float)

    def run_episode(self, p_start: float, p_end: float) -> np.ndarray:
        trajectory = generate_base_trajectory(p_start, p_end)
        
        action = -1
        prev_state = self.state_mapper(trajectory[0])
        for prob in trajectory[1:]:
            noisy_p = prob + (self.action_to_noise_mapper(action) if action >= 0 else 0.0)
            state = self.state_mapper(noisy_p)
            action = self.apply_policy(prev_state, state)
            prev_state = state

        return self.q_table

    def reset(self, q_table: np.ndarray):
        self.q_table = q_table

    def apply_policy(self, previous_state: int, current_state: int) -> int:
        try:
            self.rl_flow.set_input(input={
                "previous_state": previous_state,
                "current_state": current_state
                })
            return self.rl_flow.kickoff()
        
        except Exception as e:
            return
        

class PureQLearningEnvironment:

    def __init__(
            self, 
            exploration_strategy: Callable[[np.ndarray], int],
            state_mapper: Callable[[float], int],
            action_to_noise_mapper: Callable[[int], float],
            reward_function: Callable[[int, int], float],
            gamma: float,
            alpha: float):
        self.exploration_strategy = exploration_strategy
        self.state_mapper = state_mapper
        self.action_to_noise_mapper = action_to_noise_mapper
        self.reward_function = reward_function
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = np.zeros((10, 7), dtype=float)
        self.visit_counts = np.zeros((10, 7), dtype=float)


    def run_episode(self, p_start: float, p_end: float) -> np.ndarray:
        trajectory = generate_base_trajectory(p_start, p_end)

        action = -1
        prev_state = self.state_mapper(trajectory[0])
        for prob in trajectory[1:]:
            noisy_p = prob + (self.action_to_noise_mapper(action) if action >= 0 else 0)
            state = self.state_mapper(noisy_p)
            reward = self.reward_function(prev_state, state)
            action = self.exploration_strategy(prev_state)
            self.visit_counts[prev_state, action] += 1

            self.update_q_table(prev_state, reward, action, state)
            prev_state = state

        return self.q_table

    def update_q_table(self, prev_state: int, reward: float, action: int, next_state: int):
        best_next = self.q_table[next_state].max()
        td_error  = (reward + self.gamma * best_next) - self.q_table[prev_state, action]
        self.q_table[prev_state, action] += self.alpha * td_error

    def reset(self, q_table: np.ndarray):
        self.q_table = q_table
        self.visit_counts[:] = 0

class DriftOfflineEnvironment:
    def __init__(
        self,
        state_mapper: Callable[[float], int],
        action_to_noise_mapper: Callable[[int], float],
        reward_function: Callable[[int,int], float],
    ):
        self.state_mapper = state_mapper
        self.action_to_noise_mapper = action_to_noise_mapper
        self.reward_function = reward_function

    def reset(self, episode: List[Dict]) -> int:
        self.episode = episode
        self.idx = 0
        p0 = episode[0]["prediction"]
        self.prev_state = self.state_mapper(p0)
        return self.prev_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        entry = self.episode[self.idx]
        p_next = entry["next_prediction"]

        noisy_p = p_next + (self.action_to_noise_mapper(action) if action >= 0 else 0.0)
        noisy_p = float(np.clip(noisy_p, 0.0, 1.0))

        next_state = self.state_mapper(noisy_p)
        reward = self.reward_function(self.prev_state, next_state)

        self.idx += 1
        done = self.idx >= len(self.episode)
        self.prev_state = next_state

        return next_state, reward, done

    