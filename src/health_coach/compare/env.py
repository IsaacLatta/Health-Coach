import random
import numpy as np
from typing import Any, List, Dict, Callable, Tuple
from pydantic import BaseModel
from crewai.flow.flow import Flow

from health_coach.compare.flows import RLFlow

import health_coach.config as cfg

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

    def run_episode(self, trajectory: List[float]):
        action: int = -1
        prev_state = self.state_mapper(trajectory[0])
        for prob in trajectory[1:]:
            noisy_p = prob + (self.action_to_noise_mapper(action) if action >= 0 else 0.0)
            
            state = self.state_mapper(noisy_p)
            action = self.update_policy(prev_state, state) if action is not None else action
            prev_state = state

    def update_policy(self, previous_state: int, current_state: int) -> int:
        try:
            self.rl_flow.set_state(previous_state, current_state)
            res = self.rl_flow.kickoff()
            if res is None:
                return 0
            return res
        except Exception as e:
            print(f"Got exception: {e}")
            return 0

class PureQLearningEnvironment:
    def __init__(
            self, 
            exploration_strategy: Callable[[int, np.ndarray], int],
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
        self.q_table = np.zeros((cfg.Q_STATES, cfg.Q_ACTIONS), dtype=float)
        self.visit_counts = np.zeros((cfg.Q_STATES, cfg.Q_ACTIONS), dtype=float)

    def run_episode(self, probs: List[float]) -> np.ndarray:
        action = -1
        prev_state = self.state_mapper(probs[0])

        for p in probs[1:]:
            noise = (self.action_to_noise_mapper(action) if action >= 0 else 0.0)
            # print(f"Got noise: {noise}\n")
            noisy_p = p + noise
            
            noisy_p = float(np.clip(noisy_p, 0.0, 1.0))

            state  = self.state_mapper(noisy_p)
            reward = self.reward_function(prev_state, state)

            action = self.exploration_strategy(prev_state, self.q_table)
            self.visit_counts[prev_state, action] += 1
            self.update_q_table(prev_state, reward, action, state)

            prev_state = state

        return self.q_table

    def update_q_table(self, prev_state: int, reward: float, action: int, next_state: int):
        best_next = self.q_table[next_state].max()
        self.q_table[prev_state, action]  = self.alpha * (reward + self.gamma * best_next) + (1 - self.alpha)*self.q_table[prev_state, action]

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

    def reset(self, episode: List[float]) -> int:
        self.episode = episode
        self.idx = 0
        p0 = episode[0]
        self.prev_state = self.state_mapper(p0)
        return self.prev_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        p_next = self.episode[self.idx]

        noisy_p = p_next + (self.action_to_noise_mapper(action) if action >= 0 else 0.0)
        noisy_p = float(np.clip(noisy_p, 0.0, 1.0))

        next_state = self.state_mapper(noisy_p)
        reward = self.reward_function(self.prev_state, next_state)

        self.idx += 1
        done = self.idx >= len(self.episode)
        self.prev_state = next_state

        return next_state, reward, done

    