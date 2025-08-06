from typing import Any, List, Optional, Callable
from pydantic import BaseModel
import numpy as np
from crewai import Crew, Process
from crewai.tools import BaseTool

from threading import Lock
from abc import ABC, abstractmethod

import health_coach.tools.data as data_tools
import health_coach.tools.rl as rl_tools
from health_coach.tools.prediction import _make_prediction
import health_coach.tasks as tasks
import health_coach.agents as agents
import health_coach.config as cfg
from health_coach.compare.explorer_selector import execute_explorer
from health_coach.compare.context_engine import ContextEngine

class QLearningState(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    q_table: Optional[np.ndarray] = None
    previous_state: int = 0

class RLEngine(ABC):
    @abstractmethod
    def possible_actions(self) -> List[int]:
        ...
    @abstractmethod
    def compute_env_reward(self, prev: int, cur: int) -> float:
        ...
    @abstractmethod
    def generate_context(self, prev: int, cur: int, reward: float) -> str:
        ...
    @abstractmethod
    def shape_action(self, prev: int, cur: int, reward: float, context: str) -> int:
        ...
    @abstractmethod
    def shape_reward(self, prev: int, cur: int, reward: float, context: str) -> float:
        ...
    @abstractmethod
    def save_state(self, prev: int, action: int, reward: float, cur: int):
        ...

class SimpleQLearningEngine(RLEngine):
    def __init__(
            self, 
            exploration_tools: List[Any], 
            reward_fn: Callable[[int, int], float],
            alpha: float, 
            gamma: float
        ):
        self.q_table = np.zeros((cfg.Q_STATES, cfg.Q_ACTIONS), dtype=float)
        self.exploration_tools = exploration_tools
        self.reward_fn = reward_fn
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = True
        self.ctx_engine = ContextEngine()

    def compute_env_reward(self, prev: int, curr: int):
        return self.reward_fn(prev, curr)

    def possible_actions(self) -> List[int]:
        return list(range(cfg.Q_ACTIONS))

    def generate_context(self, prev, cur, reward) -> str:
        try:
            return self.ctx_engine.generate_context()
        except Exception as e:
            return "Unavailable"

    def safe_select(self, result: dict, num_strats: int, default: int = 0) -> int:
        try:
            val = result.get("select")
            if isinstance(val, int) and 0 <= val < num_strats:
                return val
            if isinstance(val, str) and val.isdigit():
                iv = int(val)
                if 0 <= iv < num_strats:
                    return iv
        except Exception:
            pass
        return default

    def shape_action(self, prev_state: int, current_state: int, reward: float, context: str) -> int:
        inputs = {
            "previous_state": prev_state,
            "current_state": current_state,
            "pure_reward": reward,
            "context": context,
        }

        agent = agents.RewardShapingAgent().create(max_iter=1, verbose=True)
        action_t = tasks.ShapeAction().create(agent)
        crew = Crew(
            agents=[agent],
            tasks=[action_t],
            process=Process.sequential,
            verbose=self.verbose,
        )
        json_result = crew.kickoff(inputs=inputs).json_dict
        explorer_idx = self.safe_select(json_result, 6)
        action = execute_explorer(current_state, explorer_idx, self.q_table)
        self.ctx_engine.update(prev_state, current_state, reward, action, self.q_table)
        print(f"\n\nACTION: {action}\nCONTEXT: {context}\n\n")
        return action

    def shape_reward(self, prev, cur, reward, context):
        return reward
    
    def save_state(self, prev, action, reward, cur):
        best_next = self.q_table[cur].max()
        self.q_table[prev, action] = self.alpha * (reward + self.gamma * best_next) + (1 - self.alpha)*self.q_table[prev, action]

class QLearningEngine(RLEngine):
    def __init__(
        self,
        exploration_tools: List[Any],
        context_tools: List[Any],
        shaping_tools: List[Any],
        max_iter: int = 3,
        verbose: bool = True,
    ):
        self.exploration_tools = exploration_tools
        self.context_tools = context_tools
        self.shaping_tools = shaping_tools
        self.max_iter = max_iter
        self.verbose = verbose
        self.q_table = ...

    def possible_actions(self) -> List[int]:
        return list(range(7))

    def encode_prev_state(self, raw_input: Any) -> int:
        return raw_input["previous_state"]

    def encode_curr_state(self, raw_input: Any) -> int:
        return raw_input["current_state"]

    def compute_env_reward(self, prev: int, cur: int) -> float:
        return rl_tools._compute_reward(prev, cur)

    def generate_context(self, prev: int, cur: int, reward: float) -> str:
        inputs = {
            "previous_state": prev,
            "current_state": cur,
            "pure_reward": reward,
            "possible_actions": self.possible_actions(),
        }
        
        agent = agents.ContextProvidingAgent().create(max_iter=self.max_iter, verbose=self.verbose)
        task = tasks.ShapeRewardContext().create(agent, tools=self.context_tools)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        out = crew.kickoff(inputs=inputs)
        return out.json_dict.get("context", "Unavailable")

    def shape_action(self, prev_state: int, current_state: int, reward: float, context: str) -> int:
        inputs = {
            "previous_state":   prev_state,
            "current_state":    current_state,
            "pure_reward":      reward,
            "context":          context,
            "possible_actions": self.possible_actions(),
        }

        agent = agents.RewardShapingAgent().create(max_iter=self.max_iter, verbose=self.verbose)
        action_t = tasks.ShapeAction().create(agent, tools=self.exploration_tools)
        crew = Crew(
            agents=[agent],
            tasks=[action_t],
            process=Process.sequential,
            verbose=self.verbose,
        )
        json_result = crew.kickoff(inputs=inputs).json_dict
        return json_result["action"]

    def shape_reward(self, prev: int, cur: int, reward: float, context: str) -> float:
            inputs = {
                "previous_state": prev,
                "current_state": cur,
                "pure_reward": reward,
                "context": context,
                "possible_actions": self.possible_actions(),
            }

            agent = agents.RewardShapingAgent().create(max_iter=self.max_iter, verbose=self.verbose)
            reward_t = tasks.ShapeReward().create(agent, tools=self.shaping_tools)
            crew = Crew(
                agents=[agent],
                tasks=[reward_t],
                process=Process.sequential,
                verbose=self.verbose,
            )
            out = crew.kickoff(inputs=inputs)
            return out.json_dict["shaped_reward"]

    def save_state(self, prev: int, action: int, reward: float, cur: int):
        # self.q_table = rl_tools._update_q_table(self.q_state.q_table, prev, action, reward, cur)
        return action