from typing import Any, List, Optional
from pydantic import BaseModel
import numpy as np
from crewai import Crew, Process
from crewai.tools import BaseTool

from abc import ABC, abstractmethod

import health_coach.tools.data as data_tools
import health_coach.tools.rl as rl_tools
from health_coach.tools.prediction import _make_prediction
import health_coach.tasks as tasks
import health_coach.agents as agents

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
    def encode_prev_state(self, raw_input: Any) -> int:
        ...
    @abstractmethod
    def encode_curr_state(self, raw_input: Any) -> int:
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
        self.q_table = rl_tools._update_q_table(self.q_state.q_table, prev, action, reward, cur)
        return