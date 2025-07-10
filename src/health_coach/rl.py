from typing import Any, List, Optional
from pydantic import BaseModel
import numpy as np
from crewai import Crew, Process
from crewai.tools import BaseTool

from abc import ABC, abstractmethod

import inspect
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
    def save_state(self, prev: int, action: int, reward: float, cur: int) -> QLearningState:
        ...

class QLearningEngine(RLEngine):
    def __init__(
        self,
        exploration_tools: List[Any],
        context_tools: List[Any],
        shaping_tools: List[Any],
        use_crews: bool = True,
        max_iter: int = 3,
        verbose: bool = False,
    ):
        self.exploration_tools = exploration_tools
        self.context_tools = context_tools
        self.shaping_tools = shaping_tools
        self.use_crews = use_crews
        self.max_iter = max_iter
        self.verbose = verbose

    def _run_tool(tool_fn, **inputs):
        sig = inspect.signature(tool_fn)
        kwargs = {
            name: inputs[name]
            for name in sig.parameters
            if name in inputs
        }
        return tool_fn(**kwargs)

    def possible_actions(self) -> List[int]:
        return list(range(7))

    def encode_prev_state(self, raw_input: Any) -> int:
        patient_info = data_tools._load_patient_data(raw_input.patient_id, timeout_base=0,)
        return patient_info["State"]

    def encode_curr_state(self, raw_input: Any) -> int:
        pred = _make_prediction(raw_input.patient_features)
        return rl_tools._discretize_probability(pred)

    def compute_env_reward(self, prev: int, cur: int) -> float:
        return rl_tools._compute_reward(prev, cur)

    def generate_context(self, prev: int, cur: int, reward: float) -> str:
        inputs = {
            "previous_state": prev,
            "current_state": cur,
            "pure_reward": reward,
            "possible_actions": self.possible_actions(),
        }

        if not self.use_crews:
            return "Unavailable"
        
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

        if self.use_crews:
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

        action = self.exploration_tools[0](**inputs)
        shaped = action
        for tool in self.exploration_tools:
            shaped = self._run_tool(tool, **inputs)
        return shaped

    def shape_reward(self, prev: int, cur: int, reward: float, context: str) -> float:
            inputs = {
                "previous_state": prev,
                "current_state": cur,
                "pure_reward": reward,
                "context": context,
                "possible_actions": self.possible_actions(),
            }

            if self.use_crews:
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
            
            shaped = self.shaping_tools[0](**inputs)
            for tool in self.shaping_tools:
                shaped = self._run_tool(tool, **inputs)
            return shaped

    def save_state(self, prev: int, action: int, reward: float, cur: int) -> QLearningState:
        print(f"FINAL OUTPUT:\n\tPrevious State: {prev}\n\tAction: {action}\n\tReward: {reward}\n\tCurrent State: {cur}")
        state = QLearningState()
        state.previous_state = prev
        state.q_table = np.zeros(shape=(10, 4), dtype=float)
        return state