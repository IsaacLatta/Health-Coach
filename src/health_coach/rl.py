from typing import Any, List, Tuple, Protocol, TypeVar, Generic
from crewai import Crew, Process
from crewai.tools import BaseTool

from abc import ABC, abstractmethod

import health_coach.tools.data as data_tools
import health_coach.tools.rl as rl_tools
from health_coach.tools.prediction import _make_prediction
import health_coach.tools.rl_tools.context as context_tools
import health_coach.tools.rl_tools.action  as action_tools
import health_coach.tools.rl_tools.reward  as shape_reward_tools
import health_coach.tasks as tasks
import health_coach.agents as agents

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
    def update_model(self, prev: int, action: int, reward: float, cur: int) -> bool:
        ...

class QLearningEngine(RLEngine):
    def __init__(
        self,
        exploration_tools: List[Any],
        context_tools: List[BaseTool],
        shaping_tools: List[Any],
        use_crews: bool = True,
        max_iter: int = 3,
        verbose: bool = False,
    ):
        self.exploration_tools = exploration_tools
        self.context_tools     = context_tools
        self.shaping_tools     = shaping_tools
        self.use_crews         = use_crews
        self.max_iter          = max_iter
        self.verbose           = verbose

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

        if self.use_crews:
            agent = agents.ContextProvidingAgent().create(
                max_iter=self.max_iter,
                verbose=self.verbose,
            )
            task = tasks.ShapeRewardContext().create(
                agent,
                tools=self.context_tools,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=self.verbose,
            )
            out = crew.kickoff(inputs=inputs)
            return out.json_dict.get("context", "Unavailable")

        for tool in self.context_tools:
            try:
                ctx = tool(**inputs)
                if isinstance(ctx, str):
                    return ctx
            except Exception:
                continue

        return "Unavailable"

    def shape_action(self, prev: int, cur: int, reward: float, context: str) -> int:
        inputs = {
            "previous_state":   prev,
            "current_state":    cur,
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
        for tool in self.shaping_tools:
            shaped = tool(state=cur, reward=shaped)
        return action, shaped

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
            
            action = self.exploration_tools[0](**inputs)
            shaped = action
            for tool in self.shaping_tools:
                shaped = tool(state=cur, reward=shaped)
            return shaped

    def update_model(self, prev: int, action: int, reward: float, cur: int) -> bool:
        print(f"FINAL OUTPUT:\n\tPrevious State: {prev}\n\tAction: {action}\n\t Reward: {reward}\n\tCurrent State: {cur}")
        # would call 
        # return rl_tools._update_rl_model(prev, action, reward, cur)
        return True