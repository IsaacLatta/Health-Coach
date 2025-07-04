from typing import Dict
from abc import ABC, abstractmethod
from crewai import Crew, Process
from typing import Protocol, TypeVar, List, Tuple, Any

import health_coach.tasks as tasks
import health_coach.agents as agents

import health_coach.tools.rl as rl_tools
import health_coach.tools.rl_tools.reward as shape_reward_tools
import health_coach.tools.rl_tools.context as context_tools
import health_coach.tools.rl_tools.action as action_tools
import health_coach.tools.data as data_tools
from health_coach.tools.prediction import _make_prediction

StateType  = TypeVar("StateType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

class RLEngine(Protocol[StateType, ActionType, RewardType]):
    @property
    def possible_actions(self) -> List[ActionType]:
        ...
    @property
    def encode_state(self, raw_input: Any) -> StateType:
        ...
    @property
    def encode_prev_state(self, input: Any) -> StateType:
        ...
    @property
    def compute_env_reward(self, prev: StateType, cur: StateType) -> RewardType:
        ...
    @property
    def generate_context(self, prev: StateType, cur: StateType, reward: RewardType) -> str:
        ...
    @property
    def shape_reward_and_action(self, prev: StateType, cur: StateType, reward: RewardType, context: str) -> Tuple[ActionType, RewardType]:
        ...
    @property
    def update_model(self, prev: StateType, action: ActionType, reward: RewardType, cur: StateType) -> bool:
        ...

class CrewFactory(ABC):
    @abstractmethod
    def create() -> Crew:
        ...

class QLearningEngine(RLEngine[int, int, float]):
    def __init__(self, context_crew_factory: CrewFactory, shaping_crew_factory: CrewFactory):
        self.context_crew_factory = context_crew_factory
        self.shape_crew_factory = shaping_crew_factory

    def possible_actions(self) -> List[int]:
        return [0, 1, 2, 3, 4, 5, 6]

    def encode_prev_state(self, input: Any) -> int:
        patient_info = data_tools._load_patient_data(self.state.raw_input.patient_id, timeout_base=0,)
        return patient_info["State"]

    def encode_current_state(self, input: Any) -> int:
        pred = _make_prediction(input.patient_info)
        return rl_tools._discretize_probability(pred)

    def compute_env_reward(self, prev: int, cur: int) -> float:
        return rl_tools._compute_reward(prev, cur)

    def generate_context(self, prev: int, cur: int, reward: float) -> str:
        inputs = {
            "previous_state":   prev,
            "current_state":    cur,
            "pure_reward":      reward,
            "possible_actions": self.possible_actions,
        }
        crew = self.context_crew_factory.create()
        out = crew.kickoff(inputs=inputs)
        return out.json_dict.get("context", "Unavailable")

    def shape_reward_and_action(self, prev: int, cur: int, reward: float, context: str) -> Tuple[int, float]:
        inputs = {
            "previous_state":   prev,
            "current_state":    cur,
            "pure_reward":      reward,
            "context":          context,
            "possible_actions": self.possible_actions,
        }
        out = self.shape_crew_factory.kickoff(inputs=inputs)
        action_task, reward_task = out.tasks_output
        action = action_task.json_dict["action"]
        shaped_reward = reward_task.json_dict["shaped_reward"]
        return action, shaped_reward

    def update_model(self, prev: int, action: int, reward: float, cur: int) -> bool:
        result = {"prev_state" : prev, "action" : int, "reward": float, "current_state": cur}
        print(f"Result: {result}")
        return True