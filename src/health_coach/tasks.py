from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from crewai import Task, Agent
from crewai.tools import BaseTool

from health_coach.tools.data import (
    load_patient_history,
    load_config,
    update_patient_history,
    update_configuration,
)
from health_coach.tools.rl import (
    select_action,
    compute_reward,
    update_rl_model,
    discretize_probability,
)


class TaskProxy(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def default_description(self) -> Optional[str]:
        ...

    @abstractmethod
    def default_tools(self) -> Optional[List[BaseTool]]:
        ...

    @abstractmethod
    def default_context(self) -> Optional[List[Task]]:
        ...

    @abstractmethod
    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def default_output_json(self) -> Optional[Type[BaseModel]]:
        ...

    @abstractmethod
    def default_expected_output(self) -> Optional[str]:
        ...

    def create(
        self,
        agent: Agent,
        *,
        tools: Optional[List[BaseTool]] = None,
        context: Optional[List[Task]] = None,
        tools_input: Optional[Dict[str, Any]] = None,
        output_json: Optional[Type[BaseModel]] = None,
        expected_output: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Task:
        kwargs: Dict[str, Any] = {
            "agent": agent,
            "name": self.name(),
        }
        
        desc = description if description is not None else self.default_description()
        if desc is not None:
            kwargs["description"] = desc
        tls = tools if tools is not None else self.default_tools()
        if tls is not None:
            kwargs["tools"] = tls
        ctx = context if context is not None else self.default_context()
        if ctx is not None:
            kwargs["context"] = ctx
        ti = tools_input if tools_input is not None else self.default_tools_input()
        if ti is not None:
            kwargs["tools_input"] = ti
        oj = output_json if output_json is not None else self.default_output_json()
        if oj is not None:
            kwargs["output_json"] = oj
        eo = expected_output if expected_output is not None else self.default_expected_output()
        if eo is not None:
            kwargs["expected_output"] = eo

        return Task(**kwargs)

class NewHistoryResponse(BaseModel):
    new_history: Dict[str, Any]

class UpdatePatientHistory(TaskProxy):
    def name(self) -> str:
        return "update_patient_data"

    def default_description(self) -> Optional[str]:
        return "Append a new row for the patient’s history as valid JSON."

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [update_patient_history]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return None

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return NewHistoryResponse

    def default_expected_output(self) -> Optional[str]:
        return "{new_history: dict}"

class HistoryResponse(BaseModel):
    history: Dict[str, Any]

class FetchPatientHistory(TaskProxy):
    def name(self) -> str:
        return "fetch_patient_data"

    def default_description(self) -> Optional[str]:
        return (
            "Given patient info {patient_info}, invoke `load_patient_history` and "
            "RETURN ONLY `{ \"history\": <last_row_dict> }` as valid JSON."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [load_patient_history]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return None

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return HistoryResponse

    def default_expected_output(self) -> Optional[str]:
        return "{history: dict}"

class ConfigResponse(BaseModel):
    config: Dict[str, Any]

class LoadConfiguration(TaskProxy):
    def name(self) -> str:
        return "load_configuration"

    def default_description(self) -> Optional[str]:
        return (
            "Invoke `load_config` on the YAML path and RETURN ONLY "
            "`{ \"config\": <parsed_dict> }` as valid JSON."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [load_config]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return None

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return ConfigResponse

    def default_expected_output(self) -> Optional[str]:
        return "{config: dict}"


class NewConfigResponse(BaseModel):
    new_config: Dict[str, Any]


class UpdateConfiguration(TaskProxy):
    def name(self) -> str:
        return "update_configuration"

    def default_description(self) -> Optional[str]:
        return (
            "Given a partial config update, validate and write to YAML, then "
            "RETURN ONLY `{ \"new_config\": <updated_config_dict> }`."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [update_configuration]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "update_configuration": {
                "new_config": "{load_configuration.config}"
            }
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return NewConfigResponse

    def default_expected_output(self) -> Optional[str]:
        return "{new_config: dict}"


class StateResponse(BaseModel):
    state: int

class ComputeCurrentState(TaskProxy):
    def name(self) -> str:
        return "compute_current_state"

    def default_description(self) -> Optional[str]:
        return (
            "Given a raw probability {prediction}, invoke `discretize_probability` "
            "and RETURN ONLY `{ \"state\": <int> }`."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [discretize_probability]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "discretize_probability": {
                "state": "{prediction}"
            }
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return StateResponse

    def default_expected_output(self) -> Optional[str]:
        return "{state: int}"


class ActionResponse(BaseModel):
    action: int


class ComputeAction(TaskProxy):
    def name(self) -> str:
        return "compute_action"

    def default_description(self) -> Optional[str]:
        return (
            "Given state {compute_current_state.state} and exploration rate {epsilon}, "
            "invoke `select_action` and RETURN ONLY `{ \"action\": <int> }`."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [select_action]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "select_action": {
                "state": "{compute_current_state.state}",
                "epsilon": "{epsilon}"
            }
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return ActionResponse

    def default_expected_output(self) -> Optional[str]:
        return "{action: int}"


class RewardResponse(BaseModel):
    reward: float

class ComputeReward(TaskProxy):
    def name(self) -> str:
        return "compute_reward"

    def default_description(self) -> Optional[str]:
        return (
            "Given prev_state and current_state, invoke `compute_reward` and "
            "RETURN ONLY `{ \"reward\": <float> }`."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [compute_reward]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "compute_reward": {
                "prev_state": "{fetch_patient_data.history.state}",
                "current_state": "{compute_current_state.state}"
            }
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return RewardResponse

    def default_expected_output(self) -> Optional[str]:
        return "{reward: float}"


class ShapedRewardResponse(BaseModel):
    shaped_reward: float


class ShapeReward(TaskProxy):
    def name(self) -> str:
        return "shape_reward"

    def default_description(self) -> Optional[str]:
        return (
            "Given a suite of tools shaping rewards, the raw computed reward, "
            "and the provided context about the reinforcement system's state, shape the reward."
            " Note that simply passing along the pure computed reward is also valid."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return None

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "See tool comments"
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return ShapedRewardResponse

    def default_expected_output(self) -> Optional[str]:
        return "{shaped_reward: float}"

class ContextResponse(BaseModel):
    context: str

class ShapeRewardContext(TaskProxy):
    def name(self) -> str:
        return "provide_context"

    def default_description(self) -> Optional[str]:
        return (
            "Your responsibility is to provide a textual summary of this reinforcement learning system's state." 
            " This summary will be used upstream by a Reinforcement Learning specialist to augment the reward function" 
            " Use the provided context tools to analyze the data source and RETURN ONLY "
            "{ \"context\": <textual_summary> } ALWAYS. If UNABLE TO PROVIDE CONTEXT, RETURN ONLY { \"context\": \"Unavailable\"}"
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return None

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "get_visit_count":       {"state": "{compute_current_state.state}"},
            "get_transition_count":  {
                "prev_state":  "{fetch_patient_data.history.state}",
                "action":      "{compute_current_action.action}",
                "next_state":  "{compute_current_state.state}"
            },
            "get_moving_avg":        {"window": "5"},
            "get_episode_length":    {}
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return ContextResponse

    def default_expected_output(self) -> Optional[str]:
        return "{context: str}"

class ShapeAction(TaskProxy):
    def name(self) -> str:
        return "shape_action"

    def default_description(self) -> Optional[str]:
        return (
            "Given the current state and optional system context, select an action by dynamically choosing among "
            "provided exploration strategies (epsilon-greedy, softmax, UCB, Thompson sampling, count bonus, or maxent). "
            "Return ONLY { \"action\": <int> } as valid JSON."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return None

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "epsilon_greedy":    {"state": "{compute_current_state.state}", "epsilon": "{epsilon}"},
            "softmax":           {"state": "{compute_current_state.state}", "temperature": "{temperature}"},
            "ucb":               {"state": "{compute_current_state.state}", "c": "{ucb_coefficient}"},
            "thompson":          {"state": "{compute_current_state.state}"},
            "count_bonus":       {"state": "{compute_current_state.state}", "beta": "{beta}"},
            "maxent":            {"state": "{compute_current_state.state}", "alpha": "{alpha}"},
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return ActionResponse

    def default_expected_output(self) -> Optional[str]:
        return "{action: int}"

class UpdateRLResponse(BaseModel):
    success: bool

class UpdateRLModel(TaskProxy):
    def name(self) -> str:
        return "update_rl_model"

    def default_description(self) -> Optional[str]:
        return (
            "Invoke the necessary tools to update the RL model."
            "RETURN ONLY `{ \"success\": <bool> }`. "
            "If a shaped_reward is provided, prefer the shaped_reward over the reward."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return [update_rl_model]

    def default_context(self) -> Optional[List[Task]]:
        return None

    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return {
            "update_rl_model": {
                "state": "{fetch_patient_data.history.state}",
                "action": "{compute_action.action}",
                "reward": "{compute_reward.reward}",
                "next_state": "{compute_current_state.state}",
            }
        }

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return UpdateRLResponse

    def default_expected_output(self) -> Optional[str]:
        return "{success: bool}"
