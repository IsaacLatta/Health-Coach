from crewai import Task, Agent
from health_coach.tools.data import load_patient_history, load_config, update_patient_history, update_configuration

from health_coach.tools.rl import select_action, compute_reward, update_rl_model, discretize_probability, amplify_reward

from pydantic import BaseModel
from typing import Any, Dict

class NewHistoryResponse(BaseModel):
    new_history: Dict[str, Any]

def get_update_patient_history_task(agent: Agent):
    return Task(
        agent=agent,
        name="update_patient_data",
        tools=[update_patient_history],
        description=(
            "Append a new row for the patient’s history"
            "as valid JSON."
        ),
        expected_output="{new_history: dict}",
        output_json=NewHistoryResponse,
    )

class HistoryResponse(BaseModel):
    history: Dict[str, Any]

def get_fetch_patient_history_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="fetch_patient_data",
        tools=[load_patient_history],
        description=(
            "Given patient info {patient_info}, invoke the `load_patient_history` tool "
            "and RETURN ONLY a JSON object of the form "
            "`{ \"history\": <last_row_dict> }`.  No extra prose; keys and strings "
            "must be valid JSON."
        ),
        output_json=HistoryResponse,
        expected_output="{history: dict}"
    )

class ConfigResponse(BaseModel):
    config: Dict[str, Any]

def get_load_configuration_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="load_configuration",
        tools=[load_config],
        description=(
            "Invoke the `load_config` tool on the YAML path and RETURN ONLY "
            "`{ \"config\": <parsed_dict> }`.  No surrounding text; must be valid JSON."
        ),
        output_json=ConfigResponse,
        expected_output="{config: dict}"
    )

class NewConfigResponse(BaseModel):
    new_config: Dict[str, Any]

def get_update_configuration_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="update_configuration",
        tools=[update_configuration],
        tools_input={
            "update_configuration": {
                "new_config": "{load_configuration.config}"
            }
        },
        description=(
            "Given a partial config update, validate and write it to the YAML file, "
            "then **return only**:\n"
            "`{ \"new_config\": <updated_config_dict> }` as valid JSON."
        ),
        expected_output="{new_config: dict}",
        output_json=NewConfigResponse,
    )

class StateResponse(BaseModel):
    state: int

def get_compute_current_state_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="compute_current_state",
        tools=[discretize_probability],
        description=(
            "Given a raw probability {prediction}, invoke discretize_probability and "
            "RETURN ONLY JSON {\"state\": <int>}."
        ),
        tools_input={
            "discretize_probability": {
                "state" : "{prediction}"
            }
        },
        output_json=StateResponse,
        expected_output="{state: int}",
    )

class ActionResponse(BaseModel):
    action: int

def get_compute_action_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="compute_action",
        tools=[select_action],
        tools_input={
            "select_action": {
                "state": "{compute_current_state.state}",
                "epsilon": "{epsilon}"
            }
        },
        description=(
            "Given the discrete state from compute_current_state and exploration rate {epsilon}, "
            "invoke select_action and RETURN ONLY JSON {\"action\": <int>}."
        ),
        output_json=ActionResponse,
        expected_output="{action: int}",
    )

class RewardResponse(BaseModel):
    reward: float

def get_compute_reward_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="compute_reward",
        tools=[compute_reward],
        tools_input={
            "compute_reward": {
                "prev_state": "{fetch_patient_data.history.state}",
                "current_state": "{compute_current_state.state}"
            }
        },
        description=(
            "Given previous state {fetch_patient_data.history.State} from history and current state {compute_current_state.state}, invoke compute_reward "
            "and RETURN ONLY JSON {\"reward\": <float>}."
        ),
        output_json=RewardResponse,
        expected_output="{reward: float}",
    )

class ShapedRewardResponse(BaseModel):
    shaped_reward: float

def get_shape_reward_task(agent: Agent, ctx: list[Task]) -> Task:
    return Task(
        agent=agent,
        name="shape_reward",
        tools=[amplify_reward],
        tools_input={
            "amplify_reward" : {
                "reward" : "{compute_reward.reward}",
                "factor" : "A factor of your choosing."
            }
        },
        description=("Given the patient's features found in their history, "
        "their previous MDP values, you have the opportunity to augment this reward using the provided tools."),
        output_json=ShapedRewardResponse,
        expected_output="{shaped_reward: float}",
        context=ctx
    )

class UpdateRLResponse(BaseModel):
    success: bool

def get_update_rl_model_task(agent: Agent) -> Task:
    return Task(
        agent=agent,
        name="update_rl_model",
        tools=[update_rl_model],
        tools_input={
            "update_rl_model": {
                "state": "{fetch_patient_history.history.state}",
                "action": "{compute_action.action}",
                "reward": "{compute_reward.reward}",
                "next_state": "{compute_current_state.state}"
            }
        },
        description=(
            "Using state[{fetch_patient_history.history.state}], action[{compute_action.action}], reward[{compute_reward.reward}], "
            "and next_state[{compute_current_state.state}], invoke update_rl_model "
            "and RETURN ONLY JSON {\"success\": <bool>}."
        ),
        output_json=UpdateRLResponse,
        expected_output="{success: bool}",
    )

