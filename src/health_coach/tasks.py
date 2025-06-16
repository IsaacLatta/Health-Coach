from crewai import Task, Agent
from health_coach.tools.data import load_patient_history, load_config

# src/health_coach/schemas.py
from pydantic import BaseModel
from typing import Any, Dict

class HistoryResponse(BaseModel):
    history: Dict[str, Any]

class ConfigResponse(BaseModel):
    config: Dict[str, Any]


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
