from crewai import Crew, Process, Task, Agent

from health_coach.tools.data import load_config
from health_coach.tools.data import load_patient_history

def get_fetch_patient_history_task(agent: Agent):
    return Task(
            agent=agent,
            name="fetch_patient_data",
            tools=[load_patient_history],
            description=(
                "Given a patient_id: {patient_info}, load and return the patient's history. "
                " The expected output of the tool is a dictionary representing the patient's " \
                " last history entry. If this tool fails, do attempt to retry, "
                " simply forward along the empty dictionary."
            ),
            expected_output="{\"history\": dict}"
        )

def get_load_configuration_task(agent: Agent):
    return Task(
        agent=agent,
        name="load_configuration",
        tools=[load_config],
        description=(
            "Load and return the parsed dictionary, the expected output from load_config" \
            "should be a dictionary representing the systems configuration."
            "If this tool fails, do attempt to retry, simply forward along the empty dictionary"
        ),
        expected_output="{config: dict}"
    )
