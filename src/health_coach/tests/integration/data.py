from pathlib import Path
import yaml

from crewai import Crew, Process

import health_coach.tools.data as data_mod
from health_coach.agents import data_input_agent, data_export_agent
from health_coach.tasks import (
    get_load_configuration_task,
    get_fetch_patient_history_task,
    get_update_configuration_task,
)

# point to your real resources
BASE = Path(__file__).resolve().parents[4] / "resources"
CONFIG_PATH = BASE / "config.yaml"
PATIENT_DIR = BASE / "patient_data"

# load the "ground truth" from disk
EXPECTED_CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
EXPECTED_HISTORY_0 = {
    "Age": 72, "Sex": 0, "Chest pain type": 4, "BP": 130,
    "Cholesterol": 250, "FBS over 120": 0, "EKG results": 1,
    "Max HR": 140, "Exercise angina": 1, "ST depression": 1.4,
    "Slope of ST": 2, "Number of vessels fluro": 1, "Thallium": 3,
    "Heart Disease": 0, "Id": 0, "Date": "2025-06-10",
    "Prediction": 0.45, "State": 4,
    "Action": "DECREASE_MODERATE_THRESHOLD", "Reward": -1,
    "Eval_Date": "2025-06-11", "Eval_Prediction": 0.38,
    "Next_State": 3, "True_Outcome": 0
}

def test_loading_pipeline():
    t1 = get_load_configuration_task(data_input_agent)
    t2 = get_fetch_patient_history_task(data_input_agent)

    crew = Crew(
        agents=[data_input_agent],
        tasks=[t1, t2],
        process=Process.sequential,
        verbose=False,
    )
    out = crew.kickoff({
        "patient_info": {"patient_id": "0"},
    })

    config_out, history_out = out.tasks_output
    assert config_out.json_dict["config"] == EXPECTED_CONFIG, (
        f"CONFIG mismatch:\nexpected {EXPECTED_CONFIG}\ngot {config_out}"
    )
    assert history_out.json_dict["history"] == EXPECTED_HISTORY_0, (
        f"HISTORY mismatch:\nexpected {EXPECTED_HISTORY_0}\ngot {history_out}"
    )
    print("test_loading_pipeline passed.")


def test_loading_and_saving_pipeline():
    new_cfg = {
        "thresholds":  {"moderate": 0.33, "high": 0.66},
        "explanation": {"top_k": 5},
    }
    data_mod._load_config = lambda path: new_cfg.copy()

    t1 = get_load_configuration_task(data_input_agent)
    t2 = get_fetch_patient_history_task(data_input_agent)
    t3 = get_update_configuration_task(data_export_agent)

    crew = Crew(
        agents=[data_input_agent, data_export_agent],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False,
    )
    out = crew.kickoff({
        "patient_info": {"patient_id": "0"}
    })

    cfg1, hist, cfg2 = out.tasks_output

    assert cfg1.json_dict["config"] == new_cfg, (
        f"STUBBED CONFIG mismatch:\nexpected {new_cfg}\ngot {cfg1}"
    )
    assert hist.json_dict["history"] == EXPECTED_HISTORY_0, (
        f"HISTORY mismatch:\nexpected {EXPECTED_HISTORY_0}\ngot {hist}"
    )
    print(f"\nNEW CONFIG RAW: {cfg2.raw}\n")
    print(f"\nNEW CONFIG JSON: {cfg2.json_dict}\n")


    assert cfg2.json_dict["new_config"] == new_cfg, (
        f"UPDATED CONFIG mismatch:\nexpected {new_cfg}\ngot {cfg2}"
    )
    print("test_loading_and_saving_pipeline passed.")
