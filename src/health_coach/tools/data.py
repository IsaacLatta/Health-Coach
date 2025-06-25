import os
import tempfile
import time
import yaml
import pandas as pd
import csv
from pathlib import Path
import datetime
from crewai import Task
from crewai.tools import tool

_PATIENT_DATA_DIRECTORY_ =  Path(__file__).resolve().parents[3]/"resources"/"patient_data"
_CONFIG_PATH_ = Path(__file__).resolve().parents[3]/"resources"/"config.yaml"

def _load_config(config_path: str) -> dict:
    for attempt in range(3):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            time.sleep(2 ** attempt)
    return {}

@tool("load_configuration")
def load_config() -> dict:
    """
    Loads and parses a YAML config file returning a dictionary representing the file's contents.
    Retries up to 3 times with exponential backoff on I/O errors.
    Returns an empty dict if all attempts fail.
    """
    return _load_config(_CONFIG_PATH_)

def _load_patient_data(patient_id: str, patient_folder: str =  _PATIENT_DATA_DIRECTORY_, timeout_base: int = 1) -> dict:
    csv_path = os.path.join(patient_folder, f"{patient_id}.csv")
    for attempt in range(3):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return {}
            # take only the last row
            last_row = df.iloc[-1]
            return last_row.to_dict()
        except Exception:
            time.sleep(timeout_base*(2 ** attempt))
    return {}

@tool("load_patient_history")
def load_patient_history(patient_id: str) -> dict:
    """
    patient_id: The id of the patient to load.
    Description: Reads only the last row of <patient_id>.csv and returns it as a dict.
                 Retries up to 3 times on I/O errors; returns {} if the file is missing,
                 malformed, empty, or if all attempts fail.
    """
    return _load_patient_data(patient_id)

def _update_patient_history(
    patient_id: str,
    patient_folder: str,
    feature_names: list[str],
    features: list,
    prediction: float,
    state: int,
    action: str,
    reward: int,
) -> dict:
    csv_path = Path(patient_folder) / f"{patient_id}.csv"

    rl_cols = [
        "Id", "Date", "Prediction", "State",
        "Action", "Reward",
        "Eval_Date", "Eval_Prediction",
        "Next_State", "True_Outcome",
    ]
    header = feature_names + rl_cols

    today = datetime.date.today().isoformat()
    row = {name: val for name, val in zip(feature_names, features)}
    row.update({
        "Id": patient_id,
        "Date": today,
        "Prediction": prediction,
        "State": state,
        "Action": action,
        "Reward": reward,
        "Eval_Date": "",
        "Eval_Prediction": "",
        "Next_State": "",
        "True_Outcome": "",
    })

    for attempt in range(3):
        try:
            os.makedirs(patient_folder, exist_ok=True)
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            return row

        except Exception:
            time.sleep(2 ** attempt)
    return {}

@tool("update_patient_history")
def update_patient_history(
    patient_id: str,
    feature_names: list[str],
    features: list,
    prediction: float,
    state: int,
    action: str,
    reward: int,
) -> dict:
    """
    Append a new CSV row for <patient_id>.csv in `patient_folder`.
    - feature_names/features define the original columns.
    - prediction/state/action/reward fill the RL columns.
    Leaves Eval_Date, Eval_Prediction, Next_State, True_Outcome blank.
    Retries up to 3× on I/O errors; returns the appended row as a dict on success or {}
    on persistent failure.
    """
    return _update_patient_history(
        patient_id, 
        _PATIENT_DATA_DIRECTORY_, 
        feature_names, 
        features, 
        prediction, 
        state, 
        action, 
        reward)

def _update_configuration(new_config: dict, config_path: str) -> dict:
    for attempt in range(3):
        try:
            try:
                existing_cfg = yaml.safe_load(config_path.read_text()) or {}
            except FileNotFoundError:
                existing_cfg = {}

            if "thresholds" in new_config:
                th_in = new_config["thresholds"]
                mod = th_in.get("moderate", existing_cfg.get("thresholds", {}).get("moderate"))
                high = th_in.get("high",   existing_cfg.get("thresholds", {}).get("high"))

                if mod is None or high is None:
                    raise ValueError("Both 'moderate' and 'high' thresholds must be provided")
                if not (0.0 <= mod < high <= 1.0):
                    raise ValueError(f"Thresholds must satisfy 0.0 <= moderate < high <= 1.0 (got {mod}, {high})")

                existing_cfg.setdefault("thresholds", {})
                existing_cfg["thresholds"]["moderate"] = float(mod)
                existing_cfg["thresholds"]["high"]     = float(high)

            if "explanation" in new_config and "top_k" in new_config["explanation"]:
                tk = new_config["explanation"]["top_k"]
                if not isinstance(tk, int):
                    raise ValueError(f"top_k must be an integer (got {type(tk)})")
                if not (1 <= tk < 13):
                    raise ValueError(f"top_k must be in [1, 12] (got {tk})")

                existing_cfg.setdefault("explanation", {})
                existing_cfg["explanation"]["top_k"] = tk

            yaml.safe_dump(
                existing_cfg,
                config_path.open("w"),
                sort_keys=False,
                default_flow_style=False,
            )
            return existing_cfg
        except Exception:
            time.sleep(2 ** attempt)
    return {}

@tool("update_configuration")
def update_configuration(new_config: dict) -> dict:
    """
    Update and validate the YAML config for the system.
    Retries up to 3 times on I/O or validation errors. On success, writes the updated
    config (preserving YAML structure) and returns the new dict representing the new structure.
    An empty return {} means an error occurred, do not reattempt, simply output the empty dict,
    The empty dict is expected by the system, not an exception, the system is designed to handle it accordingly.
    """
    return _update_configuration(new_config, _CONFIG_PATH_)
