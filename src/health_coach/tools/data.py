import os
import tempfile
import time
import yaml
import pandas as pd
from pathlib import Path
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

def test_load_config():
    valid_data = {
        "thresholds": {"moderate": 0.3, "high": 0.7},
        "explanation": {"top_k": 3},
    }

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(valid_data, tmp)
        valid_path = tmp.name
    result = _load_config(valid_path)
    assert result == valid_data, f"VALID YAML failed: got {result}"
    print("Example YAML passed.")
    os.remove(valid_path)

    result = _load_config(_CONFIG_PATH_)
    assert valid_data == result, f"REAL YAML failed: got {result}"
    print("REal YAML file passed.")

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        empty_path = tmp.name
    result = _load_config(empty_path)
    assert result is None, f"EMPTY YAML failed: expected None, got {result}"
    print("Empty YAML passed.")
    os.remove(empty_path)

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        tmp.write("::: not valid YAML :::")
        bad_path = tmp.name
    result = _load_config(bad_path)
    assert result == {}, f"MALFORMED YAML failed: expected {{}}, got {result}"
    print("Malformed YAML passed.")
    os.remove(bad_path)

    missing_path = os.path.join(tempfile.gettempdir(), "dne_file.yaml")
    try:
        os.remove(missing_path)
    except OSError:
        pass
    result = _load_config(missing_path)
    assert result == {}, f"MISSING file failed: expected {{}}, got {result}"
    print("Missing YAML file passed")

    print("All load_config tests passed.")

@tool("load_configuration")
def load_config() -> dict:
    """
    Loads and parses a YAML config file returning a dictionary representing the file's contents.
    Retries up to 3 times with exponential backoff on I/O errors.
    Returns an empty dict if all attempts fail.
    """
    return _load_config(_CONFIG_PATH_)

# @tool("load_patient_history")
# def load_patient_history(patient_id: str) -> dict:
#     """
#     patient_id: The id of the patient to load.
#     Description: Reads only the last row of <patient_id>.csv and returns it as a dict.
#                  Retries up to 3 times on I/O errors; returns {} if the file is missing,
#                  malformed, empty, or if all attempts fail.
#     """
#     csv_path = os.path.join(_PATIENT_DATA_DIRECTORY_, f"{patient_id}.csv")
#     for attempt in range(3):
#         try:
#             df = pd.read_csv(csv_path)
#             if df.empty:
#                 return {}
#             # take only the last row
#             last_row = df.iloc[-1]
#             return last_row.to_dict()
#         except Exception:
#             time.sleep(2 ** attempt)
#     return {}