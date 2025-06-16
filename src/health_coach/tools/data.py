import os
import tempfile
import time
import yaml
import pandas as pd
import csv
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
    print("Real YAML file passed.")

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

def _load_patient_data(patient_id: str, patient_folder: str, timeout_base: int = 1) -> dict:
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

def test_load_patient_history():
    patient_id = "1"
    print("Running load_patient_history tests ...")
    tmpdir = tempfile.mkdtemp()

    result = _load_patient_data("missing_patient", tmpdir)
    assert result == {}, f"Missing file failed: expected {{}}, got {result}"
    print("Missing file passed.")

    headers = [
        "Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120","EKG results",
        "Max HR","Exercise angina","ST depression","Slope of ST","Number of vessels fluro",
        "Thallium","Heart Disease",
        "Id","Date","Prediction","State","Action","Reward",
        "Eval_Date","Eval_Prediction","Next_State","True_Outcome"
    ]

    empty_path = os.path.join(tmpdir, f"{patient_id}.csv")
    with open(empty_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    result = _load_patient_data(patient_id, tmpdir, 0)
    assert result == {}, f"Empty file failed: expected {{}}, got {result}"
    print("Empty file passed.")

    one_path = os.path.join(tmpdir, f"{patient_id}.csv")
    row1 = {h: f"r1_{i}" for i, h in enumerate(headers)}
    with open(one_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row1)
    result = _load_patient_data(patient_id, tmpdir, 0)
    assert result == row1, f("Single row failed: expected {row1}, got {result}")
    print("Single row passed")

    row2 = {h: f"r2_{i}" for i, h in enumerate(headers)}
    with open(one_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row2)
    result = _load_patient_data(patient_id, tmpdir, 0)
    assert result == row2, f"Multiple rows failed: expected {row2}, got {result}"
    print("Multiple rows passed.")

    bad_path = os.path.join(tmpdir, f"{patient_id}.csv")
    with open(bad_path, "w") as f:
        f.write("not,a,proper,csv\n\"unterminated,quote")
    result = _load_patient_data(patient_id, tmpdir, 0)
    assert result == {}, f"Malformed CSV failed: expected {{}}, got {result}"
    print("Malformed CSV passed.")

    dummy_id = "0"
    dummy_id = "0"
    expected = {
        "Age": 72,
        "Sex": 0,
        "Chest pain type": 4,
        "BP": 130,
        "Cholesterol": 250,
        "FBS over 120": 0,
        "EKG results": 1,
        "Max HR": 140,
        "Exercise angina": 1,
        "ST depression": 1.4,
        "Slope of ST": 2,
        "Number of vessels fluro": 1,
        "Thallium": 3,
        "Heart Disease": 0,
        "Id": 0,
        "Date": "2025-06-10",               # stays a string
        "Prediction": 0.45,
        "State": 4,
        "Action": "DECREASE_MODERATE_THRESHOLD",
        "Reward": -1,
        "Eval_Date": "2025-06-11",
        "Eval_Prediction": 0.38,
        "Next_State": 3,
        "True_Outcome": 0
    }
    result = _load_patient_data(dummy_id, _PATIENT_DATA_DIRECTORY_, 0)
    assert result == expected, f"Dummy 0.csv: expected {expected}, got {result}"
    print("Dummy 0.csv test passed.")

    print("All load_patient_history tests passed.")

@tool("load_patient_history")
def load_patient_history(patient_id: str) -> dict:
    """
    patient_id: The id of the patient to load.
    Description: Reads only the last row of <patient_id>.csv and returns it as a dict.
                 Retries up to 3 times on I/O errors; returns {} if the file is missing,
                 malformed, empty, or if all attempts fail.
    """
    return _load_patient_data(patient_id, _PATIENT_DATA_DIRECTORY_)
