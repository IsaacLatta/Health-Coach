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
        "Date": "2025-06-10",
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

def test_update_patient_history():
    tmpdir = tempfile.mkdtemp()
    patient_id = "test"
    feature_names = [
        "Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120","EKG results",
        "Max HR","Exercise angina","ST depression","Slope of ST","Number of vessels fluro",
        "Thallium","Heart Disease"
    ]
    features1 = list(range(14))                 
    prediction1 = 0.5
    state1 = 5
    action1 = "ACTION1"
    reward1 = 1

    today = datetime.date.today().isoformat()
    expected_row1 = {
        **{feature_names[i]: features1[i] for i in range(14)},
        "Id": patient_id,
        "Date": today,
        "Prediction": prediction1,
        "State": state1,
        "Action": action1,
        "Reward": reward1,
        "Eval_Date": "",
        "Eval_Prediction": "",
        "Next_State": "",
        "True_Outcome": "",
    }

    result1 = _update_patient_history(
        patient_id, tmpdir,
        feature_names, features1,
        prediction1, state1, action1, reward1
    )
    assert result1 == expected_row1, f"Returned row1 mismatch: {result1}"

    csv_path = os.path.join(tmpdir, f"{patient_id}.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_header = feature_names + [
        "Id","Date","Prediction","State","Action","Reward",
        "Eval_Date","Eval_Prediction","Next_State","True_Outcome"
    ]
    assert reader.fieldnames == expected_header, f"Header mismatch: {reader.fieldnames}"

    expected_str_row1 = {k: ("" if v == "" else str(v)) for k,v in expected_row1.items()}
    assert rows[0] == expected_str_row1, f"CSV row1 mismatch: {rows[0]}"

    print("First row appended and verified")

    features2   = [x + 10 for x in features1]
    prediction2 = 0.6
    state2      = 6
    action2     = "ACTION2"
    reward2     = 0

    expected_row2 = {
        **{feature_names[i]: features2[i] for i in range(14)},
        "Id": patient_id,
        "Date": today,
        "Prediction": prediction2,
        "State": state2,
        "Action": action2,
        "Reward": reward2,
        "Eval_Date": "",
        "Eval_Prediction": "",
        "Next_State": "",
        "True_Outcome": "",
    }

    result2 = _update_patient_history(
        patient_id, tmpdir,
        feature_names, features2,
        prediction2, state2, action2, reward2
    )
    assert result2 == expected_row2, f"Returned row2 mismatch: {result2}"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_str_row2 = {k: ("" if v == "" else str(v)) for k,v in expected_row2.items()}
    assert rows[1] == expected_str_row2, f"CSV row2 mismatch: {rows[1]}"

    print("Second row appended and verified")
    print("All update_patient_history tests passed.")

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

def test_update_configuration():
    tmpdir = tempfile.mkdtemp()
    config_path = Path(tmpdir) / "config.yaml"

    # 1) New file: update both thresholds and top_k
    new_cfg = {"thresholds": {"moderate": 0.2, "high": 0.8}, "explanation": {"top_k": 5}}
    result = _update_configuration(new_cfg, config_path)
    expected = {"thresholds": {"moderate": 0.2, "high": 0.8}, "explanation": {"top_k": 5}}
    assert result == expected, f"New file update failed: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, f"Config file mismatch: {file_cfg}"
    print("✅ new file update")

    # 2) Existing file has explanation only; update thresholds only
    initial = {"explanation": {"top_k": 4}}
    config_path.write_text(yaml.dump(initial))
    new_thresh = {"thresholds": {"moderate": 0.1, "high": 0.9}}
    result = _update_configuration(new_thresh, config_path)
    expected = {"explanation": {"top_k": 4}, "thresholds": {"moderate": 0.1, "high": 0.9}}
    assert result == expected, f"Update thresholds only failed: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, f"Config after thresholds update mismatch: {file_cfg}"
    print("✅ update thresholds only")

    # 3) Existing file has thresholds only; update explanation only
    initial = {"thresholds": {"moderate": 0.3, "high": 0.7}}
    config_path.write_text(yaml.dump(initial))
    new_exp = {"explanation": {"top_k": 8}}
    result = _update_configuration(new_exp, config_path)
    expected = {"thresholds": {"moderate": 0.3, "high": 0.7}, "explanation": {"top_k": 8}}
    assert result == expected, f"Update explanation only failed: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, f"Config after explanation update mismatch: {file_cfg}"
    print("✅ update explanation only")

    # 4) Missing threshold field → should return {} and not create file
    path2 = Path(tmpdir) / "config2.yaml"
    result = _update_configuration({"thresholds": {"moderate": 0.4}}, path2)
    assert result == {}, f"Missing high threshold should fail: got {result}"
    assert not path2.exists(), "Config file should not be created on failure"
    print("✅ missing threshold field validation")

    # 5) Invalid threshold range → high <= moderate
    config_path.write_text(yaml.dump(expected))
    bad = {"thresholds": {"moderate": 0.6, "high": 0.5}}
    result = _update_configuration(bad, config_path)
    assert result == {}, f"Invalid threshold range should fail: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, "Config file should remain unchanged on failure"
    print("✅ threshold range validation")

    # 6) Invalid threshold bounds → outside [0,1]
    bad = {"thresholds": {"moderate": -0.1, "high": 1.1}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "Out-of-bounds thresholds should fail"
    print("✅ threshold bounds validation")

    # 7) Non-int top_k → should fail
    bad = {"explanation": {"top_k": "five"}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "Non-int top_k should fail"
    print("✅ top_k type validation")

    # 8) Out-of-range top_k → 0 or >=13
    bad = {"explanation": {"top_k": 0}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "top_k = 0 should fail"
    bad = {"explanation": {"top_k": 13}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "top_k >= 13 should fail"
    print("✅ top_k range validation")

    print("All _update_configuration tests passed!")

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
