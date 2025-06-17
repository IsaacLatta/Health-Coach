import os
import tempfile
import time
import yaml
import pandas as pd
import csv
from pathlib import Path
import datetime

from health_coach.tools.data import _load_config, _load_patient_data, _update_patient_history, _update_configuration

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

    print("All load_patient_history tests passed.")

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
    print("All load_patient_history tests passed.")

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


def test_update_configuration():
    tmpdir = tempfile.mkdtemp()
    config_path = Path(tmpdir) / "config.yaml"

    new_cfg = {"thresholds": {"moderate": 0.2, "high": 0.8}, "explanation": {"top_k": 5}}
    result = _update_configuration(new_cfg, config_path)
    expected = {"thresholds": {"moderate": 0.2, "high": 0.8}, "explanation": {"top_k": 5}}
    assert result == expected, f"New file update failed: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, f"Config file mismatch: {file_cfg}"
    print("New file update passed.")

    initial = {"explanation": {"top_k": 4}}
    config_path.write_text(yaml.dump(initial))
    new_thresh = {"thresholds": {"moderate": 0.1, "high": 0.9}}
    result = _update_configuration(new_thresh, config_path)
    expected = {"explanation": {"top_k": 4}, "thresholds": {"moderate": 0.1, "high": 0.9}}
    assert result == expected, f"Update thresholds only failed: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, f"Config after thresholds update mismatch: {file_cfg}"
    print("Update thresholds only passed.")

    initial = {"thresholds": {"moderate": 0.3, "high": 0.7}}
    config_path.write_text(yaml.dump(initial))
    new_exp = {"explanation": {"top_k": 8}}
    result = _update_configuration(new_exp, config_path)
    expected = {"thresholds": {"moderate": 0.3, "high": 0.7}, "explanation": {"top_k": 8}}
    assert result == expected, f"Update explanation only failed: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, f"Config after explanation update mismatch: {file_cfg}"
    print("Update explanation only passed")

    path2 = Path(tmpdir) / "config2.yaml"
    result = _update_configuration({"thresholds": {"moderate": 0.4}}, path2)
    assert result == {}, f"Missing high threshold should fail: got {result}"
    assert not path2.exists(), "Config file should not be created on failure"
    print("Missing threshold field validation passed.")

    config_path.write_text(yaml.dump(expected))
    bad = {"thresholds": {"moderate": 0.6, "high": 0.5}}
    result = _update_configuration(bad, config_path)
    assert result == {}, f"Invalid threshold range should fail: got {result}"
    file_cfg = yaml.safe_load(config_path.read_text())
    assert file_cfg == expected, "Config file should remain unchanged on failure"
    print("Threshold range validation passed.")

    bad = {"thresholds": {"moderate": -0.1, "high": 1.1}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "Out-of-bounds thresholds should fail"
    print("Threshold bounds validation passed.")

    bad = {"explanation": {"top_k": "five"}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "Non-int top_k should fail"
    print("top_k type validation passed.")

    bad = {"explanation": {"top_k": 0}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "top_k = 0 should fail"
    bad = {"explanation": {"top_k": 13}}
    result = _update_configuration(bad, config_path)
    assert result == {}, "top_k >= 13 should fail"
    print("top_k range validation passed.")

    print("All _update_configuration tests passed!")

def test_all():
    test_load_config()
    test_load_patient_history()
    test_update_configuration()
    test_update_patient_history()