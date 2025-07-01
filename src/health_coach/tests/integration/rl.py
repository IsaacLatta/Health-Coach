import numpy as np
import tempfile
from pathlib import Path
import pytest
from crewai import Crew, Process

from health_coach.rl import QLearningImplementation

import health_coach.tools.rl as rl_mod
import health_coach.tools.data as data_mod

def stub_discretize_probability(prediction: float) -> int:
    return int(np.floor(prediction * 10))

def stub_select_action(state: int, epsilon: float) -> int:
    return 3

def stub_compute_reward(prev_state: int, current_state: int) -> float:
    return float(current_state - prev_state)

stub_update_rl_model_called = {}
def stub_update_rl_model(state: int, action: int, reward: float, next_state: int) -> bool:
    global stub_update_rl_model_called
    stub_update_rl_model_called['params'] = (state, action, reward, next_state)
    return True

def test_rl(monkeypatch):
    monkeypatch.setattr(rl_mod, "_update_rl_model", stub_update_rl_model)
    monkeypatch.setattr(rl_mod, "_select_action", stub_select_action)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "0.csv").write_text(
            "Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,"
            "Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,"
            "Thallium,Heart Disease,Id,Date,Prediction,State,Action,Reward,"
            "Eval_Date,Eval_Prediction,Next_State,True_Outcome\n"
            "72,0,4,130,250,0,1,140,1,1.4,2,1,3,0,0,2025-06-10,0.45,"
            "4,DECREASE_MODERATE_THRESHOLD,,,,\n"
        )

        func = data_mod._load_patient_data
        func.__defaults__ = (str(tmpdir), func.__defaults__[1])
        monkeypatch.setattr(data_mod, "_load_patient_data", func)

        result = QLearningImplementation().create_crew().kickoff({
            "prediction": 0.7,
            "patient_info": {"patient_id": "0"},
            "epsilon": 0.1
        })

        print(f"RAW: {result.raw}")
        assert result.json_dict["success"] is True
        assert stub_update_rl_model_called['params'] == (4, 3, -3.0, 7)

def test_all():
    mp = pytest.MonkeyPatch()
    tests = [
        test_rl
    ]

    passed = 0
    for fn in tests:
        try:
            if fn.__code__.co_argcount == 1:
                fn(mp)
            else:
                fn()
        except AssertionError as e:
            print(f"{fn.__name__}: FAILED: {e}")
        else:
            print(f"{fn.__name__}: PASS")
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed.")
    mp.undo()
