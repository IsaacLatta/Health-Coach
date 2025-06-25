import numpy as np
import tempfile
from pathlib import Path
import pytest
from crewai import Crew, Process

import health_coach.tools.rl as rl_mod
import health_coach.tools.data as data_mod

from health_coach.tasks import (
    get_compute_current_state_task,
    get_compute_action_task,
    get_compute_reward_task,
    get_update_rl_model_task,
    get_fetch_patient_history_task,
    get_shape_reward_task
)

from health_coach.agents import policy_agent, reward_shaping_agent, data_input_agent

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

def test_compute_current_state_task(monkeypatch):
    task = get_compute_current_state_task(policy_agent)
    crew = Crew(
        agents=[policy_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff({"prediction": 0.37})
    state = result.tasks_output[0].json_dict["state"]
    assert state == 3

def test_compute_action_task(monkeypatch):
    t1 = get_compute_current_state_task(policy_agent)
    t2 = get_compute_action_task(policy_agent)
    crew = Crew(
        agents=[policy_agent],
        tasks=[t1, t2],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff({"prediction": 0.57, "epsilon": 0.1})
    print("result raw: {result.raw}")
    action = result.json_dict["action"]
    assert action == 1

# TODO
def test_compute_reward_task(monkeypatch):
    # monkeypatch.setattr(rl_mod, "compute_reward", stub_compute_reward)

    t0 = get_fetch_patient_history_task(data_input_agent)
    t1 = get_compute_current_state_task(policy_agent)
    t2 = get_compute_reward_task(reward_shaping_agent)
    crew = Crew(
        agents=[reward_shaping_agent],
        tasks=[t0, t1, t2],
        process=Process.sequential,
        verbose=False,
    )

    inputs = {
        "prediction" : 0.7,
        "patient_info": {"patient_id": "0"},
    }

    result = crew.kickoff(inputs)
    print(f"Raw: {result.raw}")
    reward = result.json_dict["reward"]
    assert reward == -3.0

def test_update_rl_model_task(monkeypatch):
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

        t0 = get_fetch_patient_history_task(data_input_agent)
        t1 = get_compute_current_state_task(policy_agent)
        t2 = get_compute_reward_task(reward_shaping_agent)
        t3 = get_compute_action_task(policy_agent)
        t4 = get_update_rl_model_task(reward_shaping_agent)
        crew = Crew(agents=[reward_shaping_agent],
                    tasks=[t0, t1, t2, t3, t4],
                    process=Process.sequential,
                    verbose=False)

        result = crew.kickoff({
            "prediction": 0.7,
            "patient_info": {"patient_id": "0"},
            "epsilon": 0.1
        })

        assert result.json_dict["success"] is True
        assert stub_update_rl_model_called['params'] == (4, 3, -3.0, 7)

def test_augment_reward(monkeypatch):
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

        t0 = get_fetch_patient_history_task(data_input_agent)
        t1 = get_compute_current_state_task(policy_agent)
        t2 = get_compute_reward_task(reward_shaping_agent)
        t3 = get_compute_action_task(policy_agent)
        t4 = get_shape_reward_task(reward_shaping_agent, [t0, t2])
        t5 = get_update_rl_model_task(reward_shaping_agent)
        crew = Crew(agents=[reward_shaping_agent],
                    tasks=[t0, t1, t2, t3, t4, t5],
                    process=Process.sequential,
                    verbose=False)

        result = crew.kickoff({
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
        # test_compute_current_state_task,
        # test_compute_action_task,
        # test_compute_reward_task,
        # test_update_rl_model_task,
        test_augment_reward,
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
