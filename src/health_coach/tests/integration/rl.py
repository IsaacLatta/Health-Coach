import numpy as np
import pytest
from crewai import Crew, Process

import health_coach.tools.rl as rl_mod

from health_coach.tasks import (
    get_compute_current_state_task,
    get_compute_action_task,
    get_compute_reward_task,
    get_update_rl_model_task,
)

from health_coach.agents import policy_agent, reward_shaping_agent

def stub_discretize_probability(prediction: float) -> int:
    return int(np.floor(prediction * 10))

def stub_select_action(state: int, epsilon: float) -> int:
    return state % 2

def stub_compute_reward(prev_state: int, current_state: int) -> float:
    return float(current_state - prev_state)

stub_update_rl_model_called = {}
def stub_update_rl_model(state: int, action: int, reward: float, next_state: int) -> bool:
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
    monkeypatch.setattr(rl_mod, "compute_reward", stub_compute_reward)

    task = get_compute_reward_task(reward_shaping_agent)
    crew = Crew(
        agents=[reward_shaping_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    inputs = {
        "fetch_patient_data.history.State": 4,
        "compute_current_state.state": 7
    }
    result = crew.kickoff(inputs)
    reward = result.tasks_output[0].json_dict["reward"]
    assert reward == 3.0

def test_update_rl_model_task(monkeypatch):
    monkeypatch.setattr(rl_mod, "update_rl_model", stub_update_rl_model)

    task = get_update_rl_model_task(reward_shaping_agent)
    crew = Crew(
        agents=[reward_shaping_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    inputs = {
        "compute_current_state.state": 2,
        "compute_action.action": 1,
        "apply_reward.reward": -1.0,
        "compute_current_state.state": 2,
    }
    result = crew.kickoff(inputs)
    success = result.tasks_output[0].json_dict["success"]
    assert success is True
    assert stub_update_rl_model_called["params"] == (2, 1, -1.0, 2)

def test_mock_learning_loop(monkeypatch):
    monkeypatch.setattr(rl_mod, "discretize_probability", stub_discretize_probability)
    monkeypatch.setattr(rl_mod, "select_action", stub_select_action)
    monkeypatch.setattr(rl_mod, "compute_reward", stub_compute_reward)
    monkeypatch.setattr(rl_mod, "update_rl_model", stub_update_rl_model)

    tasks = [
        get_compute_current_state_task(policy_agent),
        get_compute_action_task(policy_agent),
        get_compute_reward_task(reward_shaping_agent),
        get_update_rl_model_task(reward_shaping_agent),
    ]
    crew = Crew(
        agents=[policy_agent, reward_shaping_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    inputs = {
        "prediction": 0.32,
        "epsilon": 0.5,
        "fetch_patient_data.history.State": 1
    }
    out = crew.kickoff(inputs)

    state   = out.tasks_output[0].json_dict["state"]
    action  = out.tasks_output[1].json_dict["action"]
    reward  = out.tasks_output[2].json_dict["reward"]
    success = out.tasks_output[3].json_dict["success"]

    assert state  == stub_discretize_probability(0.32)
    assert action == state % 2
    assert reward == float(state - 1)
    assert success is True

    expected = (state, action, reward, state)
    assert stub_update_rl_model_called["params"] == expected

def test_all():
    mp = pytest.MonkeyPatch()
    tests = [
        test_compute_current_state_task,
        test_compute_action_task,
        # test_compute_reward_task,
        # test_update_rl_model_task,
        # test_mock_learning_loop,
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
