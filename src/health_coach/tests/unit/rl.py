# File: tests/test_rl.py

import sys
from pathlib import Path
import tempfile
import pickle
import math
import copy

import numpy as np
import health_coach.tools.rl as rl_mod

def assert_equal(actual, expected, name):
    """Compare actual to expected with 2-decimal tolerance for floats."""
    if isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, abs_tol=1e-2):
            print(f"FAIL: {name} -> expected {expected:.2g}, got {actual:.2g}")
            raise AssertionError(name)
    else:
        if actual != expected:
            print(f"FAIL: {name} -> expected {expected!r}, got {actual!r}")
            raise AssertionError(name)

def test_shape_reward():
    r = rl_mod._shape_reward(3.0, 1.5)
    assert_equal(r, 4.5, "_shape_reward")
    print("test_shape_reward: PASS")

def test_compute_reward():
    r = rl_mod._compute_reward(80, 50)
    assert_equal(r, 30.0, "_compute_reward")
    print("test_compute_reward: PASS")

def test_select_action_greedy():
    Q = np.array([[0.5, 0.1], [0.2, 0.8]])
    a = rl_mod._select_action(Q, state=1, epsilon=0.0)
    assert_equal(a, 1, "_select_action_greedy")
    print("test_select_action_greedy: PASS")

def test_update_q_table():
    Q0 = np.zeros((2,2))
    Q1 = rl_mod._update_q_table(
        Q0.copy(), state=0, action=0,
        reward=10.0, next_state=1,
        alpha=0.1, gamma=0.99
    )
    assert_equal(Q1[0,0], 1.0, "_update_q_table")
    print("test_update_q_table: PASS")

def test_persistent_update_and_save():
    tmp = tempfile.TemporaryDirectory()
    fake_path = Path(tmp.name) / "q.pkl"
    rl_mod._MODEL_PATH = fake_path

    with open(fake_path, "wb") as f:
        pickle.dump(np.zeros((2,2)), f)

    ok = rl_mod._update_rl_model(
        state=0, action=0, reward=10.0, next_state=1,
        alpha=0.1, gamma=0.99
    )
    assert_equal(ok, True, "_update_rl_model_return")

    with open(fake_path, "rb") as f:
        Q2 = pickle.load(f)
    assert_equal(Q2[0,0], 1.0, "_update_rl_model_effect")
    print("test_persistent_update_and_save: PASS")

    tmp.cleanup()

def test_apply_action():
    cfg = {"thresholds": {"moderate": 0.50, "high": 0.80}, "top_k": 3}
    out = rl_mod._apply_action(copy.deepcopy(cfg), 0)
    assert_equal(out["thresholds"]["moderate"], 0.55, "_apply_action_inc_mod")
    out2 = rl_mod._apply_action(copy.deepcopy(cfg), -1)
    assert_equal(out2, cfg, "_apply_action_invalid")
    print("test_apply_action: PASS")

def test_all():
    tests = [
        test_shape_reward,
        test_compute_reward,
        test_select_action_greedy,
        test_update_q_table,
        test_persistent_update_and_save,
        test_apply_action,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError:
            print(f"{fn.__name__}: FAILED")
        else:
            passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")

if __name__ == "__main__":
    test_all()
