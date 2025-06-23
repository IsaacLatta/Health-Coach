import copy
import math

from health_coach.tools.actions import (
    increase_moderate_threshold,
    decrease_moderate_threshold,
    increase_high_threshold,
    decrease_high_threshold,
    increase_top_k,
    decrease_top_k,
    nop,
    ACTIONS,
    ACTION_HANDLERS,
)

def make_base_config():
    return {
        "thresholds": {
            "moderate": 0.50,
            "high":     0.80,
        },
        "top_k": 3,
    }

def assert_equal(actual, expected, test_name):
    if isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-2):
            print(f"FAIL: {test_name} -> expected {expected:.2g}, got {actual:.2g}")
            raise AssertionError(test_name)
    else:
        if actual != expected:
            print(f"FAIL: {test_name} -> expected {expected!r}, got {actual!r}")
            raise AssertionError(test_name)

def test_increase_moderate_threshold_normal():
    cfg = make_base_config()
    out = increase_moderate_threshold(copy.deepcopy(cfg), step=0.10)
    assert_equal(out["thresholds"]["moderate"], 0.60, "increase_moderate_threshold_normal")

def test_increase_moderate_threshold_cap():
    cfg = make_base_config()
    cfg["thresholds"]["moderate"] = 0.95
    out = increase_moderate_threshold(copy.deepcopy(cfg), step=0.10)
    assert_equal(out["thresholds"]["moderate"], 0.95, "increase_moderate_threshold_cap")

def test_decrease_moderate_threshold_normal():
    cfg = make_base_config()
    out = decrease_moderate_threshold(copy.deepcopy(cfg), step=0.20)
    assert_equal(out["thresholds"]["moderate"], 0.30, "decrease_moderate_threshold_normal")

def test_decrease_moderate_threshold_floor():
    cfg = make_base_config()
    cfg["thresholds"]["moderate"] = 0.05
    out = decrease_moderate_threshold(copy.deepcopy(cfg), step=0.10)
    assert_equal(out["thresholds"]["moderate"], 0.05, "decrease_moderate_threshold_floor")

def test_increase_high_threshold_normal():
    cfg = make_base_config()
    out = increase_high_threshold(copy.deepcopy(cfg), step=0.10)
    assert_equal(out["thresholds"]["high"], 0.90, "increase_high_threshold_normal")

def test_increase_high_threshold_cap():
    cfg = make_base_config()
    cfg["thresholds"]["high"] = 0.98
    out = increase_high_threshold(copy.deepcopy(cfg), step=0.05)
    assert_equal(out["thresholds"]["high"], 0.98, "increase_high_threshold_cap")

def test_decrease_high_threshold_normal():
    cfg = make_base_config()
    out = decrease_high_threshold(copy.deepcopy(cfg), step=0.50)
    assert_equal(out["thresholds"]["high"], 0.30, "decrease_high_threshold_normal")

def test_decrease_high_threshold_floor():
    cfg = make_base_config()
    cfg["thresholds"]["high"] = 0.02
    out = decrease_high_threshold(copy.deepcopy(cfg), step=0.10)
    assert_equal(out["thresholds"]["high"], 0.02, "decrease_high_threshold_floor")

def test_increase_top_k():
    cfg = make_base_config()
    out = increase_top_k(copy.deepcopy(cfg), step=2)
    assert_equal(out["top_k"], 5, "increase_top_k")

def test_decrease_top_k_normal():
    cfg = make_base_config()
    out = decrease_top_k(copy.deepcopy(cfg), step=1)
    assert_equal(out["top_k"], 2, "decrease_top_k_normal")

def test_decrease_top_k_floor():
    cfg = make_base_config()
    cfg["top_k"] = 1
    out = decrease_top_k(copy.deepcopy(cfg), step=1)
    assert_equal(out["top_k"], 1, "decrease_top_k_floor")

def test_nop():
    cfg = make_base_config()
    clone = copy.deepcopy(cfg)
    out = nop(clone)
    assert_equal(out, cfg, "nop")

def test_action_handlers_mapping():
    expected = set(ACTIONS)
    got = set(ACTION_HANDLERS.keys())
    assert_equal(got, expected, "action_handlers_keys")
    for name, func in ACTION_HANDLERS.items():
        if not callable(func):
            print(f"FAIL: handler for {name} is not callable")
            raise AssertionError(f"{name} not callable")

def test_all():
    tests = [
        test_increase_moderate_threshold_normal,
        test_increase_moderate_threshold_cap,
        test_decrease_moderate_threshold_normal,
        test_decrease_moderate_threshold_floor,
        test_increase_high_threshold_normal,
        test_increase_high_threshold_cap,
        test_decrease_high_threshold_normal,
        test_decrease_high_threshold_floor,
        test_increase_top_k,
        test_decrease_top_k_normal,
        test_decrease_top_k_floor,
        test_nop,
        test_action_handlers_mapping,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError:
            print(f"{fn.__name__}: FAILED")
        else:
            print(f"{fn.__name__}: PASS")
            passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")

if __name__ == "__main__":
    test_all()
