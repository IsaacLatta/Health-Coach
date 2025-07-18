import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

ACTIONS = [
    "INCREASE_MODERATE_THRESHOLD",
    "DECREASE_MODERATE_THRESHOLD",
    "INCREASE_HIGH_THRESHOLD",
    "DECREASE_HIGH_THRESHOLD",
    "INCREASE_TOP_K",
    "DECREASE_TOP_K",
    "NOP",
]

def discretize_probability(p: float) -> int:
    return min(max(int(p * 10), 0), 9)

def action_to_noise(action: int, gaussian_std: float = 0.02) -> float:
    noise: int = np.random.normal(0, gaussian_std)
    return noise if action % 2 == 0 else noise*(-1)

def generate_trend_step(
    t: int,
    T: int,
    p_start: float,
    p_end: float,
    gaussian_std: float,
    delta_min: float,
    delta_max: float
) -> Dict[str, Any]:
    alpha = t / (T - 1) if T > 1 else 0.0
    p_base = (1 - alpha) * p_start + alpha * p_end + np.random.normal(0, gaussian_std)
    p_base = max(0.0, min(1.0, p_base))

    action = random.choice(ACTIONS)
    delta = random.uniform(delta_min, delta_max)

    if action.startswith("DECREASE_"):
        p_next = p_base - delta
    elif action.startswith("INCREASE_"):
        p_next = p_base + delta
    else:
        p_next = p_base

    p_next = max(0.0, min(1.0, p_next))

    state = discretize_probability(p_base)
    next_state = discretize_probability(p_next)

    if p_next < p_base:
        reward = 1
    elif p_next > p_base:
        reward = -1
    else:
        reward = 0

    return {
        "prediction": float(p_base),
        "state": state,
        "action": action,
        "reward": reward,
        "next_prediction": float(p_next),
        "next_state": next_state,
    }


def generate_random_walk_step(p_base: float, delta_max: float) -> Tuple[Dict[str, Any], float]:
    action = random.choice(ACTIONS)
    delta = random.uniform(-delta_max, delta_max)
    p_next = max(0.0, min(1.0, p_base + delta))

    state = discretize_probability(p_base)
    next_state = discretize_probability(p_next)

    if next_state < state:
        reward = 1
    elif next_state > state:
        reward = -1
    else:
        reward = 0

    step = {
        "prediction": float(p_base),
        "state": state,
        "action": action,
        "reward": reward,
        "next_prediction": float(p_next),
        "next_state": next_state,
    }
    return step, p_next


def simulate_drift(
    p_start: float,
    p_end: float,
    length: Optional[int] = None,
    *,
    min_steps: int = 20,
    max_steps: int = 50,
    gaussian_std: float = 0.02,
    delta_min: float = 0.01,
    delta_max: float = 0.05,
    seed: Optional[int] = None,
    mode: str = "trend"
) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = length if length is not None else random.randint(min_steps, max_steps)
    trajectory: List[Dict[str, Any]] = []

    if mode == "random_walk":
        p_curr = (p_start + p_end) / 2.0
        for _ in range(T):
            step, p_curr = generate_random_walk_step(p_curr, delta_max)
            trajectory.append(step)
    else:
        for t in range(T):
            step = generate_trend_step(
                t, T, p_start, p_end,
                gaussian_std, delta_min, delta_max
            )
            trajectory.append(step)

    return trajectory

