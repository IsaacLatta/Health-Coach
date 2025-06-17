import random
import numpy as np
from typing import List, Dict, Any, Optional

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
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = length if length is not None else random.randint(min_steps, max_steps)
    trajectory: List[Dict[str, Any]] = []

    for t in range(T):
        alpha = 1.0 / (t + 1)
        p_base = (1 - alpha) * p_start + alpha * p_end + np.random.normal(0, gaussian_std)
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

        trajectory.append({
            "prediction": float(p_base),
            "state": state,
            "action": action,
            "reward": reward,
            "next_prediction": float(p_next),
            "next_state": next_state,
        })

    return trajectory