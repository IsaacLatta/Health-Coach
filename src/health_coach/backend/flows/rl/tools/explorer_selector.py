from typing import Callable, Union
import numpy as np
import health_coach.config as cfg

from health_coach.backend.flows.rl.tools.exploration import (
    QLike,
    epsilon_greedy_fn,
    softmax_fn,
    ucb_fn,
    count_bonus_fn,
    thompson_fn,
    maxent_fn,
)

def get_explorer(idx: int, q_table: QLike) -> Callable[[int], int]:
    if not isinstance(idx, int) or idx < 0 or idx > 5:
        idx = 0

    if idx == 0:
        return lambda state: epsilon_greedy_fn(state, q_table)
    elif idx == 1:
        return lambda state: softmax_fn(state, q_table, temperature=cfg.SOFTMAX_TEMP)
    elif idx == 2:
        return lambda state: ucb_fn(state, q_table, c=cfg.UCB_C)
    elif idx == 3:
        return lambda state: thompson_fn(state, q_table, sigma=cfg.THOMPSON_SIGMA)
    elif idx == 4:
        return lambda state: count_bonus_fn(state, q_table)
    else:
        return lambda state: maxent_fn(state, q_table, alpha=cfg.MAXENT_ALPHA)

def execute_explorer(state: int, idx: int, q_table: QLike) -> int:
    try:
        explorer = get_explorer(idx, q_table)
        return explorer(state)
    except Exception:
        return 0
