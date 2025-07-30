import numpy as np
from typing import Tuple, Dict, Callable, List, Any

import health_coach.config as cfg
import health_coach.compare.env as env
import health_coach.rl_data_gen.drift as drift

def reward_function(prev_state: int, current_state: int) -> float:
    res = prev_state - current_state
    return res

def train_q_table(
    explorer_fn: Callable[[int, np.ndarray], int],
    train_eps:   List[List[float]],
    alpha: float,
    gamma: float,
    state_to_bin_fn: Callable[[float], int] = drift.discretize_probability,
    action_noise_fn: Callable[[int], float] = drift.action_to_noise,
    reward_fn: Callable[[int, int], float] = reward_function,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    environ = env.PureQLearningEnvironment(
        exploration_strategy=explorer_fn,
        state_mapper= state_to_bin_fn,
        action_to_noise_mapper= action_noise_fn,
        reward_function=reward_fn,
        gamma=gamma,
        alpha=alpha
    )

    q_table = np.zeros((cfg.Q_STATES, cfg.Q_ACTIONS), dtype=float)
    environ.reset(q_table)

    for ep in train_eps:
        q_table = environ.run_episode(ep)
        cfg.update_epsilon() # MOVE TO STEP_FUNC

    top2 = np.partition(q_table, -2, axis=1)[:, -2:]
    gaps = top2[:,1] - top2[:,0]
    mean_gap = float(np.mean(gaps))

    return q_table, {
        "visit_counts": environ.visit_counts.copy(),
        "mean_action_gap": mean_gap
    }
