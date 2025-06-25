# File: health_coach/tools/rl.py

import numpy as np
import random
import pickle
import math
from pathlib import Path
from crewai.tools import tool
from health_coach.tools.actions import ACTION_HANDLERS, ACTIONS

_BASE_DIR = Path(__file__).resolve().parents[3]
_MODEL_PATH = _BASE_DIR / "models" / "q_table.npy"

def _load_model(model_path: Path = None) -> np.ndarray:
    path = model_path or _MODEL_PATH
    return np.load(path, allow_pickle=False)

def _shape_reward(reward: float, factor: float) -> float:
    return reward * factor

@tool
def amplify_reward(reward: float, factor: float) -> float:
    """
    Amplify the given reward by the specified factor.
    """
    return _shape_reward(reward, factor)

def _select_action(state: int, epsilon: float = 0.1, q_table: np.ndarray = _load_model()) -> int:
    n_actions = q_table.shape[1]
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q_table[state]))

@tool
def select_action(state: int, epsilon: float) -> int:
    """
    Loads the RL model internally and selects an action index for the given state.
    """
    return _select_action(state, epsilon)

def _update_q_table(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    best_next = q_table[next_state].max()
    td_target = reward + gamma * best_next
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return q_table

def _save_np_array(q_table: np.ndarray, path: Path = None) -> bool:
    target = path or _MODEL_PATH
    try:
        # this will write a .npy file
        np.save(target, q_table)
        return True
    except Exception:
        return False

def _update_rl_model(
    state: int,
    action: int,
    reward: float,
    next_state: int,
    alpha: float = 0.1,
    gamma: float = 0.99
) -> bool:
    q_table = _load_model()
    updated_q = _update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
    return _save_np_array(updated_q)

@tool
def update_rl_model(prev_state: int, action: int, reward: float, current_state: int) -> bool:
    """
    Loads the respective model, performs a single update,
    and resave the model to disk.
    Returns True on success, False on failure.
    """
    return _update_rl_model(prev_state, action, reward, current_state)

def _compute_reward(prev_state: int, current_state: int) -> float:
    return float(prev_state - current_state)

@tool
def compute_reward(prev_state: int, current_state: int) -> float:
    """
    Compute the scalar reward from previous and current states.
    """
    return _compute_reward(prev_state, current_state)

def _apply_action(old_config: dict, action_idx: int) -> dict:
    if action_idx < 0 or action_idx >= len(ACTIONS):
        return old_config
    handler = ACTION_HANDLERS.get(ACTIONS[action_idx])
    if not callable(handler):
        return old_config
    return handler(old_config)

@tool
def apply_action(old_config: dict, action_idx: int) -> dict:
    """
    Given a config dict and an action index, applies the corresponding handler
    from ACTION_HANDLERS and returns the updated config.
    If the index is invalid or the handler isn’t callable, returns the original.
    """
    return _apply_action(old_config, action_idx)

@tool
def discretize_probability(prediction: float) -> int:
    """
    Given a continuous prediction value, returns the discretized state 
    """
    return min(max(int(prediction * 10), 0), 9)