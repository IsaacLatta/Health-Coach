import numpy as np
import random
import pickle
from pathlib import Path
from crewai.tools import tool

from health_coach.tools.actions import ACTION_HANDLERS, ACTIONS

_BASE_DIR = Path(__file__).resolve().parents[1]
_MODEL_PATH = _BASE_DIR / "models" / "cv_pred_log_reg.pkl"

def _load_model(model_path: Path = _MODEL_PATH) -> np.ndarray:
    with open(model_path, "rb") as f:
        return pickle.load(f)

def _shape_reward(reward: float, factor: float) -> float:
    return reward*factor

@tool
def amplify_reward(reward: float, factor: float) -> float:
    """
    Amplify the given reward by the specified factor.
    """
    return _shape_reward(reward, factor)

def _select_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    n_actions = q_table.shape[1]
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q_table[state]))

@tool
def select_action(state: int, epsilon: float) -> int:
    """
    Loads the rl model internally and selects an action index for the given state.
    """
    q_table = _load_model()
    return _select_action(q_table, state, epsilon)

# Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])
def _update_q(q_table: np.ndarray,
              state: int,
              action: int,
              reward: float,
              next_state: int,
              alpha: float,
              gamma: float) -> np.ndarray:
    best_next = q_table[next_state].max()
    td_target = reward + gamma * best_next
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return q_table

@tool
def update_rl_model(state: int, action: int, reward: float, next_state: int) -> list:
    """
    Load the Q-table, perform a single update, and return as nested lists.
    """
    alpha = 0.1
    gamma = 0.99
    q_table = _load_model()
    updated = _update_q(q_table, state, action, reward, next_state, alpha, gamma)
    return updated.tolist()

def _compute_reward(prev_state: int, current_state: int) -> float:
    return (prev_state - current_state)*1.0

@tool
def compute_reward(prev_state: int, current_state: int) -> float:
    """
    Compute the scalar reward from previous and current states.
    """
    return _compute_reward(prev_state, current_state)

def _generate_new_config(old_config: dict, action_idx: int) -> dict:
    if action_idx < 0 or action_idx >= len(ACTIONS):
        return old_config
    
    try:
        handler = ACTION_HANDLERS.get(ACTIONS[action_idx])
        
        if not callable(handler):
            return old_config
        else:
            return handler(old_config)
    except:
        return old_config

@tool
def apply_action(old_config: dict, action_idx: int) -> dict:
    """
    Given a config dict and an action index, applies the action and returns the updated config.
    """
    return _generate_new_config(old_config, action_idx)