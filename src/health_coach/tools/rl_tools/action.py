import math
import random
from typing import Union, Any
import numpy as np
from crewai.tools import tool
from health_coach.tools.rl_tools.context import _get_q_table, _get_episode_path, _load_episode
from health_coach.tools.actions import retrieve_action_index

def retrieve_state_index(state: Union[int, str]) -> int:
    """
    Convert state identifiers (int) to int index, or raise.
    """
    if isinstance(state, int):
        return state
    try:
        return int(state)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid state index: {state}")

def _get_visit_counts() -> np.ndarray:
    """
    Count occurrences of each (state, action) in the current episode CSV.
    Returns an array shape [num_states, num_actions].
    """
    q_table = _get_q_table()
    num_states, num_actions = q_table.shape
    counts = np.zeros((num_states, num_actions), dtype=float)
    df = _load_episode(_get_episode_path())
    for _, row in df.iterrows():
        s = retrieve_state_index(row.get("state"))
        a = retrieve_action_index(row.get("action"))
        if 0 <= s < num_states and 0 <= a < num_actions:
            counts[s, a] += 1
    return counts


def _get_q_posterior(default_std: float = 1.0) -> np.ndarray:
    """
    Build a synthetic posterior over Q-values: mean=Q, std=default_std.
    Returns shape [num_states, num_actions, 2] array.
    """
    q_table = _get_q_table()
    means = q_table
    stds = np.full_like(q_table, default_std, dtype=float)
    return np.stack([means, stds], axis=-1)

# ------------------ Exploration Tools ------------------

@tool
def epsilon_greedy(state: int, epsilon: float = 0.1) -> int:
    """
    Standard ε-greedy: with prob ε pick random action, else the argmax.
    """
    q_table = _get_q_table()
    s = retrieve_state_index(state)
    n_actions = q_table.shape[1]
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q_table[s]))

@tool
def softmax(state: int, temperature: float) -> int:
    """
    Boltzmann exploration: sample action ∼ softmax(Q(s,·)/temperature).
    """
    q_table = _get_q_table()
    s = retrieve_state_index(state)
    logits = q_table[s] / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / float(np.sum(exp_logits))
    return int(np.random.choice(len(probs), p=probs))

@tool
def ucb(state: int, c: float) -> int:
    """
    Upper-Confidence Bound: argmax_a [Q(s,a) + c * sqrt(ln N(s)+1)/(N(s,a)+1)].
    """
    q_table = _get_q_table()
    s = retrieve_state_index(state)
    counts = _get_visit_counts()[s]
    total = float(np.sum(counts))
    bonus = c * np.sqrt(np.log(total + 1.0) / (counts + 1.0))
    scores = q_table[s] + bonus
    return int(np.argmax(scores))

@tool
def count_bonus(state: int, beta: float) -> int:
    """
    Count-based bonus: argmax_a [Q(s,a) + beta / sqrt(N(s,a)+1)].
    """
    q_table = _get_q_table()
    s = retrieve_state_index(state)
    counts = _get_visit_counts()[s]
    bonus = beta / np.sqrt(counts + 1.0)
    scores = q_table[s] + bonus
    return int(np.argmax(scores))

@tool
def thompson(state: int) -> int:
    """
    Thompson Sampling: sample Q~N(mean,std) and pick argmax.
    Uses a synthetic posterior with default std.
    """
    s = retrieve_state_index(state)
    posterior = _get_q_posterior()
    means = posterior[s, :, 0]
    stds  = posterior[s, :, 1]
    samples = np.random.normal(loc=means, scale=stds)
    return int(np.argmax(samples))

@tool
def maxent(state: int, alpha: float) -> int:
    """
    Maximum-Entropy policy: alias for softmax(state, temperature=alpha).
    """
    return softmax(state=state, temperature=alpha)


def get_all_tools():
    """
    Return all exploration tools.
    """
    return [epsilon_greedy, softmax, ucb, count_bonus, thompson, maxent]
    