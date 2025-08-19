import math
import random
from typing import Union, Any, Optional, List, Iterable
import numpy as np
from crewai.tools import tool
import health_coach.config as cfg

QTable = List[List[float]]
QLike = Union[np.ndarray, QTable]

def to_nd(q: QLike | None) -> np.ndarray:
    if q is None:
        return np.zeros((int(cfg.Q_STATES), int(cfg.Q_ACTIONS)), dtype=float)
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Q-table must be 2D")
    return arr

Q_TABLE: Optional[np.ndarray] = None

def num_actions(q_table: Optional[QLike] = None) -> int:
    if q_table is not None:
        try:
            return int(np.asarray(q_table).shape[1])
        except Exception:
            pass
    if hasattr(cfg, "Q_ACTIONS"):
        return int(cfg.Q_ACTIONS)
    if hasattr(cfg, "ACTIONS"):
        return len(cfg.ACTIONS)
    return 0

def retrieve_action_index(action: Union[int, str],
                          q_table: Optional[QLike] = None) -> int:
    """
    Convert an action (int index or name) into a **validated** index.
    - If int: must be within [0, n_actions).
    - If str: must exist in cfg.ACTIONS and resolve to an index within range.
    Returns -1 when invalid.
    """
    n = num_actions(q_table)
    if isinstance(action, int):
        return action if 0 <= action < n else -1

    try:
        actions = getattr(cfg, "ACTIONS", None)
        if isinstance(actions, (list, tuple)):
            idx = actions.index(action)
            return idx if 0 <= idx < n else -1
    except Exception:
        pass
    return -1


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

def get_q_table() -> np.ndarray:
    global Q_TABLE
    if Q_TABLE is None:
        Q_TABLE = np.zeros((int(cfg.Q_STATES), int(cfg.Q_ACTIONS)), dtype=float)
    return Q_TABLE

def set_q_table(q_table: QLike):
    global Q_TABLE
    Q_TABLE = to_nd(q_table)

def _get_q_posterior(q_table: QLike, sigma: float = cfg.THOMPSON_SIGMA) -> np.ndarray:
    """
    Build a synthetic posterior over Q-values: mean=Q, std=sigma.
    Returns shape [num_states, num_actions, 2] array.
    """
    q = to_nd(q_table)
    means = q
    stds = np.full_like(q, float(sigma), dtype=float)
    return np.stack([means, stds], axis=-1)

def _get_visit_counts(q_table: QLike) -> np.ndarray:
    q = to_nd(q_table)
    num_states, num_actions_ = q.shape
    return np.zeros((num_states, num_actions_), dtype=float)

def epsilon_greedy_fn(current_state: int, q_table: np.ndarray, epsilon: float = None) -> int:
    q = to_nd(q_table)
    epsilon = cfg.EPSILON_CURRENT if epsilon is None else float(epsilon)
    s = retrieve_state_index(current_state)
    n_actions = q.shape[1]
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q[s]))


@tool
def epsilon_greedy(current_state: int, epsilon: float = None) -> int:
    """Standard ε-greedy: with prob ε pick random action, else the argmax."""
    return epsilon_greedy_fn(current_state, q_table=get_q_table(), epsilon=epsilon)


def softmax_fn(current_state: int, q_table: np.ndarray, temperature: float = cfg.SOFTMAX_TEMP) -> int:
    q = to_nd(q_table)
    s = retrieve_state_index(current_state)
    t = max(float(temperature), 1e-8)
    logits = q[s] / t
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    z = float(np.sum(exp_logits))
    probs = exp_logits / (z if z > 0 else 1.0)
    return int(np.random.choice(len(probs), p=probs))


@tool
def softmax(current_state: int, temperature: float = cfg.SOFTMAX_TEMP) -> int:
    """Boltzmann exploration: sample action ∼ softmax(Q(s,·)/temperature)."""
    return softmax_fn(current_state, q_table=get_q_table(), temperature=temperature)


def ucb_fn(current_state: int, q_table: np.ndarray, c: float = cfg.UCB_C) -> int:
    q = to_nd(q_table)
    s = retrieve_state_index(current_state)
    counts = _get_visit_counts(q)[s]
    total = float(np.sum(counts))
    bonus = float(c) * np.sqrt(np.log(total + 1.0) / (counts + 1.0))
    scores = q[s] + bonus
    return int(np.argmax(scores))


@tool
def ucb(current_state: int, c: float = cfg.UCB_C) -> int:
    """UCB: argmax_a [Q(s,a) + c * sqrt(ln N(s)+1)/(N(s,a)+1)]."""
    return ucb_fn(current_state, q_table=get_q_table(), c=c)


def count_bonus_fn(current_state: int, q_table: np.ndarray, beta: float = 1.0) -> int:
    q = to_nd(q_table)
    s = retrieve_state_index(current_state)
    counts = _get_visit_counts(q)[s]
    bonus = float(beta) / np.sqrt(counts + 1.0)
    scores = q[s] + bonus
    return int(np.argmax(scores))


@tool
def count_bonus(current_state: int, beta: float = 1.0) -> int:
    """Count-based bonus: argmax_a [Q(s,a) + beta / sqrt(N(s,a)+1)]."""
    return count_bonus_fn(current_state, q_table=get_q_table(), beta=beta)


def thompson_fn(current_state: int, q_table: np.ndarray, sigma: float = cfg.THOMPSON_SIGMA) -> int:
    q = to_nd(q_table)
    s = retrieve_state_index(current_state)
    posterior = _get_q_posterior(q, sigma=float(sigma))
    means = posterior[s, :, 0]
    stds  = posterior[s, :, 1]
    samples = np.random.normal(loc=means, scale=stds)
    return int(np.argmax(samples))


@tool
def thompson(current_state: int) -> int:
    """Thompson Sampling over a synthetic posterior with default std."""
    return thompson_fn(current_state, q_table=get_q_table(), sigma=cfg.THOMPSON_SIGMA)


def maxent_fn(current_state: int, q_table: np.ndarray, alpha: float = cfg.MAXENT_ALPHA) -> int:
    # Alias to softmax with temperature=alpha
    return softmax_fn(current_state=current_state, q_table=q_table, temperature=float(alpha))


@tool
def maxent(current_state: int, alpha: float = cfg.MAXENT_ALPHA) -> int:
    """Maximum-Entropy policy: alias for softmax(state, temperature=alpha)."""
    return maxent_fn(current_state, q_table=get_q_table(), alpha=alpha)


def get_all_tools():
    """Return all exploration tools."""
    return [epsilon_greedy, softmax, ucb, count_bonus, thompson, maxent]


def get_all_funcs():
    """Return core exploration implementations (extend as needed)."""
    return [epsilon_greedy_fn, softmax_fn, ucb_fn]
    # return [epsilon_greedy_fn, softmax_fn, ucb_fn, count_bonus_fn, thompson_fn, maxent_fn]
