from typing import Callable
import numpy as np

from health_coach.tools.rl_tools.action import (
    epsilon_greedy_fn,
    softmax_fn,
    ucb_fn,
    count_bonus_fn,
    thompson_fn,
    maxent_fn,
)

def make_epsilon_greedy_fn(epsilon: float) -> Callable[[int, np.ndarray], int]:
    """
    Returns an ε-greedy explorer that always uses the given fixed epsilon.
    """
    def explorer(state: int, q_table: np.ndarray) -> int:
        return epsilon_greedy_fn(state, q_table, epsilon=epsilon)
    return explorer


def make_softmax_fn(temperature: float) -> Callable[[int, np.ndarray], int]:
    """
    Returns a softmax/Boltzmann explorer with the given temperature.
    """
    def explorer(state: int, q_table: np.ndarray) -> int:
        return softmax_fn(state, q_table, temperature=temperature)
    return explorer


def make_ucb_fn(c: float) -> Callable[[int, np.ndarray], int]:
    """
    Returns a UCB explorer with confidence coefficient c.
    """
    def explorer(state: int, q_table: np.ndarray) -> int:
        return ucb_fn(state, q_table, c=c)
    return explorer


def make_count_bonus_fn(beta: float) -> Callable[[int, np.ndarray], int]:
    """
    Returns a count-based bonus explorer with scale β.
    """
    def explorer(state: int, q_table: np.ndarray) -> int:
        return count_bonus_fn(state, q_table, beta=beta)
    return explorer


def make_thompson_fn(std: float) -> Callable[[int, np.ndarray], int]:
    """
    Returns a Thompson-sampling explorer using posterior std=std.
    (Note: _get_q_posterior currently uses default_std; you'd need to pass std to that function if you want to tune it.)
    """
    def explorer(state: int, q_table: np.ndarray) -> int:
        return thompson_fn(state, q_table)
    return explorer


def make_maxent_fn(alpha: float) -> Callable[[int, np.ndarray], int]:
    """
    Alias for softmax_fn with temperature=alpha.
    """
    return make_softmax_fn(alpha)

def get_factories():
    return {
        "epsilon_greedy_fn": make_epsilon_greedy_fn,
        "softmax_fn":        make_softmax_fn,
        "ucb_fn":            make_ucb_fn,
        "count_bonus_fn":    make_count_bonus_fn,
        "thompson_fn":       make_thompson_fn,
        "maxent_fn":         make_maxent_fn,
    }