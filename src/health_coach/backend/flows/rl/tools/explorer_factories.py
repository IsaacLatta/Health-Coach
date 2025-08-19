from typing import Callable, Union, List
import numpy as np

from .exploration import (
    QLike,
    epsilon_greedy_fn,
    softmax_fn,
    ucb_fn,
    count_bonus_fn,
    thompson_fn,
    maxent_fn,
)

def make_epsilon_greedy_fn(epsilon: float) -> Callable[[int, QLike], int]:
    def explorer(state: int, q_table: QLike) -> int:
        return epsilon_greedy_fn(state, q_table, epsilon=epsilon)
    return explorer

def make_softmax_fn(temperature: float) -> Callable[[int, QLike], int]:
    def explorer(state: int, q_table: QLike) -> int:
        return softmax_fn(state, q_table, temperature=temperature)
    return explorer

def make_ucb_fn(c: float) -> Callable[[int, QLike], int]:
    def explorer(state: int, q_table: QLike) -> int:
        return ucb_fn(state, q_table, c=c)
    return explorer

def make_count_bonus_fn(beta: float) -> Callable[[int, QLike], int]:
    def explorer(state: int, q_table: QLike) -> int:
        return count_bonus_fn(state, q_table, beta=beta)
    return explorer

def make_thompson_fn(sigma: float) -> Callable[[int, QLike], int]:
    def explorer(state: int, q_table: QLike) -> int:
        return thompson_fn(state, q_table, sigma)
    return explorer

def make_maxent_fn(alpha: float) -> Callable[[int, QLike], int]:
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
