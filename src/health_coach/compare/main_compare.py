# TODO:
#   Add another trend mode for data, piecewise might need a mid point
#   Sweep alpha and gamma

import random
import itertools
import statistics
from typing import List, Dict, Tuple, Callable, Any

import pandas as pd
import numpy as np
import health_coach.rl_data_gen.generate as gen
from health_coach.rl_train.env import HeartDiseaseEnv
from health_coach.rl_train.offline_train import evaluate_policy as eval_offline
from health_coach.rl import QLearningEngine
from health_coach.tools.rl_tools.action import (
    get_all_tools as get_action_tools,
    get_all_tool_funcs as get_action_funcs
)

import health_coach.config as cfg

import health_coach.compare.oracle as oracle

import health_coach.rl_data_gen.drift as drift
from health_coach.tools.rl_tools.context import get_all_tools as get_context_tools
from health_coach.tools.rl_tools.reward import get_all_tools as get_reward_tools
from health_coach.tools.rl import _compute_reward
from health_coach.flows import RLFlow

from .env import AgentQLearningEnvironment, PureQLearningEnvironment, DriftOfflineEnvironment, generate_base_trajectory
from .eval import print_results, evaluate_metrics, reward_function

OFF_ALPHA = 0.1
OFF_GAMMA = 0.9
RUNS = 1

SEEDS = list(range(1))
PURE_OUTPUT_CSV = "pure_results.csv"


def train_q_table(
    explorer_fn: Callable[[int, np.ndarray], int],
    train_eps:   List[List[Dict[str, float]]],
    alpha: float,
    gamma: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    env = PureQLearningEnvironment(
        exploration_strategy=explorer_fn,
        state_mapper=drift.discretize_probability,
        action_to_noise_mapper=drift.action_to_noise,
        reward_function=reward_function,
        gamma=gamma,
        alpha=alpha
    )

    q_table = np.zeros((cfg.Q_STATES, cfg.Q_ACTIONS), dtype=float)
    env.reset(q_table)

    for ep in train_eps:
        q_table = env.run_episode(ep)
        cfg.update_epsilon()

    top2 = np.partition(q_table, -2, axis=1)[:, -2:]
    gaps = top2[:,1] - top2[:,0]
    mean_gap = float(np.mean(gaps))

    return q_table, {
        "visit_counts": env.visit_counts.copy(),
        "mean_action_gap": mean_gap
    }

def run_pure(train_eps: List[List[float]], val_eps: List[List[float]],
             seeds: List[int] = SEEDS, output_path: str = PURE_OUTPUT_CSV) -> None:
    all_results: List[Dict[str, Any]] = []
    explorers = get_action_funcs()

    for explorer in explorers:
        
        best_score = -float("inf")
        best_hyperparams = (None, None)
        for _ in range(cfg.N_TRIALS):
            alpha, gamma = cfg.sample_hyperparams()

            trial_scores = []
            for seed in range(cfg.N_SEEDS):
                random.seed(seed)
                np.random.seed(seed)

                q_table, train_metrics = train_q_table(explorer, train_eps,  alpha, gamma)
                eval_metrics = evaluate_metrics(q_table, val_eps)

                metrics = {
                    "strategy": explorer.__name__,
                    "seed": seed,
                    "alpha": alpha,
                    "gamma": gamma,
                    **eval_metrics
                }
                all_results.append(metrics)
                trial_scores.append(eval_metrics["mean_normalized_capture_ratio"])

            mean_score = sum(trial_scores) / len(trial_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_hyperparams = (alpha, gamma)

        a_best, g_best = best_hyperparams
        print(f"\nExplorer {explorer.__name__} best hyperparams:")
        print(f"  alpha = {a_best:.4f}, gamma = {g_best:.4f} → score = {best_score:.4f}\n")

        best_runs = [
            r for r in all_results
            if r["strategy"] == explorer.__name__
            and r["alpha"]    == a_best
            and r["gamma"]    == g_best
        ]
        print_results(best_runs, explorer.__name__)
        
def run_comparison():
    cfg.print_config()

    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)

    random.shuffle(episodes)
    split = int(len(episodes) * cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")

    run_pure(train_eps, val_eps)