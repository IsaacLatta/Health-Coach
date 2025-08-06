import statistics
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any
from pathlib import Path
from datetime import datetime

from health_coach.compare.env import DriftOfflineEnvironment
import health_coach.rl_data_gen.drift as drift

import health_coach.config as cfg

RESULTS_DIR = Path(__file__).parents[3] / "results" / "comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def make_output_path(prefix: str = "pure_results") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{ts}_{prefix}.csv"

def reward_function(prev_state: int, current_state: int) -> float:
    res = prev_state - current_state
    return res

def save_results(results: List[Dict[str, Any]], prefix: str = "pure_results") -> Path:
    output_path = make_output_path(prefix)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return output_path

def print_results(results: List[Dict[str, Any]], explorer_name: str) -> None:
    runs = [r for r in results if r.get("strategy") == explorer_name]
    if not runs:
        print(f"No results for explorer '{explorer_name}'\n")
        return

    print(f"Explorer: {explorer_name}\n")

    metrics_to_show = [
        ("total_raw_trajectory_regret",      "Raw Trajectory Regret"),
        ("total_raw_optimal_regret",         "Raw Optimal Regret"),
        ("mean_normalized_trajectory_regret","Norm Trajectory Regret"),
        ("mean_normalized_optimal_regret",   "Norm Optimal Regret"),
        ("mean_raw_capture_ratio",           "Raw Capture Ratio"),
        ("mean_normalized_capture_ratio",    "Norm Capture Ratio"),
    ]

    for r in runs:
        seed = r["seed"]
        values = [
            f"{label}={r.get(key, 0.0):.4f}"
            for key, label in metrics_to_show
        ]
        print(f"  Seed {seed}: " + ", ".join(values))

    print("\nAggregated across seeds:")
    for key, label in metrics_to_show:
        vals = [r.get(key, 0.0) for r in runs]
        mean_val = statistics.mean(vals)
        std_val  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label}: {mean_val:.4f} ± {std_val:.4f}")
    print()
    mean_raw = statistics.mean([r["total_raw_trajectory_regret"] for r in runs])
    per_step = mean_raw / cfg.EPISODE_LENGTH
    print(f" Mean per-step trajectory regret: {per_step:.4f}\n")


def evaluate_metrics(
    q_table: np.ndarray,
    validation_episodes: List[List[float]],
    state_to_bin_fn: Callable[[float], int] = drift.discretize_probability,
    action_noise_fn: Callable[[int], float] = drift.action_to_noise,
    reward_fn: Callable[[int, int], float] = reward_function
) -> Dict[str, Any]:
    """
    Evaluate a Q-table on held-out episodes, returning:
      - raw and normalized trajectory regret
      - raw and normalized optimal regret
      - raw and normalized capture ratios
      - visit counts per (state, action)
    """
    n_states, n_actions = q_table.shape
    visit_counts = np.zeros((n_states, n_actions), dtype=int)

    raw_traj_regrets = []
    raw_opt_regrets = []
    norm_traj_regrets = []
    norm_opt_regrets = []
    raw_capture_ratios = []
    norm_capture_ratios = []

    noises = [action_noise_fn(a) for a in range(n_actions)]
    best_noise = min(noises)
    worst_noise = max(noises)

    for episode in validation_episodes:
        agent_return = 0.0
        trajectory_return = 0.0
        optimal_return = 0.0
        worst_return = 0.0

        state = state_to_bin_fn(episode[0])

        for true_prob in episode:
            action = int(np.argmax(q_table[state]))
            noisy_prob = np.clip(true_prob + action_noise_fn(action), 0.0, 1.0)
            next_state = state_to_bin_fn(noisy_prob)
            agent_return += reward_fn(state, next_state)

            traj_state = state_to_bin_fn(true_prob)
            trajectory_return += reward_fn(state, traj_state)

            opt_prob = np.clip(true_prob + best_noise, 0.0, 1.0)
            optimal_state = state_to_bin_fn(opt_prob)
            optimal_return += reward_fn(state, optimal_state)

            bad_prob = np.clip(true_prob + worst_noise, 0.0, 1.0)
            worst_state = state_to_bin_fn(bad_prob)
            worst_return += reward_fn(state, worst_state)

            visit_counts[state, action] += 1
            state = next_state

        raw_traj = trajectory_return - agent_return
        raw_opt = optimal_return - agent_return
        raw_traj_regrets.append(raw_traj)
        raw_opt_regrets.append(raw_opt)

        denom_traj = trajectory_return - worst_return
        denom_opt = optimal_return - worst_return

        norm_traj_regrets.append(raw_traj / denom_traj if denom_traj > 0 else 0.0)
        norm_opt_regrets.append(raw_opt  / denom_opt  if denom_opt  > 0 else 0.0)

        raw_capture = agent_return / optimal_return if optimal_return != 0 else 0.0
        raw_capture_ratios.append(raw_capture)

        norm_capture = (
            (agent_return - worst_return) / denom_opt
            if denom_opt > 0 else 0.0
        )
        norm_capture_ratios.append(norm_capture)

    return {
        "total_raw_trajectory_regret": sum(raw_traj_regrets),
        "mean_raw_trajectory_regret": statistics.mean(raw_traj_regrets),
        "std_raw_trajectory_regret": (
            statistics.stdev(raw_traj_regrets) if len(raw_traj_regrets) > 1 else 0.0
        ),

        "total_raw_optimal_regret": sum(raw_opt_regrets),
        "mean_raw_optimal_regret": statistics.mean(raw_opt_regrets),
        "std_raw_optimal_regret": (
            statistics.stdev(raw_opt_regrets) if len(raw_opt_regrets) > 1 else 0.0
        ),

        "mean_normalized_trajectory_regret": statistics.mean(norm_traj_regrets),
        "std_normalized_trajectory_regret": (
            statistics.stdev(norm_traj_regrets) if len(norm_traj_regrets) > 1 else 0.0
        ),
        "mean_normalized_optimal_regret": statistics.mean(norm_opt_regrets),
        "std_normalized_optimal_regret": (
            statistics.stdev(norm_opt_regrets) if len(norm_opt_regrets) > 1 else 0.0
        ),

        "mean_raw_capture_ratio": statistics.mean(raw_capture_ratios),
        "std_raw_capture_ratio": (
            statistics.stdev(raw_capture_ratios) if len(raw_capture_ratios) > 1 else 0.0
        ),
        "mean_normalized_capture_ratio": statistics.mean(norm_capture_ratios),
        "std_normalized_capture_ratio": (
            statistics.stdev(norm_capture_ratios) if len(norm_capture_ratios) > 1 else 0.0
        ),

        "visit_counts": visit_counts,
    }
