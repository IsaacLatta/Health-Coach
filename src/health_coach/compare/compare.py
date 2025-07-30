import numpy as np
import random
import os

from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import health_coach.config as cfg
import health_coach.compare.train as train
import health_coach.compare.eval as eval


import statistics
from typing import List, Dict, Any, Tuple

def process_results(results: List[Dict[str, Any]]) -> None:
    """
    For each explorer (strategy):
      1. Group runs by (alpha, gamma)
      2. Compute mean_norm_capture for each group
      3. Identify best (alpha, gamma)
      4. Print best hyperparams + score
      5. Print detailed per-seed metrics for that combo
    """
    explorers = sorted({r["strategy"] for r in results})

    for explorer in explorers:
        runs = [r for r in results if r["strategy"] == explorer]

        groups: Dict[Tuple[float,float], List[Dict[str, Any]]] = {}
        for r in runs:
            key = (r["alpha"], r["gamma"])
            groups.setdefault(key, []).append(r)

        best_score = -float("inf")
        best_params: Tuple[float, float] = (None, None)
        for params, grp in groups.items():
            scores = [x.get("mean_normalized_capture_ratio", 0.0) for x in grp]
            mean_score = statistics.mean(scores)
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        a_best, g_best = best_params

        print(f"\nExplorer {explorer} best hyperparams:")
        print(f"  alpha = {a_best:.4f}, gamma = {g_best:.4f} → "
              f"mean_norm_capture = {best_score:.4f}\n")

        best_runs = groups[best_params]
        eval.print_results(best_runs, explorer)

def reward_function(prev_state: int, current_state: int) -> float:
    res = prev_state - current_state
    return res

def explorer_job(
    explorer,
    train_eps,
    val_eps,
    seeds: List[int],
    alpha: float,
    gamma: float
) -> List[Dict[str, Any]]:
 
    results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        q_table, train_metrics = train.train_q_table(explorer, train_eps, alpha, gamma)
        eval_metrics = eval.evaluate_metrics(q_table, val_eps)

        results.append({
            "strategy": explorer.__name__,
            "seed": seed,
            "alpha": alpha,
            "gamma": gamma,
            **eval_metrics
        })

    return results

def run_pure_comparison(
    train_eps,
    val_eps,
    explorers
) -> None:
    all_results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for explorer in explorers:
            for _ in range(cfg.N_TRIALS):
                alpha, gamma = cfg.sample_hyperparams()
                futures.append(executor.submit(
                    explorer_job,
                    explorer,
                    train_eps,
                    val_eps,
                    cfg.SEEDS,
                    alpha,
                    gamma
                ))

        for future in as_completed(futures):
            job_results = future.result()
            all_results.extend(job_results)

    process_results(all_results)
