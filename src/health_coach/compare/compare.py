import numpy as np
import random
import os
import json
import statistics
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import health_coach.config as cfg
import health_coach.compare.train as train
import health_coach.compare.train_agents as agent_train
import health_coach.compare.eval as eval
from health_coach.compare.rl import SimpleQLearningEngine
from health_coach.backend.flows.rl.tools.explorer_selector import get_hyperparams_for_explorer

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
    best_results = {}

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
        
        json_safe_runs = []
        for run in best_runs:
            json_run = run.copy()
            # Convert any numpy arrays to lists
            for key, value in json_run.items():
                if isinstance(value, np.ndarray):
                    json_run[key] = value.tolist()
            json_safe_runs.append(json_run)

        
        
        best_results[explorer] = {
            "alpha": a_best,
            "gamma": g_best,
            "score": best_score,
            "runs": json_safe_runs
        }
        
        eval.print_results(best_runs, explorer)
    
    with open(cfg.COMPARISION_RESULTS_DIR + "/pure/" + cfg.COMPARISON_RESULTS_FILE, "w") as f:
        json.dump(best_results, f, indent=2)

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
    for seed in range(seeds):
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

def run_agent_comparison(train_eps, val_eps, explorers):
    engine = SimpleQLearningEngine(
        explorers, 
        reward_function, 
        cfg.ALPHA, # Defaults, get overriden on a per explorer bases
        cfg.GAMMA
    )

    with ThreadPoolExecutor(max_workers=1) as exe:
        futures = [
            exe.submit(agent_train.train_agent_worker, engine, ep)
            for ep in train_eps
        ]
        for f in as_completed(futures):
            _ = f.result()

    q_table = engine.q_table.copy()
    metrics = eval.evaluate_metrics(q_table, val_eps)

    safe_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            safe_metrics[key] = value.tolist()
        else:
            safe_metrics[key] = value
    
    print("Agent-augmented eval:", safe_metrics)
    with open(cfg.COMPARISION_RESULTS_DIR + "/agent/" + cfg.COMPARISON_RESULTS_FILE, 'w') as f:
        json.dump(safe_metrics, f, indent=2)
  

def run_pure_comparison(train_eps, val_eps, explorers) -> None:
    all_results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(len(explorers)):
            explorer = explorers[i]
            for _ in range(cfg.N_TRIALS):
                alpha, gamma = (
                    cfg.sample_hyperparams()
                    if cfg.SHOULD_SAMPLE_HYPERPARAMS
                    else get_hyperparams_for_explorer(i)
                )

                futures.append(executor.submit(
                    explorer_job,
                    explorer,
                    train_eps,
                    val_eps,
                    cfg.N_SEEDS,
                    alpha,
                    gamma
                ))

        for future in as_completed(futures):
            job_results = future.result()
            all_results.extend(job_results)

    process_results(all_results)
