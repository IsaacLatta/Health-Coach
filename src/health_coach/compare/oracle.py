import itertools
import numpy as np
import pandas as pd

import health_coach.config as cfg
from health_coach.compare.env import PureQLearningEnvironment
from rl_data_gen.drift import discretize_probability, action_to_noise

from typing import List, Callable

def reward_fn(prev: int, cur: int) -> float:
        return 1.0 if cur < prev else (-1.0 if cur > prev else 0.0)

def build_oracle(env: PureQLearningEnvironment, n_samples: int = 50) -> Callable[[int, np.ndarray], int]:
    oracle = {}
    for s in range(cfg.Q_STATES):
        avg_rewards = np.zeros(cfg.Q_ACTIONS)
        p0 = (s + 0.5) / cfg.Q_STATES
        for a in range(cfg.Q_ACTIONS):
            total = 0.0
            for _ in range(n_samples):
                # simulate one drift step
                noise = action_to_noise(a)
                p1 = np.clip(p0 + noise, 0.0, 1.0)
                s1 = discretize_probability(p1)
                total += env.reward_function(s, s1)
            avg_rewards[a] = total / n_samples
        oracle[s] = int(np.argmax(avg_rewards))
    return lambda state, Q: oracle[state]

def sweep_oracle(
    noise_factors: List[float],
    bin_counts: List[int],
    ep_length: int,
    train_eps: List[List[float]],
    val_eps: List[List[float]],
    oracle_builder: Callable[[PureQLearningEnvironment], Callable[[int, np.ndarray],int]] = build_oracle
):
    results = []
    for noise_factor, bins in itertools.product(noise_factors, bin_counts):
        cfg.Q_STATES = bins
        cfg.NOISE_STD = noise_factor * (1.0 / bins)
        env = PureQLearningEnvironment(
            exploration_strategy=None,  # placeholder
            state_mapper=discretize_probability,
            action_to_noise_mapper=action_to_noise,
            reward_function=reward_fn,
            gamma=cfg.EPSILON_CURRENT,  # not used by oracle
            alpha=cfg.EPSILON_CURRENT   # not used by oracle
        )
        oracle = oracle_builder(env)
        regrets = []
        for ep in val_eps:
            prev_state = discretize_probability(ep[0])
            G_opt = 0.0
            for p in ep[1:ep_length+1]:  # limit to fixed length
                a = oracle(prev_state, None)
                noise = action_to_noise(a)
                p_noisy = np.clip(p + noise, 0.0, 1.0)
                s1 = discretize_probability(p_noisy)
                G_opt += env.reward_function(prev_state, s1)
                prev_state = s1
            G_true = 0.0
            prev = discretize_probability(ep[0])
            for p in ep[1:ep_length+1]:
                s_true = discretize_probability(p)
                G_true += env.reward_function(prev, s_true)
                prev = s_true
            regrets.append((G_true - G_opt) / ep_length)
        results.append({
            'bins': bins,
            'noise_factor': noise_factor,
            'mean_regret': np.mean(regrets),
            'std_regret': np.std(regrets)
        })
    return results

def print_oracle_sweep_results(results):
    print(f"{'Bins':<6} {'Noise':<10} {'Mean Regret':<12} {'Std Regret':<12}")
    print("-" * 44)
    for r in results:
        bins = r['bins']
        noise = r['noise_factor']
        mean = r['mean_regret']
        std  = r['std_regret']
        print(f"{bins:<6} {noise:<10.3f} {mean:<12.3f} {std:<12.3f}")
