# TODO:
#   Generate more episodes
#   Generate more steps per episode
#   Possibly add more state bins, pair with more aggressive trajectory
#   Add another trend mode for data, piecewise might need a mid point
#   Sweep alpha and gamma

import random
import itertools
import statistics
from typing import List, Dict, Tuple, Callable, Any

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

TRAIN_FRACTION = 0.8
OFF_ALPHA = 0.1
OFF_GAMMA = 0.9
EPSILON = 1.0
EPS_DECAY = 0.1
RUNS = 1

DIVIDER = "\n************************\n"

def evaluate_q_table(q_table: np.ndarray, episodes):
    env = DriftOfflineEnvironment(
        drift.discretize_probability, 
        drift.action_to_noise,
        _compute_reward)

    returns = []
    for episode in episodes:
        state = env.reset(episode)
        total_rewards = 0.0
        done = False
        while not done:
            state, reward, done = env.step(int(np.argmax(q_table[state])))
            total_rewards += reward
        returns.append(total_rewards)

    return statistics.mean(returns), statistics.stdev(returns)

def train_q_table(
    explorer_fn: Callable[[int, np.ndarray], int],
    train_eps:   List[List[Dict[str, float]]]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    env = PureQLearningEnvironment(
        exploration_strategy=explorer_fn,
        state_mapper=drift.discretize_probability,
        action_to_noise_mapper=drift.action_to_noise,
        reward_function=_compute_reward,
        gamma=OFF_GAMMA,
        alpha=OFF_ALPHA
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

def evaluate_metrics(
        q_table: np.ndarray, 
        val_eps, 
        state_fn: Callable[[float], int] = drift.discretize_probability,
        action_to_noise_fn: Callable[[int], float] = drift.action_to_noise,
        reward_fn: Callable[[int, int], float] = _compute_reward
        ) -> Dict[str, Any]:

    n_states, n_actions = q_table.shape
    eval_visits = np.zeros_like(q_table, dtype=int)

    returns, regrets, captures, entropies = [], [], [], []

    for ep in val_eps:
        prev_state = state_fn(ep[0])
        G_agent = 0.0
        G_opt = 0.0
        H_acc = 0.0
        T = 0

        for step in ep:
            p_next = step
            best_a = int(np.argmax(q_table[prev_state]))

            noisy_p = p_next + action_to_noise_fn(best_a)
            noisy_p = np.clip(noisy_p, 0.0, 1.0)
            s_next = state_fn(noisy_p)
            r = reward_fn(prev_state, s_next)
            G_agent += r

            s_true = state_fn(p_next)
            r_true = reward_fn(prev_state, s_true)
            G_opt += r_true

            eval_visits[prev_state, best_a] += 1
            prev_state = s_next
            T += 1

        returns.append(G_agent)
        regrets.append(G_opt - G_agent)
        captures.append(G_agent / G_opt if G_opt != 0 else 0.0)
        entropies.append(H_acc / T if T>0 else 0.0)
    
    return {
        "mean_return":    statistics.mean(returns),
        "std_return":     statistics.stdev(returns),
        "mean_regret":    statistics.mean(regrets),
        "std_regret":     statistics.stdev(regrets),
        "mean_capture":   statistics.mean(captures),
        "std_capture":    statistics.stdev(captures),
        "mean_entropy":   statistics.mean(entropies),
        "std_entropy":    statistics.stdev(entropies),
        "eval_visit_counts": eval_visits
    }

def mock_explorer(state: int, q_table: np.ndarray):
    return 2

def run_pure(train_eps, val_eps):
    for explorer in get_action_funcs():
        q_table, train_metrics = train_q_table(explorer, train_eps)
        eval_metrics = evaluate_metrics(q_table, val_eps)

        print(f"\n=== Strategy: {explorer.__name__} ===")
        # print(">>> Training metrics:")
        # print(f"  Mean action‐gap: {train_metrics['mean_action_gap']:.3f}")
        # print("  Visit counts:\n", train_metrics["visit_counts"])
        print("\n>>> Evaluation metrics:")
        # print(f"  Return    = {eval_metrics['mean_return']:.3f} ± {eval_metrics['std_return']:.3f}")
        print(f"  Regret    = {eval_metrics['mean_regret']:.3f} ± {eval_metrics['std_regret']:.3f}")
        # print(f"  Capture % = {eval_metrics['mean_capture']:.3f} ± {eval_metrics['std_capture']:.3f}")
        # print("  Eval visits:\n", eval_metrics["eval_visit_counts"])

def run_agent(train_episodes, val_episodes):
    action_tools = get_action_tools()
    context_tools = get_context_tools()
    reward_tools  = get_reward_tools()

    for action_fn, context_fn, reward_fn in itertools.product(
        action_tools, context_tools, reward_tools
    ):

        engine = QLearningEngine(
                    exploration_tools=[action_fn],
                    context_tools=[context_fn],
                    shaping_tools=[reward_fn],
                )

        flow = RLFlow()
        flow.set_rl_engine(engine)
        env = AgentQLearningEnvironment(flow, drift.discretize_probability, drift.action_to_noise)

        for seed in range(RUNS):
            q_table = np.zeros((cfg.Q_STATES, cfg.Q_ACTIONS), dtype=float)
            env.reset(q_table)

            for episode in train_episodes:
                p_start = episode[0]["prediction"]
                p_end = episode[-1]["next_prediction"]
                q_table = env.run_episode(p_start, p_end)

            mean, std = evaluate_q_table(q_table, val_episodes)
        return # this returns early, fine for now, just want to test it

def run_comparison():
    cfg.print_config()
    
    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)

    random.shuffle(episodes)
    split = int(len(episodes)*cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")

    # noise_factors = [0.02, 0.05, 0.08, 0.1, 0.2, 0.5]
    # bin_counts    = [11, 10, 20, 30, 50]
    # results1 = oracle.sweep_oracle(noise_factors, bin_counts, cfg.EPISODE_LENGTH, train_eps, val_eps)
    # always_two = lambda env: (lambda state, Q: 2)
    # results2 = oracle.sweep_oracle(noise_factors, bin_counts, cfg.EPISODE_LENGTH, train_eps, val_eps, always_two)
    # oracle.print_oracle_sweep_results(results1)
    # oracle.print_oracle_sweep_results(results2)
    run_pure(train_eps, val_eps)