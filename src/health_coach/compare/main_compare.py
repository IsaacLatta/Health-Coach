# compare.py
import random
import itertools
import statistics
from typing import List, Dict, Tuple, Callable, Any

import numpy as np
import health_coach.rl_data_gen.generate as gen
from health_coach.rl_train.env import HeartDiseaseEnv
from health_coach.rl_train.offline_train import train_q_table, evaluate_policy as eval_offline
from health_coach.rl import QLearningEngine
from health_coach.tools.rl_tools.action import (
    get_all_tools as get_action_tools,
    get_all_tool_funcs as get_action_funcs
)

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
    explorer_fn: Callable[[int], int],
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

    q_table = np.zeros((10,7), dtype=float)
    env.reset(q_table)

    for ep in train_eps:
        p0 = ep[0]["prediction"]
        pT = ep[-1]["next_prediction"]
        q_table = env.run_episode(p0, pT)

    top2 = np.partition(q_table, -2, axis=1)[:, -2:]
    gaps = top2[:,1] - top2[:,0]
    mean_gap = float(np.mean(gaps))

    return q_table, {
        "visit_counts": env.visit_counts.copy(),
        "mean_action_gap": mean_gap
    }

def evaluate_metrics(q_table: np.ndarray, val_eps) -> Dict[str, Any]:
    """Greedy‐replay held‐out episodes to compute:
       - mean/std return
       - mean/std regret
       - mean/std capture fraction
       - mean/std policy entropy
       - eval visit_counts heatmap
    """
    
    n_states, n_actions = q_table.shape
    eval_visits = np.zeros_like(q_table, dtype=int)

    returns, regrets, captures, entropies = [], [], [], []

    for ep in val_eps:
        prev_state = drift.discretize_probability(ep[0]["prediction"])
        G_agent = 0.0
        G_opt = 0.0
        H_acc = 0.0
        T = 0

        for step in ep:
            # Greedy policy & entropy
            probs = np.zeros(n_actions, dtype=float)
            best_a = int(np.argmax(q_table[prev_state]))
            probs[best_a] = 1.0
            H_acc += -np.sum(probs * np.log(probs + 1e-12))

            # replay transition
            p_next = step["next_prediction"]
            # no noise at eval time
            s_next = drift.discretize_probability(p_next)
            r = _compute_reward(prev_state, s_next)

            # agent return
            G_agent += r

            # oracle return: pick action maximizing immediate reward
            # (since future reward depends on subsequent states, this is
            #  actually a one‐step regret metric)
            r_opts = []
            for a2 in range(n_actions):
                noisy_p2 = p_next + drift.action_to_noise(a2)
                s2 = drift.discretize_probability(np.clip(noisy_p2, 0.0, 1.0))
                r_opts.append(_compute_reward(prev_state, s2))
            G_opt += max(r_opts)

            # count visits
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

def run_pure(train_eps, val_eps):
    for explorer in get_action_funcs():
        q_table, train_metrics = train_q_table(explorer, train_eps)
        eval_metrics = evaluate_metrics(q_table, val_eps)

        print(f"\n=== Strategy: {explorer.__name__} ===")
        print(">>> Training metrics:")
        print(f"  Mean action‐gap: {train_metrics['mean_action_gap']:.3f}")
        print("  Visit counts:\n", train_metrics["visit_counts"])
        print("\n>>> Evaluation metrics:")
        print(f"  Return    = {eval_metrics['mean_return']:.3f} ±{eval_metrics['std_return']:.3f}")
        print(f"  Regret    = {eval_metrics['mean_regret']:.3f} ±{eval_metrics['std_regret']:.3f}")
        print(f"  Capture % = {eval_metrics['mean_capture']:.3f} ±{eval_metrics['std_capture']:.3f}")
        print(f"  Entropy   = {eval_metrics['mean_entropy']:.3f} ±{eval_metrics['std_entropy']:.3f}")
        print("  Eval visits:\n", eval_metrics["eval_visit_counts"])

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
            q_table = np.zeros((10, 7), dtype=float)
            env.reset(q_table)

            for episode in train_episodes:
                p_start = episode[0]["prediction"]
                p_end = episode[-1]["next_prediction"]
                q_table = env.run_episode(p_start, p_end)

            mean, std = evaluate_q_table(q_table, val_episodes)
        return # this returns early, fine for now, just want to test it

def run_comparison():
    episodes = gen.get_episodes()
    random.shuffle(episodes)
    split = int(len(episodes)*TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")
    run_pure(train_eps, val_eps)