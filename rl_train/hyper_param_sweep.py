import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple

import warnings
from sklearn.exceptions import InconsistentVersionWarning

from rl_data_gen.generate import load_model, FEATURE_NAMES, extract_anchor_pairs
from rl_train.env import HeartDiseaseEnv
from rl_train.online_train import q_learning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)

ALPHAS = [0.05, 0.1, 0.2]
GAMMAS = [0.8, 0.9, 0.99]
EPSILON = 1.0
EPS_DECAY = 0.995
NUM_EPISODES = 1000
EVAL_EPISODES = 20

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "Heart_Disease_Prediction.csv"

OUTPUT_CSV = Path(__file__).resolve().parent /"results"/ "sweep_results.csv"

def evaluate_env_policy(env: HeartDiseaseEnv, Q: np.ndarray, n_runs: int) -> float:
    total_reward = 0.0
    for seed in range(n_runs):
        state, _ = env.reset(seed=seed)
        done = False
        R = 0.0
        while not done:
            action = int(np.argmax(Q[state]))
            next_state, reward, done, truncated, _ = env.step(action)
            R += reward
            state = next_state
        total_reward += R
    return total_reward / n_runs if n_runs else 0.0


def main():
    anchors = extract_anchor_pairs(DATASET_PATH)
    print(f"Loaded {len(anchors)} anchor pairs.")

    model = load_model()
    env = HeartDiseaseEnv(
        anchors=anchors,
        model=model,
        feature_names=FEATURE_NAMES,
    )

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["alpha", "gamma", "avg_eval_reward"])

        best_reward = -float('inf')
        best_cfg: Tuple[float, float] = (0.0, 0.0)

        for alpha in ALPHAS:
            for gamma in GAMMAS:
                print(f"Training Q-table (alpha={alpha}, gamma={gamma})...")
                Q = q_learning(
                    env,
                    num_episodes=NUM_EPISODES,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=EPSILON,
                    eps_decay=EPS_DECAY,
                )

                avg_reward = evaluate_env_policy(env, Q, EVAL_EPISODES)
                print(f" -> Eval reward: {avg_reward:.3f}")

                writer.writerow([alpha, gamma, f"{avg_reward:.4f}"])

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_cfg = (alpha, gamma)

    print(f"\nBest config: alpha={best_cfg[0]}, gamma={best_cfg[1]}, reward={best_reward:.3f}")
    print(f"Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    print("called.")
    main()
