import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.9
DEFAULT_EPSILON = 1.0
DEFAULT_EPS_DECAY = 0.999

ACTIONS = [
    "INCREASE_MODERATE_THRESHOLD",
    "DECREASE_MODERATE_THRESHOLD",
    "INCREASE_HIGH_THRESHOLD",
    "DECREASE_HIGH_THRESHOLD",
    "INCREASE_TOP_K",
    "DECREASE_TOP_K",
    "NOP",
]
N_STATES  = 10
N_ACTIONS = len(ACTIONS)


def load_episodes(folder: Path) -> List[List[Dict[str, Any]]]:
    episodes = []
    for fp in sorted(folder.glob("*.csv")):
        df = pd.read_csv(fp)
        episodes.append(df.to_dict(orient="records"))
    return episodes


def train_q_table(
    episodes: List[List[Dict[str, Any]]],
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA
) -> np.ndarray:
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=float)
    for ep in episodes:
        for step in ep:
            s      = int(step["state"])
            a_idx  = ACTIONS.index(step["action"])
            r      = float(step["reward"])
            s_next = int(step["next_state"])
            best_next = Q[s_next].max()
            td_error = (r + gamma * best_next) - Q[s, a_idx]
            Q[s, a_idx] += alpha * td_error
    return Q


def evaluate_policy(
    Q: np.ndarray,
    episodes: List[List[Dict[str, Any]]]
) -> float:
    total_reward = 0.0
    for ep in episodes:
        R = 0.0
        for step in ep:
            R += float(step["reward"])
        total_reward += R
    return total_reward / len(episodes) if episodes else 0.0

def offline_train(
    train_dir: str = "../data/synthetic/train",
    val_dir:   str = "../data/synthetic/val",
    alpha:     float = DEFAULT_ALPHA,
    gamma:     float = DEFAULT_GAMMA
):
    train_eps = load_episodes(Path(train_dir))
    val_eps = load_episodes(Path(val_dir))

    print(f"Loaded {len(train_eps)} training and {len(val_eps)} validation episodes.")
    Q = train_q_table(train_eps, alpha=alpha, gamma=gamma)

    r_train = evaluate_policy(Q, train_eps)
    r_val   = evaluate_policy(Q, val_eps)
    print(f"Average cumulative reward (TRAIN): {r_train:.3f}")
    print(f"Average cumulative reward (VAL):   {r_val:.3f}")

    out_path = Path(__file__).resolve().parents[1] / "models/"/"q_table.npy"
    np.save(out_path, Q)
    print(f"Saved Q-table to {out_path}")

if __name__ == "__main__":
    offline_train()
