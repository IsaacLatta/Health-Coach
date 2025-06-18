import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def q_learning(
    env,
    num_episodes,
    alpha: float,
    gamma: float,
    epsilon: float,
    eps_decay: float
) -> np.ndarray:
    
    # initialize Q-table
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep)
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            # interact with environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-update
            best_next = Q[next_state].max()
            td_target = reward + gamma * best_next
            td_error  = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        # decay exploration
        epsilon = max(0.01, epsilon * eps_decay)

    return Q