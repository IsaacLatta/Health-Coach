# compare.py
import random
import itertools
import statistics
from typing import List, Dict

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

from .env import AgentQLearningEnvironment, PureQLearningEnvironment, generate_base_trajectory

TRAIN_FRACTION = 0.8
OFF_ALPHA = 0.1
OFF_GAMMA = 0.9
EPSILON = 1.0
EPS_DECAY = 0.1
RUNS = 1

def run_pure(train_episodes, val_episodes):

    for explorer in get_action_funcs():
        env = PureQLearningEnvironment(
                explorer,
                drift.discretize_probability, 
                drift.action_to_noise,
                _compute_reward,
                OFF_GAMMA,
                OFF_ALPHA,
                )
        
        for seed in range(RUNS):
            q_table = np.zeros((10, 7), dtype=float)
            env.reset(q_table)
            
            for episode in train_episodes:
                p_start = episode[0]["prediction"]
                p_end = episode[-1]["next_prediction"]
                q_table = env.run_episode(p_start, p_end)
        
        print(f"Explorer strategy {explorer.__name__} QTable:\n{q_table}\n")            

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

                env.run_episode(p_start, p_end)
                # trajectory = generate_base_trajectory(p_start, p_end)
                # print(f"TRAJECTORY of Length({len(trajectory)})\n{trajectory}")

            Q_agent = env.q_table
            print(f"QTable\n {Q_agent}")
            # score = eval_offline(Q_agent, val_episodes)
        return # this returns early, fine for now, just want to test it

def run_comparison():
    episodes = gen.get_episodes()
    random.shuffle(episodes)
    split = int(len(episodes)*TRAIN_FRACTION)
    train_episodes, val_episodes = episodes[:split], episodes[split:]

    print(f"Train eps: {len(train_episodes)}, Val eps: {len(val_episodes)}")

    # run_agent(train_episodes, val_episodes)
    run_pure(train_episodes, val_episodes)
    return