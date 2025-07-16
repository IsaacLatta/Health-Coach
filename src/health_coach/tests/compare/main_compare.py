# compare.py
import random
import itertools
import statistics
from typing import List, Dict

from health_coach.rl_data_gen.generate import extract_anchor_pairs, generate_episodes, load_model
from health_coach.rl_train.env import HeartDiseaseEnv
from health_coach.rl_train.offline_train import train_q_table, evaluate_policy as eval_offline
from health_coach.rl import QLearningEngine
from health_coach.tools.rl_tools.action import (
    get_all_tools as get_action_tools,
    get_all_tool_funcs as get_action_funcs
)
from health_coach.tools.rl_tools.context import get_all_tools as get_context_tools
from health_coach.tools.rl_tools.reward import get_all_tools as get_reward_tools
from health_coach.flows import RLFlow, RLInput

TRAIN_FRAC = 0.8
OFF_ALPHA = 0.1
OFF_GAMMA = 0.9
EPSILON = 1.0
EPS_DECAY = 0.995
NUM_OFF_EPISODES = 500
RUNS = 5

def run_pure(train_eps, val_eps):
    results: List[Dict] = []

    for explorer in get_action_funcs():
        scores = []
        for seed in range(RUNS):
            Q = train_q_table(
                episodes=train_eps,
                alpha=OFF_ALPHA,
                gamma=OFF_GAMMA,
                explorer=explorer,
                epsilon=EPSILON,
                eps_decay=EPS_DECAY,
                num_episodes=NUM_OFF_EPISODES,
                seed=seed
            )
            score = eval_offline(Q, val_eps)
            scores.append(score)
        results.append({
            'mode': 'pure',
            'explorer': explorer.__name__,
            'mean_score': statistics.mean(scores),
            'std_score': statistics.stdev(scores),
        })

        return results

def run_agent(train_eps, val_eps):

    results: List[Dict] = []

    action_tools = get_action_tools()
    context_tools = get_context_tools()
    reward_tools  = get_reward_tools()

    for action_fn, context_fn, reward_fn in itertools.product(
        action_tools, context_tools, reward_tools
    ):
        scores = []
        for seed in range(RUNS):
            
            engine = QLearningEngine(
                exploration_tools=[action_fn],
                context_tools=[context_fn],
                shaping_tools=[reward_fn],
                alpha=OFF_ALPHA,
                gamma=OFF_GAMMA,
            )

            flow = RLFlow()
            flow.set_rl_implementation(engine)

            for ep in train_eps:
                flow.set_input(RLInput(**ep))
                flow.kickoff()

            Q_agent = engine.q_state.q_table
            score = eval_offline(Q_agent, val_eps)
            scores.append(score)

        results.append({
            'mode':   'agent',
            'action':  action_fn.__name__,
            'context': context_fn.__name__,
            'reward':  reward_fn.__name__,
            'mean_score': statistics.mean(scores),
            'std_score':  statistics.stdev(scores),
        })
        return results

def run_comparison() -> List[Dict]:
    model = load_model()
    anchors = extract_anchor_pairs()
    episodes = generate_episodes(anchors, model)
    random.shuffle(episodes)
    split = int(len(episodes) * TRAIN_FRAC)
    train_eps, val_eps = episodes[:split], episodes[split:]

    results: List[Dict] = []

    pure_results = run_pure(train_eps, val_eps)
    results.append(pure_results)

    agent_results = run_agent(train_eps, val_eps)
    results.append(agent_results)

    return results
