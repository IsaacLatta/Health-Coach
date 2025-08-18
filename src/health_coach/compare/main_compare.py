import random
import health_coach.config as cfg
import health_coach.compare.compare as compare
import health_coach.backend.agents.flows.rl.tools.exploration as exploration
import health_coach.rl_data_gen.generate as gen
from health_coach.compare.explorers import tune_explorers

def run_pure():
    cfg.print_config()

    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)
    random.shuffle(episodes)
    split = int(len(episodes) * cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")
    compare.run_pure_comparison(train_eps, val_eps,  exploration.get_all_funcs())

def run_agent():
    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)
    random.shuffle(episodes)
    split = int(len(episodes) * cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")
    compare.run_agent_comparison(train_eps, val_eps, exploration.get_all_tools())
    