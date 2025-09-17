import random
import argparse
import health_coach.config as cfg
import health_coach.compare.compare as compare
import health_coach.compare.rl_data_gen.generate as gen

from health_coach.compare.explorers import tune_explorers
from health_coach.backend.flows.rl.tools.exploration import get_all_tool_funcs

def run_pure():
    cfg.print_config()

    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)
    random.shuffle(episodes)
    split = int(len(episodes) * cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")
    compare.run_pure_comparison(train_eps, val_eps,  get_all_tool_funcs())

def run_agent():
    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)
    random.shuffle(episodes)
    split = int(len(episodes) * cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")
    compare.run_agent_comparison(train_eps, val_eps, get_all_tool_funcs())

def run_explorer_optimization():
    tune_explorers()

def main():
    parser = argparse.ArgumentParser(description='Health Coach Comparison Harness')
    
    parser.add_argument('--explorer', action='store_true', 
                       help='Run explorer optimization')
    parser.add_argument('--pure', action='store_true', 
                       help='Run pure comparison')
    parser.add_argument('--agent', action='store_true', 
                       help='Run agent comparison')
    parser.add_argument('--all', action='store_true', 
                       help='Run all comparisons (explorer, pure, and agent)')
    
    args = parser.parse_args()
    
    if not any([args.explorer, args.pure, args.agent, args.all]):
        print("No arguments provided, running agent comparison (default)")
        run_agent()
        return
    
    if args.all:
        print("Running all comparisons...")
        run_explorer_optimization()
        run_pure()
        run_agent()
    else:
        if args.explorer:
            print("Running explorer optimization...")
            run_explorer_optimization()
        if args.pure:
            print("Running pure comparison...")
            run_pure()
        if args.agent:
            print("Running agent comparison...")
            run_agent()

if __name__ == "__main__":
    main()