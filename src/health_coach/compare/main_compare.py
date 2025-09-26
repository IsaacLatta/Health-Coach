import random
import argparse
import health_coach.config as cfg
import health_coach.compare.compare as compare
import health_coach.compare.rl_data_gen.generate as gen

from health_coach.compare.explorers import tune_explorers
from health_coach.backend.flows.rl.tools.exploration import get_all_tool_funcs

import csv
from health_coach.backend.flows.rl.rl_flow import call_rl_flow
from health_coach.backend.flows.insights.insights_flow import call_insights_flow
from health_coach.compare.rl_data_gen.generate import get_episodes
from health_coach.compare.env import DriftOfflineEnvironment
from health_coach.backend.flows.rl.dependencies import RLDeps
from health_coach.backend.flows.insights.dependencies import InsightsDeps
from health_coach.backend.services.prediction import MockPredictionService
from health_coach.backend.services.context import InMemContextService
from health_coach.backend.stores.qtable import SQLiteQTables
from health_coach.backend.stores.transitions import SQLiteTransitions
from health_coach.backend.stores.config import SQLiteConfigs
from health_coach.backend.services.rl import PureRLService
import health_coach.config as cfg

import json

def run_logging_loop(csv_path: str = "/home/isaac/Projects/Health-Coach/results/rl_insights_log.csv"):
    reward_fn = lambda prev, cur: 1.0 if cur < prev else -1.0 if cur > prev else 0.0
    rl_service = PureRLService(reward_fn, *cfg.SOFTMAX_HYPERPARAMS)

    cfg.print_config()

    # --- Episodes first
    episodes = get_episodes(num_trend=int(cfg.NUM_EPISODES*2/3),
                            num_random=int(cfg.NUM_EPISODES*1/3))

    # Now mock pred can replay them
    mock_pred = MockPredictionService(episodes)

    # --- Build deps
    rl_deps = (
        RLDeps.make()
        .with_qtables(SQLiteQTables(cfg.Q_STATES, cfg.Q_ACTIONS))
        .with_transitions(SQLiteTransitions())
        .with_prediction(mock_pred)
        .with_context(InMemContextService())
        .with_rl(rl_service)
        .with_configs(SQLiteConfigs())
        .ensure()
    )

    insights_deps = (
        InsightsDeps.make()
        .with_prediction(mock_pred)
        .with_configs(SQLiteConfigs())
        .with_transitions(SQLiteTransitions())
        .with_qtables(SQLiteQTables(cfg.Q_STATES, cfg.Q_ACTIONS))
        .with_context(InMemContextService())
        .ensure()
    )

    # --- CSV header
    fieldnames = [
        "episode", "step", "prob", "state", "action", "reward",
        "step_ctx", "received_reward", "visit_count", "action_count", "action_ratio",
        "reward_mean", "reward_variance", "reward_trend",
        "chosen_q_value", "mean_q_value", "q_gap", "q_advantage",
        "entropy", "td_error_mean", "td_error_variance",
        "narrative"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        env = DriftOfflineEnvironment(
            state_mapper=lambda p: int(p * cfg.Q_STATES),
            action_to_noise_mapper=lambda a: 0.01 * (a or 0),
            reward_function=reward_fn,
        )

        for epi_idx, ep in enumerate(episodes):
            env.reset(ep)
            done, action = False, -1
            step = 0
            while not done:
                next_state, reward, done = env.step(action)

                # Pure RL step
                rl_res = call_rl_flow(rl_deps, f"p{epi_idx}", env.prev_state, next_state)
                action = rl_res["action"]

                # Current prob + state (replayed from trajectory)
                prob = ep[min(step, len(ep)-1)]
                state = int(prob * cfg.Q_STATES)

                # Insights
                features = {"id": f"p{epi_idx}"}
                insights_res = call_insights_flow({"id": f"p{epi_idx}", "features": features}, insights_deps)

                # Context snapshot
                ctx_json = rl_res.get("context_json")
                ctx_metrics = {}
                if ctx_json:
                    try:
                        ctx_metrics = json.loads(ctx_json)
                    except Exception:
                        pass

                narrative = insights_res.get("summary", {}).get("narrative", "")

                row = {
                    "episode": epi_idx,
                    "step": step,
                    "prob": prob,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "narrative": narrative,
                }
                for k in fieldnames:
                    if k in ctx_metrics:
                        row[k] = ctx_metrics[k]

                writer.writerow(row)
                step += 1




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
    
    parser.add_argument('--logloop', action='store_true',
                   help='Run RL+Insights logging loop')

    args = parser.parse_args()
    
    if args.all:
        print("Running all comparisons...")
        run_explorer_optimization()
        run_pure()
        run_agent()
    else:
        if args.logloop:
            print("Running log loop ...")
            run_logging_loop()
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