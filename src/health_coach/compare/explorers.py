import random
import numpy as np

from typing import Callable, List, Dict, Any
import health_coach.config as cfg
import health_coach.rl_data_gen.generate as gen

from health_coach.compare.train import train_q_table
from health_coach.compare.eval import evaluate_metrics
from health_coach.compare.explorer_factories import get_factories

class ExplorerTuner:
    def __init__(
        self,
        name: str,
        factory: Callable[[float], Callable[[int, np.ndarray], int]],
        param_name: str,
        param_range: tuple[float, float]
    ):
        self.name        = name
        self.factory     = factory
        self.param_name  = param_name
        self.param_range = param_range

    def sample(self) -> float:
        low, high = self.param_range
        return random.uniform(low, high)

    def tune(self, train_eps: List[List[float]], val_eps:   List[List[float]]) -> float:
        best_score = -float("inf")
        best_val = None

        print(f"\nTuning {self.name} ({self.param_name}) "
              f"in range {self.param_range}:")

        for i in range(cfg.N_TRIALS):
            v = self.sample()
            fn = self.factory(v)
            scores = []

            for seed in range(cfg.N_SEEDS):
                random.seed(seed)
                np.random.seed(seed)

                q_table, _  = train_q_table(fn, train_eps,  cfg.ALPHA, cfg.GAMMA)
                metrics = evaluate_metrics(q_table, val_eps)
                scores.append(metrics["mean_normalized_capture_ratio"])

            mean_score = sum(scores) / len(scores)
            print(f"  Trial {i+1:2d}: {self.param_name}={v:.4f} -> "
                  f"mean_norm_capture={mean_score:.4f}")

            if mean_score > best_score:
                best_score, best_val = mean_score, v

        print(f"-> Best for {self.name}: {self.param_name}={best_val:.4f} "
              f"with score={best_score:.4f}")
        return best_val

def tune_explorers():
    num_trend = int(cfg.NUM_EPISODES * 2/3)
    num_random = int(cfg.NUM_EPISODES * 1/3)
    episodes = gen.get_episodes(num_trend=num_trend, num_random=num_random)
    random.shuffle(episodes)
    split = int(len(episodes) * cfg.TRAIN_FRACTION)
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Train eps: {len(train_eps)}, Val eps: {len(val_eps)}")

    factories = get_factories()

    tuners = [
        ExplorerTuner(
            name="epsilon_greedy_fn",
            factory= factories["epsilon_greedy_fn"],
            param_name="decay_rate",
            param_range=(0.995, 0.9999)
        ),
        # ExplorerTuner(
        #     name="softmax_fn",
        #     factory=make_softmax_fn,
        #     param_name="temperature",
        #     param_range=(0.1, 5.0)
        # ),
        # ExplorerTuner(
        #     name="ucb_fn",
        #     factory=make_ucb_fn,
        #     param_name="c",
        #     param_range=(0.1, 2.0)
        # ),
        # TODO: Later add other explorers
    ]

    tuned: Dict[str, Dict[str, float]] = {}
    for tuner in tuners:
        best_val = tuner.tune(train_eps, val_eps)
        tuned[tuner.name] = {tuner.param_name: best_val}

    print(f"Tuning Results:\n{tuned}")
