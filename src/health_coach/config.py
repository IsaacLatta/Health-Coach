Q_STATES: int = 10
Q_ACTIONS: int = 7

STATE_TO_NOISE_FACTOR: float = 0.08
NOISE_STD: float = (1 / Q_STATES) * STATE_TO_NOISE_FACTOR

NUM_EPISODES: int = 5000
EPISODE_LENGTH = 100
TRAIN_FRACTION: float = 0.8
EVAL_FRACTION: float = 0.2

EPSILON_START: float = 1.0
EPSILON_END: float = 0.01
EPSILON_DECAY_RATE: float = (EPSILON_END / EPSILON_START) ** (1.0 / (TRAIN_FRACTION * NUM_EPISODES - 1))

EPSILON_CURRENT: float = EPSILON_START

def update_epsilon() -> None:
    global EPSILON_CURRENT
    EPSILON_CURRENT = max(EPSILON_END, EPSILON_CURRENT * EPSILON_DECAY_RATE)

def print_config() -> None:
    print("=== Configuration for RL Comparison ===\n"
        f"  Environment:\n"  
        f"    Action Noise STD: {NOISE_STD}\n"    
        f"    Q States: {Q_STATES}\n"
        f"    Q Actions: {Q_ACTIONS}\n"
        f"  Episodes: {NUM_EPISODES}\n"
        f"    Length: {EPISODE_LENGTH}\n"
        f"    Train: {TRAIN_FRACTION*100}\n"
        f"    Eval: {EVAL_FRACTION*100}\n"
        "\n"
        f"  Explorers:\n"
        f"    Epsilon Greedy:\n"
        f"      Epsilon Start: {EPSILON_START}\n"
        f"      Epsilon End: {EPSILON_END}\n"
        f"      Epsilon Decay Rate: {EPSILON_DECAY_RATE}\n")