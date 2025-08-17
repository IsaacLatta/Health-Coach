import random

# LLM_MODEL: str = "ollama/gemma3:latest"
# LLM_MODEL: str = "ollama/qwen3:1.7b"
LLM_MODEL: str = "ollama/qwen3:4b"
LLM_BASE_URL: str = "http://192.168.1.202:80"
# LLM_BASE_URL: str = "http://127.0.0.1:11434"

Q_STATES: int = 30
Q_ACTIONS: int = 7

STATE_TO_NOISE_FACTOR: float = 0.1
NOISE_STD: float = (1 / Q_STATES) * STATE_TO_NOISE_FACTOR

NUM_EPISODES: int = 5
EPISODE_LENGTH = 5
TRAIN_FRACTION: float = 0.8
EVAL_FRACTION: float = 0.2

# Global hyperparams
ALPHA: float = 0.5
GAMMA: float = 0.5
SHOULD_SAMPLE_HYPERPARAMS: bool = True

ALPHA_RANGE: float = (0.01, 1.0)
GAMMA_RANGE: float = (0.90, 1.0)
N_TRIALS = 10
N_SEEDS = 3

# Explorer params
SOFTMAX_TEMP: float = 1.0
MAXENT_ALPHA: float =  0.5
THOMPSON_SIGMA: float = 0.1
UCB_C: float = 1.0

EPSILON_START: float = 1.0
EPSILON_END: float = 0.01
EPSILON_DECAY_RATE: float = (EPSILON_END / EPSILON_START) ** (1.0 / (TRAIN_FRACTION * NUM_EPISODES - 1))

EPSILON_CURRENT: float = EPSILON_START

def update_epsilon() -> None:
    global EPSILON_CURRENT
    EPSILON_CURRENT = max(EPSILON_END, EPSILON_CURRENT * EPSILON_DECAY_RATE)

def sample_hyperparams():
    return (random.uniform(*ALPHA_RANGE), random.uniform(*GAMMA_RANGE))

def print_config() -> None:
    print("=== Configuration for RL Comparison ===\n"
          "  Environment:\n"
          f"    Action Noise STD: {NOISE_STD}\n"
          f"    Q States: {Q_STATES}\n"
          f"    Q Actions: {Q_ACTIONS}\n"
          "  Episodes:\n"
          f"    Total: {NUM_EPISODES}\n"
          f"    Length: {EPISODE_LENGTH}\n"
          f"    Train Fraction: {TRAIN_FRACTION*100:.1f}%\n"
          f"    Eval  Fraction: {EVAL_FRACTION*100:.1f}%\n")

    print("  Global Hyperparameters:")
    if SHOULD_SAMPLE_HYPERPARAMS:
        print(f"    α ∈ [{ALPHA_RANGE[0]:.2f}, {ALPHA_RANGE[1]:.2f}], "
              f"γ ∈ [{GAMMA_RANGE[0]:.2f}, {GAMMA_RANGE[1]:.2f}]")
        print(f"    Trials: {N_TRIALS}, Seeds per trial: {N_SEEDS}")
    else:
        print(f"    α = {ALPHA:.4f}, γ = {GAMMA:.4f}")
    print()

    print("  Explorer Parameters (fixed):")
    print("    Epsilon-Greedy:")
    print(f"      ε start = {EPSILON_START:.2f}, "
          f"ε end = {EPSILON_END:.2f}, "
          f"decay = {EPSILON_DECAY_RATE:.6f}")
    print(f"    Softmax temperature = {SOFTMAX_TEMP:.4f}")
    print(f"    MaxEnt α = {MAXENT_ALPHA:.4f}")
    print(f"    Thompson σ = {THOMPSON_SIGMA:.4f}")
    print(f"    UCB c = {UCB_C:.4f}")
    print()
