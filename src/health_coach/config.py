import random
import os

# LLM_MODEL: str = "ollama/qwen3:4b"
# LLM_BASE_URL: str = "http://localhost:11434"

ACTIONS = [
    "inc_top_k",       # 0
    "dec_top_k",       # 1
    "lower_moderate",  # 2
    "raise_moderate",  # 3
    "lower_high",      # 4
    "raise_high",      # 5
]

Q_STATES: int = 10
Q_ACTIONS = int(os.getenv("Q_ACTIONS", "6"))

STATE_TO_NOISE_FACTOR: float = 0.1
NOISE_STD: float = (1 / Q_STATES) * STATE_TO_NOISE_FACTOR

NUM_EPISODES: int = 5
EPISODE_LENGTH = 5
TRAIN_FRACTION: float = 0.8
EVAL_FRACTION: float = 0.2

COMPARISION_RESULTS_DIR="../results"
COMPARISON_RESULTS_FILE="my_results.json"
ALPHA: float = 0.4
GAMMA: float = 0.9
SHOULD_SAMPLE_HYPERPARAMS: bool = True

ALPHA_RANGE: float = (0.01, 1.0)
GAMMA_RANGE: float = (0.90, 1.0)
N_TRIALS = 10
N_SEEDS = 3

SOFTMAX_TEMP: float = 1.0
MAXENT_ALPHA: float =  0.5
THOMPSON_SIGMA: float = 0.1
UCB_C: float = 1.0

EPSILON_START: float = 1.0
EPSILON_END: float = 0.01
EPSILON_DECAY_RATE: float = (EPSILON_END / EPSILON_START) ** (1.0 / (TRAIN_FRACTION * NUM_EPISODES - 1))

EPSILON_CURRENT: float = EPSILON_START

SOFTMAX_HYPERPARAMS: tuple[float, float] = (0.5076, 0.9117)
SOFTMAX_TEMP: float = 4.784567

MAXENT_ALPHA: float =  1.29426
MAXENT_HYPERPARAMS: tuple[float, float] = (0.9180, 0.9052)

THOMPSON_HYPERPARAMS: tuple[float, float] = (0.4053, 0.9168)
THOMPSON_SIGMA: float = 0.18264

COUNT_BONUS_HYPERPARAMS: tuple[float, float] = (0.6769, 0.9145)
COUNT_BONUS_BETA: float = 3.72038

UCB_C: float = 1.916465
UCB_HYPERPARAMS: tuple[float, float] = (0.3113, 0.9119)

EPSILON_GREEDY_HYPERPARAMS: tuple[float, float] = (0.0366, 0.9836)
EPSILON_START: float = 1.0
EPSILON_END: float = 0.01
# EPSILON_DECAY_RATE: float = (EPSILON_END / EPSILON_START) ** (1.0 / (TRAIN_FRACTION * NUM_EPISODES - 1))
EPSILON_DECAY_RATE: float = 0.9965

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

DEFAULT_REPORT_CFG = {
    "moderate": float(os.getenv("THRESH_MODERATE", "0.33")),
    "high":     float(os.getenv("THRESH_HIGH", "0.66")),
    "top_k":    int(os.getenv("REPORT_TOP_K", "5")),
}

def default_patient_config() -> dict:
    """Return a fresh copy of the default report config."""
    return dict(DEFAULT_REPORT_CFG)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def apply_config_action(cfg: dict, action_idx: int) -> dict:
    n = max(1, Q_ACTIONS)
    a = int(action_idx) % n

    new = dict(cfg or DEFAULT_REPORT_CFG)
    step_k = int(os.getenv("TOP_K_STEP", "1"))
    step_t = float(os.getenv("THRESH_STEP", "0.02"))

    k  = int(new.get("top_k", DEFAULT_REPORT_CFG["top_k"]))
    m  = float(new.get("moderate", DEFAULT_REPORT_CFG["moderate"]))
    hi = float(new.get("high", DEFAULT_REPORT_CFG["high"]))

    if a == 0:  # inc_top_k
        k = _clamp(k + step_k, 1, 10)
    elif a == 1:  # dec_top_k
        k = _clamp(k - step_k, 1, 10)
    elif a == 2:  # lower_moderate
        m = _clamp(m - step_t, 0.0, 0.98)
        m = min(m, hi - 0.01)
    elif a == 3:  # raise_moderate
        m = _clamp(m + step_t, 0.0, 0.98)
        m = min(m, hi - 0.01)
    elif a == 4:  # lower_high
        hi = _clamp(hi - step_t, 0.01, 0.99)
        hi = max(hi, m + 0.01)
    elif a == 5:  # raise_high
        hi = _clamp(hi + step_t, 0.01, 0.99)
        hi = max(hi, m + 0.01)

    new["top_k"] = int(k)
    new["moderate"] = float(m)
    new["high"] = float(hi)
    print(f"\nGenerated new config: {new}\n")
    return new